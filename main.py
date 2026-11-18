"""Forecast a launch day's Twitch viewers hour by hour.

Pure Python standard library, no network, no clock, no randomness.

Method: a pooled hour-fixed-effects regression over past big launches
("comparables") shipped with each case. For every clean comparable and
every hour h = 2..24 the target is the log ratio log(viewers_h /
viewers_1). The regression has one free intercept per hour, a shared
slope on centred log(viewers_1) with a linear hour modulation, and two
launch-level features: log(channels at hour 1) and log(viewers_1 /
pre-launch 7-day median baseline), both centred on their training
medians. The forecast adds the hour's median training residual to the
fitted log ratio; the spread is the hour's residual standard deviation
(floored at 0.20 log units, scaled by 0.5), applied with the
standard-normal quantiles. The target's features are clamped to the
training range, so an extreme hour-1 print cannot extrapolate wildly.

Why those two extra features: how explosive a launch is relative to
its own pre-launch baseline, and how concentrated its hour-1 audience
is across channels, predict the day's shape (overnight dip, evening
ramp) far better than hour-1 size alone.

Clean comparables are the ones whose unlock note classifies them as a
plain paid launch (Tier: comparable): no early access, no
free-to-play or live-service launch, and a real launch day (the GTA VI
category itself is not a launch). Fallbacks keep every case valid: if
the pooled fit fails, fall back to a per-hour OLS on log(viewers_1)
(round 1's model), then to the house baseline's persistence forecast.

I/O: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for
hours 2..24, quantiles finite, > 0, <= 1e9, non-decreasing.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))
SPREAD_SCALE = 0.5
SPREAD_FLOOR = 0.20
ROUND1_SPREAD_SCALE = 0.8
PERSIST_SIGMA = 0.15


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_buckets(points, day_start, time_key="from"):
    """Group points by launch-day hour: hour 1 = [day_start, +1h)."""
    buckets = {}
    for p in points:
        off = (parse_time(p[time_key]) - day_start).total_seconds() / 3600.0
        h = int(math.floor(off)) + 1
        if h < 1:
            continue
        buckets.setdefault(h, []).append(p)
    return buckets


def mean_viewers(points):
    vals = [max(1, p["viewers"]) for p in points]
    return sum(vals) / len(vals)


def mean_channels(points):
    vals = [max(0, p.get("channels", 0)) for p in points]
    return sum(vals) / len(vals) if vals else 0.0


def baseline_median(points, day_start, time_key="from"):
    """Median viewers over the 7 days before the launch day."""
    start = day_start - timedelta(days=7)
    vals = [max(1, p["viewers"]) for p in points
            if start <= parse_time(p[time_key]) < day_start]
    if not vals:
        return None
    return statistics.median(vals)


def is_clean(comp):
    """Clean comparable: rolling-midnight launch, no early access, not weak.

    Classified from the comparable's unlock text: Tier: comparable,
    unlock at midnight local (rolling) -- the day-1 shape GTA VI is
    expected to follow -- no early-access contamination and no campaign
    early access (their hour-1 is not a cold launch), no weak analogues.
    """
    unlock = comp.get("unlock", "")
    if "Tier: comparable" not in unlock:
        return False
    if "Tier: weak" in unlock:
        return False
    if "midnight local" not in unlock.lower():
        return False
    if "early-access-contaminated" in unlock:
        return False
    low = unlock.lower()
    if "early access" in low and "no early access" not in low:
        return False
    return True


def extract_comparable(comp):
    """Pull the regression inputs for one comparable, or None to skip it."""
    if not is_clean(comp):
        return None
    day_start = parse_time(comp["day_start"])
    buckets = hour_buckets(comp["hourly"], day_start)
    if 1 not in buckets:
        return None
    v1 = mean_viewers(buckets[1])
    c1 = mean_channels(buckets[1])
    if v1 <= 0 or c1 <= 0:
        return None
    base = baseline_median(comp["hourly"], day_start)
    if base is None or base <= 0:
        return None
    rows = {}
    for h in HOURS:
        if h in buckets:
            rows[h] = math.log(mean_viewers(buckets[h]) / v1)
    if len(rows) < 12:
        return None
    return {"u": math.log(v1), "ch": math.log(c1),
            "base": math.log(v1 / base), "y": rows}


def solve_linear(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for k in range(col, n + 1):
                    m[r][k] -= f * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_pooled(comps):
    """Pooled OLS: 23 hour intercepts + u, u*t, ch, base slopes."""
    u0 = statistics.median([c["u"] for c in comps])
    ch0 = statistics.median([c["ch"] for c in comps])
    b0 = statistics.median([c["base"] for c in comps])
    npar = len(HOURS) + 4
    xtx = [[0.0] * npar for _ in range(npar)]
    xty = [0.0] * npar
    for c in comps:
        u = c["u"] - u0
        ch = c["ch"] - ch0
        bs = c["base"] - b0
        for h, y in c["y"].items():
            t = (h - 13) / 11.0
            row = [0.0] * npar
            row[h - 2] = 1.0
            row[23] = u
            row[24] = u * t
            row[25] = ch
            row[26] = bs
            for i in range(npar):
                if row[i] == 0.0:
                    continue
                ri = row[i]
                for j in range(npar):
                    xtx[i][j] += ri * row[j]
                xty[i] += ri * y
    for i in range(npar):
        xtx[i][i] += 1e-9
    beta = solve_linear(xtx, xty)
    med = {}
    sig = {}
    for h in HOURS:
        res = []
        for c in comps:
            if h not in c["y"]:
                continue
            u = c["u"] - u0
            ch = c["ch"] - ch0
            bs = c["base"] - b0
            t = (h - 13) / 11.0
            pred = (beta[h - 2] + beta[23] * u + beta[24] * u * t
                    + beta[25] * ch + beta[26] * bs)
            res.append(c["y"][h] - pred)
        if len(res) < 5:
            raise ValueError("too few residuals for hour %d" % h)
        med[h] = statistics.median(res)
        mu = sum(res) / len(res)
        sig[h] = math.sqrt(sum((r - mu) ** 2 for r in res) / len(res))
    ranges = {
        "u": (min(c["u"] - u0 for c in comps), max(c["u"] - u0 for c in comps)),
        "ch": (min(c["ch"] - ch0 for c in comps), max(c["ch"] - ch0 for c in comps)),
        "base": (min(c["base"] - b0 for c in comps), max(c["base"] - b0 for c in comps)),
    }
    return {"beta": beta, "u0": u0, "ch0": ch0, "b0": b0,
            "med": med, "sig": sig, "ranges": ranges, "n": len(comps)}


def clamp(x, lo_hi):
    lo, hi = lo_hi
    return lo if x < lo else hi if x > hi else x


def make_quantiles(mid, spread):
    """Five normal quantiles around mid, rounded, valid, non-decreasing."""
    qs = []
    prev = 0.0
    for z in Z:
        q = round2(mid * math.exp(z * spread))
        if q <= 0:
            q = 0.01
        if q > 1e9:
            q = 1e9
        if q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def predict_pooled(model, v1, c1, base):
    u = clamp(math.log(v1) - model["u0"], model["ranges"]["u"])
    ch = clamp(math.log(c1) - model["ch0"], model["ranges"]["ch"])
    bs = clamp(math.log(v1 / base) - model["b0"], model["ranges"]["base"])
    beta = model["beta"]
    out = {}
    for h in HOURS:
        t = (h - 13) / 11.0
        yhat = (beta[h - 2] + beta[23] * u + beta[24] * u * t
                + beta[25] * ch + beta[26] * bs + model["med"][h])
        mid = v1 * math.exp(yhat)
        spread = max(model["sig"][h], SPREAD_FLOOR) * SPREAD_SCALE
        out[h] = make_quantiles(mid, spread)
    return out


def fit_per_hour(comps):
    """Round 1 fallback: per hour, y ~ a + b * log(viewers_1)."""
    out = {}
    for h in HOURS:
        pts = [(c["u"], c["y"][h]) for c in comps if h in c["y"]]
        if len(pts) < 5:
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        if sxx < 1e-12:
            continue
        b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        a = my - b * mx
        res = [p[1] - (a + b * p[0]) for p in pts]
        mu = sum(res) / n
        sd = math.sqrt(sum((r - mu) ** 2 for r in res) / n)
        out[h] = (a, b, max(sd, SPREAD_FLOOR) * ROUND1_SPREAD_SCALE)
    return out


def predict_per_hour(ph, v1):
    out = {}
    for h in HOURS:
        if h not in ph:
            continue
        a, b, spread = ph[h]
        mid = v1 * math.exp(a + b * math.log(v1))
        out[h] = make_quantiles(mid, spread)
    return out


def persistence_quantiles(v1, h):
    spread = PERSIST_SIGMA * math.sqrt(h - 1)
    return make_quantiles(v1, spread)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError("%s asks for levels %s, not %s"
                         % (name, case["levels"], LEVELS))
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    v1 = max(1, first["viewers"])
    c1 = max(1, first.get("channels", 1))
    day_start = parse_time(case["day_start"])
    hist = case.get("history_hourly") or case.get("history") or []
    tkey = "from" if hist and "from" in hist[0] else "at"
    base = baseline_median(hist, day_start, tkey)
    comps = []
    for comp in case.get("comparables", []):
        ex = extract_comparable(comp)
        if ex is not None:
            comps.append(ex)
    hours = [row["hour"] for row in case["forecast"]]
    if len(comps) >= 8 and base is not None and base > 0:
        try:
            model = fit_pooled(comps)
            qmap = predict_pooled(model, v1, c1, base)
            return {"hours": [{"hour": h, "q": qmap[h]} for h in hours]}
        except ValueError:
            pass
    if len(comps) >= 5:
        ph = fit_per_hour(comps)
        qmap = predict_per_hour(ph, v1)
        if len(qmap) == len(HOURS):
            return {"hours": [{"hour": h, "q": qmap[h]} for h in hours]}
    return {"hours": [{"hour": h, "q": persistence_quantiles(v1, h)}
                      for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
