"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun).

Entry for gta6-twitch-ccu-forecast-01 (harness program/v1). Pure Python
standard library only: no network, no clock, no randomness, so two runs on
the same input produce byte-identical output.

The program reads every `<id>.in` in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes `<id>.out` to the output
directory ($MAKEFUN_OUTPUT_DIR, else /output): one object
`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for
hours 2..24 of the release day, each quantile a finite number of viewers
above 0 and at most 1e9, non-decreasing.

Model (round 2, replaces the round-1 per-hour OLS)
-----------------------------------------------
A pooled hour-fixed-effects regression over the clean comparables'
release-day curves. For every clean comparable and every hour h = 2..24,
the log ratio log(viewers_h / viewers_1) is modelled as

    log_ratio = a_h + b*u + c*u*t + d*ch + e*base

where u = log(viewers_1) centred on the training mean, t = (h - 13) / 11,
a_h is a free intercept per hour, ch = log(channels at hour 1) centred
on the training median, and base = log(viewers_1 / prelaunch) centred on
the training median, with prelaunch = the comparable's median hourly
viewers over the 7 days before its release day. The two extra features
are the round-2 finding: how explosive a launch is relative to its own
category's pre-launch baseline, and how concentrated its hour-1 audience
is across channels, both predict the shape of the day (how deep the
overnight dip runs, how steep the evening ramp is) far better than
hour-1 size alone. A comparable with no channels print or no pre-launch
history gets the training median for that feature (centred value 0);
the target case's features are clamped to the training range so an
extreme GTA VI print cannot blow the forecast up.

The forecast for an hour is the fitted log ratio plus that hour's median
training residual (the residuals are right-skewed, so the median recentres
better than the mean), with spread = that hour's residual standard
deviation (floored at 0.20 log units) scaled by 0.5, applied with the
standard-normal quantiles: q_tau = p1 * exp(log_ratio + z_tau * spread).

The clean subset is the same as round 1: rolling-midnight launches with
no early access and no free-to-play/live-service distortion, classified
from each comparable's unlock text (plus the kit's four practice launches
by name).

Local validation (practice cases scored against SullyGnome's hourly
averages, the same source the answers come from; pinball loss on the log
scale, lower is better):
  house baseline (persistence):            0.387
  round 1 (per-hour OLS, clean subset):    0.154
  round 2 (this model), kit comparables:   0.120
  round 2 (this model), final-like set:    0.120
  per case (kit protocol): Wolverine 0.108, RDR2 0.103, God of War
  Ragnarok 0.092, Spider-Man 2 0.177. Typical miss ~21%, 90% coverage
  ~0.88. The kit protocol is the relevant estimate for the dev score:
  round 1's kit-protocol local number (0.164) matched its dev score
  (0.164) exactly, so the dev score of this entry is expected near 0.120
  (round 1: 0.164; top-20 line was 0.139 at last check).

Research notes (sources in NOTES.md): GTA VI releases 19 Nov 2026 on PS5
and Xbox Series X|S (Rockstar, 6 Nov 2025; previously dated 26 May 2026);
midnight-local rolling unlock; Trailer 1 drew 93m YouTube views in 24h
(Guinness record), Trailer 2 475m views across platforms in 24h; the
single-player Twitch peak record is Hogwarts Legacy at ~1.28m concurrent
viewers, ahead of Cyberpunk 2077 (~1.14m) and ELDEN RING (~0.91m).

Fallbacks keep every case valid: if the pooled fit has too few points or
the linear solve fails, the regression falls back to round 1's per-hour
OLS; with fewer than three comparables for an hour it falls back to the
house baseline's persistence forecast for that hour. If hour 1 has no
print, the anchor falls back to the last pre-checkpoint hourly point,
then to the top live streams' viewers, then to 1.
"""

import datetime
import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file would import (none is needed).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
# The spread of an hour's move in the house baseline (kept only as fallback).
BASELINE_SIGMA = 0.15
# Floor on the regression's residual spread, in log units.
SPREAD_FLOOR = 0.20
# The residual spread is scaled by this much: 0.5 sits in the middle of
# the flat optimum (0.45-0.55) on both local protocols; 0.4 is too narrow
# (90% coverage falls to ~0.80) and 0.7 too wide.
SPREAD_SCALE = 0.5
# A regression needs at least this many comparables to be trusted.
MIN_FIT = 3
# The pooled fit needs at least this many (game, hour) points.
MIN_POOLED = 30
# Quantiles are capped at the contest's maximum (1e9 viewers).
Q_MAX = 999999999.99
# Pre-launch baseline window, in days before the release day.
BASELINE_DAYS = 7

# The kit's four practice launches (input/NOTICE.txt): all rolling-midnight,
# console-only, single-player releases with no early access. Their unlock
# text in the final case may phrase "midnight local" differently
# ("12 AM local"), so they are included by name as well as by the text rule.
PRACTICE_LAUNCHES = frozenset({
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
})


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Returns the solution list, or None if the matrix is singular.
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for cc in range(col, n + 1):
                m[r][cc] -= f * m[col][cc]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def wls(X, y, w):
    """Weighted least squares. X: list of feature vectors, y, w lists."""
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, w):
        for i in range(p):
            b[i] += wi * xi[i] * yi
            for j in range(p):
                A[i][j] += wi * xi[i] * xi[j]
    return solve(A, b)


def is_clean(comp):
    """A comparable launch shaped like GTA VI's: rolling midnight unlock,
    no early access, not free-to-play / live-service.

    Classified from the comparable's own unlock text (and, for the kit's
    four practice launches, by name -- see PRACTICE_LAUNCHES). Launches
    with a single simultaneous global unlock later in the UTC day
    (e.g. Silksong at 14:00Z, Baldur's Gate 3, Black Myth: Wukong) or with
    early access already running (Hogwarts Legacy, Diablo IV, Starfield)
    have hour-1 prints near zero or already at peak, so their ratio
    curves answer a different question and are left out.
    """
    if comp.get("name") in PRACTICE_LAUNCHES:
        return True
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" not in unlock and "12 am local" not in unlock:
        return False
    # "no early access" must not count as early access.
    if re.search(r"(?<!no )early access", unlock) or "early-access" in unlock:
        return False
    if "free-to-play" in unlock or "f2p" in unlock:
        return False
    return True


def release_day_hours(comp):
    """{hour: viewers} for the comparable's UTC release day, hours 1..24.

    A point at `from` covers [from, from + 1h); an hour the source did not
    sample has no point and is left out (per-hour, not whole-curve).
    """
    day = (comp.get("day_start") or "")[:10]
    hours = {}
    for row in comp.get("hourly") or []:
        start = row.get("from") or ""
        if not start.startswith(day):
            continue
        try:
            hour = int(start[11:13]) + 1
            viewers = float(row["viewers"])
        except (ValueError, TypeError, KeyError):
            continue
        if 1 <= hour <= 24 and viewers > 0:
            hours[hour] = viewers
    return hours


def channels_at(comp, hour=1):
    """Channels streaming at the comparable's release-day hour, or None."""
    day = (comp.get("day_start") or "")[:10]
    for row in comp.get("hourly") or []:
        start = row.get("from") or ""
        if not start.startswith(day):
            continue
        try:
            if int(start[11:13]) + 1 != hour:
                continue
        except (ValueError, TypeError):
            continue
        try:
            ch = row.get("channels")
            return float(ch) if ch is not None else None
        except (ValueError, TypeError):
            return None
    return None


def prelaunch_baseline(comp):
    """Median hourly viewers over the 7 days before the release day.

    Computed from the comparable's own hourly rows before `day_start`.
    Returns None when there is no usable pre-launch history.
    """
    day = (comp.get("day_start") or "")[:10]
    try:
        d0 = datetime.date.fromisoformat(day)
    except ValueError:
        return None
    vals = []
    for row in comp.get("hourly") or []:
        start = row.get("from") or ""
        try:
            d = datetime.date.fromisoformat(start[:10])
            v = float(row["viewers"])
        except (ValueError, TypeError, KeyError):
            continue
        if 0 < (d0 - d).days <= BASELINE_DAYS and v > 0:
            vals.append(v)
    return median(vals) if vals else None


def target_baseline(case):
    """Median hourly viewers over the 7 days before the case's release day.

    Computed from the case's `history_hourly` (the category's pre-launch
    hourly points). Returns None when there is no usable history.
    """
    day = (case.get("day_start") or "")[:10]
    try:
        d0 = datetime.date.fromisoformat(day)
    except ValueError:
        return None
    vals = []
    for row in case.get("history_hourly") or []:
        start = row.get("from") or ""
        try:
            d = datetime.date.fromisoformat(start[:10])
            v = float(row["viewers"])
        except (ValueError, TypeError, KeyError):
            continue
        if 0 < (d0 - d).days <= BASELINE_DAYS and v > 0:
            vals.append(v)
    return median(vals) if vals else None


def hour1_anchor(case):
    """Hour 1's print, with fallbacks that keep the case valid."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (ValueError, TypeError):
                pass
    # Last pre-checkpoint hourly point.
    hourly = case.get("history_hourly") or []
    if hourly:
        try:
            return max(1.0, float(hourly[-1]["viewers"]))
        except (ValueError, TypeError, KeyError):
            pass
    # Top live streams at the checkpoint.
    streams = case.get("streams") or []
    total = 0.0
    for s in streams:
        try:
            total += float(s.get("viewers") or 0)
        except (ValueError, TypeError):
            continue
    if total > 0:
        return total
    return 1.0


def fit_hour(points, log_p1):
    """OLS fit of log(viewers_h / viewers_1) on log(viewers_1).

    points: [(log_h1, log_ratio), ...] for one hour. Returns
    (predicted log ratio at log_p1, residual std). With fewer than
    MIN_FIT points, or a degenerate fit, returns (0.0, spread floor)
    -- i.e. the persistence forecast. (Round-1 fallback.)
    """
    if len(points) < MIN_FIT:
        return 0.0, SPREAD_FLOOR
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return my, max(stdev(ys), SPREAD_FLOOR)
    slope = sum((x - mx) * (y - my) for x, y in points) / den
    intercept = my - slope * mx
    log_ratio = intercept + slope * log_p1
    resid = [y - (intercept + slope * x) for x, y in points]
    return log_ratio, max(stdev(resid), SPREAD_FLOOR)


def hour_quantiles(p1, log_ratio, spread, hour):
    """The five quantiles for one hour, rounded, capped, non-decreasing."""
    zs = [Z_TABLE.get(level, 0.0) for level in LEVELS]
    out = []
    for z in zs:
        q = p1 * math.exp(log_ratio + z * spread)
        q = round2(min(max(q, 1.0), Q_MAX))
        out.append(q)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out


def forecast_case(case, name):
    p1 = hour1_anchor(case)
    log_p1 = math.log(p1)

    # Clean comparables with a usable hour-1 print, in name order so the
    # sums accumulate identically on every run. Per-comparable features
    # (channels, pre-launch baseline) are computed from the same rows.
    comparables = sorted(case.get("comparables") or [],
                         key=lambda c: c.get("name") or "")
    curves = {}       # name -> {hour: viewers}
    feat = {}         # name -> {"ch": log channels or None, "base": log ratio or None}
    for comp in comparables:
        hours = release_day_hours(comp)
        if 1 not in hours:
            continue
        if not is_clean(comp):
            continue
        cname = comp.get("name") or ""
        curves[cname] = hours
        ch = channels_at(comp, 1)
        base = prelaunch_baseline(comp)
        feat[cname] = {
            "ch": math.log(ch) if ch else None,
            "base": math.log(hours[1] / base) if (hours[1] and base) else None,
        }

    rows = []
    pooled = _fit_pooled(case, curves, feat, p1, log_p1)
    for row in case.get("forecast") or []:
        hour = row["hour"]
        if pooled is not None:
            log_ratio, spread = pooled(hour)
            qs = hour_quantiles(p1, log_ratio, spread, hour)
        else:
            # Round-1 per-hour OLS fallback.
            points = []
            for hours in curves.values():
                if hour in hours:
                    points.append((math.log(hours[1]),
                                   math.log(hours[hour] / hours[1])))
            if len(points) >= MIN_FIT:
                log_ratio, resid = fit_hour(points, log_p1)
                qs = hour_quantiles(p1, log_ratio, SPREAD_SCALE * resid, hour)
            else:
                # House baseline persistence for this hour.
                qs = hour_quantiles(
                    p1, 0.0, BASELINE_SIGMA * math.sqrt(hour - 1), hour)
        rows.append({"hour": hour, "q": qs})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def _fit_pooled(case, curves, feat, p1, log_p1):
    """Fit the hour-fixed-effects pooled model.

    Returns a function hour -> (log_ratio, spread), or None when the fit
    cannot be trusted (too few points or a singular design), in which
    case the caller falls back to per-hour OLS.
    """
    # Pooled (game, hour) points with per-game feature values.
    named = []
    for cname, hours in curves.items():
        lh1 = math.log(hours[1])
        for h in range(2, 25):
            if h in hours:
                named.append((cname, lh1, h, math.log(hours[h] / hours[1])))
    if len(named) < MIN_POOLED:
        return None
    mu = mean([pt[1] for pt in named])
    # Centres for the extra features: medians over the training games
    # that have a value; games without one get the median (centred 0).
    med = {}
    lo = {}
    hi = {}
    for key in ("ch", "base"):
        vals = [feat[n][key] for n in feat if feat[n][key] is not None]
        if vals:
            med[key] = median(vals)
            lo[key] = min(vals) - med[key]
            hi[key] = max(vals) - med[key]
        else:
            med[key] = 0.0
            lo[key] = 0.0
            hi[key] = 0.0
    hours_list = list(range(2, 25))
    hidx = {h: i for i, h in enumerate(hours_list)}

    def xrow(uu, hh, cname=None, tfeat=None):
        t = (hh - 13.0) / 11.0
        row = [0.0] * 23
        row[hidx[hh]] = 1.0
        row += [uu, uu * t]
        for key in ("ch", "base"):
            if tfeat is not None:
                fv = tfeat.get(key)
            else:
                fv = feat.get(cname, {}).get(key)
            centred = (fv - med[key]) if fv is not None else 0.0
            centred = min(max(centred, lo[key]), hi[key])
            row.append(centred)
        return row

    # The target case's own extra features.
    t_ch = next((k.get("channels") for k in case.get("known") or []
                 if k.get("hour") == 1), None)
    t_base = target_baseline(case)
    tfeat = {
        "ch": math.log(t_ch) if t_ch else None,
        "base": math.log(p1 / t_base) if (p1 and t_base) else None,
    }
    X = [xrow(lh1 - mu, h, cname=cname) for cname, lh1, h, _ in named]
    y = [pt[3] for pt in named]
    beta = wls(X, y, [1.0] * len(X))
    if beta is None:
        return None
    # Per-hour residual median and std of the pooled fit.
    by_h = {}
    for (cname, lh1, h, r) in named:
        pred = sum(b * v for b, v in zip(beta, xrow(lh1 - mu, h, cname=cname)))
        by_h.setdefault(h, []).append(r - pred)
    stats = {h: (median(rs), max(stdev(rs), SPREAD_FLOOR))
             for h, rs in by_h.items()}
    u_t = log_p1 - mu

    def pooled(hour):
        medr, s = stats.get(hour, (0.0, SPREAD_FLOOR))
        logr = sum(b * v for b, v in zip(beta, xrow(u_t, hour, tfeat=tfeat)))
        return logr + medr, s * SPREAD_SCALE

    return pooled


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
