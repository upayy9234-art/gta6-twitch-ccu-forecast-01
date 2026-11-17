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

Model
-----
The forecast is anchored on the case's hour-1 print p1 and on the release-day
curves of past big launches shipped in the case file (`comparables`).

Why a ratio model, and why hour 1's size matters: a launch-day curve is
shaped by *when* the world unlocks. GTA VI unlocks at midnight local time,
rolling (New Zealand 11:00Z on 18 Nov, central Europe 23:00Z, the UK 00:00Z,
the US east coast 05:00Z, the west coast 08:00Z), so at 00:00Z only
New Zealand / Australia / Asia / Europe are playing and the Americas are
still locked. Past launches show the same pattern: when hour 1 is small
(only a few regions live yet) the day ramps several-fold into the US
evening; when hour 1 is already huge (a simultaneous global unlock, or
early access already running) the day stays flat. Across the clean
comparables the peak-to-hour-1 ratio runs from ~1.0x (ELDEN RING,
h1 = 875k) to ~12x (Resident Evil Requiem, h1 = 46k) and ~9x (Marvel's
Wolverine, h1 = 23k).

So, per hour h = 2..24, the model regresses the log ratio
log(viewers_h / viewers_1) on log(viewers_1) over a clean subset of
comparables -- rolling-midnight launches with no early access and no
free-to-play/live-service distortion -- and evaluates the fit at this
case's own log(p1). The spread is the regression's residual standard
deviation (floored at 0.20 log units), scaled by 0.8, applied with the
standard-normal quantiles: q_tau = p1 * exp(log_ratio_h + z_tau * spread_h).

Local validation (practice cases scored against SullyGnome's hourly
averages, the same source the answers come from; pinball loss on the log
scale, lower is better):
  house baseline (persistence):            0.387
  this model (regression, clean subset):    0.154
  per case: Wolverine 0.176, RDR2 0.107, God of War Ragnarok 0.111,
  Spider-Man 2 0.225. Typical miss ~25%, 90% coverage ~0.7-0.9.

Revision round, 17 Nov 2026 (leave-one-out over the 4 practice cases,
final-like comparable set of 42): the submitted configuration was
re-tested against every variant tried during the build. Spread scale
0.7-0.8 is a flat optimum (mean loss 0.154 at 0.75-0.8); 3-point
smoothing of the log-ratio profile (0.157), clamping log(p1) to the
comparables' range (0.158), shrinking the slope toward the median
profile (0.172-0.178), and dropping the two "midnight local expected"
comparables from the clean subset (0.163) are all worse, so none was
adopted. The classifier was checked against all 38 unlock texts: the
clean set is the 9 rolling-midnight, no-early-access, non-F2P launches
(including the two Sony "midnight local expected" ones, which the test
above shows belong), and every excluded launch is out for a stated
reason (simultaneous global unlock, early access, F2P/live-service, or
no usable unlock text). No forecast changed in this revision.

Research notes (sources in NOTES.md): GTA VI releases 19 Nov 2026 on PS5
and Xbox Series X|S (Rockstar, 6 Nov 2025; previously dated 26 May 2026);
midnight-local rolling unlock; Trailer 1 drew 93m YouTube views in 24h
(Guinness record), Trailer 2 475m views across platforms in 24h; the
single-player Twitch peak record is Hogwarts Legacy at ~1.28m concurrent
viewers, ahead of Cyberpunk 2077 (~1.14m) and ELDEN RING (~0.91m).

Fallbacks keep every case valid: if the clean subset has fewer than three
comparables with data for an hour, the regression falls back to all
comparables; with fewer than three still, it falls back to the house
baseline's persistence forecast for that hour. If hour 1 has no print, the
anchor falls back to the last pre-checkpoint hourly point, then to the
top live streams' viewers, then to 1.
"""

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
# The residual spread is scaled by this much: 1.0 was too wide on the
# practice cases (mean loss 0.168), 0.7-0.8 best (0.154); 0.8 is the
# middle of the flat optimum and keeps the 90% band honest.
SPREAD_SCALE = 0.8
# A regression needs at least this many comparables to be trusted.
MIN_FIT = 3
# Quantiles are capped at the contest's maximum (1e9 viewers).
Q_MAX = 999999999.99

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
    -- i.e. the persistence forecast.
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

    # Per-hour regression points, clean subset first, all comparables
    # as fallback. Comparables are visited in name order so the sums
    # accumulate identically on every run.
    comparables = sorted(case.get("comparables") or [],
                         key=lambda c: c.get("name") or "")
    clean, allc = [], []
    for comp in comparables:
        hours = release_day_hours(comp)
        if 1 not in hours:
            continue
        if is_clean(comp):
            clean.append(hours)
        allc.append(hours)

    rows = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        points = []
        for hours in clean:
            if hour in hours:
                points.append((math.log(hours[1]),
                               math.log(hours[hour] / hours[1])))
        if len(points) < MIN_FIT:
            points = []
            for hours in allc:
                if hour in hours:
                    points.append((math.log(hours[1]),
                                   math.log(hours[hour] / hours[1])))
        if len(points) >= MIN_FIT:
            log_ratio, resid = fit_hour(points, log_p1)
            spread = SPREAD_SCALE * resid
            qs = hour_quantiles(p1, log_ratio, spread, hour)
        else:
            # House baseline persistence for this hour.
            qs = hour_quantiles(
                p1, 0.0, BASELINE_SIGMA * math.sqrt(hour - 1), hour)
        rows.append({"hour": hour, "q": qs})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


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
