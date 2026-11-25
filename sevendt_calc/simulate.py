"""
Monte-Carlo, night-by-night scheduling of the saved plan: fixed programmes (e.g. the IMS field every clear
night, the TVF field every 14 days) are placed first, then the open-time target parts in order of urgency
(few remaining feasible nights per remaining epoch first), each in the earliest free slot of the night in
which the target is above the limit. Nights are lost to weather at random. Many runs give, per target part,
the probability of an epoch on every night, the chance of finishing all epochs and typical first / last
dates. No Streamlit here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import overhead as oh
from . import phase2
from . import visibility as vis
from .tiles import angular_separation, tile_centres
from .trigger import request_obsmode


@dataclass
class FixedProgramme:
    """A programme outside the open-time plan that takes its time first: `n_tiles` around a tile, one visit per tile."""
    name: str
    tile_ids: list
    epoch_h: float                 # hours of one tile visit (exposure plus overheads)
    cadence_days: int              # 0: every clear night; n: the next visit at least n nights after the last one
    starts: np.ndarray             # (nights, tiles, 4) intervals above the altitude limit, hours after twilight
    ends: np.ndarray
    visits: list = field(default_factory=list)      # visits per night of every tile (1 unless listed under revisits)
    gap_hours: float = 0.0         # minimum hours between two visits of the same tile in one night

    def intervals(self, i: int, j: int) -> list:
        return [(float(s), float(e)) for s, e in zip(self.starts[i, j], self.ends[i, j]) if not np.isnan(s)]

    @property
    def n_visits(self) -> int:
        return int(sum(self.visits)) if self.visits else len(self.tile_ids)


def fixed_programme(name: str, centre_tile: str | None, n_tiles: int, grid: vis.SeasonGrid, cadence_days: int = 0, exptime: float = 100.0, count: int = 3,
                    submode: str = 'specall', cfg: dict | None = None, min_alt: float = 30.0, tiles: list | None = None, revisits: dict | None = None) -> FixedProgramme:
    """
    The given `tiles`, or the `n_tiles` 7DS tiles nearest to `centre_tile` (itself included); one visit of `count`
    frames of `exptime` per filter of the Spec submode, more for tiles in `revisits` ({tile: (visits, gap_hours)}).
    """
    centres = tile_centres()
    if tiles:
        tiles = [str(t) for t in tiles]
    else:
        ra0, dec0 = centres[centre_tile]
        ids = np.array(list(centres))
        ras_all = np.array([centres[t][0] for t in ids])
        decs_all = np.array([centres[t][1] for t in ids])
        order = np.argsort(angular_separation(ras_all, decs_all, ra0, dec0))[:n_tiles]
        tiles = [str(ids[k]) for k in order]
    ras = np.array([centres[t][0] for t in tiles])
    decs = np.array([centres[t][1] for t in tiles])
    om = request_obsmode('Spec', submode, cfg)
    if om is not None:
        epoch_h = oh.estimate([oh.ObsRequest(name, float(exptime), int(count), om)]).total / 3600.0
    else:
        epoch_h = float(exptime) * int(count) * 2 / 3600.0
    starts, ends = vis.intervals_above(grid, ras, decs, min_alt)
    revisits = revisits or {}
    visits = [int(revisits[t][0]) if t in revisits else 1 for t in tiles]
    gap = max([float(v[1]) for v in revisits.values()] + [0.0])
    return FixedProgramme(name=name, tile_ids=tiles, epoch_h=epoch_h, cadence_days=int(cadence_days), starts=starts, ends=ends, visits=visits, gap_hours=gap)


FIXED_JSON = None


def load_fixed_programmes(grid: vis.SeasonGrid, cfg: dict | None = None, path=None) -> list:
    """The programmes of data/fixed_programmes.json as FixedProgramme objects ([] when the file is absent)."""
    import json
    from pathlib import Path
    from .config import DATA_DIR
    path = Path(path) if path else DATA_DIR / 'fixed_programmes.json'
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    out = []
    for item in raw.get('programmes', []):
        revisits = {str(r['tile']): (int(r.get('visits', 2)), float(r.get('gap_hours', 0))) for r in item.get('revisits', [])}
        out.append(fixed_programme(str(item['name']), item.get('centre_tile'), int(item.get('n_tiles', 7)), grid, int(item.get('cadence_days', 0) or 0),
                                   float(item.get('exptime', 100)), int(item.get('count', 3)), str(item.get('submode', 'specall')), cfg,
                                   tiles=item.get('tiles'), revisits=revisits))
    return out


def _place(allowed: list, free: list, dur: float):
    """Earliest start inside one of the `allowed` intervals that is also inside a `free` interval of length >= dur; the free list after taking it."""
    best = None
    for lo, hi in allowed:
        for a, b in free:
            s, e = max(lo, a), min(hi, b)
            if e - s >= dur - 1e-9 and (best is None or s < best):
                best = s
    if best is None:
        return None, free
    new = []
    for a, b in free:
        if b <= best + 1e-9 or a >= best + dur - 1e-9:
            new.append((a, b))
        else:
            if a < best - 1e-9:
                new.append((a, best))
            if b > best + dur + 1e-9:
                new.append((best + dur, b))
    return best, new


@dataclass
class SimulationResult:
    dates: list
    keys: list                          # part keys in the order of the arrays
    prob: np.ndarray                    # (parts, nights) probability of an epoch on that night
    completed: np.ndarray               # (parts,) fraction of runs in which every epoch was observed
    epochs_done: np.ndarray             # (parts,) mean number of epochs observed
    epochs_wanted: np.ndarray           # (parts,)
    first_night: np.ndarray             # (runs, parts) night index of the first epoch, -1 when none
    last_night: np.ndarray              # (runs, parts) night index of the last epoch, -1 when none
    clear: np.ndarray                   # (runs, nights) weather
    fixed_done: dict = field(default_factory=dict)      # programme name -> (runs, nights) every visit of the programme made that night
    runs: int = 0
    epoch_nights: np.ndarray = None     # (runs, parts, max epochs) night index of the e-th epoch, -1 when it did not happen

    def percentile_date(self, which: str, k: int, q: float):
        arr = (self.first_night if which == 'first' else self.last_night)[:, k]
        arr = arr[arr >= 0]
        return self.dates[int(np.percentile(arr, q))] if arr.size else None

    def expected_nights(self, k: int, min_share: float = 0.5) -> list:
        """Night index of every epoch of part k that happens in at least `min_share` of the runs: the median night of that epoch over the runs."""
        out = []
        for e in range(int(self.epochs_wanted[k])):
            nights = self.epoch_nights[:, k, e]
            nights = nights[nights >= 0]
            if nights.size >= min_share * self.runs:
                out.append(int(np.median(nights)))
        return out

    def expected_by_key(self, min_share: float = 0.5) -> dict:
        return {key: self.expected_nights(k, min_share) for k, key in enumerate(self.keys)}


def simulate(plans: dict, grid: vis.SeasonGrid, fixed: list, runs: int = 100, bad_weather: float = 0.1, seed: int = 0, jitter: float = 0.25) -> SimulationResult:
    """
    plans: part key -> TargetPlan (feasibility and per-night intervals already computed). Each run draws the
    weather, then walks the nights: fixed programmes first (every tile, earliest free slot inside its
    time above the limit), then the parts with remaining epochs whose cadence allows a visit, most urgent
    first (feasible nights ahead per remaining epoch, with random jitter so that ties vary between runs).
    """
    rng = np.random.default_rng(seed)
    keys = [k for k, p in plans.items() if p.settings.enabled and p.epoch_h > 0 and p.settings.epochs >= 1]
    n_nights = len(grid.dates)
    night_len = (grid.end_jd - grid.start_jd) * 24.0
    ahead = {}
    for k in keys:
        f = plans[k].feasible.astype(int)
        ahead[k] = np.cumsum(f[::-1])[::-1]                        # feasible nights from this one to the end
    done = np.zeros((runs, len(keys), n_nights), bool)
    max_epochs = max([int(plans[k].settings.epochs) for k in keys] + [1])
    epoch_nights = np.full((runs, len(keys), max_epochs), -1, int)
    clear = rng.random((runs, n_nights)) >= bad_weather
    fixed_done = {prog.name: np.zeros((runs, n_nights), bool) for prog in fixed}
    index = {k: n for n, k in enumerate(keys)}
    for r in range(runs):
        remaining = {k: int(plans[k].settings.epochs) for k in keys}
        last = {k: None for k in keys}
        last_fixed = {prog.name: None for prog in fixed}
        for i in range(n_nights):
            if not clear[r, i]:
                continue
            free = [(0.0, float(night_len[i]))]
            for prog in fixed:
                if prog.cadence_days and last_fixed[prog.name] is not None and i - last_fixed[prog.name] < prog.cadence_days:
                    continue
                placed = 0
                for j in range(len(prog.tile_ids)):
                    visits = prog.visits[j] if prog.visits else 1
                    prev = None
                    for _ in range(visits):
                        allowed = prog.intervals(i, j)
                        if prev is not None:                            # the next visit of the same tile at least gap_hours later
                            allowed = [(max(lo, prev + prog.gap_hours), hi) for lo, hi in allowed if hi - max(lo, prev + prog.gap_hours) > 1e-9]
                        start, free = _place(allowed, free, prog.epoch_h)
                        if start is None:
                            break
                        prev = start
                        placed += 1
                if placed:
                    last_fixed[prog.name] = i
                fixed_done[prog.name][r, i] = placed == prog.n_visits
            cands = []
            for k in keys:
                p = plans[k]
                if remaining[k] <= 0 or not p.intervals[i]:
                    continue
                cad = p.settings.cadence
                if cad > 0 and last[k] is not None and i - last[k] < cad:
                    continue
                urgency = ahead[k][i] / (remaining[k] * max(cad, 1)) * (1.0 + jitter * (2 * rng.random() - 1))
                cands.append((urgency, k))
            cands.sort()
            for _, k in cands:
                start, free = _place(plans[k].intervals[i], free, plans[k].epoch_h)
                if start is not None:
                    e = int(plans[k].settings.epochs) - remaining[k]
                    epoch_nights[r, index[k], e] = i
                    remaining[k] -= 1
                    last[k] = i
                    done[r, index[k], i] = True
    wanted = np.array([plans[k].settings.epochs for k in keys])
    counts = done.sum(axis=2)                                        # (runs, parts)
    first = np.where(done.any(axis=2), done.argmax(axis=2), -1)
    last_ = np.where(done.any(axis=2), n_nights - 1 - done[:, :, ::-1].argmax(axis=2), -1)
    return SimulationResult(dates=list(grid.dates), keys=keys, prob=done.mean(axis=0), completed=(counts >= wanted[None, :]).mean(axis=0),
                            epochs_done=counts.mean(axis=0), epochs_wanted=wanted, first_night=first, last_night=last_, clear=clear, fixed_done=fixed_done, runs=runs,
                            epoch_nights=epoch_nights)


def summary_rows(result: SimulationResult, plans: dict) -> list:
    """One row per part: completion chance, epochs, typical first / last dates (10th, 50th, 90th percentiles)."""
    rows = []
    for k, key in enumerate(result.keys):
        p = plans[key]
        def pct(which, q):
            d = result.percentile_date(which, k, q)
            return d.isoformat() if d else ''
        rows.append({'project': p.req.key, 'target': p.target.display, 'part': p.part, 'epochs': int(result.epochs_wanted[k]), 'epochs_done_mean': round(float(result.epochs_done[k]), 2),
                     'p_complete': round(float(result.completed[k]), 3), 'p_any': round(float((result.first_night[:, k] >= 0).mean()), 3),
                     'expected_nights': ', '.join(result.dates[i].isoformat() for i in result.expected_nights(k)),
                     'first_10': pct('first', 10), 'first_50': pct('first', 50), 'first_90': pct('first', 90),
                     'last_10': pct('last', 10), 'last_50': pct('last', 50), 'last_90': pct('last', 90),
                     'epoch_h': round(p.epoch_h, 2), 'obs_starttime': p.settings.start_text, 'obs_endtime': p.settings.end_text, 'cadence': p.settings.cadence})
    return rows
