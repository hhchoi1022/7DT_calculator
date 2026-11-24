"""
Phase-2 observation request files of the 7DS open-time programme, and the operator's schedule built on them.

A request file (one per proposal, plain text) holds a [PROPOSAL] block (PROPOSAL_TITLE, PI_NAME, PI_EMAIL)
and one [TARGET] ... [END TARGET] block with one set of KEY: value lines per target, sets separated by a
blank line. Keys: TARGET_TYPE (Target / Survey), OBJ_NAME, RA, DEC, TILE_ID, OBS_MODE, OBS_SUBMODE,
EXPTIME_PER_FRAME [s], NUM_FRAMES (per filter over the whole programme), OBS_WINDOW (repeatable;
'YYYY-MM-DD to YYYY-MM-DD' or 'YYYY-MM-DD hh:mm to YYYY-MM-DD hh:mm' in UTC), NOTE (may continue on
indented lines). Lines starting with '#' are comments. Values may carry units ('100 s', '83.8 deg').

Files are matched to the registered projects (data/projects.json) by the PI name, or by an optional
mapping.json in the request folder ({"<file name>": "<Project ID>"}).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.time import Time

from . import overhead as oh
from . import visibility as vis
from .config import PROJECT_DIR, settings
from .targets import parse_coordinates
from .tiles import normalise_tile_id, tile_centre
from .trigger import request_obsmode

PROGRAM_START = date(2026, 12, 1)
PROGRAM_END = date(2027, 8, 31)
PROGRAM_NIGHTS = (PROGRAM_END - PROGRAM_START).days + 1
MOON_CLASSES = (('Dark', 0.0, 0.2), ('Gray', 0.2, 0.8), ('Bright', 0.8, 1.0))     # illuminated fraction (template definition)
GOOD_HOURS = 3.0                                                                   # 'dark' band of the observable-hours plot
KEY_RE = re.compile(r'^([A-Za-z_]+)\s*:\s*(.*)$')
WINDOW_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})(?:[ T](\d{1,2}:\d{2}))?\s*(?:to|-|~|until)\s*(\d{4}-\d{2}-\d{2})(?:[ T](\d{1,2}:\d{2}))?\s*(?:UTC?)?$', re.I)
NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?')
SCHEDULE_FILE = 'schedule.json'
MAPPING_FILE = 'mapping.json'


def requests_dir() -> Path:
    """SEVENDT_PHASE2_DIR if set, else settings()['phase2_requests_dir'] (relative to the repository), else Opentime/2612."""
    env = os.environ.get('SEVENDT_PHASE2_DIR', '').strip()
    text = env or str(settings().get('phase2_requests_dir') or 'Opentime/2612')
    path = Path(text).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


# ------------------------------------------------------------------ data
@dataclass
class Window:
    """One requested observing window (UTC). Date-only windows cover whole nights, start and end dates inclusive."""
    start: datetime
    end: datetime
    timed: bool
    text: str

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    @property
    def start_text(self) -> str:
        return time_text(self.start)

    @property
    def end_text(self) -> str:
        """Inclusive end as written in the file: the last date for a date window, the UTC time for a timed one."""
        return time_text(self.end) if self.timed else self.last_date.isoformat()

    @property
    def first_date(self) -> date:
        return self.start.date()

    @property
    def last_date(self) -> date:
        """Last night date of the window (date windows end at midnight after their last day)."""
        return (self.end - timedelta(days=1)).date() if not self.timed else self.end.date()

    def night_mask(self, grid: vis.SeasonGrid) -> np.ndarray:
        """Nights of the grid touched by the window (date windows: by the local night date; timed: by overlap)."""
        if not self.timed:
            lo, hi = self.first_date, self.last_date
            return np.array([lo <= d <= hi for d in grid.dates], bool)
        s, e = Time(self.start).jd, Time(self.end).jd
        return (grid.end_jd > s) & (grid.start_jd < e)

    def night_intervals(self, grid: vis.SeasonGrid) -> tuple[np.ndarray, np.ndarray]:
        """For a timed window: its start and end in hours after evening twilight of every night (NaN: not that night)."""
        s, e = Time(self.start).jd, Time(self.end).jd
        length = (grid.end_jd - grid.start_jd) * 24.0
        lo = np.clip((s - grid.start_jd) * 24.0, 0.0, length)
        hi = np.clip((e - grid.start_jd) * 24.0, 0.0, length)
        ok = (hi - lo) > 1e-9
        return np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)


@dataclass
class RequestTarget:
    index: int
    target_type: str = ''
    name: str = ''
    ra: float | None = None
    dec: float | None = None
    tile_id: str = ''
    mode: str = ''
    submode: str = ''
    exptime: float | None = None
    nframes: int | None = None
    windows: list = field(default_factory=list)
    note: str = ''
    issues: list = field(default_factory=list)
    n_orders: int = 1
    exposure_s: float = 0.0
    total_s: float = 0.0
    display: str = ''              # label made unique within the file (set by parse_request)
    obsmode: object = None         # ObsMode of the request (set by compute_times; None when unknown)

    @property
    def label(self) -> str:
        return self.name or self.tile_id or f'target {self.index}'

    @property
    def timed(self) -> bool:
        return any(w.timed for w in self.windows)

    @property
    def window_text(self) -> str:
        return '; '.join(w.text for w in self.windows)

    @property
    def window_starts_text(self) -> str:
        return ', '.join(w.start_text for w in self.windows)

    @property
    def window_ends_text(self) -> str:
        return ', '.join(w.end_text for w in self.windows)

    @property
    def mode_text(self) -> str:
        return f'{self.mode}/{self.submode}' if self.submode else self.mode

    @property
    def has_coordinates(self) -> bool:
        return self.ra is not None and self.dec is not None

    @property
    def window_start(self) -> date | None:
        return min((w.first_date for w in self.windows), default=None)

    @property
    def window_end(self) -> date | None:
        return max((w.last_date for w in self.windows), default=None)


@dataclass
class RequestFile:
    path: str
    title: str = ''
    pi: str = ''
    emails: str = ''
    targets: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    project_id: str = ''
    project: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def key(self) -> str:
        """Identifier used in the schedule: the Project ID, or the file name when the file could not be matched."""
        return self.project_id or Path(self.path).stem

    @property
    def lead_pi(self) -> str:
        return re.split(r',|;|\band\b|&', self.pi)[0].strip()

    @property
    def observation(self) -> str:
        if self.project.get('observation'):
            return str(self.project['observation'])
        types = {t.target_type for t in self.targets}
        return 'Survey' if types == {'Survey'} else 'Target'

    @property
    def mode(self) -> str:
        return str(self.project.get('mode') or ', '.join(sorted({t.mode for t in self.targets if t.mode})))

    @property
    def submodes(self) -> str:
        return ', '.join(sorted({t.submode for t in self.targets if t.submode}))

    @property
    def sky_background(self) -> str:
        return str(self.project.get('sky_background') or 'Any')

    @property
    def allocated_hours(self) -> float | None:
        hours = self.project.get('allocated_hours')
        return float(hours) if hours is not None else None

    @property
    def dither_arcsec(self) -> float:
        """Dithering step of the project (data/projects.json 'dither_arcsec'; 0 when absent)."""
        try:
            return float(self.project.get('dither_arcsec') or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def exposure_h(self) -> float:
        return sum(t.exposure_s for t in self.targets) / 3600.0

    @property
    def total_h(self) -> float:
        return sum(t.total_s for t in self.targets) / 3600.0

    @property
    def requested_start(self) -> date:
        starts = [t.window_start for t in self.targets if t.window_start]
        return max(min(starts), PROGRAM_START) if starts else PROGRAM_START

    @property
    def requested_end(self) -> date:
        ends = [t.window_end for t in self.targets if t.window_end]
        return min(max(ends), PROGRAM_END) if ends else PROGRAM_END

    @property
    def timed(self) -> bool:
        return any(t.timed for t in self.targets)

    @property
    def all_issues(self) -> list:
        return list(self.issues) + [f'{t.label}: {i}' for t in self.targets for i in t.issues]


# ------------------------------------------------------------------ parsing
def _number(text: str, kind: str = 'float'):
    m = NUMBER_RE.search(str(text))
    if not m:
        return None
    value = float(m.group(0))
    return int(round(value)) if kind == 'int' else value


def parse_window(text: str) -> Window:
    m = WINDOW_RE.match(text.strip())
    if not m:
        raise ValueError(f"OBS_WINDOW '{text}' is not 'YYYY-MM-DD to YYYY-MM-DD' (optionally with hh:mm UTC)")
    d0, t0, d1, t1 = m.groups()
    timed = bool(t0 or t1)
    start = datetime.strptime(f'{d0} {t0 or "00:00"}', '%Y-%m-%d %H:%M')
    end = datetime.strptime(f'{d1} {t1 or "00:00"}', '%Y-%m-%d %H:%M')
    if not timed:
        end = end + timedelta(days=1)            # whole end day
    if end <= start:
        raise ValueError(f"OBS_WINDOW '{text}' ends before it starts")
    return Window(start=start, end=end, timed=timed, text=text.strip())


def _build_target(index: int, values: dict) -> RequestTarget:
    def first(key):
        items = values.get(key) or []
        return items[0].strip() if items else ''

    t = RequestTarget(index=index)
    kind = first('TARGET_TYPE').strip().title()
    t.target_type = kind if kind in ('Target', 'Survey') else (kind or 'Target')
    if kind and kind not in ('Target', 'Survey'):
        t.issues.append(f"TARGET_TYPE '{kind}' is not Target or Survey")
    t.name = re.sub(r'\s+', ' ', first('OBJ_NAME'))
    t.tile_id = first('TILE_ID')
    t.mode = first('OBS_MODE').strip().title()
    t.submode = first('OBS_SUBMODE').strip()
    if t.mode == 'Spec':
        t.submode = t.submode.lower()
    t.note = ' '.join(v.strip() for v in values.get('NOTE', []) if v.strip())
    if t.target_type == 'Survey' or (t.tile_id and not first('RA')):
        tid = normalise_tile_id(t.tile_id) if t.tile_id else None
        if tid is None:
            t.issues.append(f"TILE_ID '{t.tile_id}' is not a 7DS tile ID")
        else:
            try:
                t.ra, t.dec = tile_centre(tid)
                t.tile_id = tid
            except ValueError as exc:
                t.issues.append(str(exc))
        if not t.name:
            t.name = t.tile_id
    else:
        ra, dec = re.sub(r'(?i)deg', '', first('RA')).strip(), re.sub(r'(?i)deg', '', first('DEC')).strip()
        if ra and dec:
            try:
                t.ra, t.dec = parse_coordinates(ra, dec)
            except ValueError as exc:
                t.issues.append(str(exc))
        else:
            t.issues.append('RA / DEC missing')
    t.exptime = _number(first('EXPTIME_PER_FRAME'))
    t.nframes = _number(first('NUM_FRAMES'), 'int')
    if t.exptime is None:
        t.issues.append('EXPTIME_PER_FRAME missing')
    if t.nframes is None:
        t.issues.append('NUM_FRAMES missing')
    for text in values.get('OBS_WINDOW', []):
        if not text.strip():
            continue
        try:
            t.windows.append(parse_window(text))
        except ValueError as exc:
            t.issues.append(str(exc))
    if not t.windows:
        t.issues.append('no OBS_WINDOW: the whole programme period is assumed')
    if not t.mode or not t.submode:
        t.issues.append('OBS_MODE / OBS_SUBMODE missing')
    return t


def parse_request(text: str, path: str = '') -> RequestFile:
    """Parse one request file (its text). Problems are collected in `issues`, never raised."""
    req = RequestFile(path=path)
    proposal, sets, section, current, last_key = {}, [], None, None, None

    def close():
        nonlocal current
        if current:
            sets.append(current)
        current = None

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            if section == 'target':
                close()
            continue
        if s.startswith('#'):
            continue
        tag = s.upper()
        if tag == '[PROPOSAL]':
            section = 'proposal'
            continue
        if tag == '[END PROPOSAL]':
            section = None
            continue
        if tag == '[TARGET]':
            section, current = 'target', None
            continue
        if tag == '[END TARGET]':
            close()
            section = None
            continue
        m = KEY_RE.match(s)
        if m and m.group(1).upper() == m.group(1):
            key, value = m.group(1), m.group(2).strip()
            if section == 'proposal':
                proposal[key] = value
                last_key = key
            elif section == 'target':
                if key == 'TARGET_TYPE' and current and 'TARGET_TYPE' in current:
                    close()
                if current is None:
                    current = {}
                current.setdefault(key, []).append(value)
                last_key = key
            else:
                req.issues.append(f"line outside a block ignored: '{s[:60]}'")
        elif section == 'target' and current and last_key in current:
            current[last_key][-1] = (current[last_key][-1] + ' ' + s).strip()      # continuation line
        elif section == 'proposal' and last_key in proposal:
            proposal[last_key] = (proposal[last_key] + ' ' + s).strip()
        else:
            req.issues.append(f"line not understood: '{s[:60]}'")
    close()
    req.title = proposal.get('PROPOSAL_TITLE', '')
    req.pi = proposal.get('PI_NAME', '')
    req.emails = proposal.get('PI_EMAIL', '')
    if not req.title and not req.pi:
        req.issues.append('no [PROPOSAL] block')
    req.targets = [_build_target(i, values) for i, values in enumerate(sets, start=1)]
    counts = {}
    for t in req.targets:
        counts[t.label] = counts.get(t.label, 0) + 1
    for t in req.targets:
        t.display = f'{t.label} #{t.index}' if counts[t.label] > 1 else t.label
    if not req.targets:
        req.issues.append('no targets')
    return req


# ------------------------------------------------------------------ projects and time accounting
def _name_forms(name: str) -> set:
    """Spellings of a person's name that identify the same person: 'Kim Mi-Ryang' matches 'Miryang Kim'."""
    name = re.sub(r'\(.*?\)', ' ', str(name))
    name = re.split(r',|;|\band\b|&', name)[0]
    tokens = re.findall(r'[a-z]+', name.lower())
    if not tokens:
        return set()
    forms = {''.join(tokens), ''.join(reversed(tokens))}
    if len(tokens) > 2:
        forms |= {tokens[-1] + ''.join(tokens[:-1]), ''.join(tokens[1:]) + tokens[0]}
    return forms


def match_project(req: RequestFile, projects: dict | None) -> str:
    """The Project ID whose PI wrote the file ('' when none or several match)."""
    if not projects:
        return ''
    forms = _name_forms(req.pi)
    cands = [pid for pid, p in projects.items() if isinstance(p, dict) and _name_forms(p.get('pi', '')) & forms]
    if len(cands) > 1:
        is_too = re.search(r'\bToO\b', req.title + ' ' + ' '.join(t.note for t in req.targets)) is not None
        narrowed = [pid for pid in cands if (str(projects[pid].get('observation', '')) == 'ToO') == is_too]
        cands = narrowed or cands
    if len(cands) > 1:
        spec = all(t.mode == 'Spec' for t in req.targets)
        narrowed = [pid for pid in cands if (str(projects[pid].get('mode', '')) == 'Spec') == spec]
        cands = narrowed or cands
    return cands[0] if len(cands) == 1 else ''


def compute_times(req: RequestFile, cfg: dict | None = None, oh_config: oh.OverheadConfig | None = None) -> None:
    """Open-shutter and total (with overheads) seconds of every target, from the overhead model of the mode."""
    oh_config = oh_config or oh.OverheadConfig.load()
    for t in req.targets:
        if t.exptime is None or t.nframes is None or t.nframes < 1:
            t.n_orders, t.exposure_s, t.total_s = 1, 0.0, 0.0
            continue
        om = request_obsmode(t.mode, t.submode, cfg) if t.mode and t.submode else None
        t.obsmode = om
        t.n_orders = max(om.n_filters_per_unit, 1) if om is not None else 1
        t.exposure_s = float(t.exptime) * int(t.nframes) * t.n_orders
        if om is None:
            t.total_s = t.exposure_s
            if t.mode and t.submode:
                t.issues.append(f'mode {t.mode_text} unknown to the instrument: no overhead estimated')
            continue
        try:
            t.total_s = oh.estimate([oh.ObsRequest(t.label, float(t.exptime), int(t.nframes), om, config=oh_config)], oh_config).total
        except ValueError as exc:
            t.total_s = t.exposure_s
            t.issues.append(f'overhead not estimated: {exc}')


def load_mapping(folder: Path | str) -> dict:
    path = Path(folder) / MAPPING_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in raw.items() if not str(k).startswith('_')}


def load_requests(folder: Path | str, projects: dict | None = None, cfg: dict | None = None) -> list:
    """Every *.txt request file of the folder, matched to the projects and with its times computed, sorted by Project ID."""
    folder = Path(folder)
    mapping = load_mapping(folder)
    oh_config = oh.OverheadConfig.load()
    out = []
    for path in sorted(folder.glob('*.txt')):
        req = parse_request(path.read_text(encoding='utf-8', errors='replace'), str(path))
        pid = mapping.get(path.name) or match_project(req, projects)
        if pid and projects and pid in projects:
            req.project_id = pid
            req.project = {k: v for k, v in projects[pid].items() if k != 'key'}
        elif projects:
            req.issues.append('no registered project matches the PI name (add the file to mapping.json)')
        compute_times(req, cfg, oh_config)
        out.append(req)
    out.sort(key=lambda r: (r.project_id == '', r.project_id, r.name))
    return out


def folder_signature(folder: Path | str) -> tuple:
    """Changes when a request file (or the mapping) is added, removed or edited."""
    folder = Path(folder)
    if not folder.is_dir():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in list(folder.glob('*.txt')) + [folder / MAPPING_FILE] if p.is_file()))


# ------------------------------------------------------------------ nights
def moon_allowed(sky_background: str, fraction: np.ndarray) -> np.ndarray:
    """Nights whose Moon illumination suits the proposal's sky background ('Dark', 'Dark & Gray', 'Any', ...)."""
    text = str(sky_background or 'Any').lower()
    if 'any' in text or not text.strip():
        return np.ones_like(fraction, bool)
    ok = np.zeros_like(fraction, bool)
    for name, lo, hi in MOON_CLASSES:
        if name.lower() in text or (name == 'Gray' and 'grey' in text):
            ok |= (fraction >= lo) & (fraction <= hi)
    return ok if ok.any() else np.ones_like(fraction, bool)


@dataclass
class NightTable:
    """Per-night numbers of one request on the season grid: rows nights, columns targets."""
    dates: list
    night_hours: np.ndarray        # length of every night [h]
    max_alt: np.ndarray            # (nights, targets) highest altitude during the night
    transit: np.ndarray            # (nights, targets) the transit falls inside the night
    at_hours: np.ndarray           # (nights, targets) hours after twilight of the nightly maximum
    culmination: np.ndarray        # (targets,)
    hours: np.ndarray              # (nights, targets) hours above the altitude limit
    starts: np.ndarray             # (nights, targets, 4) intervals above the limit, hours after twilight
    ends: np.ndarray
    requested: np.ndarray          # (nights, targets) inside one of the target's OBS_WINDOWs
    win_starts: np.ndarray         # (nights, targets, k) timed windows as intervals (NaN when none / untimed)
    win_ends: np.ndarray
    moon: np.ndarray               # (nights,) Moon illumination
    moon_ok: np.ndarray            # (nights,) suits the proposal's sky background
    hours_any: np.ndarray = None   # (nights,) hours during which at least one target is above the limit (requested windows ignored)
    timed: np.ndarray = None       # (targets,) the target's windows carry times of day
    grid: object = None            # the SeasonGrid the table was built on

    def target_intervals(self, i: int, j: int, respect_dates: bool = True) -> list:
        """Intervals (hours after twilight) of target j during night i in which it is above the limit and
        inside a requested timed window (all night when the target's windows are date-only). With
        `respect_dates=False` the date-only windows are ignored (the operator's own range applies instead)."""
        up = [(s, e) for s, e in zip(self.starts[i, j], self.ends[i, j]) if not math.isnan(s)]
        timed = self.timed[j] if self.timed is not None else False
        if respect_dates and not self.requested[i, j]:
            return []
        if timed and not self.requested[i, j]:
            return []
        wins = [(s, e) for s, e in zip(self.win_starts[i, j], self.win_ends[i, j]) if not math.isnan(s)]
        if not wins:
            return up
        out = []
        for s, e in up:
            for ws, we in wins:
                lo, hi = max(s, ws), min(e, we)
                if hi - lo > 1e-9:
                    out.append((lo, hi))
        return out

    def available(self, i: int, cols=None) -> list:
        """Union of the targets' intervals during night i (hours after twilight)."""
        cols = range(self.max_alt.shape[1]) if cols is None else cols
        return merge_intervals([iv for j in cols for iv in self.target_intervals(i, j)])


def merge_intervals(intervals: list) -> list:
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def interval_hours(intervals: list) -> float:
    return float(sum(hi - lo for lo, hi in intervals))


def night_table(req: RequestFile, grid: vis.SeasonGrid, moon: np.ndarray, min_alt: float = 30.0) -> NightTable:
    n, m = len(grid.dates), len(req.targets)
    ra = np.array([t.ra if t.has_coordinates else np.nan for t in req.targets], float)
    dec = np.array([t.dec if t.has_coordinates else np.nan for t in req.targets], float)
    if m:
        alt = vis.nightly_max_altitude(grid, ra, dec)
        hours = vis.hours_above(grid, ra, dec, min_alt)
        starts, ends = vis.intervals_above(grid, ra, dec, min_alt)
    else:
        alt = {'max_alt': np.zeros((n, 0)), 'transit': np.zeros((n, 0), bool), 'at_hours': np.zeros((n, 0)), 'culmination': np.zeros(0)}
        hours, starts, ends = np.zeros((n, 0)), np.full((n, 0, 4), np.nan), np.full((n, 0, 4), np.nan)
    hours = np.where(np.isnan(hours), 0.0, hours)
    requested = np.zeros((n, m), bool)
    k = max([sum(1 for w in t.windows if w.timed) for t in req.targets] + [1])
    win_s = np.full((n, m, k), np.nan)
    win_e = np.full((n, m, k), np.nan)
    for j, t in enumerate(req.targets):
        if not t.windows:
            requested[:, j] = True
            continue
        kk = 0
        for w in t.windows:
            requested[:, j] |= w.night_mask(grid)
            if w.timed:
                win_s[:, j, kk], win_e[:, j, kk] = w.night_intervals(grid)
                kk += 1
    hours_any = np.zeros(n)
    for i in range(n):
        ups = [(s_, e_) for j in range(m) for s_, e_ in zip(starts[i, j], ends[i, j]) if not math.isnan(s_)]
        hours_any[i] = interval_hours(merge_intervals(ups)) if ups else 0.0
    return NightTable(dates=list(grid.dates), night_hours=(grid.end_jd - grid.start_jd) * 24.0, max_alt=alt['max_alt'], transit=alt['transit'],
                      at_hours=alt['at_hours'], culmination=alt['culmination'], hours=hours, starts=starts, ends=ends, requested=requested,
                      win_starts=win_s, win_ends=win_e, moon=moon, moon_ok=moon_allowed(req.sky_background, moon), hours_any=hours_any,
                      timed=np.array([t.timed for t in req.targets], bool), grid=grid)


STRIP_CATEGORIES = (('none', 'not observable', '#ffffff'), ('obs', 'observable (> 0 h)', '#c7e9c0'), ('good', f'≥ {GOOD_HOURS:g} h observable', '#41ab5d'),
                    ('moon', 'Moon unsuitable', '#fff1a8'), ('conflict', 'conflict', '#d62728'))


def strip_categories(hours: np.ndarray, moon_ok: np.ndarray | None = None) -> np.ndarray:
    """
    Category index per night for the strip charts (see STRIP_CATEGORIES): observable hours in shades of green,
    yellow when the Moon condition fails (only when `moon_ok` is given). Windows are drawn separately, as a
    red line through the middle of the strip, so that they never hide the observability colours.
    """
    hours = np.asarray(hours, float)
    cat = np.zeros(hours.shape, int)
    cat[hours > 1e-9] = 1
    cat[hours >= GOOD_HOURS] = 2
    if moon_ok is not None:
        cat[~np.asarray(moon_ok, bool)] = 3
    return cat


def target_rows(requests: list) -> list:
    """One row per requested target set of every file: the combined table of the programme."""
    rows = []
    for req in requests:
        for t in req.targets:
            rows.append({'Project': req.key, 'PI': req.lead_pi, 'Request type': req.observation, 'Mode': t.mode, 'Submode': t.submode,
                         'Sky background': req.sky_background, 'Target': t.display, 'Type': t.target_type, 'Tile': t.tile_id,
                         'RA (°)': None if t.ra is None else round(t.ra, 5), 'Dec (°)': None if t.dec is None else round(t.dec, 5),
                         'Exptime (s)': t.exptime, 'Frames': t.nframes, 'Orders': t.n_orders, 'Total exposure (s)': round(t.exposure_s, 1),
                         'Total exposure (h)': round(t.exposure_s / 3600.0, 2), 'With overhead (h)': round(t.total_s / 3600.0, 2),
                         'dither_arcsec': req.dither_arcsec, 'obs_window_start': t.window_starts_text, 'obs_window_end': t.window_ends_text, 'Requester_note': t.note,
                         'Issues': '; '.join(t.issues), 'File': req.name})
    return rows


# ------------------------------------------------------------------ the operator's plan (per target)
_EPOCH_CACHE: dict = {}


def epoch_hours(t: RequestTarget, count: int, oh_config: oh.OverheadConfig | None = None) -> float:
    """Hours of one visit of the target with `count` frames per filter (exposure plus the overhead model)."""
    if t.exptime is None or count < 1:
        return 0.0
    key = (t.mode, t.submode, float(t.exptime), int(count))
    if key not in _EPOCH_CACHE:
        if t.obsmode is None:
            _EPOCH_CACHE[key] = float(t.exptime) * int(count) * max(t.n_orders, 1) / 3600.0
        else:
            cfg = oh_config or oh.OverheadConfig.load()
            try:
                _EPOCH_CACHE[key] = oh.estimate([oh.ObsRequest(t.label or 'target', float(t.exptime), int(count), t.obsmode, config=cfg)], cfg).total / 3600.0
            except ValueError:
                _EPOCH_CACHE[key] = float(t.exptime) * int(count) * max(t.n_orders, 1) / 3600.0
    return _EPOCH_CACHE[key]


def as_datetime(value) -> datetime:
    """A date or datetime (or their ISO strings) as a naive UTC datetime; a date means midnight (= 'that whole local night')."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = re.sub(r'\s+', ' ', str(value).strip().replace('T', ' '))
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"'{value}' is not a date or a date with a UTC time")


def time_text(value: datetime) -> str:
    """'2027-02-24' for midnight (a whole local night), '2027-02-24T03:30' otherwise."""
    return value.strftime('%Y-%m-%d') if (value.hour, value.minute, value.second) == (0, 0, 0) else value.strftime('%Y-%m-%dT%H:%M')


def as_datetime_list(value) -> list:
    """One or more UTC datetimes: a list, or a comma-separated string ('2027-02-24, 2027-05-01 03:30'), or a single value."""
    if isinstance(value, (list, tuple)):
        return [as_datetime(v) for v in value]
    if isinstance(value, str):
        return [as_datetime(part) for part in value.split(',') if part.strip()]
    return [as_datetime(value)]


@dataclass
class TargetSettings:
    """
    What the operator enters for one requested target set. obs_starttime / obs_endtime are UTC and may list
    several ranges (comma-separated, the n-th start with the n-th end); a value without a time (midnight)
    means the whole local night of that date (so a date-only range covers the nights of both dates), a
    value with a time cuts exactly there.
    """
    start: list                    # obs_starttime: one datetime per range
    end: list                      # obs_endtime: the matching end of every range
    count: int                     # frames per filter in one epoch (visit)
    epochs: int                    # number of epochs (target_epoch)
    enabled: bool = True
    cadence: int = 0               # days between consecutive epochs (0: no constraint)
    note: str = ''                 # the operator's own note for the row

    def __post_init__(self):
        self.note = '' if self.note is None else str(self.note)
        self.start = as_datetime_list(self.start)
        self.end = as_datetime_list(self.end)
        if not self.start or not self.end:
            raise ValueError('obs_starttime and obs_endtime are required')
        if len(self.start) != len(self.end):
            raise ValueError(f'{len(self.start)} start time(s) but {len(self.end)} end time(s): list the same number of ranges')

    @property
    def ranges(self) -> list:
        return sorted(zip(self.start, self.end))

    @property
    def first(self) -> datetime:
        return min(self.start)

    @property
    def last(self) -> datetime:
        return max(self.end)

    @property
    def start_night(self) -> date:
        return self.first.date()

    @property
    def end_night(self) -> date:
        return self.last.date()

    @property
    def start_text(self) -> str:
        return ', '.join(time_text(x) for x in self.start)

    @property
    def end_text(self) -> str:
        return ', '.join(time_text(x) for x in self.end)

    def as_dict(self) -> dict:
        return {'obs_starttime': self.start_text, 'obs_endtime': self.end_text, 'count': int(self.count), 'target_epoch': int(self.epochs),
                'cadence': int(self.cadence), 'enabled': bool(self.enabled), 'note': self.note}

    @classmethod
    def from_dict(cls, d: dict) -> 'TargetSettings':
        start = d['obs_starttime'] if 'obs_starttime' in d else d['obs_startdate']          # files written before the rename
        end = d['obs_endtime'] if 'obs_endtime' in d else d['obs_enddate']
        return cls(start=start, end=end, count=int(d.get('count', 1)), epochs=int(d.get('target_epoch', 1)), enabled=bool(d.get('enabled', True)),
                   cadence=int(d.get('cadence', 0) or 0), note=str(d.get('note', '') or ''))

    @staticmethod
    def _bound(value: datetime, grid: vis.SeasonGrid, is_end: bool) -> float:
        if (value.hour, value.minute) == (0, 0):                      # a bare date: the whole local night
            i = max(0, min((value.date() - grid.dates[0]).days, len(grid.dates) - 1))
            if is_end:
                return float(grid.end_jd[i] if value.date() <= grid.dates[-1] else grid.end_jd[-1] + 1.0)
            return float(grid.start_jd[i] if value.date() >= grid.dates[0] else grid.start_jd[0] - 1.0)
        return float(Time(value).jd)

    def bounds_jd(self, grid: vis.SeasonGrid) -> list:
        """Every range as (start, end) UTC Julian dates: midnight values expand to the evening / morning twilight of that local night."""
        return [(self._bound(s, grid, False), self._bound(e, grid, True)) for s, e in self.ranges]

    def inverted(self) -> bool:
        """A range that ends before it starts (a bare date equal on both sides is one whole night, not inverted)."""
        return any(e < s or (e == s and (s.hour, s.minute) != (0, 0)) for s, e in zip(self.start, self.end))


def target_key(req: RequestFile, t: RequestTarget) -> str:
    return f'{req.key}|{t.display}'


def part_key(tkey: str, part: int) -> str:
    return f'{tkey}#{part}'


def default_target_settings(req: RequestFile, t: RequestTarget, table: NightTable, j: int, use_moon: bool = True) -> TargetSettings:
    """
    A first entry: the requested window span, all frames in one epoch when one visit fits into the best
    night of the span, otherwise as many epochs as needed so that one epoch fits.
    """
    start, end = t.window_start or PROGRAM_START, t.window_end or PROGRAM_END
    start, end = max(start, PROGRAM_START), min(end, PROGRAM_END)
    frames = int(t.nframes or 1)
    in_range = np.array([start <= d <= end for d in table.dates], bool)
    if use_moon:
        in_range &= table.moon_ok
    best = 0.0
    for i in np.flatnonzero(in_range):
        for lo, hi in table.target_intervals(i, j, respect_dates=False):
            best = max(best, hi - lo)
    count, epochs = frames, 1
    if best > 0 and epoch_hours(t, frames) > best:
        for epochs in range(2, frames + 1):
            count = math.ceil(frames / epochs)
            if epoch_hours(t, count) <= best:
                break
    enabled = req.observation != 'ToO' and t.has_coordinates and t.exptime is not None and t.nframes is not None
    m = re.search(r'cadence\s*[=:]?\s*(\d+)\s*(?:d\b|days?)', t.note, re.I)
    cadence = int(m.group(1)) if m else 0
    return TargetSettings(start=datetime.combine(start, datetime.min.time()), end=datetime.combine(end, datetime.min.time()), count=max(count, 1), epochs=max(epochs, 1),
                          enabled=enabled, cadence=cadence)


@dataclass
class TargetPlan:
    key: str                       # part key: '<project>|<target>#<part>'
    req: RequestFile
    j: int
    settings: TargetSettings
    epoch_h: float                 # hours of one epoch, overhead included
    feasible: np.ndarray           # (nights,) the epoch fits into the target's time above the limit that night
    placed: np.ndarray             # (nights,) nights on which an epoch is assumed (spread evenly over the feasible nights)
    intervals: list                # per night: the target's intervals (hours after twilight) usable for the epoch
    issues: list                   # blocking: the plan cannot be carried out as entered
    warnings: list = field(default_factory=list)   # informational (frames differ from the request)
    part: int = 1                  # part number within the target set (1 unless the operator split the set)
    in_range: np.ndarray = None    # (nights,) the night overlaps obs_starttime .. obs_endtime

    @property
    def target(self) -> RequestTarget:
        return self.req.targets[self.j]

    @property
    def target_key(self) -> str:
        return target_key(self.req, self.target)

    @property
    def frames_planned(self) -> int:
        return int(self.settings.count * self.settings.epochs)

    @property
    def planned_h(self) -> float:
        return float(self.epoch_h * self.settings.epochs)

    @property
    def required_h(self) -> float:
        return self.target.total_s / 3600.0

    @property
    def n_feasible(self) -> int:
        return int(self.feasible.sum())

    @property
    def n_placed(self) -> int:
        return int(self.placed.sum())

    @property
    def ok(self) -> bool:
        return self.settings.enabled and not self.issues

    @property
    def status(self) -> str:
        if not self.settings.enabled:
            return 'off'
        return '; '.join(self.issues + self.warnings) or 'OK'


def make_target_plan(req: RequestFile, j: int, table: NightTable, settings: TargetSettings, use_moon: bool = True,
                     oh_config: oh.OverheadConfig | None = None, part: int = 1, check_frames: bool = True) -> TargetPlan:
    """
    The plan of one part of a target set (most targets have a single part; see make_target_plans). The
    operator's obs_starttime .. obs_endtime is the master range: the requested OBS_WINDOW dates play no
    role here (timed windows still fix the time of night).
    """
    t = req.targets[j]
    n = len(table.dates)
    grid = table.grid
    issues, warnings = [], []
    epoch_h = epoch_hours(t, settings.count, oh_config) if settings.count >= 1 else 0.0
    bounds = settings.bounds_jd(grid)
    night_start = grid.start_jd
    in_range = np.zeros(n, bool)
    for lo_jd, hi_jd in bounds:
        in_range |= (grid.end_jd > lo_jd) & (grid.start_jd < hi_jd)
    usable = in_range.copy()
    if use_moon:
        usable &= table.moon_ok
    intervals = []
    for i in range(n):
        if not (usable[i] and settings.enabled):
            intervals.append([])
            continue
        ups = table.target_intervals(i, j, respect_dates=False)
        clipped = []
        for lo_jd, hi_jd in bounds:
            lo_h, hi_h = (lo_jd - night_start[i]) * 24.0, (hi_jd - night_start[i]) * 24.0      # the range inside this night, hours after twilight
            clipped += [(max(a, lo_h), min(b, hi_h)) for a, b in ups if min(b, hi_h) - max(a, lo_h) > 1e-9]
        intervals.append(merge_intervals(clipped))
    longest = np.array([max((hi - lo for lo, hi in iv), default=0.0) for iv in intervals])
    feasible = (longest >= epoch_h - 1e-9) & (longest > 0) if epoch_h > 0 else np.zeros(n, bool)
    placed = np.zeros(n, bool)
    if settings.enabled:
        if settings.inverted():
            issues.append('obs_endtime before obs_starttime')
        if settings.count < 1 or settings.epochs < 1:
            issues.append('count and target_epoch must be at least 1')
        if not t.has_coordinates:
            issues.append('no coordinates')
        if check_frames and t.nframes is not None and settings.count * settings.epochs != t.nframes:
            warnings.append(f'{settings.count * settings.epochs} frames planned, {t.nframes} requested')
        idx = np.flatnonzero(feasible)
        if idx.size >= settings.epochs >= 1:
            for chunk in np.array_split(idx, settings.epochs):                 # provisional: the middle night of each part of the range
                placed[chunk[len(chunk) // 2]] = True
        else:
            placed[idx] = True
            if epoch_h > 0 and settings.count >= 1 and settings.epochs >= 1 and t.has_coordinates:
                best = float(longest.max()) if longest.size else 0.0
                if idx.size == 0:
                    issues.append(f'no night in the range with {epoch_h:.2f} h of the target above {MIN_ALT_TEXT} (longest {best:.2f} h)')
                else:
                    issues.append(f'only {idx.size} of {settings.epochs} epochs fit (nights with {epoch_h:.2f} h of the target up)')
    return TargetPlan(key=part_key(target_key(req, t), part), req=req, j=j, settings=settings, part=part, epoch_h=epoch_h, feasible=feasible, placed=placed,
                      intervals=intervals, issues=issues, warnings=warnings, in_range=in_range)


def make_target_plans(req: RequestFile, j: int, table: NightTable, parts: list, use_moon: bool = True, oh_config: oh.OverheadConfig | None = None) -> list:
    """
    The plans of every part of a target set. A target set is usually one part (count x epochs); when the
    operator splits it, the parts carry their own ranges, counts and epochs and the frames of the enabled
    parts are compared with the request as a whole.
    """
    plans = [make_target_plan(req, j, table, s, use_moon, oh_config, part=n, check_frames=False) for n, s in enumerate(parts, start=1)]
    t = req.targets[j]
    on = [p for p in plans if p.settings.enabled]
    if on and t.nframes is not None:
        total = sum(p.frames_planned for p in on)
        if total != t.nframes:
            text = f'{total} frames planned' + (f' in {len(on)} parts' if len(on) > 1 else '') + f', {t.nframes} requested'
            for p in on:
                p.warnings.append(text)
    return plans


def plans_of(plans: dict, req: RequestFile, j: int) -> list:
    """Every part plan of target j of the request, in part order."""
    return sorted((p for p in plans.values() if p.req is req and p.j == j), key=lambda p: p.part)


def place_epochs(plans: dict, n_nights: int) -> np.ndarray:
    """
    Assume an automatic scheduler: give every target its epochs on feasible nights, the most constrained
    targets first, each epoch in its own part of the target's range on the night with the least load so
    far. Updates `placed` of every plan and returns the load (hours) per night.
    """
    load = np.zeros(n_nights)
    order = sorted(plans.values(), key=lambda p: (p.n_feasible, -p.epoch_h))
    for p in order:
        p.placed[:] = False
        p.issues = [i for i in p.issues if 'cadence' not in i]
        if not p.settings.enabled or p.epoch_h <= 0 or p.settings.epochs < 1:
            continue
        idx = np.flatnonzero(p.feasible)
        if idx.size == 0:
            continue
        if p.settings.cadence > 0 and p.settings.epochs > 1:
            chosen = _place_with_cadence(idx, p.settings.epochs, p.settings.cadence, load)
            if len(chosen) < p.settings.epochs:
                p.issues.append(f'only {len(chosen)} of {p.settings.epochs} epochs fit with a cadence of {p.settings.cadence} d')
        else:
            chosen = [min(chunk, key=lambda i: (round(load[i], 3), abs(i - chunk[len(chunk) // 2]))) for chunk in np.array_split(idx, min(p.settings.epochs, idx.size))]
        for i in chosen:
            p.placed[i] = True
            load[i] += p.epoch_h
    return load


def _place_with_cadence(idx: np.ndarray, epochs: int, cadence: int, load: np.ndarray) -> list:
    """
    Epoch nights among the feasible nights `idx` (night indices) about `cadence` days apart: the first
    epoch on the feasible night from which the most epochs fit (then the least total load, then the
    earliest), each next one on the feasible night nearest to previous + cadence, never closer than the
    cadence, within a tolerance of a third of the cadence; otherwise the next feasible night after that.
    """
    feasible = set(int(i) for i in idx)
    tol = max(1, cadence // 3)

    def chain(start: int) -> list:
        out, prev = [start], start
        while len(out) < epochs:
            target = prev + cadence
            cands = [i for i in range(target, target + tol + 1) if i in feasible]
            if not cands:
                later = [i for i in feasible if i > target + tol]
                if not later:
                    break
                cands = [min(later)]
            nxt = min(cands, key=lambda i: (round(load[i], 3), i))
            out.append(nxt)
            prev = nxt
        return out

    best = None
    starts = [int(i) for i in idx]
    if len(starts) > 80:                                                           # enough candidates, evenly over the range
        starts = [starts[k] for k in np.round(np.linspace(0, len(starts) - 1, 80)).astype(int)]
    for start in starts:
        c = chain(start)
        score = (-len(c), round(sum(load[i] for i in c), 3), c[0])
        if best is None or score < best[0]:
            best = (score, c)
    return best[1] if best else []


MIN_ALT_TEXT = 'the altitude limit'


def pack_night_slots(jobs: list) -> tuple[dict, dict]:
    """
    Share one night between jobs = [(key, demand_h, intervals)]: one job at a time, pre-emptive, the job with
    the least slack (remaining availability minus remaining demand) first; exact for jobs available in one
    interval. Returns (key -> hours obtained, key -> [(start, end), ...] slots in hours after twilight).
    """
    remaining = {k: float(d) for k, d, _ in jobs}
    got = {k: 0.0 for k, _, _ in jobs}
    slots = {k: [] for k, _, _ in jobs}
    ivs = {k: [(float(lo), float(hi)) for lo, hi in iv] for k, _, iv in jobs}
    edges = sorted({x for iv in ivs.values() for lo, hi in iv for x in (lo, hi)})
    for t0, t1 in zip(edges[:-1], edges[1:]):
        left = t1 - t0
        while left > 1e-9:
            ready = [k for k in remaining if remaining[k] > 1e-9 and any(lo - 1e-9 <= t0 and t1 <= hi + 1e-9 for lo, hi in ivs[k])]
            if not ready:
                break
            t = t1 - left
            k = min(ready, key=lambda k: sum(max(0.0, hi - max(lo, t)) for lo, hi in ivs[k]) - remaining[k])
            dt = min(left, remaining[k])
            remaining[k] -= dt
            got[k] += dt
            left -= dt
            if slots[k] and abs(slots[k][-1][1] - t) < 1e-9:
                slots[k][-1] = (slots[k][-1][0], t + dt)
            else:
                slots[k].append((t, t + dt))
    return got, slots


def pack_night(jobs: list) -> dict:
    """Hours each job obtains when the night is shared (see pack_night_slots)."""
    return pack_night_slots(jobs)[0]


def night_jobs(plans: dict, i: int) -> list:
    """The epochs assumed on night i: [(target key, hours, intervals)]."""
    return [(k, p.epoch_h, p.intervals[i]) for k, p in plans.items() if p.placed[i]]


def night_conflicts(plans: dict, dates: list) -> list:
    """Nights on which the epochs assumed there cannot all get their hours (time-of-night overlap)."""
    out = []
    for i, d in enumerate(dates):
        jobs = night_jobs(plans, i)
        if len(jobs) < 2:
            continue
        got = pack_night(jobs)
        short = {k: demand - got[k] for k, demand, _ in jobs if got[k] < demand - 1.0 / 60.0}       # more than a minute short
        if short:
            out.append({'date': d, 'targets': [k for k, _, _ in jobs], 'demand_h': sum(x for _, x, _ in jobs), 'scheduled_h': sum(got.values()), 'short': short})
    return out


# ------------------------------------------------------------------ schedule file
def load_schedule(folder: Path | str) -> dict:
    """Target key -> [TargetSettings, ...] (its parts) saved earlier; entries in another layout are skipped."""
    path = Path(folder) / SCHEDULE_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except ValueError:
        return {}
    out = {}
    for k, v in raw.items():
        if str(k).startswith('_'):
            continue
        items = v if isinstance(v, list) else [v]
        parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                parts.append(TargetSettings.from_dict(item))
            except (KeyError, ValueError):
                continue
        if parts:
            out[str(k)] = parts
    return out


BACKUP_DIR = 'backup'


def backup_schedule(folder: Path | str) -> Path | None:
    """Copy the current schedule file to backup/schedule_<its save time>.json (nothing when there is no file or the copy exists)."""
    path = Path(folder) / SCHEDULE_FILE
    if not path.is_file():
        return None
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
    target = Path(folder) / BACKUP_DIR / f'schedule_{stamp}.json'
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return target


def save_schedule(folder: Path | str, settings: dict) -> Path:
    """settings: target key -> [TargetSettings, ...]; a single part is written as one object, several as a list. The previous file is kept in backup/."""
    backup_schedule(folder)
    path = Path(folder) / SCHEDULE_FILE
    data = {'_description': 'Operator schedule of the Phase-2 requests: per requested target set (key = project|target) the obs_starttime / obs_endtime '
                            '(local nights, inclusive), frames per epoch (count) and number of epochs (target_epoch); a list when the target is split into parts.',
            '_saved': datetime.now().isoformat(timespec='seconds')}
    for k, parts in settings.items():
        items = parts if isinstance(parts, list) else [parts]
        data[k] = items[0].as_dict() if len(items) == 1 else [p.as_dict() for p in items]
    path.write_text(json.dumps(data, indent=2))
    return path


def schedule_rows(requests: list, plans: dict) -> list:
    """One row per part of every target set with the operator's entries: what goes into the request system."""
    rows = []
    for req in requests:
        for j, t in enumerate(req.targets):
            parts = plans_of(plans, req, j) or [None]
            for p in parts:
                s = p.settings if p else None
                rows.append({'project': req.project_id or '', 'file': req.name, 'pi': req.lead_pi, 'target': t.label, 'part': p.part if p else 1, 'parts': len(parts),
                             'type': t.target_type, 'tile': t.tile_id, 'ra': None if t.ra is None else round(t.ra, 5), 'dec': None if t.dec is None else round(t.dec, 5),
                             'mode': t.mode_text, 'exptime': t.exptime, 'requested_frames': t.nframes, 'requested_h': round(t.total_s / 3600.0, 2),
                             'obs_starttime': s.start_text if s and s.enabled else '', 'obs_endtime': s.end_text if s and s.enabled else '',
                             'count': s.count if s and s.enabled else None, 'target_epoch': s.epochs if s and s.enabled else None, 'cadence': s.cadence if s and s.enabled else None,
                             'epoch_h': round(p.epoch_h, 2) if p else None, 'planned_h': round(p.planned_h, 2) if p and s.enabled else None,
                             'dither_arcsec': req.dither_arcsec, 'sky_background': req.sky_background, 'status': p.status if p else '',
                             'obs_window_start': t.window_starts_text, 'obs_window_end': t.window_ends_text, 'note': s.note if s else '', 'requester_note': t.note})
    return rows
