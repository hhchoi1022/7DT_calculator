"""
(3) Observation overhead estimate.

Model
-----
The 16 units observe in parallel, so a target takes as long as its slowest unit; targets are
observed one after another. For one unit and one target:

    slew (once per target)
    for each filter of the unit's filter list:
        filter change                       (when the filter differs from the previous one)
        autofocus                           (depending on the autofocus policy)
        count x (exposure time + readout)

Per-event durations come from data/overhead.json.

Autofocus: none for Spec/specall (the survey's default mode keeps the units focused), otherwise one
autofocus per filter and unit (the number of filters of the mode times the autofocus overhead).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, ObsMode

OVERHEAD_JSON = DATA_DIR / 'overhead.json'

AUTOFOCUS_POLICIES = {
    'per_filter': 'Once per filter (every filter of the mode, in every unit)',
    'history': 'Once per filter, skipped when that filter was focused within the history window',
    'per_target': 'Once per target (before the first exposure only)',
    'none': 'No autofocus (focus assumed to be current)',
}
DEFAULT_AUTOFOCUS_POLICY = 'per_filter'
COMPONENTS = ('exposure', 'readout', 'filter_change', 'autofocus', 'slewing', 'dispatch')
COMPONENT_LABELS = {
    'exposure': 'Exposure (open shutter)',
    'readout': 'Readout',
    'filter_change': 'Filter change',
    'autofocus': 'Autofocus',
    'slewing': 'Slewing',
    'dispatch': 'Setup / dispatch',
}


@dataclass
class OverheadConfig:
    autofocus: float = 150.0
    filter_change: float = 10.0
    slewing: float = 5.0
    readout: float = 10.0
    dispatch: float = 0.0                     # per target: observation setup and hand-over to the next target
    autofocus_history_duration_min: float = 60.0
    source: str = ''

    @classmethod
    def load(cls, path: Path | str = OVERHEAD_JSON) -> 'OverheadConfig':
        path = Path(path)
        with open(path, 'r') as f:
            raw = json.load(f)
        kwargs = {k: float(raw[k]) for k in ('autofocus', 'filter_change', 'slewing', 'readout', 'dispatch', 'autofocus_history_duration_min') if k in raw}
        return cls(source=str(path), **kwargs)

    def as_dict(self) -> dict:
        return {
            'autofocus': self.autofocus, 'filter_change': self.filter_change,
            'slewing': self.slewing, 'readout': self.readout, 'dispatch': self.dispatch,
            'autofocus_history_duration_min': self.autofocus_history_duration_min,
        }


def allowed_autofocus_policies(obsmode: ObsMode) -> list[str]:
    """Spec/specall: no autofocus; every other mode: once per filter and unit."""
    return ['none'] if obsmode.is_specall else ['per_filter']


@dataclass
class ObsRequest:
    """One target's observation plan."""
    name: str
    exptime: float                 # s, per frame (same for every filter)
    count: int                     # frames per filter
    obsmode: ObsMode
    autofocus_policy: str = DEFAULT_AUTOFOCUS_POLICY
    policy_forced: bool = field(default=False, init=False)

    def __post_init__(self):
        self.exptime = float(self.exptime)
        self.count = int(self.count)
        if self.exptime < 0 or self.count < 1:
            raise ValueError('exptime must be >= 0 and count >= 1')
        if self.autofocus_policy not in AUTOFOCUS_POLICIES:
            raise ValueError(f'Unknown autofocus policy {self.autofocus_policy!r}')
        allowed = allowed_autofocus_policies(self.obsmode)
        if self.autofocus_policy not in allowed:
            self.autofocus_policy = allowed[0]
            self.policy_forced = True


@dataclass
class UnitBreakdown:
    unit: str
    filters: tuple
    n_frames: int
    n_filter_changes: int
    n_autofocus: int
    exposure: float
    readout: float
    filter_change: float
    autofocus: float

    @property
    def total(self) -> float:
        """Unit busy time for the target, excluding the (shared) slew."""
        return self.exposure + self.readout + self.filter_change + self.autofocus


@dataclass
class TargetBreakdown:
    request: ObsRequest
    slewing: float
    units: list
    critical_unit: str
    start: float                    # seconds after the start of the whole sequence
    dispatch: float = 0.0           # setup and hand-over time of this target

    @property
    def critical(self) -> UnitBreakdown:
        return next(u for u in self.units if u.unit == self.critical_unit)

    @property
    def total(self) -> float:
        return self.slewing + self.dispatch + self.critical.total

    @property
    def end(self) -> float:
        return self.start + self.total

    def components(self) -> dict:
        c = self.critical
        return {'exposure': c.exposure, 'readout': c.readout, 'filter_change': c.filter_change,
                'autofocus': c.autofocus, 'slewing': self.slewing, 'dispatch': self.dispatch}

    def counts(self) -> dict:
        c = self.critical
        return {'n_frames': c.n_frames, 'n_filter_changes': c.n_filter_changes,
                'n_autofocus': c.n_autofocus, 'n_slews': 1}


@dataclass
class OverheadResult:
    config: OverheadConfig
    targets: list

    @property
    def total(self) -> float:
        return sum(t.total for t in self.targets)

    @property
    def science_time(self) -> float:
        return sum(t.components()['exposure'] for t in self.targets)

    @property
    def overhead(self) -> float:
        return self.total - self.science_time

    @property
    def efficiency(self) -> float:
        return self.science_time / self.total if self.total > 0 else float('nan')

    def components(self) -> dict:
        out = {k: 0.0 for k in COMPONENTS}
        for t in self.targets:
            for k, v in t.components().items():
                out[k] += v
        return out

    def counts(self) -> dict:
        out = {'n_frames': 0, 'n_filter_changes': 0, 'n_autofocus': 0, 'n_slews': 0}
        for t in self.targets:
            for k, v in t.counts().items():
                out[k] += v
        return out

    def summary_lines(self) -> list[str]:
        """Human-readable breakdown, e.g. 'Filter change: 2 times x 10 s = 20 s'."""
        comp, cnt, cfg = self.components(), self.counts(), self.config
        lines = [
            f"Filter change: {cnt['n_filter_changes']} times x {cfg.filter_change:g} s = {comp['filter_change']:g} s",
            f"Autofocus: {cnt['n_autofocus']} times x {cfg.autofocus:g} s = {comp['autofocus']:g} s",
            f"Slewing: {cfg.slewing:g} s x {cnt['n_slews']} targets = {comp['slewing']:g} s",
            f"Setup / dispatch: {cfg.dispatch:g} s x {cnt['n_slews']} targets = {comp['dispatch']:g} s",
            f"Readout: {cfg.readout:g} s x {cnt['n_frames']} frames = {comp['readout']:g} s",
            f"Exposure: {comp['exposure']:g} s over {cnt['n_frames']} frames",
            f"Total: {self.total:g} s = {format_duration(self.total)} (overhead {self.overhead:g} s, efficiency {100 * self.efficiency:.0f} %)",
        ]
        return lines


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h} h {m:02d} m {s:02d} s'
    if m:
        return f'{m} m {s:02d} s'
    return f'{s} s'


def estimate(requests, config: OverheadConfig | None = None) -> OverheadResult:
    """Simulate the requests one after another and return the timing breakdown."""
    cfg = config or OverheadConfig.load()
    history: dict = {}          # (unit, filter) -> time [s] of the last autofocus in that filter
    clock = 0.0
    targets = []
    for req in requests:
        units = []
        for unit, filters in req.obsmode.filters_by_unit.items():
            t = 0.0
            n_fc = n_af = n_frames = 0
            current = None
            if req.autofocus_policy == 'per_target':
                n_af += 1
                t += cfg.autofocus
            for filt in filters:
                if filt != current:
                    n_fc += 1
                    t += cfg.filter_change
                    current = filt
                    if req.autofocus_policy == 'per_filter':
                        n_af += 1
                        t += cfg.autofocus
                    elif req.autofocus_policy == 'history':
                        now = clock + cfg.slewing + t
                        last = history.get((unit, filt))
                        if last is None or (now - last) > cfg.autofocus_history_duration_min * 60.0:
                            n_af += 1
                            t += cfg.autofocus
                            history[(unit, filt)] = now
                n_frames += req.count
                t += req.count * (req.exptime + cfg.readout)
            units.append(UnitBreakdown(
                unit=unit, filters=tuple(filters), n_frames=n_frames,
                n_filter_changes=n_fc, n_autofocus=n_af,
                exposure=n_frames * req.exptime, readout=n_frames * cfg.readout,
                filter_change=n_fc * cfg.filter_change, autofocus=n_af * cfg.autofocus,
            ))
        if not units:
            raise ValueError(f'Observation mode {req.obsmode.label} assigns no filters to any unit')
        critical = max(units, key=lambda u: u.total)
        breakdown = TargetBreakdown(request=req, slewing=cfg.slewing, units=units,
                                    critical_unit=critical.unit, start=clock, dispatch=cfg.dispatch)
        clock += breakdown.total
        targets.append(breakdown)
    return OverheadResult(config=cfg, targets=targets)


COMPONENT_COLORS = {'slewing': '#9467bd', 'dispatch': '#c5b0d5', 'autofocus': '#ff7f0e', 'filter_change': '#d62728', 'exposure': '#2ca02c', 'readout': '#7f7f7f'}


def plot_timeline(result: OverheadResult):
    """Stacked horizontal bars (minutes): the time budget of every target and of the whole sequence."""
    import matplotlib.pyplot as plt

    order = ('dispatch', 'slewing', 'autofocus', 'filter_change', 'exposure', 'readout')
    labels = [t.request.name for t in result.targets]
    comps = [t.components() for t in result.targets]
    if len(result.targets) > 1:                      # a summary bar only when there is something to sum
        labels.append('All targets')
        comps.append(result.components())
    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.55 * len(labels)), dpi=110)
    left = [0.0] * len(labels)
    for key in order:
        widths = [c[key] / 60.0 for c in comps]
        ax.barh(labels, widths, left=left, color=COMPONENT_COLORS[key], label=COMPONENT_LABELS[key], edgecolor='white', height=0.6)
        left = [l + w for l, w in zip(left, widths)]
    totals = [t.total for t in result.targets] + ([result.total] if len(result.targets) > 1 else [])
    for i, tot in enumerate(totals):
        ax.text(tot / 60.0 * 1.01, i, format_duration(tot), va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Minutes')
    ax.set_xlim(0, max(totals) / 60.0 * 1.25 if totals else 1)
    ax.legend(fontsize=7, ncol=6, loc='lower right', bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig
