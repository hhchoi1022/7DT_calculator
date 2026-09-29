"""
(2) Exposure time / SNR calculator.

Built on the empirical 7DT depth model (fitted on the 5-sigma limiting magnitudes, UL5, of
archived 100 s frames). Per filter, at the reference exposure time t_ref:

    UL5_ref = b0 + b1*moon_phase + b2*moon_sep + b3*hour_since_sunset + b4*seeing
    UL5(t)  = UL5_ref + 1.25 * log10(t / t_ref)
    SNR_1(t) = 5 * 10**(-0.4 * (mag - UL5(t)))            single frame
    SNR_N    = SNR_1 * sqrt(N)                              N stacked frames
    t_frame  = t_ref * 10**((mag - UL5_ref) / 1.25) * (SNR / 5)**2 / N   to reach SNR with N frames

Filters without a fitted model (new medium bands, wide bands) are handled by interpolating the
model coefficients of the neighbouring medium bands in effective wavelength; such results are
flagged 'interpolated' (or 'extrapolated' outside the modelled wavelength range).
"""
from __future__ import annotations

import glob
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroplan import Observer
from astropy.coordinates import NonRotationTransformationWarning, SkyCoord, get_body
from astropy.time import Time

from .config import DATA_DIR, Site
from .filters import filter_colors, is_wide_band, sort_filters
from .photometry import effective_wavelength, get_filter_curve
from .targets import Target
from .visibility import as_time, make_observer

UL5_SNR = 5.0                       # the limiting magnitude of the model is defined at SNR = 5
DEPTH_APERTURE_DIAMETER = 10.0      # arcsec: the depth (UL5) is measured in this aperture
DEPTH_APERTURE_AREA = math.pi * (DEPTH_APERTURE_DIAMETER / 2) ** 2      # arcsec^2
APERTURE_OFFSET = 2.5 * math.log10(DEPTH_APERTURE_AREA)                  # surface brightness -> magnitude in the aperture
# For a uniform extended source the SNR over one arcsec^2 equals the point-source SNR of a magnitude
# mu - 1.25 log10(A0): the flux scales with the area, the noise with its square root.
SB_OFFSET = 0.5 * APERTURE_OFFSET
SHORT_EXPTIME = 30.0                # s: below this the scaling from 100 s frames becomes unreliable (read noise)
SHORT_EXPTIME_NOTE = (f'WARNING: exposure shorter than {SHORT_EXPTIME:g} s. The depth is scaled from 100 s frames as if the '
                      'sky noise dominated, so the noise estimate may be inaccurate.')


def _note(exptime: float, depth_note: str) -> str:
    parts = ([SHORT_EXPTIME_NOTE] if exptime < SHORT_EXPTIME else []) + ([depth_note] if depth_note else [])
    return ' | '.join(parts)


def extended_saturation_shift(seeing_arcsec: float) -> float:
    """
    Add this to the point-source saturation magnitude to get the surface brightness [mag/arcsec^2] that
    loads a pixel as much as the peak pixel of a star at that magnitude (Gaussian PSF of the given seeing).
    """
    sigma = max(float(seeing_arcsec), 0.3) / 2.355
    return 2.5 * math.log10(2 * math.pi * sigma ** 2)
MAG_ERR_FACTOR = 2.5 / math.log(10)  # sigma_mag = 1.0857 / SNR
DEFAULT_CONDITIONS = {'moon_phase': 0.5, 'moon_separation': 90.0, 'hour_since_sunset': 5.0, 'seeing': 2.0}
_MODEL_MEDIUM_RE = re.compile(r'^m\d{3}$')


def latest_model_path(directory: Path | str = DATA_DIR) -> Path:
    paths = sorted(glob.glob(str(Path(directory) / 'depth_model_*.json')))
    if not paths:
        raise FileNotFoundError(f'No depth_model_*.json in {directory}')
    return Path(paths[-1])


# ------------------------------------------------------------------ conditions
@dataclass
class Conditions:
    moon_phase: float
    moon_separation: float
    hour_since_sunset: float
    seeing: float
    obstime: Time | None = None
    source: str = 'default'
    notes: list = field(default_factory=list)

    def describe(self) -> list:
        lines = []
        if self.obstime is not None:
            lines.append(f'Obs time = {self.obstime.iso[:16]} UT')
        lines += [f'Moon phase = {self.moon_phase:.2f}', f'Moon separation = {self.moon_separation:.1f} deg',
                  f'Hours since sunset = {self.hour_since_sunset:.1f} h', f'Seeing = {self.seeing:.1f}"']
        return lines


def moon_phase_at(observer: Observer, time: Time) -> float:
    return float(observer.moon_illumination(time))


def moon_separation_at(ra: float, dec: float, time: Time) -> float:
    """Geocentric Moon-target separation [deg] (this is how the depth model was fitted)."""
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NonRotationTransformationWarning)
        return float(get_body('moon', time).separation(target).deg)


def hour_since_sunset_at(observer: Observer, time: Time) -> float:
    """
    Hours since astronomical sunset (Sun at -18 deg). The sunset is the one preceding the next
    sunrise (Sun at +10 deg), so the value stays on the same night through morning twilight.
    Identical to the definition used when the depth model was fitted; do not change.
    """
    sunrise = observer.sun_rise_time(time, which='next', horizon=10 * u.deg)
    sunset = observer.sun_set_time(sunrise, which='previous', horizon=-18 * u.deg)
    hours = (time.mjd - sunset.mjd) * 24.0
    if hours < 0:
        hours += 24.0
    return float(hours)


def conditions_for(ra: float | None = None, dec: float | None = None, obstime=None, seeing: float | None = None,
                   site: Site | None = None, observer: Observer | None = None, **overrides) -> Conditions:
    """
    Observing conditions for the depth model. With a target and time, moon phase, moon separation
    and hours since sunset are computed; explicit keyword values (moon_phase=, ...) override.
    Seeing is never predicted: it defaults to 2.0" unless given.
    """
    values = dict(DEFAULT_CONDITIONS)
    source = 'default'
    obstime = as_time(obstime) if obstime is not None else None
    if obstime is not None:
        observer = observer or make_observer(site)
        values['moon_phase'] = moon_phase_at(observer, obstime)
        values['hour_since_sunset'] = hour_since_sunset_at(observer, obstime)
        source = 'time'
        if ra is not None and dec is not None:
            values['moon_separation'] = moon_separation_at(ra, dec, obstime)
            source = 'target and time'
    if seeing is not None:
        values['seeing'] = float(seeing)
    for key, value in overrides.items():
        if key in values and value is not None:
            values[key] = float(value)
            source = 'manual'
    notes = []
    if values['hour_since_sunset'] > 13:
        notes.append(f"hours since sunset = {values['hour_since_sunset']:.1f} h: the requested time is not during the night, "
                     'so the sky-brightness term of the model is outside its fitted range (0-12 h)')
    return Conditions(obstime=obstime, source=source, notes=notes, **values)


# ------------------------------------------------------------------ depth model
@dataclass
class FilterDepth:
    filter: str
    lam_eff: float | None
    ul5_ref: float
    t_ref: float
    scatter: float
    status: str            # 'model' | 'interpolated' | 'extrapolated' | 'unavailable'
    note: str = ''

    @property
    def available(self) -> bool:
        return self.status != 'unavailable' and math.isfinite(self.ul5_ref)


class DepthModel:
    """The fitted depth model (data/depth_model_*.json) plus wavelength interpolation for other filters."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else latest_model_path()
        with open(self.path, 'r') as f:
            self.model = json.load(f)
        self.filters = sort_filters(self.model)
        anchors = []
        for name in self.model:
            if _MODEL_MEDIUM_RE.match(name):
                lam = effective_wavelength(name)
                if lam is not None:
                    anchors.append((lam, name))
        self.anchors = sorted(anchors)
        if not self.anchors:
            raise ValueError('Depth model has no medium-band filters to interpolate from')

    @property
    def name(self) -> str:
        return self.path.name

    def _params(self, name: str) -> tuple:
        m = self.model[name]
        return float(m['intercept']), np.asarray(m['coefficients'], dtype=float), float(m.get('scatter', np.nan)), float(m.get('exptime', 100))

    def parameters(self, filt: str) -> tuple:
        """(intercept, coefficients, scatter, t_ref, status, note) for any filter name."""
        filt = str(filt)
        if filt in self.model:
            b0, b, s, t = self._params(filt)
            return b0, b, s, t, 'model', ''
        lam = effective_wavelength(filt)
        if lam is None:
            return np.nan, np.full(4, np.nan), np.nan, 100.0, 'unavailable', f'no depth model and unknown wavelength for {filt}'
        wide = ' (wide band: the larger bandwidth is not accounted for)' if is_wide_band(filt) else ''
        lams = [a[0] for a in self.anchors]
        if lam <= lams[0] or lam >= lams[-1]:
            _, nearest = min(self.anchors, key=lambda a: abs(a[0] - lam))
            b0, b, s, t = self._params(nearest)
            return b0, b, s, t, 'extrapolated', f'no depth model for {filt}: using the {nearest} model (nearest medium band){wide}'
        j = int(np.searchsorted(lams, lam))
        (l1, n1), (l2, n2) = self.anchors[j - 1], self.anchors[j]
        w = (lam - l1) / (l2 - l1)
        b0a, ba, sa, ta = self._params(n1)
        b0b, bb, sb, tb = self._params(n2)
        b0 = (1 - w) * b0a + w * b0b
        b = (1 - w) * ba + w * bb
        s = (1 - w) * sa + w * sb
        return b0, b, s, ta, 'interpolated', f'no depth model for {filt}: interpolated between {n1} and {n2} at {lam:.0f} A{wide}'

    def depth(self, filt: str, cond: Conditions) -> FilterDepth:
        b0, b, s, t_ref, status, note = self.parameters(filt)
        if status == 'unavailable':
            return FilterDepth(filter=str(filt), lam_eff=None, ul5_ref=np.nan, t_ref=t_ref, scatter=np.nan, status=status, note=note)
        ul5 = b0 + b[0] * cond.moon_phase + b[1] * cond.moon_separation + b[2] * cond.hour_since_sunset + b[3] * cond.seeing
        return FilterDepth(filter=str(filt), lam_eff=effective_wavelength(filt), ul5_ref=float(ul5), t_ref=t_ref, scatter=s, status=status, note=note)


# ------------------------------------------------------------------ arithmetic
def ul5_at(ul5_ref: float, t_ref: float, exptime: float, count: int = 1) -> float:
    return ul5_ref + 1.25 * math.log10(max(exptime, 1e-9) * max(count, 1) / t_ref)


def snr_for(mag: float, ul5_ref: float, t_ref: float, exptime, count: int = 1):
    """SNR of `count` stacked frames of `exptime` seconds each (exptime may be an array)."""
    exptime = np.asarray(exptime, dtype=float)
    with np.errstate(divide='ignore'):
        ul5 = ul5_ref + 1.25 * np.log10(exptime / t_ref)
    return UL5_SNR * 10 ** (-0.4 * (mag - ul5)) * math.sqrt(max(count, 1))


def exptime_for(mag: float, ul5_ref: float, t_ref: float, snr, count: int = 1):
    """Per-frame exposure time needed to reach `snr` with `count` stacked frames (snr may be an array)."""
    snr = np.asarray(snr, dtype=float)
    return t_ref * 10 ** ((mag - ul5_ref) / 1.25) * (snr / UL5_SNR) ** 2 / max(count, 1)


@dataclass
class SNRResult:
    filter: str
    lam_eff: float | None
    mag_ab: float
    mag_source: str
    exptime: float
    count: int                 # frames per filter and unit
    ul5_single: float
    ul5_stacked: float
    snr_single: float
    snr_stacked: float
    mag_err: float
    depth: FilterDepth
    units: int = 1             # units observing this filter at the same time
    extended: bool = False     # mag_ab is a surface brightness; SNRs are per arcsec^2
    snr_target: float = float('nan')     # set by exptime_needed()
    time_needed: float = float('nan')    # total exposure time per unit [s] to reach snr_target (continuous)

    @property
    def frames(self) -> int:
        """Frames that get stacked: count x units."""
        return max(self.count, 1) * max(self.units, 1)

    @property
    def available(self) -> bool:
        return self.depth.available and math.isfinite(self.mag_ab)

    @property
    def snr_aperture(self) -> float:
        """Extended sources: SNR of the flux inside the depth aperture (10 arcsec), for reference."""
        return self.snr_stacked * math.sqrt(DEPTH_APERTURE_AREA) if self.extended else float('nan')

    def row_time_needed(self) -> dict:
        """Table row of the 'exposure time needed' mode."""
        shift = SB_OFFSET if self.extended else 0.0
        return {'filter': self.filter, 'lam_eff_A': self.lam_eff, 'mag_AB': self.mag_ab, 'mag_source': self.mag_source,
                'SNR_target': self.snr_target, 'units': self.units,
                'total_exptime_per_unit_s': self.time_needed, 'total_exptime_per_unit_min': self.time_needed / 60.0,
                f'frames_of_{self.exptime:g}s_per_unit': self.count, 'SNR_reached': self.snr_stacked,
                'UL5_single': self.ul5_single + shift, 'model': self.depth.status, 'note': _note(self.exptime, self.depth.note)}

    def row(self) -> dict:
        """
        Same columns for both source types. For an extended source every magnitude is a surface
        brightness [mag/arcsec^2]: mag_AB, the 5-sigma limits UL5_* (limit of the surface brightness
        over one arcsec^2) and mag_err, and the SNRs are per arcsec^2.
        """
        shift = SB_OFFSET if self.extended else 0.0
        return {'filter': self.filter, 'lam_eff_A': self.lam_eff, 'mag_AB': self.mag_ab, 'mag_source': self.mag_source,
                'exptime_s': self.exptime, 'count': self.count, 'units': self.units, 'frames_stacked': self.frames,
                'UL5_single': self.ul5_single + shift, 'UL5_stacked': self.ul5_stacked + shift,
                'SNR_single': self.snr_single, 'SNR_stacked': self.snr_stacked, 'mag_err': self.mag_err,
                'model': self.depth.status, 'note': _note(self.exptime, self.depth.note)}


@dataclass
class ExptimeResult:
    filter: str
    lam_eff: float | None
    mag_ab: float
    mag_source: str
    snr_target: float
    count: int                 # frames per filter and unit
    exptime_single: float      # per frame
    exptime_total: float       # per unit: exptime x count
    ul5_ref: float
    depth: FilterDepth
    units: int = 1

    @property
    def frames(self) -> int:
        return max(self.count, 1) * max(self.units, 1)

    @property
    def available(self) -> bool:
        return self.depth.available and math.isfinite(self.mag_ab)

    @property
    def mag_err(self) -> float:
        return MAG_ERR_FACTOR / self.snr_target

    def row(self) -> dict:
        return {'filter': self.filter, 'lam_eff_A': self.lam_eff, 'mag_AB': self.mag_ab, 'mag_source': self.mag_source,
                'SNR_target': self.snr_target, 'count': self.count, 'units': self.units, 'frames_stacked': self.frames,
                'exptime_per_frame_s': self.exptime_single, 'exptime_per_unit_s': self.exptime_total,
                f'UL5_at_{self.depth.t_ref:g}s': self.ul5_ref, 'model': self.depth.status, 'note': self.depth.note}


class ExposureCalculator:
    def __init__(self, model: DepthModel | None = None):
        self.model = model or DepthModel()

    @staticmethod
    def effective_magnitude(target: Target, mag: float) -> float:
        """
        Magnitude compared with the point-source depth. For an extended source the surface brightness
        (mag/arcsec^2) is shifted by 1.25 log10(A0) so that the SNR formula yields the SNR per arcsec^2.
        """
        return mag - SB_OFFSET if getattr(target, 'extended', False) else mag

    def snr(self, target: Target, filters, exptime: float, count: int, cond: Conditions, units: dict | None = None) -> list:
        """
        SNR per filter. `count` frames per filter and unit; `units` maps a filter to the number of units
        that observe it at the same time (from the observation mode), so count x units frames are stacked.
        Extended sources (surface brightness) get the SNR of the surface brightness per arcsec^2.
        """
        out = []
        ext = bool(getattr(target, 'extended', False))
        for filt in filters:
            n_units = int((units or {}).get(filt, 1)) or 1
            frames = max(int(count), 1) * n_units
            mag, source = target.ab_magnitude(filt)
            depth = self.model.depth(filt, cond)
            if depth.available and math.isfinite(mag):
                s1 = float(snr_for(self.effective_magnitude(target, mag), depth.ul5_ref, depth.t_ref, exptime, 1))
                sn = s1 * math.sqrt(frames)
                out.append(SNRResult(filt, depth.lam_eff, mag, source, float(exptime), int(count),
                                     ul5_at(depth.ul5_ref, depth.t_ref, exptime), ul5_at(depth.ul5_ref, depth.t_ref, exptime, frames),
                                     s1, sn, MAG_ERR_FACTOR / sn if sn > 0 else np.inf, depth, units=n_units, extended=ext))
            else:
                out.append(SNRResult(filt, depth.lam_eff, mag, source, float(exptime), int(count), np.nan, np.nan, np.nan, np.nan, np.nan, depth, units=n_units, extended=ext))
        return out

    def exptime_needed(self, target: Target, filters, exptime: float, snr_target: float, cond: Conditions, units: dict | None = None) -> list:
        """
        Total exposure time per unit needed to reach `snr_target` in each filter, given the number of
        units observing it, plus the equivalent number of frames of `exptime` seconds (rounded up)
        and the SNR those frames actually reach.
        """
        out = self.frames_needed(target, filters, exptime, snr_target, cond, units)
        for r in out:
            r.snr_target = float(snr_target)
            if r.available and r.snr_single > 0:
                frames_total = (snr_target / r.snr_single) ** 2          # continuous, single-unit equivalent
                r.time_needed = frames_total * r.exptime / max(r.units, 1)
        return out

    def frames_needed(self, target: Target, filters, exptime: float, snr_target: float, cond: Conditions, units: dict | None = None) -> list:
        """
        Frames of `exptime` seconds needed per filter to reach `snr_target` after stacking, given the
        number of units observing each filter. Returns SNRResult objects whose count is the number of
        frames per unit and whose SNR is the one actually reached with that many frames.
        """
        out = []
        ext = bool(getattr(target, 'extended', False))
        for filt in filters:
            n_units = int((units or {}).get(filt, 1)) or 1
            mag, source = target.ab_magnitude(filt)
            depth = self.model.depth(filt, cond)
            if depth.available and math.isfinite(mag):
                s1 = float(snr_for(self.effective_magnitude(target, mag), depth.ul5_ref, depth.t_ref, exptime, 1))
                total = max(int(math.ceil((snr_target / s1) ** 2)), 1)
                count = max(int(math.ceil(total / n_units)), 1)
                frames = count * n_units
                sn = s1 * math.sqrt(frames)
                out.append(SNRResult(filt, depth.lam_eff, mag, source, float(exptime), count,
                                     ul5_at(depth.ul5_ref, depth.t_ref, exptime), ul5_at(depth.ul5_ref, depth.t_ref, exptime, frames),
                                     s1, sn, MAG_ERR_FACTOR / sn if sn > 0 else np.inf, depth, units=n_units, extended=ext))
            else:
                out.append(SNRResult(filt, depth.lam_eff, mag, source, float(exptime), 0, np.nan, np.nan, np.nan, np.nan, np.nan, depth, units=n_units, extended=ext))
        return out

    def exptime(self, target: Target, filters, snr: float, count: int, cond: Conditions, units: dict | None = None) -> list:
        """Per-frame exposure time reaching `snr` after stacking count x units frames."""
        out = []
        for filt in filters:
            n_units = int((units or {}).get(filt, 1)) or 1
            frames = max(int(count), 1) * n_units
            mag, source = target.ab_magnitude(filt)
            depth = self.model.depth(filt, cond)
            if depth.available and math.isfinite(mag):
                t1 = float(exptime_for(self.effective_magnitude(target, mag), depth.ul5_ref, depth.t_ref, snr, frames))
                out.append(ExptimeResult(filt, depth.lam_eff, mag, source, float(snr), int(count), t1, t1 * max(int(count), 1), depth.ul5_ref, depth, units=n_units))
            else:
                out.append(ExptimeResult(filt, depth.lam_eff, mag, source, float(snr), int(count), np.nan, np.nan, np.nan, depth, units=n_units))
        return out


# ----------------------------------------------------------------------- plots
def _conditions_box(ax, cond: Conditions, extra: list | None = None):
    lines = list(extra or []) + cond.describe()
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6', alpha=0.9))


def _colors(results) -> dict:
    waves = {r.filter: r.lam_eff for r in results if r.lam_eff}
    return filter_colors([r.filter for r in results], waves)


def _style(depth: FilterDepth) -> str:
    return {'model': '-', 'interpolated': '--', 'extrapolated': ':'}.get(depth.status, '-')


def plot_snr_vs_exptime(target: Target, results: list, cond: Conditions, time_range=(10.0, 30000.0), n_points: int = 150):
    """
    SNR against the total exposure time per unit (units observing the same filter included); the dots
    mark the requested exptime x count and the SNR reached there.
    """
    avail = [r for r in results if r.available]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    if not avail:
        ax.text(0.5, 0.5, 'No filter could be evaluated', ha='center', va='center', transform=ax.transAxes)
        return fig
    colors = _colors(avail)
    total = avail[0].exptime * max(avail[0].count, 1)
    t = np.logspace(np.log10(min(time_range[0], total)), np.log10(max(time_range[1], total)), n_points)
    for r in avail:
        m_eff = r.mag_ab - SB_OFFSET if getattr(r, 'extended', False) else r.mag_ab
        s = snr_for(m_eff, r.depth.ul5_ref, r.depth.t_ref, t, r.units)
        ax.plot(t, s, color=colors[r.filter], ls=_style(r.depth), lw=1.6,
                label=f'{r.filter}: SNR = {r.snr_stacked:.1f}' + (f' ({r.units} units)' if r.units > 1 else ''))
        ax.plot([r.exptime * max(r.count, 1)], [r.snr_stacked], 'o', color=colors[r.filter], ms=5)
    ax.axvline(total, color='k', ls='--', lw=1, alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Total exposure time per unit [s]')
    ax.set_ylabel('SNR per arcsec²' if getattr(avail[0], 'extended', False) else 'SNR')
    ax.set_title(f'{target.label}: SNR vs total exposure time', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2 if len(avail) > 16 else 1, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _conditions_box(ax, cond, [f'{avail[0].exptime:g} s x {avail[0].count} = {total:g} s per unit'])
    _model_legend_note(ax, avail)
    fig.tight_layout()
    return fig


def plot_exptime_vs_snr(target: Target, results: list, cond: Conditions, snr_range=(1.0, 100.0), n_points: int = 120):
    """Required per-frame exposure time versus target SNR, one curve per filter."""
    avail = [r for r in results if r.available]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    if not avail:
        ax.text(0.5, 0.5, 'No filter could be evaluated', ha='center', va='center', transform=ax.transAxes)
        return fig
    colors = _colors(avail)
    s = np.logspace(np.log10(snr_range[0]), np.log10(snr_range[1]), n_points)
    for r in avail:
        t = exptime_for(r.mag_ab, r.ul5_ref, r.depth.t_ref, s, r.frames)
        ax.plot(s, t, color=colors[r.filter], ls=_style(r.depth), lw=1.6, label=f'{r.filter}: {r.exptime_single:.0f} s' + (f' ({r.units} units)' if r.units > 1 else ''))
        ax.plot([r.snr_target], [r.exptime_single], 'o', color=colors[r.filter], ms=5)
    ax.axvline(avail[0].snr_target, color='k', ls='--', lw=1, alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(s[0], s[-1])
    ax.set_xlabel('SNR of the stacked frames (count x units per filter)')
    ax.set_ylabel('Exposure time per frame [s]')
    ax.set_title(f'{target.label}: exposure time vs SNR', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2 if len(avail) > 16 else 1, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _conditions_box(ax, cond, [f'SNR = {avail[0].snr_target:g} with {avail[0].count} frame(s)'])
    _model_legend_note(ax, avail)
    fig.tight_layout()
    return fig


def _model_legend_note(ax, results):
    statuses = {r.depth.status for r in results}
    notes = []
    if 'interpolated' in statuses:
        notes.append('dashed: interpolated model')
    if 'extrapolated' in statuses:
        notes.append('dotted: extrapolated model')
    if notes:
        ax.text(0.98, 0.02, '\n'.join(notes), transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='0.3')


def plot_spectrum_points(target: Target, results: list, noise_seed: int | None = None, title: str | None = None):
    """
    Expected 7DT photometry: one point per filter at its effective wavelength with the error bar
    1.0857/SNR (horizontal bar = filter FWHM), on top of the target spectrum in AB magnitudes.
    With `noise_seed`, a random Gaussian realisation of the measured magnitudes is drawn.
    """
    avail = [r for r in results if r.available and r.lam_eff]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=110)
    spec = target.effective_spectrum()
    if spec is not None:
        mag = spec.ab_mag_curve()
        ax.plot(spec.wavelength, mag, color='0.4', lw=0.9, alpha=0.8, label='input spectrum')
    if not avail:
        ax.text(0.5, 0.5, 'No filter could be evaluated', ha='center', va='center', transform=ax.transAxes)
        return fig
    colors = _colors(avail)
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    for r in avail:
        err = r.mag_err
        mag = r.mag_ab + (rng.normal(0, err) if (rng is not None and math.isfinite(err)) else 0.0)
        curve = get_filter_curve(r.filter)
        xerr = curve.fwhm / 2 if curve is not None else None
        ax.errorbar(r.lam_eff, mag, yerr=err if math.isfinite(err) else None, xerr=xerr, fmt='o', color=colors[r.filter],
                    ms=5, capsize=2, lw=1.2, label=r.filter if len(avail) <= 12 else None)
    mags = [r.mag_ab for r in avail]
    errs = [r.mag_err for r in avail if math.isfinite(r.mag_err)]
    pad = float(np.clip(2.5 * np.nanmedian(errs), 0.3, 1.5)) if errs else 0.3
    lo, hi = min(mags) - pad, max(mags) + pad
    if spec is not None:
        m = spec.ab_mag_curve()
        m = m[np.isfinite(m) & (spec.wavelength > 3500) & (spec.wavelength < 9500)]
        if m.size:
            lo, hi = min(lo, np.nanpercentile(m, 2) - 0.2), max(hi, np.nanpercentile(m, 98) + 0.2)
    ax.set_ylim(hi, lo)
    ax.set_xlim(3400, 9600)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('AB magnitude / arcsec²' if getattr(avail[0], 'extended', False) else 'AB magnitude')
    ax.set_title(title or f'{target.label}: expected 7DT photometry' + (' (noise realisation)' if rng is not None else ''), fontsize=11)
    ax.grid(alpha=0.3)
    if len(avail) <= 12 or spec is not None:
        ax.legend(fontsize=7, ncol=3, loc='lower right')
    fig.tight_layout()
    return fig


def plot_snr_vs_frames(target: Target, results: list, cond: Conditions, snr_target: float, max_frames: int = 300):
    """SNR against the number of stacked frames (count x units) at the chosen per-frame exposure time."""
    avail = [r for r in results if r.available]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    if not avail:
        ax.text(0.5, 0.5, 'No filter could be evaluated', ha='center', va='center', transform=ax.transAxes)
        return fig
    colors = _colors(avail)
    n = np.arange(1, max_frames + 1)
    for r in avail:
        ax.plot(n, r.snr_single * np.sqrt(n), color=colors[r.filter], ls=_style(r.depth), lw=1.6,
                label=f'{r.filter}: {r.count} frame(s) per unit' + (f' x {r.units} units' if r.units > 1 else ''))
        ax.plot([r.frames], [r.snr_stacked], 'o', color=colors[r.filter], ms=5)
    ax.axhline(snr_target, color='k', ls='--', lw=1, alpha=0.6)
    ax.text(n[0] * 1.1, snr_target * 1.08, f'target SNR = {snr_target:g}', fontsize=8, color='0.3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(n[0], n[-1])
    ax.set_xlabel('Stacked frames (count x units)')
    ax.set_ylabel('SNR per arcsec²' if getattr(avail[0], 'extended', False) else 'SNR')
    ax.set_title(f'{target.label}: SNR vs number of frames ({avail[0].exptime:g} s each)', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2 if len(avail) > 16 else 1, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _conditions_box(ax, cond, [f'exptime = {avail[0].exptime:g} s per frame'])
    _model_legend_note(ax, avail)
    fig.tight_layout()
    return fig


def plot_snr_vs_total_time(target: Target, results: list, cond: Conditions, snr_target: float, time_range=(10.0, 30000.0), n_points: int = 150):
    """
    SNR against the total exposure time per unit (frames of any length stacked, units observing the
    same filter included). The dots mark the time needed for the target SNR in each filter.
    """
    avail = [r for r in results if r.available]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    if not avail:
        ax.text(0.5, 0.5, 'No filter could be evaluated', ha='center', va='center', transform=ax.transAxes)
        return fig
    colors = _colors(avail)
    t = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]), n_points)
    for r in avail:
        m_eff = r.mag_ab - SB_OFFSET if getattr(r, 'extended', False) else r.mag_ab
        s = snr_for(m_eff, r.depth.ul5_ref, r.depth.t_ref, t, r.units)          # SNR after T seconds on each of the units
        label = f'{r.filter}: {r.time_needed:.0f} s' + (f' ({r.units} units)' if r.units > 1 else '')
        ax.plot(t, s, color=colors[r.filter], ls=_style(r.depth), lw=1.6, label=label)
        if math.isfinite(r.time_needed):
            ax.plot([r.time_needed], [snr_target], 'o', color=colors[r.filter], ms=5)
    ax.axhline(snr_target, color='k', ls='--', lw=1, alpha=0.6)
    ax.text(t[0] * 1.1, snr_target * 1.08, f'target SNR = {snr_target:g}', fontsize=8, color='0.3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Total exposure time per unit [s]')
    ax.set_ylabel('SNR per arcsec²' if getattr(avail[0], 'extended', False) else 'SNR')
    ax.set_title(f'{target.label}: SNR vs total exposure time', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2 if len(avail) > 16 else 1, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _conditions_box(ax, cond, [f'frames of {avail[0].exptime:g} s'])
    _model_legend_note(ax, avail)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ interactive plots
def _plotly_color(c) -> str:
    from matplotlib.colors import to_hex
    return to_hex(c)


def _plotly_dash(depth: FilterDepth) -> str:
    return {'model': 'solid', 'interpolated': '7px,4px', 'extrapolated': '2px,3px'}.get(depth.status, 'solid')


def _plotly_note(fig, results, extra: str | None = None):
    statuses = {r.depth.status for r in results}
    notes = [extra] if extra else []
    if 'interpolated' in statuses:
        notes.append('dashed: interpolated model')
    if 'extrapolated' in statuses:
        notes.append('dotted: extrapolated model')
    if notes:
        fig.add_annotation(x=0.99, y=0.01, xref='paper', yref='paper', text='<br>'.join(notes), showarrow=False,
                           xanchor='right', yanchor='bottom', font=dict(size=9, color='#555'), align='right')


def _plotly_layout(fig, title: str, xlabel: str, ylabel: str, height: int = 400):
    fig.update_layout(title=dict(text=title, font=dict(size=13)), height=height, hovermode='closest', plot_bgcolor='white', font=dict(size=11),
                      margin=dict(l=50, r=20, t=45, b=45), legend=dict(font=dict(size=9), itemsizing='trace', itemwidth=45, tracegroupgap=0))
    fig.update_xaxes(title=dict(text=xlabel, font=dict(size=12)), tickfont=dict(size=10), showgrid=True, gridcolor='#e6e6e6', showline=True, linecolor='#444', mirror=True, ticks='outside')
    fig.update_yaxes(title=dict(text=ylabel, font=dict(size=12)), tickfont=dict(size=10), showgrid=True, gridcolor='#e6e6e6', showline=True, linecolor='#444', mirror=True, ticks='outside')
    return fig


def _empty_plotly(title: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No filter could be evaluated', showarrow=False)
    return _plotly_layout(fig, title, '', '')


def plot_snr_vs_exptime_interactive(target: Target, results: list, cond: Conditions, time_range=(10.0, 30000.0), n_points: int = 120):
    """Plotly version of plot_snr_vs_exptime: SNR against the total exposure time per unit, one curve per filter."""
    import plotly.graph_objects as go
    avail = [r for r in results if r.available]
    title = f'{target.label}: SNR vs total exposure time'
    if not avail:
        return _empty_plotly(title)
    colors = _colors(avail)
    extended = getattr(avail[0], 'extended', False)
    total = avail[0].exptime * max(avail[0].count, 1)
    t = np.logspace(np.log10(min(time_range[0], total)), np.log10(max(time_range[1], total)), n_points)
    fig = go.Figure()
    for r in avail:
        m_eff = r.mag_ab - SB_OFFSET if extended else r.mag_ab
        s = snr_for(m_eff, r.depth.ul5_ref, r.depth.t_ref, t, r.units)
        name = f'{r.filter}: SNR = {r.snr_stacked:.1f}' + (f' ({r.units} units)' if r.units > 1 else '')
        col = _plotly_color(colors[r.filter])
        fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name=name, legendgroup=r.filter, line=dict(color=col, width=1.1, dash=_plotly_dash(r.depth)),
                                 hovertemplate=f'<b>{r.filter}</b><br>total time = %{{x:.0f}} s<br>SNR = %{{y:.2f}}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[r.exptime * max(r.count, 1)], y=[r.snr_stacked], mode='markers', legendgroup=r.filter, showlegend=False,
                                 marker=dict(color=col, size=7, line=dict(color='white', width=1)),
                                 hovertemplate=f'<b>{r.filter}</b><br>{r.exptime:g} s x {r.count} frames' + (f' x {r.units} units' if r.units > 1 else '')
                                 + f'<br>SNR = {r.snr_stacked:.2f}<extra></extra>'))
    fig.add_vline(x=total, line=dict(color='black', dash='dash', width=1), opacity=0.6)
    _plotly_layout(fig, title, 'Total exposure time per unit [s]', 'SNR per arcsec²' if extended else 'SNR')
    fig.update_xaxes(type='log', range=[np.log10(t[0]), np.log10(t[-1])], dtick=1, minor=dict(showgrid=True, gridcolor='#f2f2f2'))
    fig.update_yaxes(type='log', dtick=1, minor=dict(showgrid=True, gridcolor='#f2f2f2'))
    _plotly_note(fig, avail, f'{avail[0].exptime:g} s x {avail[0].count} = {total:g} s per unit')
    return fig


def plot_snr_vs_total_time_interactive(target: Target, results: list, cond: Conditions, snr_target: float, time_range=(10.0, 30000.0), n_points: int = 120):
    """Plotly version of plot_snr_vs_total_time: the dots mark the total time needed for the target SNR per filter."""
    import plotly.graph_objects as go
    avail = [r for r in results if r.available]
    title = f'{target.label}: SNR vs total exposure time'
    if not avail:
        return _empty_plotly(title)
    colors = _colors(avail)
    extended = getattr(avail[0], 'extended', False)
    t = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]), n_points)
    fig = go.Figure()
    for r in avail:
        m_eff = r.mag_ab - SB_OFFSET if extended else r.mag_ab
        s = snr_for(m_eff, r.depth.ul5_ref, r.depth.t_ref, t, r.units)
        name = f'{r.filter}: {r.time_needed:.0f} s' + (f' ({r.units} units)' if r.units > 1 else '')
        col = _plotly_color(colors[r.filter])
        fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name=name, legendgroup=r.filter, line=dict(color=col, width=1.1, dash=_plotly_dash(r.depth)),
                                 hovertemplate=f'<b>{r.filter}</b><br>total time = %{{x:.0f}} s<br>SNR = %{{y:.2f}}<extra></extra>'))
        if math.isfinite(r.time_needed):
            fig.add_trace(go.Scatter(x=[r.time_needed], y=[snr_target], mode='markers', legendgroup=r.filter, showlegend=False,
                                     marker=dict(color=col, size=7, line=dict(color='white', width=1)),
                                     hovertemplate=f'<b>{r.filter}</b><br>time needed = {r.time_needed:.0f} s per unit<br>SNR = {snr_target:g}<extra></extra>'))
    fig.add_hline(y=snr_target, line=dict(color='black', dash='dash', width=1), opacity=0.6,
                  annotation_text=f'target SNR = {snr_target:g}', annotation_position='top left', annotation_font=dict(size=9, color='#555'))
    _plotly_layout(fig, title, 'Total exposure time per unit [s]', 'SNR per arcsec²' if extended else 'SNR')
    fig.update_xaxes(type='log', range=[np.log10(t[0]), np.log10(t[-1])], dtick=1, minor=dict(showgrid=True, gridcolor='#f2f2f2'))
    fig.update_yaxes(type='log', dtick=1, minor=dict(showgrid=True, gridcolor='#f2f2f2'))
    _plotly_note(fig, avail, f'frames of {avail[0].exptime:g} s')
    return fig


def plot_spectrum_points_interactive(target: Target, results: list, noise_seed: int | None = None, title: str | None = None, max_curve_points: int = 2500):
    """Plotly version of plot_spectrum_points: expected 7DT photometry (error bars 1.0857/SNR, horizontal bar = filter FWHM) over the spectrum."""
    import plotly.graph_objects as go
    avail = [r for r in results if r.available and r.lam_eff]
    extended = getattr(avail[0], 'extended', False) if avail else False
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    title = title or f'{target.label}: expected 7DT photometry' + (' (noise realisation)' if rng is not None else '')
    fig = go.Figure()
    spec = target.effective_spectrum()
    lo = hi = None
    if spec is not None:
        w = np.asarray(spec.wavelength, dtype=float)
        m = spec.ab_mag_curve()
        sel = np.isfinite(m) & (w > 3000) & (w < 10000)
        w, m = w[sel], m[sel]
        if w.size > max_curve_points:
            step = int(np.ceil(w.size / max_curve_points))
            w, m = w[::step], m[::step]
        if w.size:
            fig.add_trace(go.Scatter(x=w, y=m, mode='lines', name='input spectrum', line=dict(color='#666', width=1), opacity=0.8,
                                     hovertemplate='λ = %{x:.0f} Å<br>AB = %{y:.2f}<extra>input spectrum</extra>'))
            inside = m[(w > 3500) & (w < 9500)]
            if inside.size:
                lo, hi = float(np.nanpercentile(inside, 2) - 0.2), float(np.nanpercentile(inside, 98) + 0.2)
    if not avail:
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No filter could be evaluated', showarrow=False)
        return _plotly_layout(fig, title, 'Wavelength [Å]', 'AB magnitude')
    colors = _colors(avail)
    mags = []
    for r in avail:
        err = r.mag_err
        mag = r.mag_ab + (rng.normal(0, err) if (rng is not None and math.isfinite(err)) else 0.0)
        mags.append(r.mag_ab)
        curve = get_filter_curve(r.filter)
        xerr = curve.fwhm / 2 if curve is not None else 0.0
        col = _plotly_color(colors[r.filter])
        err_txt = f'±{err:.3f}' if math.isfinite(err) else 'n/a'
        fig.add_trace(go.Scatter(x=[r.lam_eff], y=[mag], mode='markers', name=r.filter, marker=dict(color=col, size=7),
                                 error_y=dict(type='data', array=[err], visible=math.isfinite(err), color=col, thickness=1.0, width=3),
                                 error_x=dict(type='data', array=[xerr], visible=xerr > 0, color=col, thickness=1.0, width=0),
                                 hovertemplate=f'<b>{r.filter}</b><br>λ_eff = {r.lam_eff:.0f} Å<br>AB = %{{y:.3f}} {err_txt}'
                                 + f'<br>expected = {r.mag_ab:.3f}<br>SNR = {r.snr_stacked:.1f}<extra></extra>'))
    errs = [r.mag_err for r in avail if math.isfinite(r.mag_err)]
    pad = float(np.clip(2.5 * np.nanmedian(errs), 0.3, 1.5)) if errs else 0.3
    ylo, yhi = min(mags) - pad, max(mags) + pad
    if lo is not None:
        ylo, yhi = min(ylo, lo), max(yhi, hi)
    _plotly_layout(fig, title, 'Wavelength [Å]', 'AB magnitude / arcsec²' if extended else 'AB magnitude')
    fig.update_xaxes(range=[3400, 9600])
    fig.update_yaxes(range=[yhi, ylo])
    fig.update_layout(showlegend=len(avail) <= 12 or spec is not None)
    return fig
