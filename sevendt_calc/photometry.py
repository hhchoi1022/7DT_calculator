"""
Filter transmission curves, spectra and synthetic AB photometry.

The transmission curves live in data/transmission/<filter>.csv (two columns: wavelength [A],
response). They are copied from the ezphot configuration so the calculator is self-contained.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .filters import medium_band_wavelength

C_AA = 2.99792458e18          # speed of light [A/s]
AB_ZEROPOINT = 48.60
TRANSMISSION_DIR = DATA_DIR / 'transmission'

# m_AB - m_Vega (Blanton & Roweis 2007). Bands not listed are treated as AB already.
VEGA_TO_AB = {'U': 0.79, 'B': -0.09, 'V': 0.02, 'R': 0.21, 'I': 0.45}
MAG_INPUT_FILTERS = ('B', 'V', 'R', 'I', 'g', 'r', 'i')

_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def ab_offset(band: str) -> float:
    return VEGA_TO_AB.get(str(band), 0.0)


def to_ab(mag: float, band: str) -> float:
    """Convert a magnitude given in `band` to the AB system (Vega bands get the offset)."""
    return float(mag) + ab_offset(band)


def magnitude_system(band: str) -> str:
    return 'Vega' if str(band) in VEGA_TO_AB else 'AB'


# ---------------------------------------------------------------- filter curves
def _parse_two_columns(text: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.replace(',', ' ').replace(';', ' ').split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        raise ValueError('Need at least two numeric rows with two columns')
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    keep = np.concatenate([[True], np.diff(x) > 0])
    return x[keep], y[keep]


@dataclass
class FilterCurve:
    name: str
    wavelength: np.ndarray            # A
    response: np.ndarray              # fraction (0-1)
    lam_eff: float = field(init=False)
    lam_pivot: float = field(init=False)
    lam_min: float = field(init=False)   # where response > 1 % of peak
    lam_max: float = field(init=False)
    fwhm: float = field(init=False)

    def __post_init__(self):
        w = np.asarray(self.wavelength, dtype=float)
        t = np.clip(np.asarray(self.response, dtype=float), 0, None)
        if t.max() > 1.5:                # percent -> fraction
            t = t / 100.0
        self.wavelength, self.response = w, t
        area = _trapezoid(t, w)
        self.lam_eff = float(_trapezoid(w * t, w) / area)
        self.lam_pivot = float(math.sqrt(_trapezoid(w * t, w) / _trapezoid(t / w, w)))
        above = w[t > 0.01 * t.max()]
        self.lam_min, self.lam_max = float(above.min()), float(above.max())
        half = w[t > 0.5 * t.max()]
        self.fwhm = float(half.max() - half.min()) if half.size else 0.0

    def __repr__(self):
        return f'FilterCurve({self.name}, lam_eff={self.lam_eff:.0f} A, fwhm={self.fwhm:.0f} A)'

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> 'FilterCurve':
        path = Path(path)
        w, t = _parse_two_columns(path.read_text())
        return cls(name=name or path.stem, wavelength=w, response=t)


@lru_cache(maxsize=4)
def load_filter_curves(directory: str | None = None) -> dict[str, FilterCurve]:
    directory = Path(directory) if directory else TRANSMISSION_DIR
    curves = {}
    for path in sorted(directory.glob('*.csv')):
        try:
            curves[path.stem] = FilterCurve.from_file(path)
        except ValueError:
            continue
    return curves


def get_filter_curve(name: str, directory: str | None = None) -> FilterCurve | None:
    return load_filter_curves(directory).get(str(name))


def effective_wavelength(name: str) -> float | None:
    """Effective wavelength [A] from the transmission curve, else the nominal one from the name."""
    curve = get_filter_curve(name)
    if curve is not None:
        return curve.lam_eff
    return medium_band_wavelength(name)


# --------------------------------------------------------------------- spectra
@dataclass
class Spectrum:
    """A source spectrum stored as f_lambda [erg/s/cm2/A] on an ascending wavelength grid [A]."""
    wavelength: np.ndarray
    flux: np.ndarray
    source: str = ''

    @classmethod
    def from_arrays(cls, wavelength, flux, flux_unit: str = 'flam', source: str = '') -> 'Spectrum':
        w = np.asarray(wavelength, dtype=float)
        f = np.asarray(flux, dtype=float)
        if w.shape != f.shape:
            raise ValueError('wavelength and flux must have the same length')
        good = np.isfinite(w) & np.isfinite(f)
        w, f = w[good], f[good]
        if w.size < 2:
            raise ValueError('A spectrum needs at least two finite points')
        order = np.argsort(w, kind='stable')
        w, f = w[order], f[order]
        keep = np.concatenate([[True], np.diff(w) > 0])
        w, f = w[keep], f[keep]
        unit = str(flux_unit).lower().replace('_', '')
        if unit in ('fnu', 'f_nu', 'nu'):
            f = f * C_AA / w ** 2          # erg/s/cm2/Hz -> erg/s/cm2/A
        elif unit not in ('flam', 'flambda', 'lambda', 'f_lambda'):
            raise ValueError(f"flux_unit must be 'flam' or 'fnu', got {flux_unit!r}")
        return cls(wavelength=w, flux=f, source=source)

    @classmethod
    def from_text(cls, text, flux_unit: str = 'flam', source: str = '') -> 'Spectrum':
        """Parse a two-column ASCII table (wavelength [A], flux); comments and headers are skipped."""
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        w, f = _parse_two_columns(text)
        return cls.from_arrays(w, f, flux_unit=flux_unit, source=source)

    @property
    def lam_min(self) -> float:
        return float(self.wavelength[0])

    @property
    def lam_max(self) -> float:
        return float(self.wavelength[-1])

    @property
    def fnu(self) -> np.ndarray:
        return self.flux * self.wavelength ** 2 / C_AA

    def ab_mag_curve(self) -> np.ndarray:
        """Monochromatic AB magnitude as a function of wavelength (NaN where flux <= 0)."""
        fnu = self.fnu
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = -2.5 * np.log10(np.where(fnu > 0, fnu, np.nan)) - AB_ZEROPOINT
        return mag

    def coverage(self, curve: FilterCurve) -> float:
        """Fraction of the filter throughput integral covered by the spectrum's wavelength range."""
        w, t = curve.wavelength, curve.response
        total = _trapezoid(t, w)
        inside = (w >= self.lam_min) & (w <= self.lam_max)
        if inside.sum() < 2 or total <= 0:
            return 0.0
        return float(_trapezoid(t[inside], w[inside]) / total)

    def synthetic_ab_mag(self, curve: FilterCurve, min_coverage: float = 0.95) -> float:
        """
        AB magnitude through `curve` for a photon-counting detector:
            <f_nu> = int(f_lambda * lambda * T dlambda) / int(c / lambda * T dlambda)
        NaN when the spectrum covers less than `min_coverage` of the filter throughput.
        """
        if self.coverage(curve) < min_coverage:
            return float('nan')
        sel = (curve.wavelength >= max(curve.lam_min, self.lam_min)) & (curve.wavelength <= min(curve.lam_max, self.lam_max))
        w = curve.wavelength[sel]
        t = curve.response[sel]
        if w.size < 2:
            return float('nan')
        f = np.interp(w, self.wavelength, self.flux)
        num = _trapezoid(f * w * t, w)
        den = _trapezoid(C_AA / w * t, w)
        if not (num > 0 and den > 0):
            return float('nan')
        return float(-2.5 * math.log10(num / den) - AB_ZEROPOINT)

    def scaled_to(self, mag_ab: float, curve: FilterCurve) -> 'Spectrum':
        """Copy of the spectrum rescaled so that its synthetic magnitude in `curve` equals mag_ab."""
        current = self.synthetic_ab_mag(curve)
        if not math.isfinite(current):
            raise ValueError(f'Spectrum does not cover the {curve.name} band; cannot scale to it')
        factor = 10 ** (-0.4 * (float(mag_ab) - current))
        return Spectrum(wavelength=self.wavelength.copy(), flux=self.flux * factor, source=self.source)



def flat_spectrum(mag_ab: float, lam_min: float = 3000.0, lam_max: float = 11000.0, n: int = 801) -> Spectrum:
    """A constant-f_nu spectrum with the given AB magnitude (what a magnitude-only target assumes)."""
    w = np.linspace(lam_min, lam_max, n)
    fnu = 10 ** (-0.4 * (float(mag_ab) + AB_ZEROPOINT))
    return Spectrum(wavelength=w, flux=fnu * C_AA / w ** 2, source='flat spectrum')
