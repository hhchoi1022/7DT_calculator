"""Filter-name helpers shared by the whole package."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

BROAD_ORDER = ['u', 'g', 'r', 'i', 'z']
BROAD_COLORS = {'u': '#7b3fb8', 'g': '#2ca02c', 'r': '#d62728', 'i': '#e6a700', 'z': '#222222'}

_MEDIUM_RE = re.compile(r'^m(\d{3})(w?)$')


def medium_band_wavelength(name: str) -> float | None:
    """Nominal central wavelength [A] encoded in a medium-band name (m400 -> 4000, m769w -> 7690)."""
    m = _MEDIUM_RE.match(str(name))
    return float(m.group(1)) * 10.0 if m else None


def is_wide_band(name: str) -> bool:
    m = _MEDIUM_RE.match(str(name))
    return bool(m and m.group(2) == 'w')


def is_slot(name: str) -> bool:
    """Empty filter-wheel slot entries in filtinfo.dict look like 'Slot6'."""
    return str(name).lower().startswith('slot')


def sort_key(name: str):
    name = str(name)
    if name in BROAD_ORDER:
        return (0, BROAD_ORDER.index(name), 0.0, name)
    wave = medium_band_wavelength(name)
    if wave is not None:
        return (1, 0, wave, name)
    return (2, 0, 0.0, name)


def sort_filters(names) -> list[str]:
    """Broad bands (u g r i z) first, then medium bands by wavelength, then everything else."""
    return sorted({str(n) for n in names}, key=sort_key)


def unique_in_order(names) -> list[str]:
    seen, out = set(), []
    for n in names:
        n = str(n)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def filter_colors(names, wavelengths: dict | None = None) -> dict[str, tuple]:
    """
    Colour per filter: fixed colours for broad bands, a spectral colormap for medium bands
    (ordered by wavelength). `wavelengths` may give effective wavelengths [A] for names whose
    wavelength is not encoded in the name.
    """
    names = [str(n) for n in names]
    colors = {}
    waves = {}
    for n in names:
        if n in BROAD_COLORS:
            colors[n] = BROAD_COLORS[n]
            continue
        w = (wavelengths or {}).get(n) or medium_band_wavelength(n)
        if w is not None:
            waves[n] = w
    if waves:
        cmap = plt.get_cmap('turbo')
        lo, hi = 3500.0, 9000.0
        for n, w in waves.items():
            colors[n] = cmap(float(np.clip((w - lo) / (hi - lo), 0, 1)))
    for n in names:
        colors.setdefault(n, '#7f7f7f')
    return colors
