"""
Spectral templates offered in the target table (data/templates/templates.json).

Every template is a relative f_lambda shape; the target's magnitude (in its filter) sets the
absolute level, and an optional redshift stretches the wavelength axis. Two special entries are
not files: a flat (constant f_nu) spectrum and 'Upload (custom file)'.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .photometry import Spectrum

TEMPLATE_DIR = DATA_DIR / 'templates'
MANIFEST = TEMPLATE_DIR / 'templates.json'

FLAT_KEY = 'Flat (constant f_nu)'
UPLOAD_KEY = 'Upload (custom file)'
DEFAULT_KEY = 'Star: G2V'
SPECIAL_KEYS = (FLAT_KEY, UPLOAD_KEY)


@lru_cache(maxsize=1)
def manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, 'r') as f:
        return json.load(f)


def template_keys() -> list[str]:
    """Dropdown order: the upload entry first, then the templates of the manifest, then the flat spectrum."""
    return [UPLOAD_KEY] + list(manifest()) + [FLAT_KEY]


def default_key() -> str:
    keys = template_keys()
    return DEFAULT_KEY if DEFAULT_KEY in keys else keys[0]


def is_template(key) -> bool:
    return key in manifest()


def describe(key: str) -> str:
    entry = manifest().get(key)
    if entry is None:
        return ''
    return f"{entry.get('source', '')} {entry.get('note', '')}".strip()


@lru_cache(maxsize=64)
def _load(key: str) -> Spectrum:
    entry = manifest()[key]
    data = np.loadtxt(TEMPLATE_DIR / entry['file'])
    return Spectrum.from_arrays(data[:, 0], data[:, 1], 'flam', source=key)


def load_template(key: str, redshift: float = 0.0) -> Spectrum:
    """Template spectrum (relative f_lambda) shifted to `redshift`; flux level is meaningless until scaled."""
    if key not in manifest():
        raise KeyError(f'Unknown spectrum template {key!r}')
    base = _load(key)
    z = float(redshift or 0.0)
    if z < 0:
        raise ValueError('redshift must be >= 0')
    return Spectrum(wavelength=base.wavelength * (1.0 + z), flux=base.flux / (1.0 + z), source=key if z == 0 else f'{key} at z={z:g}')
