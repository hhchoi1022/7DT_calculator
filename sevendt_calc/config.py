"""
Live access to the 7DT (TCSpy) configuration folder.

Everything is read from the folder at call time, so changes made by the observatory
(new specmode folder, new filters) show up without restarting the calculator.

The folder defaults to /lyman/data1/7dt/configuration and can be overridden with the
SEVENDT_CONFIG_DIR environment variable. Paths written inside the config files refer to
the control PC (/home/kds/TCSpy/configuration/...); they are remapped onto the folder
above.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .filters import is_slot, sort_filters, unique_in_order

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'

DEFAULT_CONFIG_DIR = '/lyman/data1/7dt/configuration'
REMOTE_PREFIXES = ('/home/kds/TCSpy/configuration', '/home/kds/tcspy/configuration')

MODES = ('Spec', 'Deep', 'Color')
DEFAULT_SUBMODE = {'Spec': 'specall', 'Color': 'gri', 'Deep': 'g,r,i'}


def config_dir() -> Path:
    return Path(os.environ.get('SEVENDT_CONFIG_DIR', DEFAULT_CONFIG_DIR))


def remap_path(path_str: str) -> Path:
    """Map a path written for the control PC onto the local configuration folder."""
    path_str = str(path_str)
    p = Path(path_str)
    if p.exists():
        return p
    for prefix in REMOTE_PREFIXES:
        if path_str.startswith(prefix):
            return config_dir() / path_str[len(prefix):].lstrip('/')
    return p


def _read_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


# --------------------------------------------------------------------------- site
@dataclass(frozen=True)
class Site:
    name: str
    latitude: float      # deg, north positive
    longitude: float     # deg, east positive
    elevation: float     # m
    timezone: str        # IANA name

    @property
    def label(self) -> str:
        return f'{self.name} (lat {self.latitude:.4f}, lon {self.longitude:.4f}, {self.elevation:.0f} m)'


def load_site() -> Site:
    cfg = _read_json(config_dir() / 'observer.config')
    return Site(
        name='7DT',
        latitude=float(cfg['OBSERVER_LATITUDE']),
        longitude=float(cfg['OBSERVER_LONGITUDE']),
        elevation=float(cfg['OBSERVER_ELEVATION']),
        timezone=str(cfg.get('OBSERVER_TIMEZONE', 'America/Santiago')),
    )


def load_target_constraints() -> dict:
    """Observability limits used by the scheduler (target.config)."""
    cfg = _read_json(config_dir() / 'target.config')
    return {
        'min_alt': float(cfg.get('TARGET_MINALT', 30)),
        'max_alt': float(cfg.get('TARGET_MAXALT', 88)),
        'moon_sep': float(cfg.get('TARGET_MOONSEP', 40)),
    }


def load_night_sun_altitude() -> float:
    """Sun altitude [deg] below which science observation is allowed (nightsession.config)."""
    try:
        cfg = _read_json(config_dir() / 'nightsession.config')
        return float(cfg.get('NIGHTSESSION_SUNALT_OBSERVATION', -18))
    except (OSError, ValueError):
        return -18.0


# ------------------------------------------------------------------------ filters
def load_filtinfo() -> dict[str, list[str]]:
    """Filters installed in each unit's wheel: {'7DT01': ['g', 'r', ...], ...}."""
    raw = _read_json(config_dir() / 'filtinfo.dict')
    return {str(unit): [str(f) for f in filters] for unit, filters in raw.items()}


def installed_filters(filtinfo: dict | None = None) -> list[str]:
    filtinfo = filtinfo if filtinfo is not None else load_filtinfo()
    names = [f for filters in filtinfo.values() for f in filters if not is_slot(f)]
    return sort_filters(names)


def units_with_filter(filter_name: str, filtinfo: dict | None = None) -> list[str]:
    filtinfo = filtinfo if filtinfo is not None else load_filtinfo()
    return [unit for unit, filters in filtinfo.items() if filter_name in filters]


# -------------------------------------------------------------- observation modes
def _mode_folder(config_name: str, key: str) -> Path:
    cfg = _read_json(config_dir() / config_name)
    return remap_path(cfg[key])


def _load_mode_files(folder: Path, suffix: str) -> dict[str, dict[str, list[str]]]:
    modes = {}
    if not folder.is_dir():
        raise FileNotFoundError(f'Observation-mode folder not found: {folder}')
    for path in sorted(folder.glob(f'*{suffix}')):
        try:
            raw = _read_json(path)
        except ValueError:
            continue
        modes[path.stem] = {str(unit): [str(f) for f in filters] for unit, filters in raw.items()}
    return modes


def specmode_folder() -> Path:
    return _mode_folder('specmode.config', 'SPECMODE_FOLDER')


def colormode_folder() -> Path:
    return _mode_folder('colormode.config', 'COLORMODE_FOLDER')


def load_specmodes() -> dict[str, dict[str, list[str]]]:
    """All *.specmode files of the active specmode folder: {name: {unit: [filters]}}."""
    return _load_mode_files(specmode_folder(), '.specmode')


def load_colormodes() -> dict[str, dict[str, list[str]]]:
    return _load_mode_files(colormode_folder(), '.colormode')


def submode_options(mode: str) -> list[str]:
    """Selectable sub-modes for a mode. Deep has no predefined list (filters are typed)."""
    if mode == 'Spec':
        names = list(load_specmodes())
        return sorted(names, key=lambda n: (n != 'specall', n))
    if mode == 'Color':
        names = list(load_colormodes())
        return sorted(names, key=lambda n: (n != 'gri', n))
    if mode == 'Deep':
        return []
    raise ValueError(f'Unknown observation mode: {mode}')


@dataclass(frozen=True)
class ObsMode:
    """A resolved observation mode: which filters each unit observes, in order."""
    mode: str
    submode: str
    filters_by_unit: dict
    warnings: tuple = ()

    @property
    def label(self) -> str:
        return f'{self.mode}/{self.submode}'

    @property
    def units(self) -> list[str]:
        return list(self.filters_by_unit)

    @property
    def filters(self) -> list[str]:
        """Every filter used by the mode (any unit), broad bands first then by wavelength."""
        return sort_filters(f for filters in self.filters_by_unit.values() for f in filters)

    def filter_multiplicity(self) -> dict:
        """Number of unit exposures per frame set for each filter: how many (unit, slot) entries observe it."""
        counts: dict = {}
        for filters in self.filters_by_unit.values():
            for f in filters:
                counts[f] = counts.get(f, 0) + 1
        return counts

    @property
    def n_filters_per_unit(self) -> int:
        return max((len(f) for f in self.filters_by_unit.values()), default=0)

    @property
    def is_specall(self) -> bool:
        return self.mode == 'Spec' and self.submode == 'specall'

    def unit_table(self) -> list[dict]:
        return [{'unit': u, 'filters': ', '.join(f)} for u, f in self.filters_by_unit.items()]


def parse_deep_filters(text) -> list[str]:
    if isinstance(text, (list, tuple)):
        parts = [str(t) for t in text]
    else:
        parts = str(text).replace(';', ',').replace(' ', ',').split(',')
    return unique_in_order(p.strip() for p in parts if p.strip())


def resolve_obsmode(mode: str, submode, *, filtinfo=None, specmodes=None, colormodes=None) -> ObsMode:
    """
    Build an ObsMode.

    mode     : 'Spec' | 'Deep' | 'Color'
    submode  : specmode name / colormode name / list (or comma string) of filters for Deep
    """
    filtinfo = filtinfo if filtinfo is not None else load_filtinfo()
    warnings = []
    if mode == 'Spec':
        specmodes = specmodes if specmodes is not None else load_specmodes()
        if submode not in specmodes:
            raise ValueError(f"Specmode '{submode}' not found in {specmode_folder()}")
        table = specmodes[submode]
    elif mode == 'Color':
        colormodes = colormodes if colormodes is not None else load_colormodes()
        if submode not in colormodes:
            raise ValueError(f"Colormode '{submode}' not found in {colormode_folder()}")
        table = colormodes[submode]
    elif mode == 'Deep':
        filters = parse_deep_filters(submode)
        if not filters:
            raise ValueError('Deep mode needs at least one filter')
        table = {}
        excluded = []
        for unit, installed in filtinfo.items():
            missing = [f for f in filters if f not in installed]
            if missing:
                excluded.append(f'{unit} (missing {", ".join(missing)})')
            else:
                table[unit] = list(filters)
        if not table:
            raise ValueError(f'No unit has all of the filters {filters} installed')
        if excluded:
            warnings.append('Units without the requested filters are left out: ' + '; '.join(excluded))
        submode = ','.join(filters)
    else:
        raise ValueError(f'Unknown observation mode: {mode}')

    clean = {}
    for unit, filters in table.items():
        filters = [str(f) for f in filters if str(f).strip()]
        if not filters:
            continue
        installed = filtinfo.get(unit)
        if installed is not None:
            not_installed = [f for f in filters if f not in installed]
            if not_installed:
                warnings.append(f'{unit}: {", ".join(not_installed)} not in filtinfo.dict')
        clean[unit] = tuple(filters)
    return ObsMode(mode=mode, submode=str(submode), filters_by_unit=clean, warnings=tuple(warnings))
