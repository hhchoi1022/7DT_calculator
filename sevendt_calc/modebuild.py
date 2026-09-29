"""
Building a new Spec / Color observation mode from the installed filters: one filter sequence per
unit, validated against filtinfo.dict, exported in the TCSpy .specmode / .colormode JSON format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ObsMode, is_slot, load_filtinfo

KINDS = {'Spec': '.specmode', 'Color': '.colormode'}
MAX_SLOTS = 8


def unit_filters(unit: str, filtinfo: dict) -> list[str]:
    """Filters installed in one unit (empty slots left out), in wheel order."""
    return [f for f in filtinfo.get(unit, []) if not is_slot(f)]


def clean_sequence(values) -> list[str]:
    """The filter sequence of one unit from editor cells: empty cells dropped, order kept."""
    out = []
    for v in values or []:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ('nan', 'none'):
            out.append(s)
    return out


def safe_name(name: str) -> str:
    """A file-name-safe mode name: letters, digits, '_' and '-' only."""
    name = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(name or '').strip()).strip('_')
    return name or 'newmode'


def filename(name: str, kind: str) -> str:
    return safe_name(name) + KINDS[kind]


@dataclass
class BuiltMode:
    kind: str
    name: str
    filters_by_unit: dict                     # every unit of filtinfo, possibly with an empty list
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def obsmode(self) -> ObsMode:
        """The mode as the other pages see it (units without filters left out)."""
        table = {u: list(f) for u, f in self.filters_by_unit.items() if f}
        return ObsMode(mode=self.kind, submode=safe_name(self.name), filters_by_unit=table)

    @property
    def ok(self) -> bool:
        return not self.errors and any(self.filters_by_unit.values())

    def to_json(self, include_empty: bool = True) -> str:
        """TCSpy layout: one unit per line, its filter list on the same line."""
        items = [(u, f) for u, f in self.filters_by_unit.items() if include_empty or f]
        lines = [f'    "{u}": [{", ".join(chr(34) + x + chr(34) for x in f)}]' for u, f in items]
        return '{\n' + ',\n'.join(lines) + '\n}\n'

    @property
    def file_name(self) -> str:
        return filename(self.name, self.kind)


def build_mode(kind: str, name: str, sequences: dict, filtinfo: dict | None = None) -> BuiltMode:
    """
    Validate {unit: [filters...]} against the installed filters.
    errors  : a filter that the unit does not have (the file would not run)
    warnings: units without any filter, unequal sequence lengths
    """
    if kind not in KINDS:
        raise ValueError(f'Unknown mode kind {kind!r}; expected one of {list(KINDS)}')
    filtinfo = filtinfo if filtinfo is not None else load_filtinfo()
    table, errors, warnings = {}, [], []
    for unit in filtinfo:
        seq = clean_sequence(sequences.get(unit, []))
        table[unit] = seq
        missing = sorted({f for f in seq if f not in unit_filters(unit, filtinfo)})
        if missing:
            errors.append(f'{unit}: {", ".join(missing)} not installed (available: {", ".join(unit_filters(unit, filtinfo))})')
    for unit in sequences:
        if unit not in filtinfo:
            errors.append(f'{unit} is not a 7DT unit')
    lengths = {u: len(f) for u, f in table.items() if f}
    empty = [u for u, f in table.items() if not f]
    if not lengths:
        warnings.append('No filter selected yet')
    else:
        if empty:
            warnings.append('Units without a filter (written as an empty list): ' + ', '.join(empty))
        if len(set(lengths.values())) > 1:
            warnings.append('Units have different numbers of filters: ' + ', '.join(f'{u} ({n})' for u, n in lengths.items())
                            + '. TCSpy modes normally give every unit the same number of frames; repeat a filter to pad a unit.')
    return BuiltMode(kind=kind, name=safe_name(name), filters_by_unit=table, errors=errors, warnings=warnings)


def parse_mode_file(data: bytes | str, filename: str = '') -> tuple[str, str, dict]:
    """
    Read a .specmode / .colormode / .mode JSON file ({unit: [filters...]}).
    Returns (kind, name, sequences); the kind comes from the extension (.colormode -> Color, else Spec).
    """
    import json
    from pathlib import Path
    text = data.decode('utf-8') if isinstance(data, bytes) else str(data)
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ValueError(f'Not a JSON mode file: {exc}') from exc
    if not isinstance(raw, dict) or not all(isinstance(v, (list, tuple)) for v in raw.values()):
        raise ValueError('A mode file must map each unit to a list of filters, e.g. {"7DT01": ["m650", "m769w"], ...}')
    seq = {str(u): [str(f) for f in fs] for u, fs in raw.items()}
    path = Path(filename or 'newmode')
    kind = 'Color' if path.suffix.lower() == '.colormode' else 'Spec'
    return kind, safe_name(path.stem), seq


def pad_sequences(sequences: dict) -> dict:
    """Repeat the last filter of the shorter units so that every observing unit has the same number of frames."""
    seqs = {u: clean_sequence(f) for u, f in sequences.items()}
    longest = max((len(f) for f in seqs.values()), default=0)
    return {u: (f + [f[-1]] * (longest - len(f)) if f else []) for u, f in seqs.items()}


def add_filter(sequences: dict, unit: str, filt: str) -> dict:
    """Click on a wheel cell: append the filter to the unit's sequence (the same filter may appear several times)."""
    seqs = {u: list(f) for u, f in sequences.items()}
    seqs.setdefault(unit, []).append(filt)
    return seqs


def remove_at(sequences: dict, unit: str, position: int) -> dict:
    """Remove the entry at `position` (0-based) of the unit's sequence; out-of-range positions are ignored."""
    seqs = {u: list(f) for u, f in sequences.items()}
    seq = seqs.get(unit, [])
    if 0 <= position < len(seq):
        seqs[unit] = seq[:position] + seq[position + 1:]
    return seqs


def remove_last(sequences: dict, unit: str) -> dict:
    """Undo for one unit: drop the last filter of its sequence."""
    seqs = {u: list(f) for u, f in sequences.items()}
    if seqs.get(unit):
        seqs[unit] = seqs[unit][:-1]
    return seqs


def filter_summary(built: BuiltMode) -> list[dict]:
    """One row per distinct filter: how many unit slots observe it and which units."""
    from .photometry import effective_wavelength
    om = built.obsmode
    mult = om.filter_multiplicity()
    rows = []
    for f in om.filters:
        units = [u for u, seq in built.filters_by_unit.items() if f in seq]
        lam = effective_wavelength(f)
        rows.append({'filter': f, 'lam_eff_A': round(lam) if lam else None, 'slots': mult.get(f, 0), 'units': ', '.join(units)})
    return rows
