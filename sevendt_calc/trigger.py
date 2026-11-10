"""
Observation requests: the spreadsheet columns, reading of uploaded target tables, validation of every
row against the request rules and the current instrument, and formatting for the request spreadsheet.
Settings come from data/trigger.json.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, ObsMode, installed_filters, load_colormodes, load_filtinfo, load_specmodes, public_submodes, resolve_obsmode
from .targets import parse_coordinates

TRIGGER_JSON = DATA_DIR / 'trigger.json'
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'
DATE_EXAMPLE = '2026-11-11T00:00:00'


@dataclass(frozen=True)
class Field:
    name: str            # column name in the spreadsheet
    label: str           # name shown on the page
    required: bool       # must be given for every target (no default)
    aliases: tuple       # column names recognised in uploaded files (case, spaces and underscores ignored)
    help: str = ''


FIELDS = (
    Field('objname', 'Object name', True, ('name', 'object', 'target', 'obj', 'id', 'source', 'objectname', 'targetname'), 'up to 100 characters'),
    Field('RA', 'Right Ascension (hh:mm:ss.s or deg)', True, ('ra', 'raj2000', 'ra_deg', 'alpha', 'right ascension', 'ra(deg)', 'ra_hms')),
    Field('De', 'Declination (±dd:mm:ss or deg)', True, ('de', 'dec', 'decj2000', 'dec_deg', 'delta', 'declination', 'dec(deg)', 'dec_dms')),
    Field('obsmode', 'Observation mode', False, ('mode', 'observation mode', 'obs_mode')),
    Field('specmode', 'Specmode', False, ('spec mode', 'spec_mode')),
    Field('colormode', 'Colormode', False, ('color mode', 'color_mode', 'colourmode')),
    Field('filter_', 'Deepmode / filter', False, ('filter', 'filters', 'deepmode', 'deep mode', 'band')),
    Field('exptime', 'Exposure time per frame [s]', False, ('exp', 'exposure', 'exposure time', 'exptime_s', 't_exp', 'texp')),
    Field('count', 'Frames per filter', False, ('counts', 'n', 'nframes', 'frames', 'ncount', 'repeat', 'nexp')),
    Field('obs_startdate', 'Observation start (UTC)', False, ('obs_starttime', 'start', 'startdate', 'start_date', 'obs_start', 'starttime')),
    Field('obs_enddate', 'Observation end (UTC)', False, ('obs_endtime', 'end', 'enddate', 'end_date', 'obs_end', 'endtime')),
    Field('binning', 'Binning', False, ('bin',)),
    Field('gain', 'Gain', False, ('egain',)),
    Field('priority', 'Priority', False, ('prio',)),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}
REQUIRED = tuple(f.name for f in FIELDS if f.required)
HIDDEN_COLUMNS = ('note', 'comment', 'weight', 'is_ToO', 'is_rapidToO', 'objtype')
SHEET_COLUMNS = tuple(f.name for f in FIELDS) + HIDDEN_COLUMNS
LABELS = {f.name: f.label for f in FIELDS}
IGNORE = '(ignore)'


def _norm(text) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


ALIAS_MAP = {}
for _f in FIELDS:
    for _a in (_f.name,) + _f.aliases:
        ALIAS_MAP.setdefault(_norm(_a), _f.name)


@dataclass
class TriggerSettings:
    spreadsheet_url: str = ''
    period: str = ''
    obsmodes: tuple = ('Spec', 'Color', 'Deep', 'Single', 'Search')
    specmodes: tuple | None = None
    colormodes: tuple = ('gr', 'ri', 'gi', 'gri')
    deepmodes: tuple = ('g', 'r', 'i')
    default_exptime: float = 100.0
    max_exptime: float = 180.0
    default_count: int = 3
    default_binning: int = 1
    default_gain: int = 2750
    priority: int = 50
    gain_table: list = field(default_factory=list)
    hidden: dict = field(default_factory=lambda: {'note': 'None', 'weight': 1, 'is_ToO': 0, 'is_rapidToO': 0, 'objtype': 'Open'})

    @classmethod
    def load(cls, path: Path | str = TRIGGER_JSON) -> 'TriggerSettings':
        raw = json.loads(Path(path).read_text())
        kw = {}
        for name in ('spreadsheet_url', 'period', 'default_exptime', 'max_exptime', 'default_count', 'default_binning', 'default_gain', 'priority', 'gain_table', 'hidden'):
            if name in raw:
                kw[name] = raw[name]
        for name in ('obsmodes', 'colormodes', 'deepmodes'):
            if raw.get(name):
                kw[name] = tuple(str(x) for x in raw[name])
        kw['specmodes'] = tuple(str(x) for x in raw['specmodes']) if raw.get('specmodes') else None
        return cls(**kw)

    def spec_options(self, specmodes: dict | None = None) -> list:
        if self.specmodes:
            return list(self.specmodes)
        return public_submodes('Spec', specmodes if specmodes is not None else load_specmodes())

    def worksheet_name(self, project_id: str) -> str:
        return f'{self.period}_{safe_project_id(project_id)}'


def safe_project_id(project_id) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(project_id or '').strip()).strip('_')


# ------------------------------------------------------------------ uploaded tables
def read_table(data: bytes | str, filename: str = '') -> pd.DataFrame:
    """CSV / TSV / whitespace-separated text with a header line (a leading '#' on the header is allowed)."""
    text = data.decode('utf-8-sig', errors='replace') if isinstance(data, bytes) else str(data)
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError('The file is empty')
    if lines[0].lstrip().startswith('#'):
        lines[0] = lines[0].lstrip('#').strip()
    lines = [l for l in lines if not l.lstrip().startswith('#')]
    head = lines[0]
    sep = ',' if ',' in head else ('\t' if '\t' in head else r'\s+')
    df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=sep, dtype=str, engine='python', skipinitialspace=True, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith('Unnamed')]]
    return df.fillna('')


def recognise_columns(columns) -> tuple[dict, list]:
    """{file column: field name} for the columns that match a field or an alias; the rest in a list."""
    mapping, unknown = {}, []
    for c in columns:
        target = ALIAS_MAP.get(_norm(c))
        if target and target not in mapping.values():
            mapping[c] = target
        else:
            unknown.append(c)
    return mapping, unknown


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Keep the mapped columns under their field names (columns mapped to IGNORE or unmapped are dropped)."""
    keep = {c: f for c, f in mapping.items() if f and f != IGNORE and c in df.columns}
    out = df[list(keep)].rename(columns=keep)
    return out.loc[:, ~out.columns.duplicated()].fillna('')


# ------------------------------------------------------------------ modes
def n_orders(obsmode: str, submode: str | None, cfg: dict | None = None) -> int:
    """How many filters each unit observes in sequence (exposure time and count are given per order)."""
    om = request_obsmode(obsmode, submode, cfg)
    return max(om.n_filters_per_unit, 1) if om is not None else 1


def request_obsmode(obsmode: str, submode: str | None, cfg: dict | None = None) -> ObsMode | None:
    """The ObsMode of a request (for the mode figure, the ETC and the overhead): None when it cannot be built."""
    filtinfo = (cfg or {}).get('filtinfo') or load_filtinfo()
    specmodes = (cfg or {}).get('specmodes') or load_specmodes()
    colormodes = (cfg or {}).get('colormodes') or load_colormodes()
    sub = str(submode or '').strip()
    try:
        if obsmode == 'Spec' and sub:
            return resolve_obsmode('Spec', sub, filtinfo=filtinfo, specmodes=specmodes, colormodes=colormodes)
        if obsmode == 'Color' and sub:
            if sub in colormodes:
                return resolve_obsmode('Color', sub, filtinfo=filtinfo, specmodes=specmodes, colormodes=colormodes)
            filters = [f for f in sub]                                    # e.g. 'gi': one filter per unit, alternating
            table, k = {}, 0
            for unit, installed in filtinfo.items():
                options = [f for f in filters if f in installed]
                if options:
                    table[unit] = [options[k % len(options)]]
                    k += 1
            return ObsMode(mode='Color', submode=sub, filters_by_unit=table) if table else None
        if obsmode in ('Deep', 'Single', 'Search') and sub:
            return resolve_obsmode('Deep', sub, filtinfo=filtinfo, specmodes=specmodes, colormodes=colormodes)
    except ValueError:
        return None
    return None


# ------------------------------------------------------------------ validation
@dataclass
class Issue:
    row: int             # 1-based row number of the target table
    field: str
    level: str           # 'error' | 'warning'
    message: str

    def __str__(self) -> str:
        return f'Row {self.row}, {LABELS.get(self.field, self.field)}: {self.message}'


def parse_sequence(value, n: int, kind: str) -> tuple[list, str | None]:
    """'100' or '100,120' -> one value per order (a single value is repeated); kind 'float' or 'int'."""
    parts = [p.strip() for p in str(value).replace(';', ',').split(',') if p.strip()]
    if not parts:
        return [], 'missing'
    try:
        vals = [float(p) for p in parts]
        if kind == 'int':
            if any(v != int(v) for v in vals):
                return [], 'must be whole numbers'
            vals = [int(v) for v in vals]
    except ValueError:
        return [], f'not a number: {value}'
    if len(vals) == 1 and n > 1:
        vals = vals * n
    if len(vals) != n:
        return vals, f'{len(vals)} value(s) given but the mode has {n} order(s)'
    return vals, None


def parse_date(value) -> tuple[datetime | None, str | None]:
    text = str(value or '').strip()
    if text.lower() in ('', 'none', 'nan', 'null'):
        return None, None
    for fmt in (DATE_FORMAT, '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(text, fmt), None
        except ValueError:
            continue
    return None, f'use the UTC format {DATE_EXAMPLE}'


def _blank(value) -> bool:
    return value is None or str(value).strip() == '' or str(value).strip().lower() == 'nan'


def validate_rows(rows: list, settings: TriggerSettings, defaults: dict, cfg: dict | None = None) -> tuple[list, list]:
    """
    rows: dicts with spreadsheet column names (strings as typed); blank cells take `defaults` (the page's
    observation settings: obsmode, specmode, colormode, filter_, exptime, count, obs_startdate, obs_enddate,
    binning, gain). Returns (normalised rows, issues). A normalised row carries RA_deg / De_deg and the
    exptime / count lists next to the text values.
    """
    filtinfo = (cfg or {}).get('filtinfo') or load_filtinfo()
    installed = installed_filters(filtinfo)
    spec_options = settings.spec_options((cfg or {}).get('specmodes'))
    out, issues = [], []
    for i, raw in enumerate(rows, start=1):
        row = {}
        for f in FIELDS:
            v = raw.get(f.name)
            row[f.name] = str(v).strip() if not _blank(v) else str(defaults.get(f.name, '') if not _blank(defaults.get(f.name)) else '')
        err = lambda fld, msg: issues.append(Issue(i, fld, 'error', msg))
        warn = lambda fld, msg: issues.append(Issue(i, fld, 'warning', msg))
        # object name
        if not row['objname']:
            err('objname', 'required')
        elif len(row['objname']) > 100:
            err('objname', f'{len(row["objname"])} characters (maximum 100)')
        # coordinates
        row['RA_deg'] = row['De_deg'] = None
        if not row['RA']:
            err('RA', 'required')
        if not row['De']:
            err('De', 'required')
        if row['RA'] and row['De']:
            try:
                row['RA_deg'], row['De_deg'] = parse_coordinates(row['RA'], row['De'])
            except ValueError as exc:
                err('RA', str(exc))
        # observation mode and its sub-mode
        mode = next((m for m in settings.obsmodes if m.lower() == row['obsmode'].lower()), None)
        if mode is None:
            err('obsmode', f'must be one of {", ".join(settings.obsmodes)}' if row['obsmode'] else 'required')
        row['obsmode'] = mode or row['obsmode']
        if mode == 'Spec':
            if row['specmode'] not in spec_options:
                err('specmode', f'must be one of {", ".join(spec_options)}' if row['specmode'] else 'required for Spec')
            row['colormode'] = row['filter_'] = ''
        elif mode == 'Color':
            if row['colormode'] not in settings.colormodes:
                err('colormode', f'must be one of {", ".join(settings.colormodes)}' if row['colormode'] else 'required for Color')
            row['specmode'] = row['filter_'] = ''
        elif mode == 'Deep':
            if row['filter_'] not in settings.deepmodes:
                err('filter_', f'must be one of {", ".join(settings.deepmodes)}' if row['filter_'] else 'required for Deep')
            row['specmode'] = row['colormode'] = ''
        elif mode in ('Single', 'Search'):
            if row['filter_'] and row['filter_'] not in installed:
                err('filter_', f'{row["filter_"]} is not installed in any unit')
            row['specmode'] = row['colormode'] = ''
        sub = row['specmode'] or row['colormode'] or row['filter_']
        n = n_orders(mode, sub, cfg) if mode else 1
        # exposure time and count per order
        vals, e = parse_sequence(row['exptime'] or settings.default_exptime, n, 'float')
        row['exptime_list'] = vals
        if e:
            err('exptime', e)
        else:
            if any(v <= 0 for v in vals):
                err('exptime', 'must be positive')
            elif any(v > settings.max_exptime for v in vals):
                err('exptime', f'above {settings.max_exptime:g} s')
            elif any(v != settings.default_exptime for v in vals):
                warn('exptime', f'differs from the standard {settings.default_exptime:g} s; the depth model was fitted on {settings.default_exptime:g} s frames')
        vals, e = parse_sequence(row['count'] or settings.default_count, n, 'int')
        row['count_list'] = vals
        if e:
            err('count', e)
        elif any(v < 1 for v in vals):
            err('count', 'must be at least 1')
        # dates
        dates = {}
        for fld in ('obs_startdate', 'obs_enddate'):
            d, e = parse_date(row[fld])
            dates[fld] = d
            if e:
                err(fld, e)
            row[fld] = d.strftime(DATE_FORMAT) if d else ''
        if dates['obs_startdate'] and dates['obs_enddate'] and dates['obs_enddate'] <= dates['obs_startdate']:
            err('obs_enddate', 'must be after the start')
        # binning, gain, priority
        for fld, default, minimum in (('binning', settings.default_binning, 1), ('gain', settings.default_gain, 0)):
            text = row[fld] or str(default)
            try:
                v = int(float(text))
                if float(text) != v:
                    raise ValueError
                if v < minimum:
                    err(fld, f'must be at least {minimum}')
                row[fld] = str(v)
            except ValueError:
                err(fld, f'must be an integer: {text}')
        row['priority'] = str(settings.priority)
        out.append(row)
    return out, issues


def format_rows(rows: list, project_id: str, settings: TriggerSettings) -> list:
    """Spreadsheet rows (strings) in SHEET_COLUMNS order from validated rows."""
    out = []
    for r in rows:
        rec = {
            'objname': r['objname'],
            'RA': f'{r["RA_deg"]:.6f}' if r.get('RA_deg') is not None else r['RA'],
            'De': f'{r["De_deg"]:.6f}' if r.get('De_deg') is not None else r['De'],
            'obsmode': r['obsmode'], 'specmode': r['specmode'] or 'None', 'colormode': r['colormode'] or 'None', 'filter_': r['filter_'] or 'None',
            'exptime': ','.join(f'{v:g}' for v in r.get('exptime_list') or []),
            'count': ','.join(str(v) for v in r.get('count_list') or []),
            'obs_startdate': r['obs_startdate'] or 'None', 'obs_enddate': r['obs_enddate'] or 'None',
            'binning': r['binning'], 'gain': r['gain'], 'priority': r['priority'],
        }
        hidden = dict(settings.hidden)
        hidden['comment'] = safe_project_id(project_id)
        for c in HIDDEN_COLUMNS:
            rec[c] = str(hidden.get(c, ''))
        out.append({c: rec.get(c, '') for c in SHEET_COLUMNS})
    return out


def sample_file() -> bytes:
    return (b'objname,RA,De,obsmode,specmode,filter_,exptime,count,obs_startdate,obs_enddate\n'
            b'NGC0253,00:47:33.12,-25:17:17.6,Spec,specall,,100,3,,\n'
            b'T01022,78.26087,-71.32075,Spec,specall,,"100,100","3,3",2026-11-11T00:00:00,2026-12-01T00:00:00\n'
            b'M31,10.68471,+41.26875,Deep,,r,100,3,,\n')
