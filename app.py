"""
7DT Observation Calculator, Streamlit user interface.

Run with:   streamlit run app.py
The calculations live in the sevendt_calc package; this file only builds the page.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import streamlit as st
from astropy.time import Time

from sevendt_calc import config, etc, overhead, tiles
from sevendt_calc import templates as tpl
from sevendt_calc import visibility as vis
from sevendt_calc.photometry import MAG_INPUT_FILTERS, Spectrum, load_filter_curves
from sevendt_calc.targets import SATURATION_MAG, Target, parse_coordinates, resolve_name

st.set_page_config(page_title='7DT Observation Calculator', page_icon='🔭', layout='wide')

FUNCTIONS = ['Visibility', 'Exposure time / SNR', 'Overhead time', '7DS tile matcher']
FUNCTION_ICONS = {'Visibility': ':material/nights_stay:', 'Exposure time / SNR': ':material/query_stats:',
                  'Overhead time': ':material/timer:', '7DS tile matcher': ':material/grid_on:'}
FUNCTION_BAR_CSS = '''
<style>
.st-key-function_bar button { min-height: 3.6rem; border-width: 2px; border-radius: 0.6rem; }
.st-key-function_bar button p { font-size: 1.15rem; font-weight: 600; }
.st-key-function_bar button span[data-testid="stIconMaterial"] { font-size: 1.6rem; }
</style>
'''
TARGET_COLUMNS = ['Name', 'RA', 'Dec', 'Mag', 'Mag filter', 'Spectrum', 'z']
EXAMPLE_ROWS = [
    {'Name': 'NGC 253', 'RA': '00:47:33.12', 'Dec': '-25:17:17.6', 'Mag': 19.0, 'Mag filter': 'r', 'Spectrum': 'Galaxy: Sc', 'z': 0.0},
    {'Name': 'SN 2026abc', 'RA': '03:12:45.0', 'Dec': '-12:30:00', 'Mag': 18.5, 'Mag filter': 'g', 'Spectrum': 'SN Ia (max)', 'z': 0.02},
    {'Name': '3C 273', 'RA': '12:29:06.7', 'Dec': '+02:03:09', 'Mag': 12.9, 'Mag filter': 'V', 'Spectrum': 'QSO (SDSS, Vanden Berk 2001)', 'z': 0.158},
    {'Name': 'Feige 110', 'RA': '23:19:58.4', 'Dec': '-05:09:56', 'Mag': 11.8, 'Mag filter': 'V', 'Spectrum': 'Star: B0V', 'z': 0.0},
    {'Name': 'AT 2026xyz', 'RA': '20:15:00.0', 'Dec': '-40:00:00', 'Mag': 20.0, 'Mag filter': 'r', 'Spectrum': 'SN II-P (max)', 'z': 0.01},
]
SPECTRUM_FILE_TYPES = ['txt', 'dat', 'csv', 'ascii', 'spec', 'tsv']
RED = 'background-color: #ffb3b3; color: #000000'
CSV_ALIASES = {
    'name': 'Name', 'objname': 'Name', 'target': 'Name', 'object': 'Name', 'id': 'Name',
    'ra': 'RA', 'ra_deg': 'RA', 'dec': 'Dec', 'de': 'Dec', 'dec_deg': 'Dec',
    'mag': 'Mag', 'magnitude': 'Mag', 'mag_filter': 'Mag filter', 'filter': 'Mag filter', 'band': 'Mag filter', 'magfilter': 'Mag filter',
    'spectrum': 'Spectrum', 'spectrum_type': 'Spectrum', 'template': 'Spectrum', 'z': 'z', 'redshift': 'z',
}
UPLOAD_WORDS = ('upload', 'custom', 'file', 'own')


def expected_spectrum_filename(name: str) -> str:
    """File name that is matched automatically to a target: the object name with blanks etc. replaced by '_', plus .txt"""
    stem = re.sub(r'[^A-Za-z0-9.+-]+', '_', str(name).strip()).strip('_')
    return f'{stem or "target"}.txt'


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner='Loading the 7DS tiles...')
def get_tiles() -> tiles.TileSet:
    return tiles.TileSet()


@st.cache_resource(show_spinner='Loading the depth model...')
def get_calculator() -> etc.ExposureCalculator:
    load_filter_curves()
    return etc.ExposureCalculator(etc.DepthModel())


@st.cache_data(ttl=60, show_spinner=False)
def live_config() -> dict:
    """The 7DT configuration folder is re-read at most once a minute."""
    return {
        'site': config.load_site(),
        'filtinfo': config.load_filtinfo(),
        'specmodes': config.load_specmodes(),
        'colormodes': config.load_colormodes(),
        'constraints': vis.Constraints.from_config(),
        'specmode_folder': str(config.specmode_folder()),
        'colormode_folder': str(config.colormode_folder()),
        'config_dir': str(config.config_dir()),
    }


# ------------------------------------------------------------------ small helpers
def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    text = str(value).strip()
    return '' if text.lower() in ('nan', 'none') else text


def empty_targets_df(rows=None) -> pd.DataFrame:
    df = pd.DataFrame(rows if rows is not None else EXAMPLE_ROWS, columns=TARGET_COLUMNS)
    for col in ('Name', 'RA', 'Dec', 'Mag filter', 'Spectrum'):
        df[col] = df[col].map(_text).astype(object)
    # an unknown Spectrum value that mentions upload/custom/file means 'Upload (custom file)'; anything else -> default
    keys = tpl.template_keys()
    lowered = {k.lower(): k for k in keys}
    for i in df.index:
        value = df.at[i, 'Spectrum']
        if value not in keys:
            low = value.lower()
            if low in lowered:
                df.at[i, 'Spectrum'] = lowered[low]
            elif any(w in low for w in UPLOAD_WORDS):
                df.at[i, 'Spectrum'] = tpl.UPLOAD_KEY
            else:
                df.at[i, 'Spectrum'] = tpl.default_key()
    df['Mag'] = pd.to_numeric(df['Mag'], errors='coerce').astype(float)
    df['z'] = pd.to_numeric(df['z'], errors='coerce').fillna(0.0).astype(float)
    return df


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    return buf.getvalue()


def show_figure(fig, name: str, stretch: bool = True):
    st.pyplot(fig, width='stretch' if stretch else 'content')
    st.download_button('Download figure (PNG)', fig_png(fig), file_name=name, mime='image/png', key=f'dl_{name}_{id(fig)}')
    matplotlib.pyplot.close(fig)


def csv_download(df: pd.DataFrame, name: str, label: str = 'Download table (CSV)'):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), file_name=name, mime='text/csv', key=f'dl_{name}')


def styled_rows(df: pd.DataFrame, flag_column: str, flag_value='WARNING'):
    return df.style.apply(lambda row: [RED if row[flag_column] == flag_value else '' for _ in row], axis=1)


def _site_key(site) -> tuple:
    return (site.name, site.latitude, site.longitude, site.elevation, site.timezone)


@st.cache_data(ttl=600, show_spinner=False)
def coming_night_date(site_key: tuple) -> date:
    """Local calendar date of the evening of the current or coming night at the site."""
    site = config.Site(*site_key)
    observer = vis.make_observer(site)
    night = vis.night_for(observer, Time.now())
    return night.sunset.to_datetime(timezone=ZoneInfo(site.timezone)).date()


@st.cache_data(show_spinner=False)
def night_midpoint_utc(site_key: tuple, night_date: date) -> datetime:
    """Middle of the night that starts on the local date `night_date`, as a naive UTC datetime."""
    site = config.Site(*site_key)
    observer = vis.make_observer(site)
    local_noon = datetime.combine(night_date, datetime.min.time().replace(hour=12), tzinfo=ZoneInfo(site.timezone))
    t = vis.night_midpoint(observer, Time(local_noon.astimezone(timezone.utc).replace(tzinfo=None)))
    return t.to_datetime().replace(second=0, microsecond=0)


def night_time_input(prefix: str, site) -> Time:
    """
    Observation time chosen as a night (local date of its evening, default: the coming night) and,
    unless 'Use the middle of the night' is ticked (default), a UTC clock time within that night.
    """
    key = _site_key(site)
    c1, c2, c3 = st.columns([1, 1.3, 1], vertical_alignment='bottom')
    night_date = c1.date_input('Night of (local date)', value=coming_night_date(key), key=f'{prefix}_night')
    use_mid = c2.checkbox('Use the middle of the night', value=True, key=f'{prefix}_mid')
    mid = night_midpoint_utc(key, night_date)
    clock = c3.time_input('Time (UTC)', value=mid.time(), key=f'{prefix}_time', disabled=use_mid, step=300)
    if use_mid:
        return Time(mid)
    # a UTC clock time before 14:00 belongs to the morning half of the night (next UTC date)
    utc_date = night_date + timedelta(days=1) if clock.hour < 14 else night_date
    return Time(datetime.combine(utc_date, clock))


def obsmode_selector(prefix: str, cfg: dict, label: str = 'Observation mode'):
    """Mode (Spec / Deep / Color) plus its sub-mode; returns a resolved ObsMode or None."""
    c1, c2 = st.columns([1, 2])
    mode = c1.selectbox(label, config.MODES, key=f'{prefix}_mode')
    if mode == 'Spec':
        options = sorted(cfg['specmodes'], key=lambda n: (n != 'specall', n))
        sub = c2.selectbox('Sub-mode (specmode file)', options, key=f'{prefix}_spec')
    elif mode == 'Color':
        options = sorted(cfg['colormodes'], key=lambda n: (n != 'gri', n))
        sub = c2.selectbox('Sub-mode (colormode file)', options, key=f'{prefix}_color')
    else:
        installed = config.installed_filters(cfg['filtinfo'])
        default = [f for f in ('g', 'r', 'i') if f in installed]
        sub = c2.multiselect('Filters (every unit observes them in this order)', installed, default=default, key=f'{prefix}_deep')
    try:
        om = config.resolve_obsmode(mode, sub, filtinfo=cfg['filtinfo'], specmodes=cfg['specmodes'], colormodes=cfg['colormodes'])
    except ValueError as exc:
        st.error(str(exc))
        return None
    for w in om.warnings:
        st.warning(w)
    with st.expander(f'{om.label}: {om.n_filters_per_unit} filter(s) per unit, {len(om.filters)} distinct filters, {len(om.units)} units'):
        st.dataframe(pd.DataFrame(om.unit_table()), hide_index=True, width='stretch')
    return om


def policy_selector(prefix: str, om) -> str:
    policy = overhead.allowed_autofocus_policies(om)[0]
    if policy == 'none':
        st.caption('Autofocus: none for Spec/specall.')
    else:
        st.caption(f'Autofocus: once per filter of the mode ({om.n_filters_per_unit} per unit).')
    return policy


# ------------------------------------------------------------------ 1. targets
def load_csv(upload) -> pd.DataFrame:
    raw = pd.read_csv(upload, dtype=str, skipinitialspace=True, comment='#')
    raw.columns = [str(c).strip() for c in raw.columns]
    rename = {c: CSV_ALIASES[c.lower()] for c in raw.columns if c.lower() in CSV_ALIASES}
    raw = raw.rename(columns=rename)
    missing = [c.lower() for c in ('Name', 'RA', 'Dec') if c not in raw.columns]
    if missing:
        raise ValueError(f'the required columns name, ra, dec must all be present (missing: {", ".join(missing)})')
    for col in TARGET_COLUMNS:
        if col not in raw.columns:
            raw[col] = ''
    return empty_targets_df(raw[TARGET_COLUMNS].to_dict('records'))


def parse_targets(df: pd.DataFrame) -> list:
    """Targets from the input table. Each target carries .row, .auto_named and .issues (list of problems)."""
    targets = []
    for i, row in df.reset_index(drop=True).iterrows():
        name, ra_t, dec_t = _text(row['Name']), _text(row['RA']), _text(row['Dec'])
        mag = None if pd.isna(row['Mag']) else float(row['Mag'])
        mag_filter = _text(row['Mag filter']) or None
        spectrum_type = _text(row['Spectrum']) or tpl.default_key()
        z = 0.0 if pd.isna(row['z']) else float(row['z'])
        if not any([name, ra_t, dec_t, mag is not None, mag_filter]):
            continue
        issues, ra, dec = [], None, None
        if ra_t and dec_t:
            try:
                ra, dec = parse_coordinates(ra_t, dec_t)
            except ValueError as exc:
                issues.append(f'cannot read the coordinates ({exc})')
        elif ra_t or dec_t:
            issues.append('RA and Dec are both needed')
        else:
            issues.append('missing RA/Dec')
        if not name:
            issues.append('missing name')
        if mag is not None and mag_filter is None:
            issues.append('magnitude ignored (no filter)')
            mag = None
        try:
            target = Target(name=name or f'Target {i + 1}', ra=ra, dec=dec, mag=mag, mag_filter=mag_filter,
                            spectrum_type=spectrum_type, redshift=z)
        except ValueError as exc:
            issues.append(str(exc))
            target = Target(name=name or f'Target {i + 1}', ra=ra, dec=dec)
        target.row = i
        target.auto_named = not name
        target.issues = issues
        target.spectrum_status = ''
        targets.append(target)
    return targets


SAMPLE_CSV = config.DATA_DIR / 'sample_targets.csv'
SAMPLE_SPECTRUM = config.DATA_DIR / 'sample_spectrum.txt'
FLUX_UNIT_LABELS = ['f_lambda [erg/s/cm²/Å]', 'f_nu [erg/s/cm²/Hz]']
DERIVED_COLUMNS = ['Status']
YELLOW = 'background-color: #fff3b0; color: #000000'


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b''


def _editor_key() -> str:
    return f'editor_{st.session_state.editor_version}'


def apply_editor_state(base: pd.DataFrame, state: dict | None) -> pd.DataFrame:
    """Fold a data_editor edit state (edited_rows / added_rows / deleted_rows) into the input table."""
    df = base.reset_index(drop=True).copy()
    if not state:
        return df
    for idx, changes in state.get('edited_rows', {}).items():
        idx = int(idx)
        if idx in df.index:
            for col, val in changes.items():
                if col in TARGET_COLUMNS:
                    df.at[idx, col] = val
    deleted = [int(i) for i in state.get('deleted_rows', []) if int(i) in df.index]
    if deleted:
        df = df.drop(index=deleted)
    added = [{c: row.get(c, '') for c in TARGET_COLUMNS} for row in state.get('added_rows', [])]
    rows = df.to_dict('records') + added
    return empty_targets_df(rows) if rows else empty_targets_df([]).iloc[0:0]


def commit_editor():
    """
    on_change of the table: fold the edits into the stored inputs. The editor keeps its key, so the
    grid (scroll position, selected cell) survives; only the read-only Status column is recomputed.
    """
    st.session_state.targets_base = apply_editor_state(st.session_state.targets_base, st.session_state.get(_editor_key()))


def remount_editor():
    """Give the editor a new key (after imports, name resolving or row changes) so stale edits are dropped."""
    st.session_state.editor_version += 1


def attach_spectra(targets: list) -> list:
    """
    Give every Upload row its spectrum from the widgets rendered in the Target Spectrum box (their values are
    read from the session state, so this can run before the box is drawn). Returns the Upload rows.
    """
    uploads = [t for t in targets if t.spectrum_type == tpl.UPLOAD_KEY]
    files = {f.name: f for f in (st.session_state.get('bulk_spectra') or [])}
    stems = {Path(name).stem.lower(): name for name in files}
    bulk_unit = 'fnu' if str(st.session_state.get('bulk_unit', FLUX_UNIT_LABELS[0])).startswith('f_nu') else 'flam'
    for t in uploads:
        expected = expected_spectrum_filename(t.name)
        single = st.session_state.get(f'spec_{t.row}_{t.label}')
        single_unit = 'fnu' if str(st.session_state.get(f'spec_unit_{t.row}_{t.label}', FLUX_UNIT_LABELS[0])).startswith('f_nu') else 'flam'
        bulk_name = stems.get(Path(expected).stem.lower()) or stems.get(t.name.lower())
        try:
            if single is not None:
                t.spectrum = Spectrum.from_text(single.getvalue(), single_unit, source=single.name)
                t.spectrum_file = single.name
                t.spectrum_status = f'✓ {single.name}'
            elif bulk_name is not None:
                t.spectrum = Spectrum.from_text(files[bulk_name].getvalue(), bulk_unit, source=bulk_name)
                t.spectrum_file = bulk_name
                t.spectrum_status = f'✓ {bulk_name} (bulk)'
            else:
                t.spectrum_status = f'no file yet (expected {expected})'
                t.issues.append('spectrum file not received')
        except ValueError as exc:
            t.spectrum_status = f'cannot read the file: {exc}'
            t.issues.append('spectrum file unreadable')
    return uploads


def build_table(base: pd.DataFrame, targets: list, installed: list):
    """The input columns plus read-only derived columns, and a Styler colouring the derived cells."""
    disp = base.reset_index(drop=True).copy()
    for col in DERIVED_COLUMNS:
        disp[col] = ''
    styles = pd.DataFrame('', index=disp.index, columns=disp.columns)
    for t in targets:
        i = t.row
        if i not in disp.index:
            continue
        note = t.saturation_note(installed)
        warnings = []
        if note:
            warnings.append('saturation risk')
        if not t.has_photometry:
            warnings.append('no magnitude: skipped by Exposure time / SNR')
        errors = list(t.issues)
        if t.spectrum_type == tpl.UPLOAD_KEY and t.has_spectrum:
            warnings.append(f'spectrum: {t.spectrum_status}')
        disp.at[i, 'Status'] = '; '.join(errors + warnings) if (errors or warnings) else 'OK'
        if errors:
            styles.at[i, 'Status'] = RED
        elif warnings:
            styles.at[i, 'Status'] = YELLOW
    return disp, disp.style.apply(lambda _: styles, axis=None)


def target_editor(left, styler):
    with left:
        st.data_editor(
            styler, num_rows='fixed', hide_index=True, width='stretch', key=_editor_key(),
            on_change=commit_editor, disabled=DERIVED_COLUMNS,
            column_config={
                'Name': st.column_config.TextColumn('Name', width='small', help='Object name; used for name resolving and for matching spectrum files'),
                'RA': st.column_config.TextColumn('RA', width='small', help='Degrees or hh:mm:ss.s'),
                'Dec': st.column_config.TextColumn('Dec', width='small', help='Degrees or ±dd:mm:ss'),
                'Mag': st.column_config.NumberColumn('Mag', width='small', help='Magnitude in the chosen filter (optional)', format='%.2f', step=0.1),
                'Mag filter': st.column_config.SelectboxColumn('Filter', width='small', options=list(MAG_INPUT_FILTERS), help='B V R I: Vega; g r i: AB'),
                'Spectrum': st.column_config.SelectboxColumn('Spectrum', width='medium', options=tpl.template_keys(), default=tpl.default_key(), required=True,
                                                             help='Spectral shape: a template, a flat spectrum, or your own file (Upload)'),
                'z': st.column_config.NumberColumn('z', width='small', min_value=0.0, max_value=10.0, step=0.01, default=0.0, format='%.3f',
                                                   help='Redshift applied to the template spectrum'),
                'Status': st.column_config.TextColumn('Status', width='large', help='Red: missing or unreadable input. Yellow: warning'),
            })


def bulk_upload_panel(right):
    """Right-hand panel: CSV bulk import and the sample file."""
    with right:
        st.markdown('**Bulk upload**')
        upload = st.file_uploader('Targets CSV', type=['csv', 'txt'], key='csv_upload')
        marker = (upload.name, upload.size) if upload is not None else None
        if upload is not None and st.session_state.get('csv_marker') != marker:
            try:
                st.session_state.targets_base = load_csv(upload)
                st.session_state.csv_marker = marker
                remount_editor()
                st.rerun()
            except Exception as exc:
                st.error(f'Could not read the CSV: {exc}')
        st.download_button('Download a sample CSV', _read_bytes(SAMPLE_CSV), file_name='7dt_targets_sample.csv', mime='text/csv',
                           width='stretch', key='dl_sample_csv')
        st.caption('Required columns: name, ra, dec. Optional: mag, mag_filter, spectrum, z.')


def resolve_names_in_base():
    base = st.session_state.targets_base.reset_index(drop=True).copy()
    failed = []
    for i, row in base.iterrows():
        name = _text(row['Name'])
        if name and not (_text(row['RA']) and _text(row['Dec'])):
            try:
                ra, dec = resolve_name(name)
                base.at[i, 'RA'], base.at[i, 'Dec'] = f'{ra:.5f}', f'{dec:+.5f}'
            except ValueError as exc:
                failed.append(str(exc))
    st.session_state.targets_base = base
    st.session_state.resolve_failures = failed
    remount_editor()
    st.rerun()


def target_spectrum_box(uploads: list):
    """Per-row spectrum files (left) and a multi-file box (right); shown only when Upload rows exist."""
    with st.container(border=True):
        st.markdown('**Target Spectrum**: two-column ASCII files, wavelength [Å] and flux. '
                    'A file named `Object_name.txt` is matched to its target automatically.')
        rows_col, bulk_col = st.columns([3, 1.3])
        with bulk_col:
            st.file_uploader('Several files at once', type=SPECTRUM_FILE_TYPES, accept_multiple_files=True, key='bulk_spectra')
            st.radio('Flux unit of these files', FLUX_UNIT_LABELS, key='bulk_unit')
            st.download_button('Download a sample spectrum', _read_bytes(SAMPLE_SPECTRUM), file_name='GRB_260901.txt',
                               mime='text/plain', width='stretch', key='dl_sample_spectrum')
        with rows_col:
            for t in uploads:
                c0, c1, c2 = st.columns([1.5, 3, 1.5], vertical_alignment='center')
                ok = t.has_spectrum
                c0.markdown(f'**{t.label}**  \n' + (f':green[{t.spectrum_status}]' if ok else f':red[{t.spectrum_status}]'))
                c1.file_uploader(f'Spectrum file of {t.label}', type=SPECTRUM_FILE_TYPES, key=f'spec_{t.row}_{t.label}', label_visibility='collapsed')
                c2.radio('Flux unit of the file', FLUX_UNIT_LABELS, key=f'spec_unit_{t.row}_{t.label}', label_visibility='collapsed')
            st.caption('A file chosen in a row overrides a bulk file. With a magnitude the file is scaled to it; without one its own flux is used.')


def target_section(cfg: dict) -> list:
    st.header('Input Target(s)')
    if 'editor_version' not in st.session_state:
        st.session_state.editor_version = 0
        st.session_state.targets_base = empty_targets_df()
    installed = config.installed_filters(cfg['filtinfo'])
    base = st.session_state.targets_base
    targets = parse_targets(base)
    uploads = attach_spectra(targets)
    disp, styler = build_table(base, targets, installed)

    left, right = st.columns([4, 1])
    target_editor(left, styler)
    with left:
        b1, b2, b3, b4 = st.columns([1, 1, 1.6, 2.4], vertical_alignment='center')
        add_clicked = b1.button('Add a row', width='stretch')
        remove_clicked = b2.button('Remove the last row', width='stretch', disabled=len(base) == 0)
        resolve_clicked = b3.button('Resolve names to RA/Dec (Sesame)', width='stretch')
        b4.caption('Red: missing or unreadable input. Yellow: warning.')
    bulk_upload_panel(right)
    if add_clicked:
        rows = base.to_dict('records') + [{c: '' for c in TARGET_COLUMNS}]
        st.session_state.targets_base = empty_targets_df(rows)
        remount_editor()
        st.rerun()
    if remove_clicked and len(base):
        rows = base.to_dict('records')[:-1]
        st.session_state.targets_base = empty_targets_df(rows) if rows else empty_targets_df([]).iloc[0:0]
        remount_editor()
        st.rerun()
    if resolve_clicked:
        resolve_names_in_base()
    for msg in st.session_state.pop('resolve_failures', []):
        st.error(msg)
    if uploads:
        target_spectrum_box(uploads)
    if targets:
        for t in targets:
            note = t.saturation_note(installed)
            if note:
                st.warning(f'{t.label}: {note}')
            for extra in t.notes:
                st.info(f'{t.label}: {extra}')
    else:
        st.info('Add at least one target above.')
    return targets


# ------------------------------------------------------------------ 2. functions
@st.cache_data(show_spinner=False, max_entries=64)
def monthly_visibility_cached(site_key: tuple, limits: tuple, start_night: date, coords: tuple):
    site = config.Site(*site_key)
    cons = vis.Constraints(min_alt=limits[0], max_alt=limits[1], moon_sep=limits[2], sun_alt=limits[3])
    targets = [Target(name=name, ra=ra, dec=dec) for name, ra, dec in coords]
    return vis.monthly_visibility(targets, start_night, site, cons)


def run_visibility(targets, cfg):
    site, cons = cfg['site'], cfg['constraints']
    obstime = night_time_input('vis', site)
    with st.expander('Observatory and observability limits (7DT defaults)'):
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input('Latitude [deg]', -90.0, 90.0, site.latitude, format='%.4f')
        lon = c2.number_input('Longitude [deg, east positive]', -180.0, 180.0, site.longitude, format='%.4f')
        elev = c3.number_input('Altitude [m]', -500.0, 9000.0, site.elevation, format='%.0f')
        c4, c5, c6 = st.columns(3)
        min_alt = c4.number_input('Minimum target altitude [deg]', 0.0, 90.0, cons.min_alt)
        moon_sep = c5.number_input('Minimum Moon separation [deg]', 0.0, 180.0, cons.moon_sep)
        sun_alt = c6.number_input('Night: Sun altitude below [deg]', -30.0, 0.0, cons.sun_alt)
    custom = (lat, lon, elev) != (site.latitude, site.longitude, site.elevation)
    site_used = config.Site('custom site' if custom else site.name, lat, lon, elev, site.timezone)
    cons_used = vis.Constraints(min_alt=min_alt, max_alt=cons.max_alt, moon_sep=moon_sep, sun_alt=sun_alt)

    valid = [t for t in targets if t.has_coord]
    for t in targets:
        if not t.has_coord:
            st.error(f'{t.label}: RA and Dec are required (red cells in the target table).')
    if not valid:
        return
    with st.spinner('Computing visibility...'):
        results = vis.compute_all(valid, obstime, site_used, cons_used)
    night = results[0].night
    st.markdown(f'**Night of {night.local_date} (local)**: sunset {night.utc(night.sunset)} UT, '
                f'Sun {cons_used.sun_alt:g}° at {night.utc(night.evening_twilight)}-{night.utc(night.morning_twilight)} UT, '
                f'sunrise {night.utc(night.sunrise)} UT. Moon illumination {100 * results[0].moon_illumination:.0f} %.')
    n_ok = sum(1 for r in results if not r.issues)
    st.caption(f'{n_ok} of {len(results)} targets are observable at the requested time; rows marked WARNING in the table '
               'list the reason in the last column.')
    df = pd.DataFrame([r.summary() for r in results])
    st.dataframe(styled_rows(df.round(2), 'status'), hide_index=True, width='stretch')
    csv_download(df, 'visibility.csv')
    st.plotly_chart(vis.plot_visibility_interactive(results, site_used), width='stretch',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'visibility', 'scale': 2}})
    st.caption('Hover for altitude, azimuth and Moon separation at any time; click legend entries to hide targets; the camera icon saves a PNG.')

    st.markdown('**The coming month**: highest altitude each night and the interval in which each target is observable.')
    start_night = st.session_state.get('vis_night') or night.sunset.to_datetime(timezone=ZoneInfo(site_used.timezone)).date()
    with st.spinner('Computing the month...'):
        monthly = monthly_visibility_cached(_site_key(site_used), (cons_used.min_alt, cons_used.max_alt, cons_used.moon_sep, cons_used.sun_alt),
                                            start_night, tuple((t.label, round(t.ra, 5), round(t.dec, 5)) for t in valid))
    st.plotly_chart(vis.plot_monthly_interactive(monthly, site_used), width='stretch',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'visibility_month', 'scale': 2}})


@st.cache_data(show_spinner=False)
def typical_seeing() -> float:
    """Median seeing of the frames the depth model was fitted on (the auto value of the seeing input)."""
    model = etc.DepthModel().model
    values = [m['medians'][3] for m in model.values() if len(m.get('medians', [])) > 3]
    return round(float(np.median(values)), 1) if values else 2.0


def _conditions_widgets(prefix: str, site):
    obstime = night_time_input(prefix, site)
    auto = etc.conditions_for(None, None, obstime, typical_seeing(), site=site)
    overrides = {}
    with st.expander('Environmental conditions (7DT defaults)', expanded=False):
        manual = st.checkbox('Edit these values', key=f'{prefix}_manual')
        c1, c2, c3, c4 = st.columns(4)
        seeing = c1.number_input('Seeing ["]', 0.5, 6.0, auto.seeing, 0.1, key=f'{prefix}_seeing', disabled=not manual,
                                 help='Median seeing of the frames behind the depth model')
        phase = c2.number_input('Moon illumination (0-1)', 0.0, 1.0, round(auto.moon_phase, 2), 0.05, key=f'{prefix}_phase', disabled=not manual)
        hss = c3.number_input('Hours since sunset', 0.0, 14.0, round(auto.hour_since_sunset, 1), 0.5, key=f'{prefix}_hss', disabled=not manual)
        msep = c4.number_input('Moon separation [deg]', 0.0, 180.0, 90.0, 5.0, key=f'{prefix}_msep', disabled=not manual,
                               help='Computed from each target\'s coordinates unless edited here')
        if manual:
            overrides = {'moon_phase': phase, 'moon_separation': msep, 'hour_since_sunset': hss}
        else:
            seeing = auto.seeing
        for note in auto.notes:
            st.warning(note)
    return obstime, seeing, overrides


def _etc_block(prefix: str, cfg: dict, calc_mode: str, label: str | None):
    with st.container(border=True):
        if label:
            st.markdown(f'**{label}**')
        c1, c2 = st.columns(2)
        if calc_mode == 'snr':
            value = c1.number_input('Exposure time per frame [s]', 1.0, 7200.0, 100.0, 10.0, key=f'{prefix}_exp')
        else:
            value = c1.number_input('Target SNR (of the stacked frames)', 1.0, 1000.0, 10.0, 1.0, key=f'{prefix}_snr')
        count = int(c2.number_input('Frames per filter and unit', 1, 999, 3, key=f'{prefix}_cnt',
                                    help='Filters observed by several units at once get count x units frames stacked'))
        om = obsmode_selector(prefix, cfg)
        filters = om.filters if om is not None else []
        if calc_mode == 'exptime' and om is not None:
            filters = st.multiselect('Filters to evaluate', om.filters, default=om.filters, key=f'{prefix}_filters')
    return {'value': value, 'count': count, 'obsmode': om, 'filters': filters}


def run_etc(targets, cfg):
    calc = get_calculator()
    st.markdown('**Predictions are for point sources.** They are derived from the measured 5σ depth of real 7DT frames '
                '(10″ aperture), scaled to your exposure time and to the stacked frames (count × units), with the '
                'magnitude in each filter taken from the chosen spectrum.')
    calc_mode = st.radio('Calculation', ['SNR from exposure time', 'Exposure time from SNR'], horizontal=True, key='etc_calc')
    mode_key = 'snr' if calc_mode.startswith('SNR') else 'exptime'
    obstime, seeing, overrides = _conditions_widgets('etc', cfg['site'])

    valid = [t for t in targets if t.has_photometry]
    skipped = [t.label for t in targets if not t.has_photometry]
    if skipped:
        st.info('Skipped (no magnitude with a filter, or no usable spectrum): ' + ', '.join(skipped))
    if not valid:
        return
    same = st.checkbox('Same settings for all targets', value=True, key='etc_same') if len(valid) > 1 else True
    if same:
        shared = _etc_block('etc_all', cfg, mode_key, None)
        settings = [shared] * len(valid)
    else:
        settings = [_etc_block(f'etc_{t.row}', cfg, mode_key, t.label) for t in valid]
    noise = st.checkbox('Draw a random noise realisation of the expected photometry', value=True, key='etc_noise')
    many = len(valid) > 8

    for i, (t, s) in enumerate(zip(valid, settings)):
        if s['obsmode'] is None or not s['filters']:
            continue
        # the first two targets open, the others collapsed (table and figures inside)
        with st.expander(t.label, expanded=i < 2):
            cond = etc.conditions_for(t.ra, t.dec, obstime, seeing, site=cfg['site'], **overrides)
            if not t.has_coord and not overrides:
                st.info('No coordinates: the Moon separation falls back to 90°.')
            st.caption(' | '.join(cond.describe()) + f' | conditions from: {cond.source}')
            for note in cond.notes:
                st.warning(note)
            note = t.saturation_note(s['filters'])
            if note:
                st.warning(note)
            units = s['obsmode'].filter_multiplicity()
            if mode_key == 'snr':
                results = calc.snr(t, s['filters'], s['value'], s['count'], cond, units=units)
            else:
                results = calc.exptime(t, s['filters'], s['value'], s['count'], cond, units=units)
            df = pd.DataFrame([r.row() for r in results])
            st.dataframe(df.round(3), hide_index=True, width='stretch')
            csv_download(df, f'{mode_key}_{t.label}.csv')
            # with many targets the figures of the collapsed ones are drawn on request to keep the page quick
            draw = (i < 2 or not many) or st.checkbox('Draw the figures', key=f'etc_fig_{t.row}')
            if draw:
                seed = int(Time.now().unix) % 10000 if noise else None
                c1, c2 = st.columns(2)
                with c1:
                    if mode_key == 'snr':
                        show_figure(etc.plot_snr_vs_exptime(t, results, cond), f'snr_vs_exptime_{t.label}.png')
                    else:
                        show_figure(etc.plot_exptime_vs_snr(t, results, cond), f'exptime_vs_snr_{t.label}.png')
                with c2:
                    title = None if mode_key == 'snr' else f'{t.label}: photometry at SNR = {s["value"]:g}'
                    show_figure(etc.plot_spectrum_points(t, results, noise_seed=seed, title=title), f'photometry_{t.label}.png')
            unavailable = [r.filter for r in results if not r.available]
            if unavailable:
                st.warning('Not evaluated (no depth model or magnitude): ' + ', '.join(unavailable))


def _overhead_block(prefix: str, cfg: dict, label: str | None):
    with st.container(border=True):
        if label:
            st.markdown(f'**{label}**')
        c1, c2 = st.columns(2)
        exptime = c1.number_input('Exposure time per frame [s]', 0.0, 7200.0, 100.0, 10.0, key=f'{prefix}_exp')
        count = int(c2.number_input('Frames per filter', 1, 999, 3, key=f'{prefix}_cnt'))
        om = obsmode_selector(prefix, cfg)
        policy = policy_selector(prefix, om) if om is not None else None
    return {'exptime': exptime, 'count': count, 'obsmode': om, 'policy': policy}


def run_overhead(targets, cfg):
    base = overhead.OverheadConfig.load()
    with st.expander('Overhead values (7DT defaults)'):
        c = st.columns(5)
        vals = {
            'autofocus': c[0].number_input('Autofocus [s]', 0.0, 3600.0, base.autofocus, key='oh_autofocus'),
            'filter_change': c[1].number_input('Filter change [s]', 0.0, 600.0, base.filter_change, key='oh_fc'),
            'slewing': c[2].number_input('Slewing [s]', 0.0, 600.0, base.slewing, key='oh_slew'),
            'readout': c[3].number_input('Readout [s]', 0.0, 600.0, base.readout, key='oh_ro'),
            'autofocus_history_duration_min': c[4].number_input('Focus history window [min]', 0.0, 600.0, base.autofocus_history_duration_min, key='oh_hist'),
        }
    cfg_oh = overhead.OverheadConfig(source=base.source, **vals)
    if not targets:
        st.error('Add at least one target (a name is enough for this function).')
        return
    same = st.checkbox('Same settings for all targets', value=True, key='oh_same') if len(targets) > 1 else True
    if same:
        shared = _overhead_block('oh_all', cfg, None)
        settings = [shared] * len(targets)
    else:
        settings = [_overhead_block(f'oh_{t.row}', cfg, t.label) for t in targets]
    requests = []
    for t, s in zip(targets, settings):
        if s['obsmode'] is None:
            return
        requests.append(overhead.ObsRequest(t.label, s['exptime'], s['count'], s['obsmode'], s['policy'] or 'per_filter'))
    result = overhead.estimate(requests, cfg_oh)

    m = st.columns(4)
    m[0].metric('Total observation time', overhead.format_duration(result.total))
    m[1].metric('Open-shutter time', overhead.format_duration(result.science_time))
    m[2].metric('Overhead', overhead.format_duration(result.overhead))
    m[3].metric('Efficiency', f'{100 * result.efficiency:.0f} %')
    m[0].caption(f'{result.total:.0f} s in total')
    st.markdown('\n'.join(f'- {line}' for line in result.summary_lines()))
    st.caption('Counts are those of the slowest unit of each target (all units observe in parallel); targets are observed one after another.')

    rows = []
    for tb in result.targets:
        r, cnt, comp = tb.request, tb.counts(), tb.components()
        rows.append({'target': r.name, 'mode': r.obsmode.label, 'exptime_s': r.exptime, 'count': r.count,
                     'autofocus_policy': r.autofocus_policy, 'slowest_unit': tb.critical_unit, 'frames': cnt['n_frames'],
                     'filter_changes': cnt['n_filter_changes'], 'autofocus_runs': cnt['n_autofocus'],
                     'exposure_s': comp['exposure'], 'readout_s': comp['readout'], 'filter_change_s': comp['filter_change'],
                     'autofocus_s': comp['autofocus'], 'slewing_s': comp['slewing'], 'total_s': tb.total,
                     'start_min': tb.start / 60, 'end_min': tb.end / 60})
    df = pd.DataFrame(rows)
    st.dataframe(df.round(2), hide_index=True, width='stretch')
    csv_download(df, 'overhead.csv')
    show_figure(overhead.plot_timeline(result), 'overhead.png')
    with st.expander('Per-unit breakdown'):
        pick = st.selectbox('Target', [tb.request.name for tb in result.targets], key='oh_pick')
        tb = next(x for x in result.targets if x.request.name == pick)
        udf = pd.DataFrame([{'unit': u.unit, 'filters': ', '.join(u.filters), 'frames': u.n_frames, 'filter_changes': u.n_filter_changes,
                             'autofocus_runs': u.n_autofocus, 'exposure_s': u.exposure, 'readout_s': u.readout,
                             'filter_change_s': u.filter_change, 'autofocus_s': u.autofocus, 'total_s': u.total} for u in tb.units])
        st.dataframe(udf, hide_index=True, width='stretch')


def run_tiles(targets, cfg):
    ts = get_tiles()
    valid = [t for t in targets if t.has_coord]
    for t in targets:
        if not t.has_coord:
            st.error(f'{t.label}: RA and Dec are required (red cells in the target table).')
    if not valid:
        return
    same = st.checkbox('Same radius for all targets', value=True, key='tile_same') if len(valid) > 1 else True
    if same:
        radius = st.number_input('Search radius [deg] (0 = point matching)', 0.0, 10.0, 0.0, 0.1, key='tile_radius_all')
        radii = [radius] * len(valid)
    else:
        radii = []
        cols = st.columns(min(len(valid), 4))
        for i, t in enumerate(valid):
            radii.append(cols[i % len(cols)].number_input(f'{t.label}: radius [deg]', 0.0, 10.0, 0.0, 0.1, key=f'tile_radius_{t.row}'))
    st.caption(f'7DT defaults: a warning when the target is within {tiles.EDGE_TOLERANCE_ARCMIN:g} arcmin of a tile edge; '
               f'with a radius, tiles with at least {100 * tiles.DEFAULT_MIN_OVERLAP:.0f} % of their area inside the circle are listed.')
    matches = [ts.match(t.ra, t.dec, radius_deg=r, target=t) for t, r in zip(valid, radii)]
    df = pd.DataFrame([m.summary() for m in matches])
    st.dataframe(styled_rows(df.round(3), 'status'), hide_index=True, width='stretch')
    csv_download(df, 'tile_matches.csv')
    with st.expander('Matched tiles in detail'):
        detail = pd.concat([pd.DataFrame(m.table()).assign(target=m.target.label) for m in matches if m.tiles], ignore_index=True) if any(m.tiles for m in matches) else pd.DataFrame()
        st.dataframe(detail.round(3), hide_index=True, width='stretch')
    show_figure(tiles.plot_tile_matches(matches, ncols=2 if len(matches) > 1 else 1), 'tiles.png', stretch=len(matches) > 1)


# ------------------------------------------------------------------ page
def function_bar() -> str:
    """Four large buttons; the selected function is the filled (primary) one."""
    st.session_state.setdefault('function', FUNCTIONS[0])
    st.markdown(FUNCTION_BAR_CSS, unsafe_allow_html=True)
    st.markdown('**Select a function**')
    with st.container(key='function_bar'):
        cols = st.columns(len(FUNCTIONS))
        for col, name in zip(cols, FUNCTIONS):
            selected = st.session_state.function == name
            if col.button(name, icon=FUNCTION_ICONS[name], type='primary' if selected else 'secondary',
                          width='stretch', key=f'fn_{name}'):
                st.session_state.function = name
                st.rerun()
    return st.session_state.function


def main():
    st.title('7DT Observation Calculator')
    try:
        cfg = live_config()
    except Exception as exc:
        st.error(f'Cannot read the 7DT configuration folder ({config.config_dir()}): {exc}. '
                 'Set SEVENDT_CONFIG_DIR to the folder holding observer.config, filtinfo.dict, specmode.config and colormode.config.')
        st.stop()
    st.markdown('Plan your 7DT observations: check when your targets are visible from the 7DT site, estimate the exposure '
                'time or the SNR in every 7DT filter, add the observing overheads, and find the 7DS survey tiles that cover '
                'your targets. Enter the target(s) once, then pick a function below.')

    targets = target_section(cfg)
    st.divider()
    func = function_bar()
    st.header(func)
    if func == FUNCTIONS[0]:
        run_visibility(targets, cfg)
    elif func == FUNCTIONS[1]:
        run_etc(targets, cfg)
    elif func == FUNCTIONS[2]:
        run_overhead(targets, cfg)
    else:
        run_tiles(targets, cfg)


if __name__ == '__main__':
    main()
