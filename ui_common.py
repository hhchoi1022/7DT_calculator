"""
Pieces shared by the stand-alone 7DT calculator pages: the observatory panel, the target table
(index, name, coordinates, plus optional extra columns), CSV bulk import and name resolving.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import astropy.units as u
import pandas as pd
import streamlit as st
from astropy.time import Time, TimeDelta

from sevendt_calc import config, modeplot
from sevendt_calc import templates as tpl
from sevendt_calc import visibility as vis
from sevendt_calc.photometry import Spectrum
from sevendt_calc.targets import Target, parse_coordinates, resolve_name

BASE_COLUMNS = ['Name', 'RA', 'Dec']
CSV_ALIASES = {'name': 'Name', 'objname': 'Name', 'target': 'Name', 'object': 'Name', 'id': 'Name',
               'ra': 'RA', 'ra_deg': 'RA', 'dec': 'Dec', 'de': 'Dec', 'dec_deg': 'Dec'}
RED = 'background-color: #ffb3b3; color: #000000'
YELLOW = 'background-color: #fff3b0; color: #000000'


FEEDBACK_EMAIL = 'hhchoi1022@gmail.com'


def feedback_note():
    """A small line at the very top of every page: where to report problems and send feedback."""
    st.markdown(f"<div style='font-size:0.78rem;color:#777;margin-bottom:-0.6rem'>Problems or any feedback: "
                f"<a href='mailto:{FEEDBACK_EMAIL}'>{FEEDBACK_EMAIL}</a></div>", unsafe_allow_html=True)


@st.cache_data(ttl=60, show_spinner=False)
def live_config() -> dict:
    """The 7DT configuration folder is re-read at most once a minute."""
    return {'site': config.load_site(), 'filtinfo': config.load_filtinfo(), 'specmodes': config.load_specmodes(),
            'colormodes': config.load_colormodes(), 'constraints': vis.Constraints.from_config()}


def site_key(site) -> tuple:
    return (site.name, site.latitude, site.longitude, site.elevation, site.timezone)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    text = str(value).strip()
    return '' if text.lower() in ('nan', 'none') else text


# ------------------------------------------------------------------ observatory
def site_panel(cfg: dict, prefix: str = 'site'):
    """Observatory and observability limits, 7DT defaults, editable. Returns (site, constraints)."""
    site, cons = cfg['site'], cfg['constraints']
    with st.expander('Configuration (7DT defaults)'):
        st.caption(f'{site.name}: lat {site.latitude:.4f}°, lon {site.longitude:.4f}°, {site.elevation:.0f} m, {site.timezone}. '
                   f'Observable when altitude ≥ {cons.min_alt:g}°, Moon separation ≥ {cons.moon_sep:g}°, Sun altitude ≤ {cons.sun_alt:g}°.')
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input('Latitude [deg]', -90.0, 90.0, site.latitude, format='%.4f', key=f'{prefix}_lat')
        lon = c2.number_input('Longitude [deg, east positive]', -180.0, 180.0, site.longitude, format='%.4f', key=f'{prefix}_lon')
        elev = c3.number_input('Altitude [m]', -500.0, 9000.0, site.elevation, format='%.0f', key=f'{prefix}_elev')
        c4, c5, c6 = st.columns(3)
        min_alt = c4.number_input('Minimum target altitude [deg]', 0.0, 90.0, cons.min_alt, key=f'{prefix}_minalt')
        moon_sep = c5.number_input('Minimum Moon separation [deg]', 0.0, 180.0, cons.moon_sep, key=f'{prefix}_moonsep')
        sun_alt = c6.number_input('Night: Sun altitude below [deg]', -30.0, 0.0, cons.sun_alt, key=f'{prefix}_sunalt')
    custom = (lat, lon, elev) != (site.latitude, site.longitude, site.elevation)
    site_used = config.Site('custom site' if custom else site.name, lat, lon, elev, site.timezone)
    cons_used = vis.Constraints(min_alt=min_alt, max_alt=cons.max_alt, moon_sep=moon_sep, sun_alt=sun_alt)
    return site_used, cons_used


# ------------------------------------------------------------------ target table
def make_table(rows, extra_columns: dict | None = None) -> pd.DataFrame:
    """A clean input table: Name, RA, Dec and the extra columns with their defaults; the row index (1..N) is the Index."""
    extra_columns = extra_columns or {}
    columns = BASE_COLUMNS + list(extra_columns)
    df = pd.DataFrame(list(rows), columns=columns)
    for col in ('Name', 'RA', 'Dec'):
        df[col] = df[col].map(_text).astype(object)
    for col, default in extra_columns.items():
        if isinstance(default, float):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(float)
        else:
            df[col] = df[col].map(lambda v, d=default: _text(v) or d).astype(object)
    df = df.reset_index(drop=True)
    df.index = pd.RangeIndex(1, len(df) + 1, name='Index')
    return df


def load_csv(upload, extra_columns: dict | None = None) -> pd.DataFrame:
    raw = pd.read_csv(upload, dtype=str, skipinitialspace=True, comment='#')
    raw.columns = [str(c).strip() for c in raw.columns]
    aliases = dict(CSV_ALIASES)
    for col in (extra_columns or {}):
        aliases[col.lower()] = col
    raw = raw.rename(columns={c: aliases[c.lower()] for c in raw.columns if c.lower() in aliases})
    missing = [c for c in ('RA', 'Dec') if c not in raw.columns]
    if missing:
        raise ValueError(f'the CSV needs the columns ra and dec (missing: {", ".join(c.lower() for c in missing)})')
    for col in BASE_COLUMNS + list(extra_columns or {}):
        if col not in raw.columns:
            raw[col] = ''
    return make_table(raw.to_dict('records'), extra_columns)


def parse_targets(df: pd.DataFrame, build=None) -> list:
    """
    Targets from the table. Rows without a name are named by their index. `build(row, issues)` may
    return a Target with extra fields; by default only name and coordinates are used.
    Every target carries .row (0-based), .index (1-based) and .issues.
    """
    targets = []
    for i, row in df.reset_index(drop=True).iterrows():
        name, ra_t, dec_t = _text(row['Name']), _text(row['RA']), _text(row['Dec'])
        if not any([name, ra_t, dec_t]):
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
        label = name or str(i + 1)
        try:
            target = build(row, label, ra, dec, issues) if build else Target(name=label, ra=ra, dec=dec)
        except ValueError as exc:
            issues.append(str(exc))
            target = Target(name=label, ra=ra, dec=dec)
        target.row = i
        target.index = i + 1
        target.issues = issues
        targets.append(target)
    return targets


def _base(prefix: str, example_rows, extra_columns):
    if f'{prefix}_version' not in st.session_state:
        st.session_state[f'{prefix}_version'] = 0
        st.session_state[f'{prefix}_base'] = make_table(example_rows, extra_columns)
    return st.session_state[f'{prefix}_base']


def _remount(prefix: str, df: pd.DataFrame, extra_columns):
    st.session_state[f'{prefix}_base'] = make_table(df.to_dict('records'), extra_columns)
    st.session_state[f'{prefix}_version'] += 1
    st.rerun()


def targets_table(prefix: str, example_rows, extra_columns: dict | None = None, column_config: dict | None = None,
                  sample_csv: bytes | None = None, csv_help: str = 'Columns: name (optional), ra, dec') -> pd.DataFrame:
    """
    Editable target table with a + row (num_rows='dynamic'), a Bulk upload panel on the right and a
    name-resolving button. Cell edits do not re-mount the table; adding or deleting rows does (to
    renumber the Index column). Returns the edited table.
    """
    extra_columns = extra_columns or {}
    base = _base(prefix, example_rows, extra_columns)
    left, right = st.columns([4, 1])
    cfg = {
        'Name': st.column_config.TextColumn('Name', help='Optional; used for name resolving: an object name (Sesame) or a 7DS tile ID such as T01234 (an empty name is replaced by the Index)'),
        'RA': st.column_config.TextColumn('RA', help='Degrees or hh:mm:ss.s'),
        'Dec': st.column_config.TextColumn('Dec', help='Degrees or ±dd:mm:ss'),
    }
    cfg.update(column_config or {})
    with left:
        edited = st.data_editor(base, num_rows='dynamic', hide_index=False, width='stretch',
                                key=f'{prefix}_editor_{st.session_state[f"{prefix}_version"]}', column_config=cfg)
        b1, b2 = st.columns([1.4, 3], vertical_alignment='center')
        resolve_clicked = b1.button('Resolve names to RA/Dec', width='stretch', key=f'{prefix}_resolve')
        b2.caption('RA/Dec in degrees or hh:mm:ss / ±dd:mm:ss. Use the last (empty) row to add a target; select rows and press Delete to remove them.')
    with right:
        st.markdown('**Bulk upload**')
        upload = st.file_uploader('Targets CSV', type=['csv', 'txt'], key=f'{prefix}_csv')
        marker = (upload.name, upload.size) if upload is not None else None
        if upload is not None and st.session_state.get(f'{prefix}_csv_marker') != marker:
            try:
                df = load_csv(upload, extra_columns)
                st.session_state[f'{prefix}_csv_marker'] = marker
                _remount(prefix, df, extra_columns)
            except Exception as exc:
                st.error(f'Could not read the CSV: {exc}')
        if sample_csv:
            st.download_button('Download a sample CSV', sample_csv, file_name='7dt_targets_sample.csv', mime='text/csv',
                               width='stretch', key=f'{prefix}_sample')
        st.caption(csv_help)
    if len(edited) != len(base):            # rows added or deleted: renumber and re-mount
        _remount(prefix, edited, extra_columns)
    if resolve_clicked:
        df = edited.copy()
        failed = []
        for i, row in df.iterrows():
            name = _text(row['Name'])
            if name:                                   # every named row is resolved, existing RA/Dec are overwritten
                try:
                    ra, dec = resolve_name(name)
                    df.at[i, 'RA'], df.at[i, 'Dec'] = f'{ra:.5f}', f'{dec:+.5f}'
                except ValueError as exc:
                    failed.append(str(exc))
        st.session_state[f'{prefix}_resolve_failures'] = failed
        _remount(prefix, df, extra_columns)
    for msg in st.session_state.pop(f'{prefix}_resolve_failures', []):
        st.error(msg)
    return edited


def report_issues(targets: list):
    """One red line per target with a problem (the table itself stays plain)."""
    for t in targets:
        if t.issues:
            st.error(f'{t.label}: ' + '; '.join(t.issues))


# ------------------------------------------------------------------ night
@st.cache_data(ttl=600, show_spinner=False)
def coming_night_date(key: tuple) -> date:
    site = config.Site(*key)
    night = vis.night_for(vis.make_observer(site), Time.now())
    return night.sunset.to_datetime(timezone=ZoneInfo(site.timezone)).date()


@st.cache_data(show_spinner=False)
def night_midpoint_utc(key: tuple, night_date: date) -> datetime:
    site = config.Site(*key)
    local_noon = datetime.combine(night_date, datetime.min.time().replace(hour=12), tzinfo=ZoneInfo(site.timezone))
    t = vis.night_midpoint(vis.make_observer(site), Time(local_noon.astimezone(timezone.utc).replace(tzinfo=None)))
    return t.to_datetime().replace(second=0, microsecond=0)


def night_date_input(prefix: str, site, label: str = 'Night of (local date)') -> date:
    return st.date_input(label, value=coming_night_date(site_key(site)), key=f'{prefix}_night')


def styled_rows(df: pd.DataFrame, flag_column: str, flag_value='WARNING'):
    return df.style.apply(lambda row: [RED if row[flag_column] == flag_value else '' for _ in row], axis=1)


# ------------------------------------------------------------------ moon
@st.cache_data(show_spinner=False, max_entries=128)
def moon_info(key: tuple, night_date: date) -> dict:
    """Illumination, waxing/waning and rise/set times of the Moon for the night starting on `night_date`."""
    site = config.Site(*key)
    observer = vis.make_observer(site)
    tz = ZoneInfo(site.timezone)
    local_noon = datetime.combine(night_date, datetime.min.time().replace(hour=12), tzinfo=tz)
    night = vis.night_for(observer, Time(local_noon.astimezone(timezone.utc).replace(tzinfo=None)))
    mid = night.evening_twilight + (night.morning_twilight - night.evening_twilight) / 2
    frac = float(observer.moon_illumination(mid))
    waxing = float(observer.moon_illumination(mid + TimeDelta(0.5, format='jd'))) > frac

    def local(t):
        return t.to_datetime(timezone=tz).strftime('%H:%M')

    # what the Moon does between the evening and morning twilights
    up_at_dusk = float(observer.moon_altaz(night.evening_twilight).alt.deg) > 0
    if up_at_dusk:
        sett = observer.moon_set_time(night.evening_twilight, which='next')
        during = f'up at dusk, sets at {local(sett)}' if sett < night.morning_twilight else 'up all night'
    else:
        rise = observer.moon_rise_time(night.evening_twilight, which='next')
        during = f'rises at {local(rise)}' if rise < night.morning_twilight else 'below the horizon all night'
    if frac < 0.03:
        name = 'new Moon'
    elif frac > 0.97:
        name = 'full Moon'
    elif abs(frac - 0.5) < 0.04:
        name = 'first quarter' if waxing else 'last quarter'
    elif frac < 0.5:
        name = 'waxing crescent' if waxing else 'waning crescent'
    else:
        name = 'waxing gibbous' if waxing else 'waning gibbous'
    return {'fraction': frac, 'waxing': waxing, 'name': name, 'during': during, 'local_date': night.local_date}


def moon_svg(fraction: float, waxing: bool, size: int = 110, southern: bool = True) -> str:
    """
    A Moon disk with the lit part drawn for the illuminated fraction. Seen from the southern
    hemisphere a waxing Moon is lit on the left, so lit side = left when waxing (mirrored otherwise).
    """
    r = size * 0.42
    cx = cy = size / 2
    f = min(max(fraction, 0.0), 1.0)
    lit_left = waxing if southern else not waxing
    rx = r * abs(1 - 2 * f)
    top, bottom = f'{cx:.1f},{cy - r:.1f}', f'{cx:.1f},{cy + r:.1f}'
    # sweep-flag 1 runs clockwise on screen: top -> right -> bottom -> left -> top
    if lit_left:
        half = f'M {top} A {r:.1f},{r:.1f} 0 0 0 {bottom}'          # semicircle through the left side
        sweep = 0 if f > 0.5 else 1                                  # gibbous: terminator bulges into the right (dark) side
    else:
        half = f'M {top} A {r:.1f},{r:.1f} 0 0 1 {bottom}'          # semicircle through the right side
        sweep = 1 if f > 0.5 else 0
    lit = f'{half} A {rx:.1f},{r:.1f} 0 0 {sweep} {top} Z'
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#3b3b46"/>'
            f'<path d="{lit}" fill="#f4e7b2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#777" stroke-width="1"/></svg>')


def moon_panel(site, night_date: date):
    """The Moon of the chosen night: a phase drawing with illumination, phase name and rise/set times."""
    info = moon_info(site_key(site), night_date)
    c1, c2 = st.columns([1, 2.2], vertical_alignment='center')
    c1.markdown(moon_svg(info['fraction'], info['waxing']), unsafe_allow_html=True)
    c2.markdown(f"**Moon on the night of {info['local_date']}**  \n"
                f"{100 * info['fraction']:.0f} % illuminated, {info['name']}  \n"
                f"{info['during']} (local time)")
    return info


# ------------------------------------------------------------------ observation modes
def obsmode_selector(prefix: str, cfg: dict, label: str = 'Observation mode'):
    """Mode (Spec / Deep / Color) plus its sub-mode; returns a resolved ObsMode or None."""
    warm_mode_figures(mode_folder_signature())
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
        png = obsmode_png(om.mode, om.submode, tuple((u, tuple(f)) for u, f in om.filters_by_unit.items()))
        st.image(png, width='stretch')
        st.download_button('Download figure (PNG)', png, file_name=f'obsmode_{om.mode}_{om.submode}.png', mime='image/png', key=f'{prefix}_modefig')
    return om


MODE_FIGURE_DIR = config.DATA_DIR / 'cache' / 'obsmode'


def _mode_figure_path(mode: str, submode: str, table: tuple) -> Path:
    import hashlib
    digest = hashlib.md5(repr((mode, submode, table)).encode()).hexdigest()[:10]
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{mode}_{submode}')
    return MODE_FIGURE_DIR / f'{safe}_{digest}.png'


def render_mode_figure(mode: str, submode: str, table: tuple) -> bytes:
    """Draw the mode figure (a few seconds) and keep the PNG on disk, keyed by the mode's content."""
    path = _mode_figure_path(mode, submode, table)
    if path.exists():
        return path.read_bytes()
    om = config.ObsMode(mode=mode, submode=submode, filters_by_unit={u: f for u, f in table})
    fig = modeplot.plot_obsmode(om)
    png = figure_png(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    MODE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(png)
    tmp.replace(path)
    return png


@st.cache_data(show_spinner='Drawing the observation mode...', max_entries=64)
def obsmode_png(mode: str, submode: str, table: tuple) -> bytes:
    """The mode figure: from the disk cache when the warm-up has drawn it, otherwise drawn now."""
    return render_mode_figure(mode, submode, table)


def mode_folder_signature() -> tuple:
    """Names and modification times of the mode files: changes here re-trigger the warm-up."""
    sig = []
    for folder, suffix in ((config.specmode_folder(), '.specmode'), (config.colormode_folder(), '.colormode')):
        try:
            sig += [(f.name, int(f.stat().st_mtime)) for f in sorted(folder.glob(f'*{suffix}'))]
        except OSError:
            continue
    return tuple(sig)


@st.cache_resource(show_spinner=False)
def warm_mode_figures(signature: tuple = ()) -> bool:
    """Draw the figures of every Spec and Color mode in a background thread, once per server and per
    state of the mode files (a changed or new mode file starts a new warm-up)."""
    import threading

    def work():
        try:
            cfg = live_config.__wrapped__() if hasattr(live_config, '__wrapped__') else live_config()
        except Exception:
            cfg = {'filtinfo': config.load_filtinfo(), 'specmodes': config.load_specmodes(), 'colormodes': config.load_colormodes()}
        jobs = [('Spec', name) for name in cfg['specmodes']] + [('Color', name) for name in cfg['colormodes']] + [('Deep', 'g,r,i')]
        for mode, sub in jobs:
            try:
                om = config.resolve_obsmode(mode, sub, filtinfo=cfg['filtinfo'], specmodes=cfg['specmodes'], colormodes=cfg['colormodes'])
                render_mode_figure(om.mode, om.submode, tuple((u, tuple(f)) for u, f in om.filters_by_unit.items()))
            except Exception:
                continue

    threading.Thread(target=work, name='mode-figure-warmup', daemon=True).start()
    return True


def figure_png(fig, dpi: int = 300) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def show_figure(fig, name: str, key: str | None = None, dpi: int = 300, download: bool = True):
    """Render a matplotlib figure as a sharp PNG (300 dpi) stretched to the container, with a download button."""
    import matplotlib.pyplot as plt
    png = figure_png(fig, dpi)
    st.image(png, width='stretch')
    if download:
        st.download_button('Download figure (PNG)', png, file_name=name, mime='image/png', key=key or f'dl_{name}')
    plt.close(fig)


# ------------------------------------------------------------------ night with a time
def night_time_input(prefix: str, site) -> Time:
    """Night (local date), 'Use the middle of the night' (default) or a UTC clock time within that night."""
    key = site_key(site)
    c1, c2, c3 = st.columns([1, 1.3, 1], vertical_alignment='bottom')
    night_date = c1.date_input('Night of (local date)', value=coming_night_date(key), key=f'{prefix}_night')
    use_mid = c2.checkbox('Use the middle of the night', value=True, key=f'{prefix}_mid')
    mid = night_midpoint_utc(key, night_date)
    clock = c3.time_input('Time (UTC)', value=mid.time(), key=f'{prefix}_time', disabled=use_mid, step=300)
    if use_mid:
        return Time(mid)
    utc_date = night_date + timedelta(days=1) if clock.hour < 14 else night_date
    return Time(datetime.combine(utc_date, clock))


# ------------------------------------------------------------------ spectra of Upload rows
SPECTRUM_FILE_TYPES = ['txt', 'dat', 'csv', 'ascii', 'spec', 'tsv']
FLUX_UNIT_LABELS = ['f_lambda [erg/s/cm²/Å]', 'f_nu [erg/s/cm²/Hz]']


def expected_spectrum_filename(name: str) -> str:
    """File name matched automatically to a target: the object name with blanks etc. replaced by '_', plus .txt"""
    stem = re.sub(r'[^A-Za-z0-9.+-]+', '_', str(name).strip()).strip('_')
    return f'{stem or "target"}.txt'


def attach_spectra(targets: list, prefix: str) -> list:
    """
    Give every Upload row its spectrum from the widgets of the Target Spectrum box (read from the
    session state, so this can run before the box is drawn). Returns the Upload rows.
    """
    uploads = [t for t in targets if t.spectrum_type == tpl.UPLOAD_KEY]
    files = {f.name: f for f in (st.session_state.get(f'{prefix}_bulk_spectra') or [])}
    stems = {Path(name).stem.lower(): name for name in files}
    bulk_unit = 'fnu' if str(st.session_state.get(f'{prefix}_bulk_unit', FLUX_UNIT_LABELS[0])).startswith('f_nu') else 'flam'
    for t in uploads:
        expected = expected_spectrum_filename(t.name)
        single = st.session_state.get(f'{prefix}_spec_{t.row}')
        single_unit = 'fnu' if str(st.session_state.get(f'{prefix}_spec_unit_{t.row}', FLUX_UNIT_LABELS[0])).startswith('f_nu') else 'flam'
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


def target_spectrum_box(uploads: list, prefix: str, sample: bytes | None = None, sample_name: str = 'sample_spectrum.txt'):
    """Per-row spectrum files (left) and a multi-file box (right); call only when Upload rows exist."""
    with st.container(border=True):
        st.markdown('**Target Spectrum**: two-column ASCII files, wavelength [Å] and flux. '
                    'A file named `Object_name.txt` is matched to its target automatically.')
        rows_col, bulk_col = st.columns([3, 1.3])
        with bulk_col:
            st.file_uploader('Several files at once', type=SPECTRUM_FILE_TYPES, accept_multiple_files=True, key=f'{prefix}_bulk_spectra')
            st.radio('Flux unit of these files', FLUX_UNIT_LABELS, key=f'{prefix}_bulk_unit')
            if sample:
                st.download_button('Download a sample spectrum', sample, file_name=sample_name, mime='text/plain',
                                   width='stretch', key=f'{prefix}_dl_sample_spectrum')
        with rows_col:
            for t in uploads:
                c0, c1, c2 = st.columns([1.5, 3, 1.5], vertical_alignment='center')
                ok = t.has_spectrum
                c0.markdown(f'**{t.label}**  \n' + (f':green[{t.spectrum_status}]' if ok else f':red[{t.spectrum_status}]'))
                c1.file_uploader(f'Spectrum file of {t.label}', type=SPECTRUM_FILE_TYPES, key=f'{prefix}_spec_{t.row}', label_visibility='collapsed')
                c2.radio('Flux unit of the file', FLUX_UNIT_LABELS, key=f'{prefix}_spec_unit_{t.row}', label_visibility='collapsed')
            st.caption('A file chosen in a row overrides a bulk file. With a magnitude the file is scaled to it; without one its own flux is used.')
