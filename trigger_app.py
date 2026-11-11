"""
7DT Observation request: targets (or 7DS survey tiles) entered in a fixed format, validated against the
request rules and the current instrument, checked against the project's allocated time, and written to
the request spreadsheet as one worksheet per project.
Run with:  streamlit run trigger_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from astropy.time import Time

from sevendt_calc import config, overhead, sheets, tiles, trigger
from sevendt_calc import visibility as vis
from sevendt_calc.targets import Target, parse_coordinates, resolve_name
from ui_common import FEEDBACK_EMAIL, FEEDBACK_NAME, feedback_note, live_config, night_date_input, night_midpoint_utc, obsmode_png, site_key, styled_rows

st.set_page_config(page_title='7DT Observation request', page_icon='📝', layout='wide')

REQUEST_TYPES = ('Target', 'Survey')
DATE_COLUMNS = ('obs_startdate', 'obs_enddate')
EDITOR_COLUMNS = {'Target': ['objname', 'RA', 'De', 'exptime', 'count', 'obs_startdate', 'obs_enddate'],
                  'Survey': ['objname', 'exptime', 'count', 'obs_startdate', 'obs_enddate']}
EXAMPLE_ROWS = {'Target': [{'objname': 'NGC0253', 'RA': '00:47:33.12', 'De': '-25:17:17.6', 'exptime': '100', 'count': '3'}],
                'Survey': [{'objname': 'T01022', 'exptime': '100', 'count': '3'}, {'objname': 'T01023', 'exptime': '100', 'count': '3'}]}
HELP = {
    'objname': 'Target name, up to 100 characters. Survey request: the 7DS tile ID, e.g. T01022 (the coordinates are the tile centre).',
    'RA': 'Right ascension: hh:mm:ss.s (e.g. 00:47:33.12) or degrees (e.g. 11.888).',
    'De': 'Declination: ±dd:mm:ss (e.g. -25:17:17.6) or degrees (e.g. -25.288).',
    'exptime': 'Exposure time per frame in seconds, one number; standard 100, 180 at most. Applied to every filter of the mode.',
    'count': 'Number of frames per filter, a whole number; standard 3.',
    'obs_startdate': 'Optional. Earliest UTC date of the observation, typed as YYYY-MM-DD (e.g. 2026-11-11) or YYYY-MM-DDTHH:MM:SS; press Enter to confirm.',
    'obs_enddate': 'Optional. Latest UTC date of the observation, typed as YYYY-MM-DD (e.g. 2026-12-01) or YYYY-MM-DDTHH:MM:SS; press Enter to confirm.',
}
RED, YELLOW = 'background-color: #ffb3b3', 'background-color: #fff3b0'
PLOT_LIMIT = 10                      # targets drawn in the visibility plots by default (the table covers all of them)


# ------------------------------------------------------------------ state helpers
def _text(v) -> str:
    return '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _date_text(v) -> str:
    """A date cell as typed (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS); a date-only value gets the time 00:00:00. Unreadable text is kept for the validation to flag."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ''
    if hasattr(v, 'strftime'):
        return pd.Timestamp(v).strftime(trigger.DATE_FORMAT)
    text = str(v).strip()
    if not text or text.lower() == 'none':
        return ''
    d, err = trigger.parse_date(text)
    return d.strftime(trigger.DATE_FORMAT) if d and not err else text


def _frame(rows, columns) -> pd.DataFrame:
    """Editor frame: every column as text; dates are shown as YYYY-MM-DD (plus the time when one was given)."""
    df = pd.DataFrame(rows).reindex(columns=columns)
    for c in columns:
        if c in DATE_COLUMNS:
            shown = []
            for v in df[c]:
                t = _date_text(v)
                shown.append(t[:-9] if t.endswith('T00:00:00') else t)
            df[c] = pd.Series(shown, index=df.index, dtype=object)
        else:
            df[c] = df[c].map(_text)
    df.index = pd.RangeIndex(1, len(df) + 1, name='Index')
    return df


def _remount(df: pd.DataFrame):
    st.session_state['tr_rows'] = df
    st.session_state['tr_version'] = st.session_state.get('tr_version', 0) + 1


def _rows_as_text(df: pd.DataFrame) -> list:
    rows = [{c: (_date_text(v) if c in DATE_COLUMNS else _text(v)) for c, v in r.items()} for r in df.to_dict('records')]
    return [r for r in rows if any(r.get(c) for c in ('objname', 'RA', 'De', 'exptime', 'count'))]


@st.cache_resource(show_spinner=False)
def get_tiles() -> tiles.TileSet:
    return tiles.TileSet()


# ------------------------------------------------------------------ project
def project_section(ts: trigger.TriggerSettings):
    st.header('Project')
    c1, c2, c3 = st.columns([1, 1, 1.6], vertical_alignment='center')
    project = c1.text_input('Project ID', key='tr_project', placeholder='e.g. GRB_followup',
                            help='Letters, digits, _ and -. The rows go to the worksheet <period>_<Project ID> of the request spreadsheet.')
    key = c2.text_input('Project key', key='tr_key', type='password', help=f'Given with the time allocation. Lost it? Contact {FEEDBACK_NAME} ({FEEDBACK_EMAIL}).')
    projects = trigger.load_projects()
    ok, msg, hours = trigger.check_project(project, key, projects)
    if not trigger.safe_project_id(project):
        c3.caption('The Project ID is required before submitting.')
    elif ok and hours is not None:
        c3.metric('Time allocated', f'{hours:g} h', help='Total observation time of the project for this period, overheads included.')
    elif ok:
        c3.caption(msg)
    else:
        c3.error(msg)
    return project, ok, hours


def load_from_sheet(ts: trigger.TriggerSettings, project: str, ok: bool):
    """Once per project: fetch its worksheet and put the rows into the table (submitting then replaces the worksheet)."""
    ss = st.session_state
    pid = trigger.safe_project_id(project)
    key_path = sheets.service_account_key_path()
    if not (ok and pid and key_path):
        return
    if ss.get('tr_loaded_for') == pid:
        return
    try:
        with st.spinner('Reading the project worksheet...'):
            rows = sheets.read_rows(ts.spreadsheet_url, ts.worksheet_name(pid), key_path)
    except Exception as exc:
        ss['tr_loaded_for'] = pid
        st.warning(f'Could not read the worksheet: {exc}')
        return
    ss['tr_loaded_for'] = pid
    ss['tr_replace'] = bool(rows)
    if not rows:
        return
    first = rows[0]
    survey = all(tiles.normalise_tile_id(r.get('objname', '')) for r in rows)
    rtype = 'Survey' if survey else 'Target'
    ss['tr_type_pending'] = rtype
    mode = first.get('obsmode', '')
    if mode in ts.obsmodes:
        ss['tr_obsmode_pending'] = mode
        sub = {'Spec': first.get('specmode', ''), 'Color': first.get('colormode', ''), 'Deep': first.get('filter_', '')}.get(mode, '')
        ss['tr_submode_pending'] = (mode, sub)
    cleaned = [{c: ('' if str(r.get(c, '')).strip() == 'None' else r.get(c, '')) for c in EDITOR_COLUMNS[rtype]} for r in rows]
    _remount(_frame(cleaned, EDITOR_COLUMNS[rtype]))
    ss['tr_loaded_note'] = f'{len(rows)} row(s) loaded from worksheet {ts.worksheet_name(pid)}; submitting replaces them.'


# ------------------------------------------------------------------ settings
def settings_section(ts: trigger.TriggerSettings, cfg: dict):
    st.header('Observation settings')
    ss = st.session_state
    if 'tr_obsmode_pending' in ss:
        ss['tr_obsmode'] = ss.pop('tr_obsmode_pending')
    pending = ss.pop('tr_submode_pending', None)
    c1, c2 = st.columns([1, 2])
    obsmode = c1.selectbox('Observation mode', list(ts.obsmodes), key='tr_obsmode',
                           help='Spec and Color use every active telescope and start only when all of them are idle.')
    if pending and pending[0] == obsmode and pending[1]:
        ss[{'Spec': 'tr_specmode', 'Color': 'tr_colormode', 'Deep': 'tr_deepmode'}[obsmode]] = pending[1]
    if obsmode == 'Spec':
        sub_field, submode = 'specmode', c2.selectbox('Specmode', ts.spec_options(cfg['specmodes']), key='tr_specmode')
    elif obsmode == 'Color':
        sub_field, submode = 'colormode', c2.selectbox('Colormode', list(ts.colormodes), key='tr_colormode')
    else:
        sub_field, submode = 'filter_', c2.selectbox('Deepmode', list(ts.deepmodes), key='tr_deepmode')
    if obsmode in ('Spec', 'Color'):
        st.caption('Uses every active telescope and starts only when all of them are idle.')
    om = trigger.request_obsmode(obsmode, submode, cfg)
    if om is not None:
        with st.expander(f'{om.label}: {om.n_filters_per_unit} filter(s) per unit, {len(om.filters)} distinct filters, {len(om.units)} units'):
            st.image(obsmode_png(om.mode, om.submode, tuple((u, tuple(f)) for u, f in om.filters_by_unit.items())), width='stretch')
    binning, gain = int(ts.default_binning), int(ts.default_gain)
    st.caption(f'Binning {binning}, gain {gain} and priority {ts.priority} are fixed by the observatory. To change the gain, contact {FEEDBACK_NAME} ({FEEDBACK_EMAIL}).')
    defaults = {'obsmode': obsmode, sub_field: submode, 'binning': str(binning), 'gain': str(gain), 'obs_startdate': '', 'obs_enddate': ''}
    return defaults, om


# ------------------------------------------------------------------ targets
def columns_help(rtype: str):
    with st.expander('Columns of a request file'):
        rows = [{'column': c, 'shown as': trigger.LABELS[c], 'required': 'yes' if trigger.FIELD_BY_NAME[c].required else 'optional', 'note': HELP[c]}
                for c in EDITOR_COLUMNS[rtype]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Column names are matched ignoring case, spaces and underscores; common aliases (name, ra, dec, exposure, frames, ...) are recognised. '
                   'Unrecognised columns can be assigned after the upload or ignored. The observation mode is set on the page, not in the file.')


def _import(df: pd.DataFrame, mapping: dict, rtype: str):
    data = trigger.apply_mapping(df, mapping)
    _remount(_frame(data.to_dict('records'), EDITOR_COLUMNS[rtype]))       # unreadable dates stay as typed and are flagged by the validation


@st.dialog('7DS Tile matcher', width='large')
def tile_matcher_dialog(ts: trigger.TriggerSettings):
    """Tiles inside a circle or an ellipse around a position; tiles can be excluded or added by clicking them in the map."""
    ss = st.session_state
    c1, c2 = st.columns(2)
    ra_text = c1.text_input('RA (hh:mm:ss.s or deg)', key='tm_ra', placeholder='e.g. 10:00:00 or 150.0')
    dec_text = c2.text_input('Dec (±dd:mm:ss or deg)', key='tm_dec', placeholder='e.g. -30:00:00 or -30.0')
    shape = st.radio('Search area', ['Circle', 'Ellipse'], horizontal=True, key='tm_shape')
    if shape == 'Circle':
        radius = st.number_input('Radius [deg]', 0.0, 30.0, 1.0, 0.1, key='tm_radius')
        ellipse = None
    else:
        e1, e2, e3 = st.columns(3)
        a = e1.number_input('Semi-major axis [deg]', 0.01, 30.0, 2.0, 0.1, key='tm_a')
        b = e2.number_input('Semi-minor axis [deg]', 0.01, 30.0, 1.0, 0.1, key='tm_b')
        pa = e3.number_input('Position angle [deg, N through E]', 0.0, 180.0, 0.0, 1.0, key='tm_pa')
        radius, ellipse = 0.0, (a, b, pa)
    if st.button('Match tiles', key='tm_match', type='primary'):
        try:
            ra, dec = parse_coordinates(ra_text, dec_text)
            ss['tm_result'] = get_tiles().match(ra, dec, radius_deg=radius, ellipse=ellipse)
            ss['tm_excluded'], ss['tm_added'] = set(), set()
            ss['tm_plot_ver'] = ss.get('tm_plot_ver', 0) + 1
        except ValueError as exc:
            st.error(str(exc))
    m = ss.get('tm_result')
    if m is None:
        return
    excluded, added = ss.setdefault('tm_excluded', set()), ss.setdefault('tm_added', set())
    chosen = [t.id for t in m.tiles if t.id not in excluded] + sorted(added)
    summary = st.empty()
    st.caption('Click the circle on a tile to exclude it (grey) or to add a neighbouring tile (green); click again to undo. Drag to pan, scroll to zoom.')
    ver = ss.get('tm_plot_ver', 0)
    state = ss.get(f'tm_plot_{ver}')                               # the click of the previous run, applied before the chart is drawn
    points = (state or {}).get('selection', {}).get('points', []) if isinstance(state, dict) else []
    clicked = {p.get('customdata') for p in points if p.get('customdata')}
    if clicked:
        matched_ids = {t.id for t in m.tiles}
        for tid in clicked:
            if tid in matched_ids:
                excluded.symmetric_difference_update({tid})
            else:
                added.symmetric_difference_update({tid})
        ver = ss['tm_plot_ver'] = ver + 1                          # a fresh chart: the next click starts from no selection
        chosen = [t.id for t in m.tiles if t.id not in excluded] + sorted(added)
    fig = tiles.plot_tile_match_interactive(m, excluded=excluded, added=added, clickable=True)
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'scrollZoom': True}, on_select='rerun', selection_mode='points', key=f'tm_plot_{ver}')
    summary.markdown(f'**{len(chosen)} tile(s) selected** around RA {m.ra:.4f}, Dec {m.dec:+.4f}'
                     + (f', radius {m.radius_deg:g}°' if m.ellipse is None else f', ellipse {m.ellipse[0]:g}° x {m.ellipse[1]:g}°, PA {m.ellipse[2]:g}°')
                     + (f' ({len(excluded)} excluded)' if excluded else '') + (f' ({len(added)} added)' if added else ''))
    if chosen:
        rows = [t.row() for t in m.tiles if t.id in chosen] + [{'tile': tid, 'ra_center': float(get_tiles().ra[get_tiles()._index[tid]]), 'dec_center': float(get_tiles().dec[get_tiles()._index[tid]]), 'tile_fraction_in_circle': None} for tid in sorted(added)]
        table = pd.DataFrame(rows)[['tile', 'ra_center', 'dec_center', 'tile_fraction_in_circle']].rename(columns={'tile_fraction_in_circle': 'fraction inside the area'})
        st.dataframe(table.round(4), hide_index=True, width='stretch', height=min(35 * (len(rows) + 1) + 3, 260))
        if st.button(f'Insert {len(chosen)} tile(s) to the table', key='tm_insert', type='primary'):
            base = ss['tr_rows']
            existing = _rows_as_text(base)
            present = {r['objname'] for r in existing}
            new_rows = [{'objname': tid} for tid in chosen if tid not in present]
            _remount(_frame(existing + new_rows, list(base.columns)))
            for k in ('tm_result', 'tm_excluded', 'tm_added'):
                ss.pop(k, None)
            st.rerun()


def targets_section(ts: trigger.TriggerSettings, rtype: str) -> pd.DataFrame:
    st.header('Input target(s)' if rtype == 'Target' else 'Input tile(s)')
    ss = st.session_state
    columns = EDITOR_COLUMNS[rtype]
    if 'tr_rows' not in ss:
        _remount(_frame(EXAMPLE_ROWS[rtype], columns))
    elif list(ss['tr_rows'].columns) != columns:                      # request type changed
        current = _rows_as_text(ss['tr_rows'])
        other = [t for t in REQUEST_TYPES if t != rtype][0]
        if [r['objname'] for r in current] == [r['objname'] for r in EXAMPLE_ROWS[other]]:
            _remount(_frame(EXAMPLE_ROWS[rtype], columns))             # still the examples: show the examples of the new type
        else:
            _remount(_frame(current, columns))                         # user data: keep what fits
    left, right = st.columns([4, 1.4])
    with right:
        st.markdown('**Upload a file**')
        upload = st.file_uploader('Targets file (csv, txt, dat, tsv)', type=['csv', 'txt', 'dat', 'tsv', 'ascii'], key='tr_upload')
        marker = (upload.name, upload.size) if upload is not None else None
        if upload is not None and ss.get('tr_upload_marker') != marker:
            ss['tr_upload_marker'] = marker
            try:
                df = trigger.read_table(upload.getvalue(), upload.name)
                mapping, unknown = trigger.recognise_columns(df.columns)
                mapping = {c: f for c, f in mapping.items() if f in columns}
                ss['tr_upload_df'], ss['tr_mapping'], ss['tr_unknown'] = df, mapping, unknown
                if not unknown:
                    _import(df, mapping, rtype)
            except ValueError as exc:
                st.error(f'Could not read the file: {exc}')
        if ss.get('tr_unknown'):
            st.markdown('Assign the unrecognised columns:')
            labels = {trigger.IGNORE: trigger.IGNORE, **{c: f'{trigger.LABELS[c]} [{c}]' for c in columns}}
            choice = {c: st.selectbox(f'column "{c}"', list(labels), format_func=lambda k: labels[k], key=f'tr_map_{c}') for c in ss['tr_unknown']}
            if st.button('Import with these columns', key='tr_import', width='stretch'):
                mapping = dict(ss['tr_mapping'])
                for c, f in choice.items():
                    if f != trigger.IGNORE and f not in mapping.values():
                        mapping[c] = f
                _import(ss['tr_upload_df'], mapping, rtype)
                ss['tr_unknown'] = []
                st.rerun()
        st.download_button('Download a sample file', trigger.sample_file(rtype), file_name=f'7dt_request_sample_{rtype.lower()}.csv', mime='text/csv', width='stretch', key='tr_sample')
        columns_help(rtype)
    with left:
        base = ss['tr_rows']
        col_cfg = {}
        for c in base.columns:
            # Only the object name is marked required in the editor: the editor withholds a new row until its required
            # cells are filled, so marking the other columns would make such rows vanish when the table is redrawn.
            # Dates are plain text (YYYY-MM-DD): the calendar editor did not accept typed dates.
            col_cfg[c] = st.column_config.TextColumn(trigger.LABELS[c], help=HELP[c], required=(c == 'objname'))
        edited = st.data_editor(base, num_rows='dynamic', hide_index=False, width='stretch', column_config=col_cfg, key=f'tr_editor_{ss["tr_version"]}')
        b1, b2 = st.columns([1.4, 3], vertical_alignment='center')
        if rtype == 'Target':
            resolve = b1.button('Resolve names to RA/Dec', width='stretch', key='tr_resolve')
            b2.caption('Object name, RA/Dec, exposure time and frames are required. Hover a column header for its format. '
                       'Use the last (empty) row to add a target; select rows and press Delete to remove them.')
        else:
            resolve = False
            if b1.button('7DS Tile matcher', width='stretch', key='tr_tilematch'):
                tile_matcher_dialog(ts)
            b2.caption('Each row is one 7DS tile (object name = tile ID, e.g. T01022); the coordinates are the tile centre. '
                       'The Tile matcher finds the tiles inside a circle or an ellipse and inserts them here.')
    if len(edited) != len(base):
        _remount(_frame(edited.to_dict('records'), list(base.columns)))
        st.rerun()
    if resolve:
        df, failed = edited.copy(), []
        for i, row in df.iterrows():
            name = _text(row['objname'])
            if name:
                try:
                    ra, dec = resolve_name(name)
                    df.at[i, 'RA'], df.at[i, 'De'] = f'{ra:.5f}', f'{dec:+.5f}'
                except ValueError as exc:
                    failed.append(str(exc))
        ss['tr_resolve_failures'] = failed
        _remount(_frame(df.to_dict('records'), list(base.columns)))
        st.rerun()
    for msg in ss.pop('tr_resolve_failures', []):
        st.error(msg)
    return edited


# ------------------------------------------------------------------ validation, helper plots, preview, submit
def validate(edited: pd.DataFrame, ts: trigger.TriggerSettings, defaults: dict, cfg: dict, project: str, survey: bool):
    rows = _rows_as_text(edited)
    if not rows:
        return [], [], []
    norm, issues = trigger.validate_rows(rows, ts, defaults, cfg, survey=survey)
    return norm, trigger.format_rows(norm, project, ts), issues


@st.cache_data(show_spinner=False, max_entries=16)
def visibility_table_cached(key: tuple, cons: dict, night, targets: tuple) -> pd.DataFrame:
    site = config.Site(*key)
    tg = [Target(name=n, ra=ra, dec=dec) for n, ra, dec in targets]
    results = vis.compute_all(tg, Time(night_midpoint_utc(key, night)), site, vis.Constraints(**cons))
    return pd.DataFrame([vis.night_summary(r) for r in results])


@st.cache_data(show_spinner=False, max_entries=16)
def visibility_plots_cached(key: tuple, cons: dict, night, targets: tuple):
    site = config.Site(*key)
    constraints = vis.Constraints(**cons)
    tg = [Target(name=n, ra=ra, dec=dec) for n, ra, dec in targets]
    results = vis.compute_all(tg, Time(night_midpoint_utc(key, night)), site, constraints)
    m = vis.monthly_visibility(tg, night, site, constraints, grid=vis.month_grid(night, site, constraints))
    return vis.plot_visibility_interactive(results, site, mark_time=False), vis.plot_monthly_interactive(m, site, show_max_alt=False)


def helper_section(norm: list, om, cfg: dict, hours: float | None):
    st.header('Helper plots')
    targets = tuple((r['objname'] or f'row {i}', float(r['RA_deg']), float(r['De_deg'])) for i, r in enumerate(norm, start=1) if r.get('RA_deg') is not None)
    site, cons = cfg['site'], cfg['constraints']
    cons_d = {'min_alt': cons.min_alt, 'max_alt': cons.max_alt, 'moon_sep': cons.moon_sep, 'sun_alt': cons.sun_alt}
    c_vis, c_time = st.columns([2, 1])
    with c_vis, st.expander('Visibility', expanded=True):
        night = night_date_input('tr', site)
        if not targets:
            st.caption('Needs targets with valid coordinates.')
        else:
            with st.spinner(f'Computing the night for {len(targets)} target(s)...'):
                df = visibility_table_cached(site_key(site), cons_d, night, targets)
            st.dataframe(styled_rows(df[['target', 'status', 'observable_hours', 'windows', 'max_alt_deg', 'note']], 'status'), hide_index=True, width='stretch',
                         height=min(35 * (len(df) + 1) + 3, 320))
            names = [t[0] for t in targets]
            chosen = st.multiselect('Targets in the plots', names, default=names[:PLOT_LIMIT], key='tr_plot_targets',
                                    help=f'The table covers every target; the plots draw the selected ones (first {PLOT_LIMIT} by default).')
            subset = tuple(t for t in targets if t[0] in set(chosen))
            if subset:
                with st.spinner('Drawing...'):
                    fig_night, fig_month = visibility_plots_cached(site_key(site), cons_d, night, subset)
                st.plotly_chart(fig_night, width='stretch', config={'displaylogo': False}, key='tr_vis_plot')
                st.plotly_chart(fig_month, width='stretch', config={'displaylogo': False}, key='tr_month_plot')
    with c_time, st.expander('Total observation time', expanded=True):
        if om is None:
            st.caption('Choose an observation mode first.')
        else:
            tt = trigger.total_time(norm, om)
            st.metric('Targets with exposure and frames', tt['n_targets'])
            st.metric('Total exposure (per unit)', overhead.format_duration(tt['exposure_s']), help='exposure time x frames x filters per unit, summed over the targets')
            st.metric('Total observation time', overhead.format_duration(tt['total_s']), help='exposure plus every overhead below')
            if tt['lines']:
                st.markdown('**Overhead details** (' + overhead.format_duration(tt['overhead_s']) + ' in total)')
                st.markdown('\n'.join(f'- {line}' for line in tt['lines']))
            if hours is not None:
                st.metric('Allocated to the project', f'{hours:g} h', delta=f'{(hours * 3600 - tt["total_s"]) / 3600:+.1f} h left', delta_color='normal')
                if tt['total_s'] > hours * 3600:
                    st.error(f'The request needs {tt["total_s"] / 3600:.1f} h but the project has {hours:g} h allocated.')
            else:
                st.caption('No allocation known for this project.')
            return hours is not None and tt['total_s'] > hours * 3600
    return False


def preview_section(formatted: list, issues: list):
    st.header('Request preview')
    if not formatted:
        st.info('Add at least one target.')
        return
    df = pd.DataFrame(formatted, columns=list(trigger.SHEET_COLUMNS))
    df.index = pd.RangeIndex(1, len(df) + 1, name='Row')
    css = pd.DataFrame('', index=df.index, columns=df.columns)
    for it in issues:
        if it.field in css.columns:
            css.at[it.row, it.field] = RED if it.level == 'error' else YELLOW
    st.dataframe(df.style.apply(lambda _: css, axis=None), width='stretch', height=min(35 * (len(df) + 1) + 3, 420))
    errors = [i for i in issues if i.level == 'error']
    warnings = [i for i in issues if i.level == 'warning']
    for it in errors[:30]:
        st.error(str(it))
    if len(errors) > 30:
        st.error(f'... and {len(errors) - 30} more error(s).')
    for it in warnings[:10]:
        st.warning(str(it))
    if len(warnings) > 10:
        st.warning(f'... and {len(warnings) - 10} more warning(s).')
    if not errors:
        st.success(f'{len(formatted)} target(s) ready.' + (f' {len(warnings)} warning(s).' if warnings else ''))


def submit_section(formatted: list, errors: list, ts: trigger.TriggerSettings, project: str, project_ok: bool, over_allocation: bool):
    ss = st.session_state
    key_path = sheets.service_account_key_path()
    name = ts.worksheet_name(project) if trigger.safe_project_id(project) else ''
    replace = bool(ss.get('tr_replace')) and ss.get('tr_loaded_for') == trigger.safe_project_id(project)
    c1, c2 = st.columns(2)
    csv = pd.DataFrame(formatted, columns=list(trigger.SHEET_COLUMNS)).to_csv(index=False).encode('utf-8') if formatted else b''
    c2.download_button('Download the request (CSV)', csv, file_name=f'{name or "request"}.csv', mime='text/csv', width='stretch', key='tr_csv', disabled=not formatted)
    if key_path is None:
        st.warning('Spreadsheet access is not configured on this server (no Google service-account key). Download the CSV and send it to the observatory.')
    disabled = bool(errors) or not name or not formatted or key_path is None or not project_ok or over_allocation
    label = f'{"Replace the worksheet with" if replace else "Submit"} {len(formatted)} target(s)'
    if c1.button(label, width='stretch', key='tr_submit', type='primary', disabled=disabled):
        try:
            with st.spinner('Writing to the spreadsheet...'):
                fn = sheets.replace_rows if replace else sheets.append_rows
                result = fn(ts.spreadsheet_url, name, list(trigger.SHEET_COLUMNS), formatted, key_path)
            ss['tr_last'] = result
            ss['tr_replace'] = True                     # the worksheet now holds exactly this table
        except Exception as exc:
            st.error(f'The spreadsheet refused the write: {exc}')
            st.caption(f'Make sure the spreadsheet is shared with {sheets.service_account_email(key_path)} (editor).')
    last = ss.get('tr_last')
    if last:
        verb = 'written to' if last.get('replaced') else 'appended to'
        st.success(f'{last["n_rows"]} row(s) {verb} worksheet {last["worksheet"]}' + (' (new worksheet)' if last['created'] else '') + f'. [Open the spreadsheet]({last["url"]})')


def main():
    feedback_note()
    st.title('7DT Observation request')
    cfg = live_config()
    ts = trigger.TriggerSettings.load()
    ss = st.session_state
    project, ok, hours = project_section(ts)
    load_from_sheet(ts, project, ok)
    if ss.get('tr_loaded_note'):
        c1, c2 = st.columns([3, 1], vertical_alignment='center')
        c1.info(ss['tr_loaded_note'])
        if c2.button('Reload from the spreadsheet', key='tr_reload', width='stretch'):
            ss.pop('tr_loaded_for', None)
            ss.pop('tr_loaded_note', None)
            st.rerun()
    if 'tr_type_pending' in ss:
        ss['tr_type'] = ss.pop('tr_type_pending')
    rtype = st.radio('Request type', list(REQUEST_TYPES), horizontal=True, key='tr_type',
                     help='Target: your own positions. Survey: 7DS tiles (object name = tile ID), coordinates fixed to the tile centres.')
    defaults, om = settings_section(ts, cfg)
    edited = targets_section(ts, rtype)
    norm, formatted, issues = validate(edited, ts, defaults, cfg, project, survey=(rtype == 'Survey'))
    over = helper_section(norm, om, cfg, hours)
    preview_section(formatted, issues)
    submit_section(formatted, [i for i in issues if i.level == 'error'], ts, project, ok, over)


if __name__ == '__main__':
    main()
