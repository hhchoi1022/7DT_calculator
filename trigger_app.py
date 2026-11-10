"""
7DT Observation request: targets entered in a fixed format (table or uploaded file), validated against the
request rules and the current instrument, and appended to the request spreadsheet as a worksheet per project.
Run with:  streamlit run trigger_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from astropy.time import Time

from sevendt_calc import config, etc, overhead, sheets, tiles, trigger
from sevendt_calc import templates as tpl
from sevendt_calc import visibility as vis
from sevendt_calc.targets import Target, resolve_name
from ui_common import FEEDBACK_EMAIL, FEEDBACK_NAME, feedback_note, figure_png, live_config, night_date_input, night_midpoint_utc, obsmode_png, site_key, styled_rows

st.set_page_config(page_title='7DT Observation request', page_icon='📝', layout='wide')

EDITOR_BASE = ['objname', 'RA', 'De', 'exptime', 'count', 'obs_startdate', 'obs_enddate']
EXAMPLE_ROWS = [{'objname': 'NGC0253', 'RA': '00:47:33.12', 'De': '-25:17:17.6', 'exptime': '100', 'count': '3'},
                {'objname': 'T01022', 'RA': '78.26087', 'De': '-71.32075', 'exptime': '100', 'count': '3'}]
RED, YELLOW = 'background-color: #ffb3b3', 'background-color: #fff3b0'


# ------------------------------------------------------------------ state helpers
DATE_COLUMNS = ('obs_startdate', 'obs_enddate')


def _frame(rows, columns) -> pd.DataFrame:
    """Editor frame: text columns as strings, the two date columns as datetimes (calendar picker in the editor)."""
    df = pd.DataFrame(rows).reindex(columns=columns)
    for c in columns:
        if c in DATE_COLUMNS:
            parsed = []
            for v in df[c]:
                if v is None or (not isinstance(v, str) and pd.isna(v)):
                    parsed.append(None)
                elif hasattr(v, 'year'):
                    parsed.append(pd.Timestamp(v).date())
                else:
                    d, _ = trigger.parse_date(v)
                    parsed.append(d.date() if d else None)
            df[c] = pd.Series(parsed, index=df.index, dtype=object)
        else:
            df[c] = df[c].map(_text)
    df.index = pd.RangeIndex(1, len(df) + 1, name='Index')
    return df


def _date_text(v) -> str:
    """A calendar date from the editor -> the spreadsheet format with the time 00:00:00."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ''
    if hasattr(v, 'strftime'):
        return pd.Timestamp(v).strftime('%Y-%m-%dT00:00:00')
    return str(v).strip()


def _remount(df: pd.DataFrame):
    st.session_state['tr_rows'] = df
    st.session_state['tr_version'] = st.session_state.get('tr_version', 0) + 1


def _text(v) -> str:
    return '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


@st.cache_resource(show_spinner=False)
def get_tiles() -> tiles.TileSet:
    return tiles.TileSet()


@st.cache_resource(show_spinner=False)
def get_calculator() -> etc.ExposureCalculator:
    return etc.ExposureCalculator(etc.DepthModel())


def typical_seeing() -> float:
    model = etc.DepthModel().model
    values = [m['medians'][3] for m in model.values() if len(m.get('medians', [])) > 3]
    return float(sorted(values)[len(values) // 2]) if values else 2.5


# ------------------------------------------------------------------ sections
def project_section(ts: trigger.TriggerSettings) -> str:
    st.header('Project')
    c1, c2 = st.columns([1, 2], vertical_alignment='center')
    project = c1.text_input('Project ID', key='tr_project', placeholder='e.g. GRB_followup',
                            help='Letters, digits, _ and -. The rows go to the worksheet <period>_<Project ID> of the request spreadsheet.')
    if trigger.safe_project_id(project):
        c2.markdown(f'Worksheet: **{ts.worksheet_name(project)}**')
    else:
        c2.caption('The Project ID is required before submitting.')
    return project


def settings_section(ts: trigger.TriggerSettings, cfg: dict):
    """Observation mode, sub-mode, per-order exposure time and count, binning, gain: the defaults for every target."""
    st.header('Observation settings')
    c1, c2 = st.columns([1, 2])
    obsmode = c1.selectbox('Observation mode', list(ts.obsmodes), key='tr_obsmode',
                           help='Spec and Color use every active telescope and start only when all of them are idle.')
    sub_field, submode = None, ''
    if obsmode == 'Spec':
        sub_field = 'specmode'
        submode = c2.selectbox('Specmode', ts.spec_options(cfg['specmodes']), key='tr_specmode')
    elif obsmode == 'Color':
        sub_field = 'colormode'
        submode = c2.selectbox('Colormode', list(ts.colormodes), key='tr_colormode')
    elif obsmode == 'Deep':
        sub_field = 'filter_'
        submode = c2.selectbox('Deepmode', list(ts.deepmodes), key='tr_deepmode')
    else:
        sub_field = 'filter_'
        submode = c2.selectbox('Filter (optional)', [''] + config.installed_filters(cfg['filtinfo']), key='tr_filter')
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


def columns_help():
    with st.expander('Columns of a request file'):
        rows = [{'column': f.name, 'shown as': f.label, 'required': 'yes' if f.required else 'optional', 'note': f.help}
                for f in trigger.FIELDS if f.name in trigger.FILE_COLUMNS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Column names are matched ignoring case, spaces and underscores; common aliases (name, ra, dec, exposure, frames, ...) are recognised. '
                   'Unrecognised columns can be assigned after the upload or ignored. The observation mode is set on the page, not in the file.')


def targets_section(ts: trigger.TriggerSettings) -> pd.DataFrame:
    st.header('Input target(s)')
    ss = st.session_state
    if 'tr_rows' not in ss:
        _remount(_frame(EXAMPLE_ROWS, EDITOR_BASE))
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
                ss['tr_upload_df'], ss['tr_mapping'], ss['tr_unknown'] = df, mapping, unknown
                if not unknown:                          # everything recognised: import at once
                    _import(df, mapping)
            except ValueError as exc:
                st.error(f'Could not read the file: {exc}')
        if ss.get('tr_unknown'):
            st.markdown('Assign the unrecognised columns:')
            labels = {trigger.IGNORE: trigger.IGNORE, **{f.name: f'{f.label} [{f.name}]' for f in trigger.FIELDS if f.name in trigger.FILE_COLUMNS}}
            choice = {}
            for c in ss['tr_unknown']:
                choice[c] = st.selectbox(f'column "{c}"', list(labels), format_func=lambda k: labels[k], key=f'tr_map_{c}')
            if st.button('Import with these columns', key='tr_import', width='stretch'):
                mapping = dict(ss['tr_mapping'])
                for c, f in choice.items():
                    if f != trigger.IGNORE and f not in mapping.values():
                        mapping[c] = f
                _import(ss['tr_upload_df'], mapping)
                ss['tr_unknown'] = []
                st.rerun()
        st.download_button('Download a sample file', trigger.sample_file(), file_name='7dt_request_sample.csv', mime='text/csv', width='stretch', key='tr_sample')
        columns_help()
    with left:
        base = ss['tr_rows']
        col_cfg = {}
        for c in base.columns:
            f = trigger.FIELD_BY_NAME.get(c)
            if c in DATE_COLUMNS:
                col_cfg[c] = st.column_config.DateColumn(trigger.LABELS.get(c, c), help='Pick a day in the calendar (UTC, 00:00:00).', format='YYYY-MM-DD')
            else:
                # Only the object name is marked as required in the editor ("Please fill out this cell"): the editor
                # withholds a new row until its required cells are filled, so marking RA/Dec, exposure time or frames
                # would make such rows vanish when the table is redrawn (e.g. by "Resolve names"). The validation
                # below still demands every required column and marks it in the preview.
                col_cfg[c] = st.column_config.TextColumn(trigger.LABELS.get(c, c), help=f.help if f else None, required=(c == 'objname'))
        edited = st.data_editor(base, num_rows='dynamic', hide_index=False, width='stretch', column_config=col_cfg, key=f'tr_editor_{ss["tr_version"]}')
        b1, b2 = st.columns([1.4, 3], vertical_alignment='center')
        resolve = b1.button('Resolve names to RA/Dec', width='stretch', key='tr_resolve')
        b2.caption('Object name, RA/Dec, exposure time and frames are required. RA/Dec in degrees or hh:mm:ss / ±dd:mm:ss. '
                   f'Exposure time and frames: one number each, applied to every filter of the mode (standard {int(ts.default_exptime)} s and {ts.default_count} frames). '
                   'Dates are optional: pick a day in the calendar (UTC, stored as 00:00:00). Use the last (empty) row to add a target; select rows and press Delete to remove them.')
    if len(edited) != len(base):                      # rows added or removed: renumber and re-mount
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
    for msg in ss.get('tr_import_notes', []):
        st.error(msg)
    return edited


def _import(df: pd.DataFrame, mapping: dict):
    data = trigger.apply_mapping(df, mapping)
    bad = []
    for c in DATE_COLUMNS:
        if c in data.columns:
            for i, v in enumerate(data[c], start=1):
                if not trigger._blank(v) and trigger.parse_date(v)[1]:
                    bad.append(f'row {i}: {trigger.LABELS[c]} "{v}" was not understood (use {trigger.DATE_EXAMPLE}); left empty')
    st.session_state['tr_import_notes'] = bad
    _remount(_frame(data.to_dict('records'), EDITOR_BASE))          # only the file columns; the mode comes from the settings


def validate(edited: pd.DataFrame, ts: trigger.TriggerSettings, defaults: dict, cfg: dict, project: str):
    """Rows of the editor -> (normalised rows, spreadsheet rows, issues). Fully empty rows are ignored."""
    rows = [{c: (_date_text(v) if c in DATE_COLUMNS else _text(v)) for c, v in r.items()} for r in edited.to_dict('records')]
    rows = [r for r in rows if any(r.get(c) for c in ('objname', 'RA', 'De'))]
    if not rows:
        return [], [], []
    norm, issues = trigger.validate_rows(rows, ts, defaults, cfg)
    return norm, trigger.format_rows(norm, project, ts), issues


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
    st.dataframe(df.style.apply(lambda _: css, axis=None), width='stretch')
    errors = [i for i in issues if i.level == 'error']
    warnings = [i for i in issues if i.level == 'warning']
    for it in errors:
        st.error(str(it))
    for it in warnings:
        st.warning(str(it))
    if not errors:
        st.success(f'{len(formatted)} target(s) ready.' + (f' {len(warnings)} warning(s).' if warnings else ''))


def submit_section(formatted: list, errors: list, ts: trigger.TriggerSettings, project: str):
    key_path = sheets.service_account_key_path()
    name = ts.worksheet_name(project) if trigger.safe_project_id(project) else ''
    c1, c2 = st.columns([1, 1])
    csv = pd.DataFrame(formatted, columns=list(trigger.SHEET_COLUMNS)).to_csv(index=False).encode('utf-8') if formatted else b''
    c2.download_button('Download the request (CSV)', csv, file_name=f'{name or "request"}.csv', mime='text/csv', width='stretch',
                       key='tr_csv', disabled=not formatted)
    if key_path is None:
        st.warning('Spreadsheet access is not configured on this server (no Google service-account key). Download the CSV and send it to the observatory.')
    disabled = bool(errors) or not name or not formatted or key_path is None
    if c1.button(f'Submit {len(formatted)} target(s) to the spreadsheet', width='stretch', key='tr_submit', type='primary', disabled=disabled):
        try:
            with st.spinner('Writing to the spreadsheet...'):
                result = sheets.append_rows(ts.spreadsheet_url, name, list(trigger.SHEET_COLUMNS), formatted, key_path)
            st.session_state['tr_last'] = result
        except Exception as exc:
            st.error(f'The spreadsheet refused the write: {exc}')
            if key_path is not None:
                st.caption(f'Make sure the spreadsheet is shared with {sheets.service_account_email(key_path)} (editor).')
    last = st.session_state.get('tr_last')
    if last:
        st.success(f'{last["n_rows"]} row(s) appended to worksheet {last["worksheet"]}' + (' (new worksheet)' if last['created'] else '') + f'. [Open the spreadsheet]({last["url"]})')


@st.cache_data(show_spinner=False, max_entries=32)
def visibility_cached(key: tuple, cons: dict, night, targets: tuple):
    site = config.Site(*key)
    constraints = vis.Constraints(**cons)
    tg = [Target(name=n, ra=ra, dec=dec) for n, ra, dec in targets]
    results = vis.compute_all(tg, Time(night_midpoint_utc(key, night)), site, constraints)
    return pd.DataFrame([vis.night_summary(r) for r in results]), vis.plot_visibility_interactive(results, site, mark_time=False)


@st.cache_data(show_spinner=False, max_entries=16)
def month_grid_cached(key: tuple, cons: dict, start_night):
    return vis.month_grid(start_night, config.Site(*key), vis.Constraints(**cons))


@st.cache_data(show_spinner=False, max_entries=32)
def monthly_cached(key: tuple, cons: dict, start_night, targets: tuple):
    site = config.Site(*key)
    tg = [Target(name=n, ra=ra, dec=dec) for n, ra, dec in targets]
    m = vis.monthly_visibility(tg, start_night, site, vis.Constraints(**cons), grid=month_grid_cached(key, cons, start_night))
    return vis.plot_monthly_interactive(m, site, show_max_alt=False)


@st.cache_data(show_spinner=False, max_entries=32)
def depth_cached(mag: float, band: str, spectrum: str, mode: str, submode: str, table: tuple, exptime: float, count: int, seeing: float):
    om = config.ObsMode(mode=mode, submode=submode, filters_by_unit={u: list(f) for u, f in table})
    target = Target(name='target', mag=float(mag), mag_filter=band, spectrum_type=spectrum)
    cond = etc.conditions_for(seeing=seeing, moon_phase=0.5, moon_separation=90.0, hour_since_sunset=5.0)
    results = get_calculator().snr(target, om.filters, exptime, count, cond, units=om.filter_multiplicity())
    df = pd.DataFrame([r.row() for r in results])
    return df, etc.plot_snr_vs_exptime_interactive(target, results, cond), etc.plot_spectrum_points_interactive(target, results, noise_seed=1, title='Expected 7DT photometry')


@st.cache_data(show_spinner=False, max_entries=32)
def overhead_cached(mode: str, submode: str, table: tuple, exptime: float, count: int):
    import matplotlib.pyplot as plt
    om = config.ObsMode(mode=mode, submode=submode, filters_by_unit={u: list(f) for u, f in table})
    cfg_oh = overhead.OverheadConfig.load()
    req = overhead.ObsRequest('request', exptime, count, om, 'per_filter', config=cfg_oh)
    result = overhead.estimate([req], cfg_oh)
    fig = overhead.plot_timeline(result)
    png = figure_png(fig, dpi=150)
    plt.close(fig)
    return {'total': result.total, 'science': result.science_time, 'overhead': result.overhead, 'efficiency': result.efficiency}, result.summary_lines(), req.autofocus_policy, png


@st.cache_data(show_spinner=False, max_entries=32)
def tiles_cached(targets: tuple):
    ts_ = get_tiles()
    matches = [ts_.match(ra, dec, radius_deg=0.0, target=Target(name=n, ra=ra, dec=dec)) for n, ra, dec in targets]
    return pd.DataFrame([m.summary() for m in matches]), [tiles.plot_tile_match_interactive(m) for m in matches]


def tools_section(norm: list, om, cfg: dict):
    """Visibility, depth, overhead and tile checks with their figures: four collapsible panels side by side, computed as the inputs change."""
    st.header('Checks')
    targets = tuple((r['objname'] or f'row {i}', float(r['RA_deg']), float(r['De_deg'])) for i, r in enumerate(norm, start=1) if r.get('RA_deg') is not None)
    first = next((r for r in norm if r.get('exptime_s') is not None and r.get('count_n') is not None), None)
    exptime = float(first['exptime_s']) if first else 100.0
    count = int(first['count_n']) if first else 3
    site, cons = cfg['site'], cfg['constraints']
    cons_d = {'min_alt': cons.min_alt, 'max_alt': cons.max_alt, 'moon_sep': cons.moon_sep, 'sun_alt': cons.sun_alt}
    table = tuple((u, tuple(f)) for u, f in om.filters_by_unit.items()) if om is not None else ()
    c_vis, c_depth, c_oh, c_tile = st.columns(4)

    with c_vis, st.expander('Visibility'):
        night = night_date_input('tr', site)
        if not targets:
            st.caption('Needs targets with valid coordinates.')
        else:
            with st.spinner('Computing...'):
                df, fig = visibility_cached(site_key(site), cons_d, night, targets)
            st.dataframe(styled_rows(df[['target', 'status', 'observable_hours', 'windows', 'note']], 'status'), hide_index=True, width='stretch')
            st.plotly_chart(fig, width='stretch', config={'displaylogo': False}, key='tr_vis_plot')
            with st.spinner('Computing the month...'):
                st.plotly_chart(monthly_cached(site_key(site), cons_d, night, targets), width='stretch', config={'displaylogo': False}, key='tr_month_plot')

    with c_depth, st.expander('Depth (SNR)'):
        if om is None:
            st.caption('Choose an observation mode with filters first.')
        else:
            mag = st.number_input('Magnitude', -2.0, 30.0, 19.0, 0.1, key='tr_mag', help='AB for g r i, Vega for B V R I')
            band = st.selectbox('Filter of the magnitude', ['g', 'r', 'i', 'B', 'V', 'R', 'I'], index=1, key='tr_band')
            keys = [k for k in tpl.template_keys() if k not in (tpl.UPLOAD_KEY, tpl.FLAT_KEY)]
            spectrum = st.selectbox('Spectrum', keys, index=keys.index('Star: G2V') if 'Star: G2V' in keys else 0, key='tr_spec')
            with st.spinner('Computing...'):
                df, fig_snr, fig_phot = depth_cached(float(mag), band, spectrum, om.mode, om.submode, table, exptime, count, typical_seeing())
            st.caption(f'50 % Moon at 90°, 5 h after sunset, typical seeing; {exptime:g} s x {count} frames per filter (first target).')
            st.dataframe(df[['filter', 'mag_AB', 'UL5_stacked', 'SNR_stacked']].round(2), hide_index=True, width='stretch')
            st.plotly_chart(fig_snr, width='stretch', config={'displaylogo': False}, key='tr_snr_plot')
            st.plotly_chart(fig_phot, width='stretch', config={'displaylogo': False}, key='tr_phot_plot')

    with c_oh, st.expander('Overhead'):
        if om is None:
            st.caption('Choose an observation mode with filters first.')
        else:
            metrics, lines, policy, png = overhead_cached(om.mode, om.submode, table, exptime, count)
            st.metric('Total observation time', overhead.format_duration(metrics['total']))
            st.metric('Open-shutter time', overhead.format_duration(metrics['science']))
            st.metric('Overhead', overhead.format_duration(metrics['overhead']), help=f'efficiency {100 * metrics["efficiency"]:.0f} %')
            st.markdown('\n'.join(f'- {line}' for line in lines))
            st.image(png, width='stretch')
            st.caption('Per target, first target\'s exposure and frames; autofocus ' + ('once per filter.' if policy == 'per_filter' else 'none.'))

    with c_tile, st.expander('7DS tiles'):
        if not targets:
            st.caption('Needs targets with valid coordinates.')
        else:
            df, figs = tiles_cached(targets)
            st.dataframe(styled_rows(df[['target', 'status', 'primary_tile', 'warnings']], 'status'), hide_index=True, width='stretch')
            for i, (fig, name) in enumerate(zip(figs, df['target'])):
                if st.toggle(f'Map: {name}', key=f'tr_tile_map_{i}'):
                    st.plotly_chart(fig, width='stretch', config={'displaylogo': False}, key=f'tr_tile_plot_{i}')


def main():
    feedback_note()
    st.title('7DT Observation request')
    cfg = live_config()
    ts = trigger.TriggerSettings.load()
    project = project_section(ts)
    defaults, om = settings_section(ts, cfg)
    edited = targets_section(ts)
    norm, formatted, issues = validate(edited, ts, defaults, cfg, project)
    tools_section(norm, om, cfg)
    preview_section(formatted, issues)
    submit_section(formatted, [i for i in issues if i.level == 'error'], ts, project)


if __name__ == '__main__':
    main()
