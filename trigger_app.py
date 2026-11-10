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
from ui_common import feedback_note, live_config, night_date_input, night_midpoint_utc, obsmode_png, site_key, styled_rows

st.set_page_config(page_title='7DT Observation request', page_icon='📝', layout='wide')

EDITOR_BASE = ['objname', 'RA', 'De', 'obs_startdate', 'obs_enddate']
EXAMPLE_ROWS = [{'objname': 'NGC0253', 'RA': '00:47:33.12', 'De': '-25:17:17.6'}, {'objname': 'T01022', 'RA': '78.26087', 'De': '-71.32075'}]
RED, YELLOW = 'background-color: #ffb3b3', 'background-color: #fff3b0'


# ------------------------------------------------------------------ state helpers
def _frame(rows, columns) -> pd.DataFrame:
    df = pd.DataFrame(rows, dtype=str).reindex(columns=columns, fill_value='').fillna('')
    df.index = pd.RangeIndex(1, len(df) + 1, name='Index')
    return df


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
    n = max(om.n_filters_per_unit, 1) if om is not None else 1

    st.markdown('**Exposure time and frames per order** (an order is one filter position of the sequence; the values are stored comma-separated)')
    cols = st.columns(max(n, 1) * 2)
    exptimes, counts = [], []
    for k in range(n):
        label = f'order {k + 1}' if n > 1 else ''
        exptimes.append(cols[2 * k].number_input(f'Exposure time [s] {label}'.strip(), 1.0, 10000.0, float(ts.default_exptime), 10.0, key=f'tr_exp_{k}'))
        counts.append(int(cols[2 * k + 1].number_input(f'Frames {label}'.strip(), 1, 999, int(ts.default_count), key=f'tr_cnt_{k}')))
    if any(e > ts.max_exptime for e in exptimes):
        st.error(f'Exposure times above {ts.max_exptime:g} s are not accepted.')
    elif any(e != ts.default_exptime for e in exptimes):
        st.warning(f'The standard exposure time is {ts.default_exptime:g} s (the depth model was fitted on {ts.default_exptime:g} s frames); other values are accepted with this warning.')

    c1, c2, c3 = st.columns(3)
    binning = int(c1.number_input('Binning', 1, 8, int(ts.default_binning), key='tr_binning'))
    gain = int(c2.number_input('Gain', 0, 10000, int(ts.default_gain), key='tr_gain'))
    c3.number_input('Priority', value=int(ts.priority), disabled=True, key='tr_priority', help='Set by the observatory.')
    if ts.gain_table:
        with st.expander('Gain settings of the camera'):
            st.dataframe(pd.DataFrame(ts.gain_table).rename(columns={'gain': 'Gain number', 'gain_db': 'Gain in dB', 'multiply': 'Gain multiply', 'conversion': 'Conversion factor',
                                                                     'read_noise': 'Read noise RMS', 'full_well': 'Full well capacity'}), hide_index=True, width='stretch')
    d1, d2 = st.columns(2)
    start = d1.text_input('Observation start (UTC), optional', key='tr_start', placeholder=trigger.DATE_EXAMPLE, help='Applied to targets without their own start date.')
    end = d2.text_input('Observation end (UTC), optional', key='tr_end', placeholder='2026-12-01T00:00:00', help='Applied to targets without their own end date.')
    defaults = {'obsmode': obsmode, sub_field: submode, 'exptime': ','.join(f'{e:g}' for e in exptimes), 'count': ','.join(str(c) for c in counts),
                'binning': str(binning), 'gain': str(gain), 'obs_startdate': start, 'obs_enddate': end}
    return defaults, om, exptimes, counts


def columns_help():
    with st.expander('Columns of a request file'):
        rows = [{'column in the file / spreadsheet': f.name, 'shown as': f.label, 'required': 'yes' if f.required else 'optional (page setting when blank)',
                 'note': f.help} for f in trigger.FIELDS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Column names are matched ignoring case, spaces and underscores; common aliases (name, ra, dec, filter, exposure, ...) are recognised. '
                   'Unrecognised columns can be assigned after the upload. Exposure time and frames may hold one value per order, comma-separated.')


def targets_section() -> pd.DataFrame:
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
            labels = {trigger.IGNORE: trigger.IGNORE, **{f.name: f'{f.label} [{f.name}]' for f in trigger.FIELDS}}
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
        col_cfg = {c: st.column_config.TextColumn(trigger.LABELS.get(c, c), help=trigger.FIELD_BY_NAME[c].help if c in trigger.FIELD_BY_NAME else None)
                   for c in base.columns}
        edited = st.data_editor(base, num_rows='dynamic', hide_index=False, width='stretch', column_config=col_cfg,
                                key=f'tr_editor_{ss["tr_version"]}')
        b1, b2 = st.columns([1.4, 3], vertical_alignment='center')
        resolve = b1.button('Resolve names to RA/Dec', width='stretch', key='tr_resolve')
        b2.caption('RA/Dec in degrees or hh:mm:ss / ±dd:mm:ss. Use the last (empty) row to add a target; select rows and press Delete to remove them. '
                   'Dates: UTC, ' + trigger.DATE_EXAMPLE + '. Blank cells take the observation settings above.')
    if len(edited) != len(base):
        _remount(_frame(edited.to_dict('records'), list(base.columns)))
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


def _import(df: pd.DataFrame, mapping: dict):
    data = trigger.apply_mapping(df, mapping)
    extra = [f.name for f in trigger.FIELDS if f.name in data.columns and f.name not in EDITOR_BASE]
    _remount(_frame(data.to_dict('records'), EDITOR_BASE + extra))


def preview_section(edited: pd.DataFrame, ts: trigger.TriggerSettings, defaults: dict, cfg: dict, project: str):
    st.header('Request preview')
    rows = [{c: _text(v) for c, v in r.items()} for r in edited.to_dict('records')]
    rows = [r for r in rows if any(r.get(c) for c in ('objname', 'RA', 'De'))]          # ignore fully empty rows
    if not rows:
        st.info('Add at least one target.')
        return [], [], []
    norm, issues = trigger.validate_rows(rows, ts, defaults, cfg)
    formatted = trigger.format_rows(norm, project, ts)
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
    return norm, formatted, errors


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


def tools_section(norm: list, om, exptimes: list, counts: list, cfg: dict):
    st.header('Checks')
    targets = [Target(name=r['objname'] or f'row {i}', ra=r['RA_deg'], dec=r['De_deg']) for i, r in enumerate(norm, start=1) if r.get('RA_deg') is not None]
    site, cons = cfg['site'], cfg['constraints']
    ss = st.session_state

    with st.expander('Visibility'):
        night = night_date_input('tr', site)
        if st.button('Calculate visibility', key='tr_vis_btn') and targets:
            obstime = Time(night_midpoint_utc(site_key(site), night))
            with st.spinner('Computing the night...'):
                results = vis.compute_all(targets, obstime, site, cons)
            ss['tr_vis'] = (night, results)
        if ss.get('tr_vis') and ss['tr_vis'][0] == night:
            results = ss['tr_vis'][1]
            st.dataframe(styled_rows(pd.DataFrame([vis.night_summary(r) for r in results]), 'status'), hide_index=True, width='stretch')
            st.plotly_chart(vis.plot_visibility_interactive(results, site, mark_time=False), width='stretch', config={'displaylogo': False}, key='tr_vis_plot')
        elif not targets:
            st.caption('Needs targets with valid coordinates.')

    with st.expander('Depth (SNR with these settings)'):
        if om is None:
            st.caption('Choose an observation mode with filters first.')
        else:
            c1, c2, c3 = st.columns(3)
            mag = c1.number_input('Magnitude (AB for g r i, Vega for B V R I)', -2.0, 30.0, 19.0, 0.1, key='tr_mag')
            band = c2.selectbox('Filter of the magnitude', ['g', 'r', 'i', 'B', 'V', 'R', 'I'], index=1, key='tr_band')
            keys = [k for k in tpl.template_keys() if k not in (tpl.UPLOAD_KEY, tpl.FLAT_KEY)]
            spectrum = c3.selectbox('Spectrum', keys, index=keys.index('Star: G2V') if 'Star: G2V' in keys else 0, key='tr_spec')
            st.caption(f'Conditions: 50 % Moon at 90°, 5 h after sunset, typical seeing; exposure {exptimes[0]:g} s x {counts[0]} frames per filter (first order).')
            if st.button('Calculate depth', key='tr_depth_btn'):
                target = Target(name='target', mag=float(mag), mag_filter=band, spectrum_type=spectrum)
                cond = etc.conditions_for(seeing=typical_seeing(), moon_phase=0.5, moon_separation=90.0, hour_since_sunset=5.0)
                results = get_calculator().snr(target, om.filters, exptimes[0], counts[0], cond, units=om.filter_multiplicity())
                ss['tr_depth'] = pd.DataFrame([r.row() for r in results])
            if isinstance(ss.get('tr_depth'), pd.DataFrame):
                st.dataframe(ss['tr_depth'].round(3), hide_index=True, width='stretch')

    with st.expander('Overhead'):
        if om is None:
            st.caption('Choose an observation mode with filters first.')
        else:
            sig = (om.label, exptimes[0], counts[0])
            if st.button('Calculate overhead', key='tr_oh_btn'):
                cfg_oh = overhead.OverheadConfig.load()
                req = overhead.ObsRequest('request', exptimes[0], counts[0], om, 'per_filter', config=cfg_oh)
                ss['tr_oh'] = (sig, overhead.estimate([req], cfg_oh), req.autofocus_policy)
        if om is not None and ss.get('tr_oh') and ss['tr_oh'][0] == (om.label, exptimes[0], counts[0]):
            _, result, policy = ss['tr_oh']
            m = st.columns(4)
            m[0].metric('Total observation time', overhead.format_duration(result.total))
            m[1].metric('Open-shutter time', overhead.format_duration(result.science_time))
            m[2].metric('Overhead', overhead.format_duration(result.overhead))
            m[3].metric('Efficiency', f'{100 * result.efficiency:.0f} %')
            st.markdown('\n'.join(f'- {line}' for line in result.summary_lines()))
            st.caption('Per target, first order only; autofocus follows the mode rule (' + ('once per filter' if policy == 'per_filter' else 'none') + ').')

    with st.expander('7DS tiles'):
        if st.button('Match tiles', key='tr_tile_btn') and targets:
            ts_ = get_tiles()
            ss['tr_tiles'] = pd.DataFrame([ts_.match(t.ra, t.dec, radius_deg=0.0, target=t).summary() for t in targets])
        if isinstance(ss.get('tr_tiles'), pd.DataFrame):
            st.dataframe(styled_rows(ss['tr_tiles'].round(3), 'status'), hide_index=True, width='stretch')
        elif not targets:
            st.caption('Needs targets with valid coordinates.')


def main():
    feedback_note()
    st.title('7DT Observation request')
    cfg = live_config()
    ts = trigger.TriggerSettings.load()
    project = project_section(ts)
    defaults, om, exptimes, counts = settings_section(ts, cfg)
    edited = targets_section()
    norm, formatted, errors = preview_section(edited, ts, defaults, cfg, project)
    submit_section(formatted, errors, ts, project)
    tools_section(norm, om, exptimes, counts, cfg)


if __name__ == '__main__':
    main()
