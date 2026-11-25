"""
7DT Scheduling (operator page): every target requested in the Phase-2 files of the open-time programme in
one editable table (obs_starttime, obs_endtime, frames per epoch, number of epochs; a target set may be split
into parts with their own values), the check that each
target is up long enough inside its range for the entered plan, a load-balanced assumption of which nights
the automatic scheduler would use, visibility of any chosen targets over the programme period, and the
nights on which the assumed epochs compete for the same time of night.
Run with:  streamlit run schedule_app.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schedule_plots import (MIN_ALT, WINDOW_COLOR, load_requests_cached, night_figure, night_tables, program_data, projects_figure, projects_signature,
                            projects_table, visibility_figure, warm_program_data, with_expected)
from sevendt_calc import phase2, sheets, simulate, trigger
from sevendt_calc import visibility as vis
from ui_common import feedback_note, live_config, site_key

st.set_page_config(page_title='7DT Scheduling', page_icon='🗓️', layout='wide')

EDIT_COLUMNS = ['obs_starttime', 'obs_endtime', 'count', 'target_epoch', 'cadence', 'visits_per_night', 'gap_hours', 'moon', 'scheduled', 'Note']
DATE_PATTERN = r'^\s*\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?(\s*,\s*\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?)*\s*$'
RED, YELLOW = 'background-color: #ffb3b3', 'background-color: #fff3b0'


# ------------------------------------------------------------------ the plan table
def base_frame(requests: list, settings: dict) -> pd.DataFrame:
    """One row per part of every target set (settings: target key -> [TargetSettings, ...])."""
    rows = []
    for req in requests:
        for t in req.targets:
            k = phase2.target_key(req, t)
            parts = settings[k]
            for n, s in enumerate(parts, start=1):
                exposure_h = (t.exptime or 0.0) * s.count * s.epochs * max(t.n_orders, 1) / 3600.0
                rows.append({'key': phase2.part_key(k, n), 'Project': req.key, 'PI': req.lead_pi, 'Target': t.display, 'Part': f'{n}/{len(parts)}' if len(parts) > 1 else '',
                             'Total exptime (h)': round(exposure_h, 2), 'With overhead (h)': round(phase2.epoch_hours(t, s.count) * s.epochs, 2),
                             'Type': t.target_type, 'Mode': t.mode_text, 'Sky': req.sky_background,
                             'RA (°)': None if t.ra is None else round(t.ra, 4), 'Dec (°)': None if t.dec is None else round(t.dec, 4),
                             'Exptime (s)': t.exptime, 'Requested frames': t.nframes, 'Requested (h)': round(t.total_s / 3600.0, 2), 'dither_arcsec': req.dither_arcsec,
                             'obs_window_start': t.window_starts_text.replace('T', ' '), 'obs_window_end': t.window_ends_text.replace('T', ' '), 'Requester_note': t.note,
                             'obs_starttime': s.start_text.replace('T', ' '), 'obs_endtime': s.end_text.replace('T', ' '),
                             'count': int(s.count), 'target_epoch': int(s.epochs), 'cadence': int(s.cadence), 'visits_per_night': int(s.visits_per_night),
                             'gap_hours': float(s.gap_hours), 'moon': bool(s.use_moon), 'scheduled': bool(s.enabled), 'Note': s.note})
    return pd.DataFrame(rows)


def _as_datetime_list(value, fallback: list) -> list:
    """Typed 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC), several separated by commas; anything else keeps the previous value."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return fallback
    try:
        return phase2.as_datetime_list(value) or fallback
    except (ValueError, TypeError):
        return fallback


def settings_from_frame(df: pd.DataFrame, previous: dict) -> dict:
    """The edited rows back into target key -> [TargetSettings, ...] (rows of a target set are consecutive, in part order)."""
    out = {k: [] for k in previous}
    bad = {}
    for _, r in df.iterrows():
        tkey, _, n = str(r['key']).rpartition('#')
        n = int(n) - 1
        old = previous[tkey][n] if n < len(previous[tkey]) else previous[tkey][-1]
        try:
            count = int(r['count']) if not pd.isna(r['count']) else old.count
            epochs = int(r['target_epoch']) if not pd.isna(r['target_epoch']) else old.epochs
            cadence = int(r['cadence']) if not pd.isna(r['cadence']) else old.cadence
            visits = int(r['visits_per_night']) if not pd.isna(r['visits_per_night']) else old.visits_per_night
            gap = float(r['gap_hours']) if not pd.isna(r['gap_hours']) else old.gap_hours
        except (TypeError, ValueError):
            count, epochs, cadence, visits, gap = old.count, old.epochs, old.cadence, old.visits_per_night, old.gap_hours
        note = '' if r['Note'] is None or (isinstance(r['Note'], float) and np.isnan(r['Note'])) else str(r['Note'])
        try:
            out[tkey].append(phase2.TargetSettings(start=_as_datetime_list(r['obs_starttime'], old.start), end=_as_datetime_list(r['obs_endtime'], old.end), count=max(count, 1),
                                                   epochs=max(epochs, 1), enabled=bool(r['scheduled']), cadence=max(cadence, 0), note=note, visits_per_night=max(visits, 1), gap_hours=max(gap, 0.0),
                                                   use_moon=bool(r['moon'])))
        except ValueError as exc:                                      # e.g. two start times but one end time: keep the previous ranges and say so
            bad[f'{tkey}#{n + 1}'] = str(exc)
            out[tkey].append(phase2.TargetSettings(start=old.start, end=old.end, count=max(count, 1), epochs=max(epochs, 1), enabled=bool(r['scheduled']), cadence=max(cadence, 0), note=note,
                                                   visits_per_night=max(visits, 1), gap_hours=max(gap, 0.0), use_moon=bool(r['moon'])))
    st.session_state['sch_bad_rows'] = bad
    return {k: v or previous[k] for k, v in out.items()}


def remount_editor(requests: list) -> None:
    ss = st.session_state
    ss['sch_editor_base'] = base_frame(requests, ss['sch_tsettings'])
    ss['sch_editor_version'] = ss.get('sch_editor_version', 0) + 1


def split_controls(requests: list) -> None:
    """Split a chosen target set into parts (each with its own range, count and epochs) or merge its parts again."""
    ss = st.session_state
    options = [(phase2.target_key(req, t), f'{req.key} {t.display}') for req in requests for t in req.targets]
    labels = dict(options)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
    tkey = c1.selectbox('Target set to split', [k for k, _ in options], format_func=lambda k: labels[k], key='sch_split_target')
    parts = ss['sch_tsettings'][tkey]
    c1.caption(f'{len(parts)} part(s) at the moment.' if len(parts) > 1 else 'One part at the moment.')
    if c2.button('Add a part', key='sch_split_add', help='Adds a part with the same range as the last one; then edit its dates, count and epochs in the table.'):
        last = parts[-1]
        parts.append(phase2.TargetSettings(start=list(last.start), end=list(last.end), count=last.count, epochs=1, enabled=last.enabled))
        remount_editor(requests)
        st.rerun()
    if c3.button('Remove last part', key='sch_split_remove', disabled=len(parts) < 2):
        parts.pop()
        remount_editor(requests)
        st.rerun()
    c4.caption('A split target set has one row per part; its parts are compared with the requested frames as a whole. Most target sets stay in one part.')


def plan_section(requests: list, tables: dict) -> tuple[dict, bool]:
    """The editable table of every requested target set; returns (plans, use_moon)."""
    ss = st.session_state
    st.subheader('Targets and plan')
    c1, c2, c3 = st.columns([1.3, 1, 3])
    use_moon = c1.checkbox('Apply the Moon condition', value=True, key='sch_use_moon', help='Only nights whose Moon illumination suits the project\'s sky background count (Dark ≤ 0.2, Gray 0.2–0.8, Bright ≥ 0.8); the moon column switches it off per row.')
    if c2.button('Reset to defaults', key='sch_reset', help='Requested window span; all frames in one epoch when one visit fits into a night of the span, otherwise split into epochs that fit.'):
        ss['sch_tsettings'] = {phase2.target_key(req, t): [phase2.default_target_settings(req, t, tables[req.key], j, use_moon)] for req in requests for j, t in enumerate(req.targets)}
        remount_editor(requests)
    c3.caption('Edit obs_starttime / obs_endtime (UTC, typed as YYYY-MM-DD for that whole local night or YYYY-MM-DD HH:MM to cut exactly there; several ranges comma-separated, '
               'the n-th start with the n-th end), count (frames per filter in one epoch), target_epoch (total epochs), cadence (days between nights, 0 = free), '
               'visits/night and gap (several epochs in one night, at least gap hours after the previous visit of the same tile, fixed programmes included), moon (off = ignore the Moon condition for this row). '
               'The range you enter is the master (the requested OBS_WINDOW is shown for reference only; timed windows still fix the time of night). '
               'One epoch = count frames in every filter of the mode, overheads included; the plan needs target_epoch nights in the range on which the target is up for that long.')
    split_controls(requests)
    column_config = {
        'key': None, 'RA (°)': st.column_config.NumberColumn(format='%.4f'), 'Dec (°)': st.column_config.NumberColumn(format='%.4f'),
        'Exptime (s)': st.column_config.NumberColumn(format='%g'),
        'dither_arcsec': st.column_config.NumberColumn(format='%.0f'), 'Requested (h)': st.column_config.NumberColumn(format='%.2f'),
        'obs_starttime': st.column_config.TextColumn('obs_starttime', validate=DATE_PATTERN, required=True, width='medium',
                                                     help='UTC, typed as YYYY-MM-DD (the whole local night of that date) or YYYY-MM-DD HH:MM (start exactly there, e.g. 2027-03-05 00:30). '
                                                          'Several ranges: separate the starts with commas and list the matching ends in obs_endtime.'),
        'obs_endtime': st.column_config.TextColumn('obs_endtime', validate=DATE_PATTERN, required=True, width='medium',
                                                   help='UTC, typed as YYYY-MM-DD (that whole local night, inclusive) or YYYY-MM-DD HH:MM (stop exactly there); one entry per start, comma-separated.'),
        'obs_window_start': st.column_config.TextColumn('obs_window_start', help='Starts of the OBS_WINDOWs the PI requested (reference only)'),
        'obs_window_end': st.column_config.TextColumn('obs_window_end', help='Ends of the OBS_WINDOWs the PI requested (inclusive dates; reference only)'),
        'count': st.column_config.NumberColumn('count', min_value=1, max_value=100000, step=1, required=True, help='Frames per filter in one epoch'),
        'target_epoch': st.column_config.NumberColumn('target_epoch', min_value=1, max_value=1000, step=1, required=True, help='Number of epochs (visits)'),
        'cadence': st.column_config.NumberColumn('cadence (d)', min_value=0, max_value=365, step=1, required=True, help='Days between consecutive epochs; 0 = no constraint. Taken from "cadence=7d" in the NOTE when present.'),
        'visits_per_night': st.column_config.NumberColumn('visits/night', min_value=1, max_value=20, step=1, required=True, help='Epochs of this row in one night (1 normally).'),
        'gap_hours': st.column_config.NumberColumn('gap (h)', min_value=0.0, max_value=12.0, step=0.5, format='%.1f', required=True,
                                                   help='Minimum hours after the previous visit of the same tile that night, by this row or by a fixed programme (e.g. IMS); 0 = none.'),
        'Total exptime (h)': st.column_config.NumberColumn('Total exptime (h)', format='%.2f', help='exptime × count × target_epoch × orders of the mode (specall: 2), updated as you edit'),
        'With overhead (h)': st.column_config.NumberColumn('With overhead (h)', format='%.2f', help='target_epoch × (one epoch of count frames with the overhead model), updated as you edit'),
        'moon': st.column_config.CheckboxColumn('moon', help='Apply the project\'s sky-background (Moon) condition to this row; off = any Moon (e.g. nightly monitoring)'),
        'scheduled': st.column_config.CheckboxColumn('scheduled', help='Off: left out of the plan and of the conflict check'),
        'Note': st.column_config.TextColumn('Note', width='medium', help='Your own note for this row (saved with the schedule)'),
        'Requester_note': st.column_config.TextColumn('Requester_note', help='The NOTE the PI wrote in the request file'),
    }
    editor_key = f"sch_editor_{ss.get('sch_editor_version', 0)}"
    state = ss.get(editor_key)
    previous = ss.get('sch_editor_base')
    if state is not None and previous is not None:            # apply the edits already made before the table is rebuilt, so the live totals do not lag one run behind
        pending = previous.copy()
        for ridx, changes in (dict(state.get('edited_rows') or {})).items():
            for col, val in dict(changes).items():
                if col in pending.columns and 0 <= int(ridx) < len(pending):
                    pending.iat[int(ridx), pending.columns.get_loc(col)] = val
        ss['sch_tsettings'] = settings_from_frame(pending, ss['sch_tsettings'])
    base = base_frame(requests, ss['sch_tsettings'])       # values follow the latest edits (live totals); a keyed fixed-row editor keeps its identity across value changes
    ss['sch_editor_base'] = base
    disabled = [c for c in base.columns if c not in EDIT_COLUMNS]
    first = ['Project', 'Target', 'Part'] + EDIT_COLUMNS + ['Total exptime (h)', 'With overhead (h)']
    order = first + [c for c in base.columns if c not in first and c != 'key']
    st.markdown('**Plan table** (one row per requested target set) — type into the cells with the pencil icon (obs_starttime, obs_endtime, count, target_epoch, cadence, visits/night, gap, scheduled, Note) '
                'and press Enter; Total exptime = exptime × count × target_epoch × orders and With overhead follow your edits; the other columns (Requester_note = the PI\'s NOTE) repeat the request and cannot be edited.')
    edited = st.data_editor(base, key=editor_key, column_config=column_config, disabled=disabled, hide_index=True,
                            width='stretch', height=min(60 + 36 * len(base), 560), column_order=order)
    ss['sch_tsettings'] = settings_from_frame(edited, ss['sch_tsettings'])
    plans = {}
    for req in requests:
        for j, t in enumerate(req.targets):
            for p in phase2.make_target_plans(req, j, tables[req.key], ss['sch_tsettings'][phase2.target_key(req, t)], use_moon):
                plans[p.key] = p
    phase2.place_epochs(plans, phase2.PROGRAM_NIGHTS)
    row_problems(plans)
    return plans, use_moon


def row_problems(plans: dict) -> None:
    """A short list of the rows that cannot be observed as entered (blocking) or carry a remark, instead of a per-row table."""
    n_parts = {}
    for p in plans.values():
        n_parts[p.target_key] = max(n_parts.get(p.target_key, 0), p.part)
    on = [p for p in plans.values() if p.settings.enabled]
    errors = [p for p in on if p.issues]
    warns = [p for p in on if not p.issues and p.warnings]
    total = sum(p.planned_h for p in on)

    def name(p):
        return f'{p.req.key} {p.target.display}' + (f' part {p.part}' if n_parts[p.target_key] > 1 else '')

    msg = f'{len(on)} of {len(plans)} rows scheduled; planned {total:.1f} h in total.'
    bad = st.session_state.get('sch_bad_rows') or {}
    if bad:
        st.error('Ranges not applied (the previous values are kept):\n\n' + '\n'.join(f'- {k.replace("#1", "") if n_parts.get(k.rpartition("#")[0], 1) == 1 else k}: {v}' for k, v in bad.items()))
    if errors:
        st.error(f'{len(errors)} row(s) cannot be observed as entered. {msg}\n\n' + '\n'.join(f'- {name(p)}: ' + '; '.join(p.issues + p.warnings) for p in errors))
    elif warns:
        st.warning(f'Every scheduled row fits; {len(warns)} with a remark. {msg}\n\n' + '\n'.join(f'- {name(p)}: ' + '; '.join(p.warnings) for p in warns))
    else:
        st.success(f'Every scheduled row fits. {msg}')
    if warns and errors:
        st.warning('\n'.join(f'- {name(p)}: ' + '; '.join(p.warnings) for p in warns))


# ------------------------------------------------------------------ page sections
def select_project(requests: list, df: pd.DataFrame) -> phase2.RequestFile:
    ss = st.session_state
    keys = [r.key for r in requests]
    event = st.dataframe(df, hide_index=True, on_select='rerun', selection_mode='single-row', key='sch_table', width='stretch',
                         column_config={'Allocated (h)': st.column_config.NumberColumn(format='%.1f'), 'dither_arcsec': st.column_config.NumberColumn(format='%.0f')})
    rows = list(getattr(getattr(event, 'selection', None), 'rows', []) or [])
    clicked = keys[rows[0]] if rows and rows[0] < len(keys) else None
    if clicked and clicked != ss.get('sch_last_click'):
        ss['sch_last_click'] = clicked
        ss['sch_project'] = clicked
    if ss.get('sch_project') not in keys:
        ss['sch_project'] = keys[0]
    choice = st.selectbox('Project', keys, format_func=lambda k: f'{k}: {next(r for r in requests if r.key == k).title[:90]}', key='sch_project')
    req = next(r for r in requests if r.key == choice)
    st.caption(f'{req.key}: {req.title} · PI {req.pi} ({req.emails}) · {req.observation} · {req.mode} {req.submodes} · sky background {req.sky_background} · dither {req.dither_arcsec:g}" · file {req.name}')
    for issue in req.all_issues:
        st.warning(issue)
    return req


def visibility_section(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid, selected: phase2.RequestFile) -> None:
    ss = st.session_state
    st.subheader('Visibility of the targets')
    keys = [r.key for r in requests]
    by_key = {r.key: r for r in requests}
    if ss.get('sch_vis_follow') != selected.key:                         # a newly chosen project becomes the default selection
        ss['sch_vis_follow'] = selected.key
        ss['sch_vis_projects'] = [selected.key]
        ss.pop('sch_vis_targets', None)
    c1, c2 = st.columns([1, 2])
    chosen_projects = c1.multiselect('Projects', keys, format_func=lambda k: f'{k} {by_key[k].lead_pi}', key='sch_vis_projects')
    options = [(k, j) for k in chosen_projects for j in range(len(by_key[k].targets)) if by_key[k].targets[j].has_coordinates]
    labels = {opt: f'{opt[0]} {by_key[opt[0]].targets[opt[1]].display}' for opt in options}
    if 'sch_vis_targets' not in ss or any(o not in options for o in ss['sch_vis_targets']):
        ss['sch_vis_targets'] = [o for o in options if o in (ss.get('sch_vis_targets') or options[:8])]
    chosen = c2.multiselect('Targets', options, format_func=lambda o: labels[o], key='sch_vis_targets')
    c3, c4 = st.columns([1.6, 1])
    mode = c3.radio('y axis', ['Maximum altitude of the night', 'Observable hours per night'], horizontal=True, key='sch_yaxis')
    mode = 'alt' if mode.startswith('Max') else 'hours'
    show_moon = c4.checkbox('Sky brightness (Moon)', value=True, key='sch_show_moon',
                            help='Moon illumination on the right axis; in the strips, nights whose Moon does not suit the project\'s sky background are yellow (Dark ≤ 0.2, Gray 0.2–0.8, Bright ≥ 0.8).')
    if not chosen:
        st.info('Pick at least one target.')
        return
    items = [(by_key[k], tables[k], j) for k, j in chosen]
    st.plotly_chart(visibility_figure(items, grid, mode, show_moon, plans), width='stretch', key='sch_vis_fig', config={'displaylogo': False})
    st.caption(f'Top: every chosen target per night; dashed grey line at {MIN_ALT:g}° / {phase2.GOOD_HOURS:g} h; red diamonds = the assumed epochs; dotted red outline = the entered obs_starttime to obs_endtime. '
               'Bottom: one bar per chosen target: light green = observable (max altitude ≥ 30° / > 0 h), dark green = at transit / ≥ 3 h, '
               'yellow = Moon unsuitable for its project; red centre line = its entered range with the assumed epochs as diamonds; thin blue line = the OBS_WINDOW the PI requested.')


def all_projects_section(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid) -> None:
    st.subheader('All projects')
    conflicts = phase2.night_conflicts(plans, list(grid.dates))
    show_moon = st.checkbox('Mark nights that fail the Moon condition', value=False, key='sch_all_moon')
    st.plotly_chart(projects_figure(requests, tables, plans, conflicts, show_moon), width='stretch', key='sch_projects_fig', config={'displaylogo': False})
    st.caption('Per project and night: greens = hours during which at least one of its targets is above the limit (light > 0 h, dark ≥ 3 h); '
               'red diamonds = nights on which an epoch of one of its targets is assumed (automatic scheduling: the most constrained targets first, '
               'each epoch in its own part of the range on the night with the least load, cadence respected).')
    if conflicts:
        st.error(f'{len(conflicts)} night(s) on which the assumed epochs compete for the same time of night (least-slack packing, one target at a time).')
        rows = [{'Night (local)': c['date'].isoformat(), 'Targets': ', '.join(c['targets']), 'Epochs (h)': round(c['demand_h'], 2),
                 'Fits (h)': round(c['scheduled_h'], 2), 'Short': ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items())} for c in conflicts]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(60 + 36 * len(rows), 330))
    else:
        st.success('No night on which the assumed epochs compete for the same time of night.')
    load = np.zeros(len(grid.dates))
    for p in plans.values():
        load[p.placed] += p.epoch_h
    st.caption(f'Nights with an epoch: {int((load > 0).sum())} of {len(grid.dates)}; busiest night {grid.dates[int(load.argmax())]} with {load.max():.1f} h.')
    default = conflicts[0]['date'] if conflicts else grid.dates[int(load.argmax())]
    night = st.date_input('Night to inspect (local date of the evening)', value=default, min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, key='sch_night')
    i = (night - phase2.PROGRAM_START).days
    st.caption('Light bars: the target above the altitude limit (inside its timed window, if any). Red: the epoch, packed one target at a time, least slack first.')
    st.plotly_chart(night_figure(plans, grid, i), width='stretch', key='sch_night_fig', config={'displaylogo': False})


def observation_list_section(requests: list, plans: dict, folder: Path) -> None:
    """The plan as an observation list (one row per scheduled part, fixed columns) written to a Google Sheet or downloaded."""
    st.subheader('Export the observation list')
    ts = trigger.TriggerSettings.load()
    hidden = ts.hidden or {}
    rows = phase2.export_rows(requests, plans, binning=ts.default_binning, gain=ts.default_gain, priority=ts.priority, weight=hidden.get('weight', 1), objtype=hidden.get('objtype', 'Open'))
    df = pd.DataFrame(rows, columns=list(phase2.EXPORT_COLUMNS))
    st.caption(f'{len(df)} scheduled rows: objname, RA, De (degrees), cadence, target_count (= target_epoch), starttime / endtime (= obs_starttime / obs_endtime), sky_brightness, exptime, count, '
               f'filter_ None, binning {ts.default_binning}, gain {ts.default_gain}, dither_arcsec, colormode / specmode, obsmode, priority {ts.priority}, weight {hidden.get("weight", 1)}, '
               f'objtype {hidden.get("objtype", "Open")}, program_id, program_PI (no spaces), note (yours). Rows with "scheduled" off are left out.')
    st.dataframe(df, hide_index=True, width='stretch', height=min(60 + 36 * len(df), 330))
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    default_ws = f'schedule_{folder.name}'
    worksheet = c1.text_input('Worksheet', value=default_ws, key='sch_sheet_ws', help=f'Worksheet of the request spreadsheet ({ts.spreadsheet_url}); it is replaced by this list.')
    c2.download_button('Download observation list (CSV)', data=df.to_csv(index=False).encode(), file_name=f'{worksheet or default_ws}.csv', mime='text/csv', key='sch_export_csv')
    key_path = sheets.service_account_key_path()
    if key_path is None:
        c3.button('Write to Google Sheet', key='sch_sheet_write', disabled=True)
        st.warning('No Google service-account key on this server: set google_service_account_key in data/settings.json (or GOOGLE_APPLICATION_CREDENTIALS) and share the spreadsheet with that account; the CSV works meanwhile.')
        return
    if c3.button('Write to Google Sheet', type='primary', key='sch_sheet_write', help='Replaces the worksheet with these rows'):
        try:
            result = sheets.replace_rows(ts.spreadsheet_url, worksheet.strip() or default_ws, list(phase2.EXPORT_COLUMNS), rows, key_path)
        except Exception as exc:                          # gspread / network errors: show them, do not crash the page
            st.error(f'Writing failed: {exc}')
        else:
            st.success(f'{result["n_rows"]} rows written to worksheet {result["worksheet"]}' + (' (created)' if result['created'] else '') + f': {result["url"]}')
    st.caption(f'Service account: {sheets.service_account_email(key_path)} (needs editor access to the spreadsheet).')


@st.cache_resource(show_spinner=False)
def fixed_programmes_cached(key: tuple, sun_alt: float, signature: tuple) -> list:
    return simulate.load_fixed_programmes(program_data(key, sun_alt)['grid'], live_config())


def simulation_section(plans: dict, grid: vis.SeasonGrid, key: tuple, sun_alt: float) -> None:
    """Monte-Carlo night-by-night scheduling of the current plan: fixed programmes first, random weather, many runs."""
    ss = st.session_state
    st.subheader('Simulated scheduling')
    fixed_path = trigger.PROJECTS_JSON.parent / 'fixed_programmes.json'
    fixed_sig = (str(fixed_path), fixed_path.stat().st_mtime_ns) if fixed_path.is_file() else (str(fixed_path), 0)
    fixed = fixed_programmes_cached(key, sun_alt, fixed_sig)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    runs = int(c1.number_input('Runs', min_value=10, max_value=1000, value=200, step=10, key='sch_sim_runs'))
    bad = float(c2.number_input('Bad-weather fraction', min_value=0.0, max_value=0.9, value=0.10, step=0.05, key='sch_sim_bad'))
    seed = int(c3.number_input('Seed', min_value=0, max_value=10000, value=1, step=1, key='sch_sim_seed'))
    c4.caption('Every clear night: ' + ('; '.join(f'{f.name}: {len(f.tile_ids)} tiles around {f.tile_ids[0]}, {f.epoch_h * 60:.0f} min per tile' + (f', every {f.cadence_days} d' if f.cadence_days else ', every night') for f in fixed) if fixed else 'no fixed programmes (data/fixed_programmes.json)')
                + ' are placed first; then the scheduled parts, most urgent first (feasible nights ahead per remaining epoch), each in the earliest free slot while the target is up.')
    if not st.button('Run the simulation', key='sch_sim_run') and 'sch_sim_result' not in ss:
        return
    if st.session_state.get('sch_sim_run'):
        with st.spinner(f'{runs} runs...'):
            result = simulate.simulate(plans, grid, fixed, runs=runs, bad_weather=bad, seed=seed)
        ss['sch_sim_result'] = (result, simulate.summary_rows(result, plans))
        ss['sch_sim_expected'] = result.expected_by_key()
        st.rerun()                                       # the figures above now show the expected epoch nights
    result, rows = ss['sch_sim_result']
    df = pd.DataFrame(rows)
    n_low = int((df['p_complete'] < 0.9).sum())
    st.caption(f'{result.runs} runs, {(1 - result.clear.mean()):.0%} of nights lost to weather. '
               + ' · '.join(f'{name}: all tiles visited on {done[result.clear].mean():.0%} of the clear nights' if not next(f for f in fixed if f.name == name).cadence_days
                            else f'{name}: {done.sum(axis=1).mean():.1f} visits per run' for name, done in result.fixed_done.items()))
    if n_low:
        st.warning(f'{n_low} part(s) finish all epochs in fewer than 90 % of the runs (p_complete): widen their range, move them, or reduce the epochs.')
    style = df.style.apply(lambda col: ['background-color: #ffb3b3' if v < 0.5 else ('background-color: #fff3b0' if v < 0.9 else '') for v in df['p_complete']], axis=0)
    st.dataframe(style, hide_index=True, width='stretch', height=min(60 + 36 * len(df), 420))
    st.download_button('Download simulation summary (CSV)', data=df.to_csv(index=False).encode(), file_name='7dt_simulation_summary.csv', mime='text/csv', key='sch_sim_csv')
    dates = [datetime.combine(d, datetime.min.time()) for d in result.dates]
    labels = [k.replace('#1', '') if k.endswith('#1') and plans[k].part == 1 and len(phase2.plans_of(plans, plans[k].req, plans[k].j)) == 1 else k for k in result.keys]
    fig = go.Figure(go.Heatmap(x=dates, y=labels, z=result.prob, zmin=0, zmax=1, colorscale='Reds', colorbar=dict(title='P(epoch)'),
                               hovertemplate='%{y} %{x|%Y-%m-%d}: %{z:.2f}<extra></extra>'))
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=16 * len(labels) + 120, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type='date', range=[x0, x1]), yaxis=dict(type='category', autorange='reversed', tickfont=dict(size=9)))
    st.plotly_chart(fig, width='stretch', key='sch_sim_fig', config={'displaylogo': False})
    st.caption('Probability that the part gets an epoch on that night over the runs. Darker = more likely; an empty row = never scheduled (nothing fits in its range, or always beaten by others).')


def export_section(requests: list, plans: dict, folder: Path) -> None:
    ss = st.session_state
    st.subheader('Save and export')
    c1, c2, c3, c4 = st.columns(4)
    saved = phase2.load_schedule(folder)
    changed = {k: v for k, v in ss['sch_tsettings'].items() if saved.get(k) != v}
    if c1.button('Save schedule', type='primary', key='sch_save', help=f'Writes {folder / phase2.SCHEDULE_FILE}; the previous file is kept in {folder / phase2.BACKUP_DIR}'):
        path = phase2.save_schedule(folder, ss['sch_tsettings'])
        st.caption(f'Saved to {path} (previous versions in {folder / phase2.BACKUP_DIR})')
        changed = {}
    elif changed:
        c1.caption(f'{len(changed)} target set(s) changed since the last save.')
    rows = phase2.schedule_rows(requests, plans)
    c2.download_button('Download plan table (CSV)', data=pd.DataFrame(rows).to_csv(index=False).encode(), file_name='7dt_schedule_targets.csv', mime='text/csv', key='sch_csv')
    payload = {k: (v[0].as_dict() if len(v) == 1 else [x.as_dict() for x in v]) for k, v in ss['sch_tsettings'].items()}
    c3.download_button('Download schedule (JSON)', data=json.dumps(payload, indent=2).encode(), file_name='7dt_schedule.json', mime='application/json', key='sch_json')
    upload = c4.file_uploader('Load a schedule JSON', type=['json'], key='sch_upload', label_visibility='collapsed')
    if upload is not None and ss.get('sch_uploaded') != upload.file_id:
        try:
            raw = json.loads(upload.getvalue().decode())
            loaded = {}
            for k, v in raw.items():
                if str(k).startswith('_'):
                    continue
                loaded[str(k)] = [phase2.TargetSettings.from_dict(x) for x in (v if isinstance(v, list) else [v])]
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f'Not a schedule file: {exc}')
        else:
            for k, v in loaded.items():
                if k in ss['sch_tsettings'] and v:
                    ss['sch_tsettings'][k] = v
            remount_editor(requests)
            ss['sch_uploaded'] = upload.file_id
            st.rerun()


# ------------------------------------------------------------------ main
def main():
    feedback_note()
    st.title('7DT Scheduling')
    cfg = live_config()
    key, sun_alt = site_key(cfg['site']), cfg['constraints'].sun_alt
    warm_program_data(key, sun_alt)
    folder = phase2.requests_dir()
    signature = phase2.folder_signature(folder)
    if not signature:
        st.error(f'No request files in {folder} (set phase2_requests_dir in data/settings.json or SEVENDT_PHASE2_DIR).')
        return
    projects_sig = projects_signature()
    requests = load_requests_cached(str(folder), signature, projects_sig)
    if trigger.load_projects() is None:
        st.warning('data/projects.json is missing: no allocations, request types or sky backgrounds are known.')
    with st.spinner('Computing the nights of the programme (first visit only)...'):
        data = program_data(key, sun_alt)
        tables = night_tables(str(folder), signature, projects_sig, key, sun_alt)
    grid = data['grid']
    ss = st.session_state
    if ss.get('sch_signature') != (signature, projects_sig) or 'sch_tsettings' not in ss:
        saved = phase2.load_schedule(folder)
        ss['sch_tsettings'] = {phase2.target_key(req, t): saved.get(phase2.target_key(req, t)) or [phase2.default_target_settings(req, t, tables[req.key], j)]
                               for req in requests for j, t in enumerate(req.targets)}
        remount_editor(requests)
        ss['sch_signature'] = (signature, projects_sig)
    st.caption(f'{len(requests)} request files in {folder} · programme {phase2.PROGRAM_START} to {phase2.PROGRAM_END} · altitude limit {MIN_ALT:g}° · '
               f'dates are local nights (evening date); times UTC.')
    plans, _ = plan_section(requests, tables)
    if ss.get('sch_sim_expected'):
        plans_fig = with_expected(plans, ss['sch_sim_expected'])
        st.caption('Diamonds and epoch counts below follow the last simulation run (expected epoch nights); run it again after editing the plan.')
    else:
        plans_fig = plans
    st.subheader('Project summary')
    st.caption('Per project: requested and planned time against the allocation, and when the assumed epochs fall. Tick a row (or pick below) to preselect it in the visibility plot.')
    req = select_project(requests, projects_table(requests, plans_fig, list(grid.dates)))
    visibility_section(requests, tables, plans_fig, grid, req)
    all_projects_section(requests, tables, plans_fig, grid)
    simulation_section(plans, grid, key, sun_alt)
    export_section(requests, plans, folder)
    observation_list_section(requests, plans, folder)


main()
