"""
7DT Schedule viewer (read-only, for the PIs of the open-time programme): with their Project ID and key a PI
sees how their requested targets are planned (dates, frames per epoch, epochs, assumed nights) and the
visibility of their targets; the other programmes appear only as anonymous "another programme has an
epoch" marks. The admin key shows every programme like the operator's page, without editing.
Run with:  streamlit run plan_app.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schedule_plots import (CAT_INDEX, MIN_ALT, SIM_BAD_WEATHER, SIM_RUNS, dot_trace, fixed_signature, heatmap, load_requests_cached, night_dates, night_figure, night_tables,
                            probability_figure, program_data, projects_figure, projects_signature, projects_table, read_operator_note, saved_plans, schedule_signature,
                            read_program_notes, save_program_note, simulated_saved, visibility_figure, warm_program_data, warm_simulation, with_expected)
from sevendt_calc import keyguard, phase2, trigger
from sevendt_calc import visibility as vis
from ui_common import feedback_note, live_config, live_text_input, site_key

st.set_page_config(page_title='7DT Schedule', page_icon='📅', layout='wide')


# ------------------------------------------------------------------ pieces
def saved_text(folder: Path) -> str:
    path = folder / phase2.SCHEDULE_FILE
    if not path.is_file():
        return 'no schedule has been saved yet: the plan shown is the default one'
    try:
        stamp = json.loads(path.read_text()).get('_saved', '')
    except ValueError:
        stamp = ''
    return f'schedule as saved by the operator on {stamp.replace("T", " ")}' if stamp else 'schedule as saved by the operator'


def plan_frame(req: phase2.RequestFile, plans: dict, dates: list, sim_rows: dict | None = None) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(req.targets):
        parts = phase2.plans_of(plans, req, j)
        for p in parts:
            s = p.settings
            nights = ', '.join(dates[i].isoformat() for i in np.flatnonzero(p.placed))
            sim = (sim_rows or {}).get(p.key, {})
            rows.append({'Target': t.display, 'Part': f'{p.part}/{len(parts)}' if len(parts) > 1 else '', 'Type': t.target_type, 'Mode': t.mode_text,
                         'obs_starttime': s.start_text.replace('T', ' '), 'obs_endtime': s.end_text.replace('T', ' '), 'count': s.count, 'target_epoch': s.epochs,
                         'cadence (d)': s.cadence, 'Exptime (s)': t.exptime, 'Total exptime (h)': round((t.exptime or 0) * s.count * s.epochs * max(t.n_orders, 1) / 3600, 2),
                         'With overhead (h)': round(p.planned_h, 2), 'Requested frames': t.nframes,
                         'P(all epochs)': sim.get('p_complete') if s.enabled else None, 'Epochs expected': sim.get('epochs_done_mean') if s.enabled else None,
                         'First epoch (10–90 %)': f"{sim['first_10']} – {sim['first_90']}" if sim.get('first_10') and s.enabled else '',
                         'Last epoch (50 %)': sim.get('last_50', '') if s.enabled else '', 'Last epoch (90 %)': sim.get('last_90', '') if s.enabled else '',
                         ('Expected observation nights' if sim_rows else 'Assumed observation nights'): nights, 'Status': 'not scheduled' if not s.enabled else p.status, 'Note': s.note,
                         'obs_window_start': t.window_starts_text.replace('T', ' '), 'obs_window_end': t.window_ends_text.replace('T', ' '), 'Requester_note': t.note})
    return pd.DataFrame(rows)


def project_header(req: phase2.RequestFile, plans: dict, dates: list) -> None:
    ps = [p for p in plans.values() if p.req is req]
    on = [p for p in ps if p.settings.enabled]
    planned = sum(p.planned_h for p in on)
    placed = np.zeros(len(dates), bool)
    for p in on:
        placed |= p.placed
    nights = np.flatnonzero(placed)
    st.markdown(f'**{req.key}: {req.title}**')
    st.caption(f'PI {req.pi} · {req.observation} · {req.mode} {req.submodes} · sky background {req.sky_background} · dither {req.dither_arcsec:g}"')
    m = st.columns(6)
    m[0].metric('Allocated', f'{req.allocated_hours:.1f} h' if req.allocated_hours is not None else 'n/a')
    m[1].metric('Requested', f'{req.total_h:.1f} h')
    m[2].metric('Planned (with overheads)', f'{planned:.1f} h')
    m[3].metric('Expected observations', f'{int(sum(p.n_placed for p in on))} of {int(sum(p.settings.epochs for p in on))}')
    m[4].metric('First expected observation', dates[nights[0]].isoformat() if nights.size else '-')
    m[5].metric('Last expected observation', dates[nights[-1]].isoformat() if nights.size else '-')
    bad = [p for p in on if p.issues]
    if bad:
        st.warning('Rows the operator still has to settle: ' + '; '.join(f'{p.target.display}: {"; ".join(p.issues)}' for p in bad))


def visibility_block(req: phase2.RequestFile, tables: dict, plans: dict, grid: vis.SeasonGrid, prefix: str) -> None:
    options = [j for j, t in enumerate(req.targets) if t.has_coordinates]
    if not options:
        return
    c1, c2, c3 = st.columns([2, 1.4, 1])
    chosen = c1.multiselect('Targets', options, default=options[:8], format_func=lambda j: req.targets[j].display, key=f'{prefix}_targets_{req.key}')
    mode = c2.radio('y axis', ['Maximum altitude of the night', 'Observable hours per night'], horizontal=True, key=f'{prefix}_yaxis')
    show_moon = c3.checkbox('Sky brightness (Moon)', value=True, key=f'{prefix}_moon')
    if not chosen:
        return
    items = [(req, tables[req.key], j) for j in chosen]
    st.plotly_chart(visibility_figure(items, grid, 'alt' if mode.startswith('Max') else 'hours', show_moon, plans, epoch_label='expected observation', range_label='OBS_WINDOW (scheduled)'),
                    width='stretch', key=f'{prefix}_vis_fig', config={'displaylogo': False})
    st.caption(f'Top: each target per night; dashed grey line at {MIN_ALT:g}° / {phase2.GOOD_HOURS:g} h; red diamonds = the expected observation nights; dotted red = OBS_WINDOW (scheduled). '
               'Bottom: one bar per target: greens = observable (light > 0 h, dark = at transit / ≥ 3 h), yellow = Moon unsuitable for your sky background; '
               'red line = the nights of OBS_WINDOW (scheduled); when it covers only part of a night, the UT times are in the hover; thin blue line = the OBS_WINDOW you requested.')


def occupancy_figure(req: phase2.RequestFile, tables: dict, plans: dict, mode: str = 'alt') -> go.Figure:
    """
    Your programme's nights next to one anonymous row: nights on which another programme has an epoch (no
    names, no hours). The greens follow the y axis of the visibility figure: 'alt' = some target reaches the
    altitude limit (light) / transits during the night (dark); 'hours' = some target observable > 0 h / >= 3 h.
    """
    table = tables[req.key]
    dates = night_dates(table)
    n = len(dates)
    own, others = np.zeros(n, bool), np.zeros(n, bool)
    for p in plans.values():
        if not p.settings.enabled:
            continue
        if p.req is req:
            own |= p.placed
        else:
            others |= p.placed
    labels = [f'{req.key} (yours)', 'other programmes']
    cat = np.zeros(n, int)
    if mode == 'alt':
        up = (table.max_alt >= MIN_ALT)
        cat[up.any(axis=1)] = CAT_INDEX['obs']
        cat[(table.transit & up).any(axis=1)] = CAT_INDEX['good']
        own_text = [f'{d:%Y-%m-%d}: best max altitude {a:.1f}°' + (', a target transits during the night' if t else '') for d, a, t in zip(table.dates, table.max_alt.max(axis=1), (table.transit & up).any(axis=1))]
        legend = [('max altitude ≥ 30°', phase2.STRIP_CATEGORIES[CAT_INDEX['obs']][2]), ('transit during the night', phase2.STRIP_CATEGORIES[CAT_INDEX['good']][2])]
    else:
        cat = phase2.strip_categories(table.hours_any)
        own_text = [f'{d:%Y-%m-%d}: {h:.2f} h with one of your targets up' for d, h in zip(table.dates, table.hours_any)]
        legend = [(phase2.STRIP_CATEGORIES[CAT_INDEX[k]][1], phase2.STRIP_CATEGORIES[CAT_INDEX[k]][2]) for k in ('obs', 'good')]
    cats = np.array([cat, np.zeros(n, int)])
    hover = [[t + (', epoch assumed' if o else '') for t, o in zip(own_text, own)],
             [f'{d:%Y-%m-%d}: ' + ('another programme has an epoch' if o else 'free of other epochs') for d, o in zip(table.dates, others)]]
    fig = go.Figure(heatmap(dates, labels, cats, hover))
    fig.add_trace(dot_trace(dates, labels, [own, others], 'expected observation', [[f'{d:%Y-%m-%d}: your observation' for d in table.dates], ['another programme' for _ in table.dates]]))
    for label, color in legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color, line=dict(color='#999', width=0.5)), name=label, hoverinfo='skip'))
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()), datetime.combine(phase2.PROGRAM_END, datetime.min.time())
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type='date', range=[x0, x1]), yaxis=dict(type='category', autorange='reversed'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def program_note_block(req: phase2.RequestFile, folder: Path | None, admin: bool, prefix: str) -> None:
    """The operator's note on this programme for its PI; with the admin key it can be edited here."""
    if folder is None:
        return
    note = read_program_notes(folder).get(req.key, '')
    if note:
        st.info("**Note from the operator**\n\n" + note)
    if not admin:
        return
    edit_key = f'{prefix}_note_edit_{req.key}'
    if not st.session_state.get(edit_key):
        if st.button('Edit the programme note' if note else 'Add a programme note', key=f'{prefix}_note_open_{req.key}'):
            st.session_state[edit_key] = True
            st.rerun()
        return
    text = st.text_area(f'Note on programme {req.key} (markdown; shown to its PI above the plan; empty = none)', value=note, height=120, key=f'{prefix}_note_text_{req.key}')
    c1, c2, _ = st.columns([1, 1, 5])
    if c1.button('Save note', type='primary', key=f'{prefix}_note_save_{req.key}'):
        if save_program_note(folder, req.key, text):
            st.session_state[edit_key] = False
            st.rerun()
        else:
            st.error('The note could not be written (the request folder is read-only on this server); edit it on the operator\'s server.')
    if c2.button('Cancel', key=f'{prefix}_note_cancel_{req.key}'):
        st.session_state[edit_key] = False
        st.rerun()


def project_view(req: phase2.RequestFile, tables: dict, plans: dict, grid: vis.SeasonGrid, prefix: str = 'pl', sim: dict | None = None,
                 folder: Path | None = None, admin: bool = False) -> None:
    dates = list(grid.dates)
    project_header(req, plans, dates)
    program_note_block(req, folder, admin, prefix)
    df = plan_frame(req, plans, dates, sim['rows'] if sim else None)
    st.dataframe(df, hide_index=True, width='stretch', height=min(60 + 36 * len(df), 480),
                 column_config={'Total exptime (h)': st.column_config.NumberColumn(format='%.2f'), 'With overhead (h)': st.column_config.NumberColumn(format='%.2f'),
                                'Exptime (s)': st.column_config.NumberColumn(format='%g'), 'P(all epochs)': st.column_config.NumberColumn(format='%.2f'),
                                'Epochs expected': st.column_config.NumberColumn(format='%.1f')})
    st.download_button('Download this plan (CSV)', data=df.to_csv(index=False).encode(), file_name=f'7dt_plan_{req.key}.csv', mime='text/csv', key=f'{prefix}_csv_{req.key}')
    st.caption('obs_starttime / obs_endtime are UTC (a date alone = that whole local night); count = frames per filter in one epoch; target_epoch = number of epochs; '
               'Total exptime = exptime × count × target_epoch × orders of the mode; the expected observation nights are the median nights of each epoch over the simulated runs '
               '(epochs that happen in fewer than half of the runs are left out).')
    if sim is not None:
        st.subheader('Expected observation dates')
        fixed_text = '; '.join(f'{name} ({n} tiles around {tile}, ' + (f'every {cad} d' if cad else 'every clear night') + ')' for name, n, tile, cad in sim['fixed'])
        st.caption(f'From {SIM_RUNS} simulated runs of the whole programme with {SIM_BAD_WEATHER:.0%} of the nights lost to weather at random; the fixed programmes '
                   f'({fixed_text}) take their time first, then every scheduled target in order of urgency. P(all epochs) = share of runs in which all epochs of the row were observed; '
                   'the dates are percentiles over the runs. The colour below is the probability of an observation on that night.')
        keys = [p.key for j in range(len(req.targets)) for p in phase2.plans_of(plans, req, j) if p.settings.enabled]
        if keys:
            labels = [f'{plans[k].target.display}' + (f' part {plans[k].part}' if len(phase2.plans_of(plans, req, plans[k].j)) > 1 else '') for k in keys]
            st.plotly_chart(probability_figure(sim['result'], keys, labels), width='stretch', key=f'{prefix}_prob_{req.key}', config={'displaylogo': False})
    st.subheader('Visibility of your targets')
    visibility_block(req, tables, plans, grid, prefix)


def admin_view(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid, sim: dict | None = None, folder: Path | None = None) -> None:
    st.subheader('All programmes')
    df = projects_table(requests, plans, list(grid.dates))
    st.dataframe(df, hide_index=True, width='stretch', column_config={'Allocated (h)': st.column_config.NumberColumn(format='%.1f'), 'dither_arcsec': st.column_config.NumberColumn(format='%.0f')})
    conflicts = phase2.night_conflicts(plans, list(grid.dates))
    st.plotly_chart(projects_figure(requests, tables, plans, conflicts, False), width='stretch', key='pl_projects_fig', config={'displaylogo': False})
    if conflicts:
        rows = [{'Night (local)': c['date'].isoformat(), 'Targets': ', '.join(c['targets']), 'Epochs (h)': round(c['demand_h'], 2), 'Fits (h)': round(c['scheduled_h'], 2),
                 'Short': ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items())} for c in conflicts]
        st.warning(f'{len(conflicts)} night(s) on which the assumed epochs compete for the same time of night.')
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(60 + 36 * len(rows), 300))
    keys = [r.key for r in requests]
    choice = st.selectbox('Programme', keys, format_func=lambda k: f'{k}: {next(r for r in requests if r.key == k).title[:90]}', key='pl_admin_project')
    req = next(r for r in requests if r.key == choice)
    project_view(req, tables, plans, grid, prefix='pla', sim=sim, folder=folder, admin=True)
    st.subheader('One night')
    night = st.date_input('Night to inspect (local date of the evening)', value=conflicts[0]['date'] if conflicts else phase2.PROGRAM_START,
                          min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, key='pl_night')
    st.plotly_chart(night_figure(plans, grid, (night - phase2.PROGRAM_START).days), width='stretch', key='pl_night_fig', config={'displaylogo': False})


# ------------------------------------------------------------------ key guessing
def client_ip() -> str:
    """The visitor's address: the first X-Forwarded-For entry behind nginx, else the socket's."""
    try:
        forwarded = str(st.context.headers.get('X-Forwarded-For', '') or '').split(',')[0].strip()
        return forwarded or str(st.context.ip_address or '?')
    except Exception:                                # no request context (tests, bare mode)
        return '?'


def key_scopes(pid: str) -> list[str]:
    """What a wrong key counts against: the visitor's address, and the Project ID when one is given."""
    scopes = [f'ip:{client_ip()}']
    pid_norm = trigger.safe_project_id(pid)
    if pid_norm:
        scopes.append(f'pid:{pid_norm}')
    return scopes


def count_wrong_key(scopes: list[str], pid: str, key: str) -> None:
    """Record a wrong key once per distinct (ID, key) of this session: the field reports while typing, so a
    PI who types slowly must not use up the free attempts with the prefixes of their own key."""
    if len(key) < 4:
        return
    tried = st.session_state.setdefault('pl_tried', set())
    if (pid, key) in tried:
        return
    tried.add((pid, key))
    keyguard.GUARD.failed(*scopes)


# ------------------------------------------------------------------ main
def main():
    feedback_note()
    st.title('7DT Schedule')
    cfg = live_config()
    key, sun_alt = site_key(cfg['site']), cfg['constraints'].sun_alt
    warm_program_data(key, sun_alt)
    folder = phase2.requests_dir()
    signature = phase2.folder_signature(folder)
    if not signature:
        st.error('No request files are registered on this server.')
        return
    try:                                                  # start loading the precomputed simulation before anyone logs in
        warm_simulation(str(folder), signature, projects_signature(), schedule_signature(str(folder)), key, sun_alt, fixed_signature())
    except Exception:
        pass
    c1, c2 = st.columns(2)
    with c1:
        pid = live_text_input('Project ID', key='pl_pid', placeholder='e.g. 26007', help='Your 5-digit Project ID')
    with c2:
        pkey = live_text_input('Project key', key='pl_key', password=True, help='The key you received with your allocation')
    st.caption(f'Open-time programme {phase2.PROGRAM_START} to {phase2.PROGRAM_END} · dates are local nights (evening date), times UTC · {saved_text(folder)}.')
    note = read_operator_note(folder)
    if note:
        st.info("**Operator's note**\n\n" + note)
    if not pkey.strip():
        st.info('Enter your Project ID and key to see how your targets are planned.')
        return
    projects = trigger.load_projects()
    if projects is None:                            # never fall open on the public server: no list, no schedule
        st.error('The project list (data/projects.json) is not configured on this server, so no schedule can be shown.')
        return
    admin = phase2.admin_key()
    scopes = key_scopes(pid)
    wait = keyguard.GUARD.wait(*scopes)
    if wait > 0:
        st.error(f'Too many wrong keys: try again in {math.ceil(wait)} s.')
        return
    is_admin = bool(admin) and pkey.strip() == admin
    if not is_admin:
        ok, msg, _, _ = trigger.check_project_entry(pid, pkey.strip(), projects)
        if not ok:
            count_wrong_key(scopes, pid, pkey.strip())
            st.error(msg)
            return
    keyguard.GUARD.succeeded(*scopes)
    projects_sig = projects_signature()
    with st.spinner('Loading the schedule...'):
        data = program_data(key, sun_alt)
        tables = night_tables(str(folder), signature, projects_sig, key, sun_alt)
        requests = load_requests_cached(str(folder), signature, projects_sig)
        plans = saved_plans(str(folder), signature, projects_sig, schedule_signature(str(folder)), key, sun_alt)
    with st.spinner('Simulating the programme (once per saved schedule)...'):
        try:
            sim = simulated_saved(str(folder), signature, projects_sig, schedule_signature(str(folder)), key, sun_alt, fixed_signature())
        except Exception:                             # the plan is still shown when the simulation fails
            sim = None
    if sim is not None:
        plans = with_expected(plans, sim['expected'])      # diamonds and epoch counts follow the simulation
    grid = data['grid']
    pid_norm = trigger.safe_project_id(pid)
    if is_admin and not pid_norm:                   # admin without a Project ID: everything; with one: exactly what that PI sees
        admin_view(requests, tables, plans, grid, sim, folder)
        return
    if is_admin:
        st.caption(f'Admin view of project {pid_norm}: this is what its PI sees.')
    req = next((r for r in requests if r.project_id == pid_norm), None)
    if req is None:
        st.warning(f'No Phase-2 request of project {pid_norm} is registered yet.')
        return
    project_view(req, tables, plans, grid, sim=sim, folder=folder, admin=is_admin)
    st.subheader('Your nights among the other programmes')
    mode = 'alt' if str(st.session_state.get('pl_yaxis', 'Max')).startswith('Max') else 'hours'
    st.plotly_chart(occupancy_figure(req, tables, plans, mode), width='stretch', key='pl_occ_fig', config={'displaylogo': False})
    st.caption(('Top row: nights on which at least one of your targets reaches 30° (light green) or transits during the night (dark green), same colours as the visibility bars above; '
                if mode == 'alt' else
                'Top row: nights on which at least one of your targets is observable (light green > 0 h, dark green ≥ 3 h), same colours as the visibility bars above; ')
               + 'diamonds = your assumed epochs. Bottom row: nights on which another programme has an epoch; which programme and for how long is not shown.')


main()
