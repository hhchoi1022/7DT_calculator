"""
7DT Scheduling (operator page): every target requested in the Phase-2 files of the open-time programme in
one table, the projects with their planned observing window and hours per night, visibility of any chosen
targets over the programme period, and the nights on which the planned projects compete for the same time
of night.
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
from plotly.subplots import make_subplots

from sevendt_calc import phase2, trigger
from sevendt_calc import visibility as vis
from ui_common import feedback_note, live_config, season_grid_for, site_key

st.set_page_config(page_title='7DT Scheduling', page_icon='🗓️', layout='wide')

MIN_ALT = 30.0
PLAN_COLOR = 'rgba(214,39,40,0.9)'
CAT_INDEX = {name: k for k, (name, _, _) in enumerate(phase2.STRIP_CATEGORIES)}
CAT_SCALE = [[k / len(phase2.STRIP_CATEGORIES), c] for k, (_, _, c) in enumerate(phase2.STRIP_CATEGORIES) for _ in (0, 1)]
CAT_SCALE = [[(k // 2 + (k % 2)) / len(phase2.STRIP_CATEGORIES), c] for k, (_, c) in enumerate(CAT_SCALE)]


# ------------------------------------------------------------------ cached data
@st.cache_resource(show_spinner=False)
def program_data(key: tuple, sun_alt: float) -> dict:
    """The twilight grid and Moon illumination of every night of the programme (about 6 s, once per server)."""
    cfg = live_config()
    grid = season_grid_for(cfg['site'], cfg['constraints'], phase2.PROGRAM_START, phase2.PROGRAM_NIGHTS)
    return {'grid': grid, 'moon': vis.moon_illumination_nights(grid)}


@st.cache_resource(show_spinner=False)
def warm_program_data(key: tuple, sun_alt: float) -> bool:
    import threading as _threading
    _threading.Thread(target=lambda: program_data(key, sun_alt), name='program-grid-warmup', daemon=True).start()
    return True


def projects_signature() -> tuple:
    p = trigger.PROJECTS_JSON
    return (str(p), p.stat().st_mtime_ns) if p.is_file() else (str(p), 0)


@st.cache_resource(show_spinner=False)
def load_requests_cached(folder: str, signature: tuple, projects_sig: tuple) -> list:
    return phase2.load_requests(folder, trigger.load_projects(), live_config())


@st.cache_resource(show_spinner=False)
def night_tables(folder: str, signature: tuple, projects_sig: tuple, key: tuple, sun_alt: float) -> dict:
    data = program_data(key, sun_alt)
    return {req.key: phase2.night_table(req, data['grid'], data['moon'], MIN_ALT) for req in load_requests_cached(folder, signature, projects_sig)}


# ------------------------------------------------------------------ figure helpers
def night_dates(table: phase2.NightTable) -> list:
    return [datetime.combine(d, datetime.min.time()) for d in table.dates]


def plan_shape(settings: phase2.PlanSettings, yref: str = 'paper', y0=0, y1=1) -> dict:
    return dict(type='rect', xref='x', yref=yref, x0=datetime.combine(settings.start, datetime.min.time()) - timedelta(hours=12),
                x1=datetime.combine(settings.end, datetime.min.time()) + timedelta(hours=12), y0=y0, y1=y1, fillcolor='rgba(0,0,0,0)',
                line=dict(color=PLAN_COLOR, width=2, dash='dot'), layer='above')


def utc_text(grid: vis.SeasonGrid, i: int, hours: float) -> str:
    t = datetime(2000, 1, 1) + timedelta(days=grid.start_jd[i] - 2451544.5 + hours / 24.0)
    return t.strftime('%H:%M')


def slot_text(table: phase2.NightTable, grid: vis.SeasonGrid, i: int, j: int) -> str:
    parts = [f'{utc_text(grid, i, s)}–{utc_text(grid, i, e)} UT' for s, e in zip(table.starts[i, j], table.ends[i, j]) if not np.isnan(s)]
    return ', '.join(parts) if parts else f'below {MIN_ALT:g}° all night'


def category_legend(names: list, fig: go.Figure, row=None, col=None) -> None:
    for name in names:
        k = CAT_INDEX[name]
        _, label, color = phase2.STRIP_CATEGORIES[k]
        tr = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color, line=dict(color='#999', width=0.5)), name=label, hoverinfo='skip')
        if row:                      # (an if statement, not a conditional expression: Streamlit's magic would st.write() the figure)
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)


def heatmap(dates: list, labels: list, cats: np.ndarray, hover: list) -> go.Heatmap:
    return go.Heatmap(x=dates, y=labels, z=cats, zmin=-0.5, zmax=len(phase2.STRIP_CATEGORIES) - 0.5, colorscale=CAT_SCALE, showscale=False,
                      xgap=0, ygap=2, hovertext=hover, hoverinfo='text')


def visibility_figure(items: list, grid: vis.SeasonGrid, mode: str, show_moon: bool, plans: dict) -> go.Figure:
    """
    items: [(req, table, j)] chosen targets. Top: nightly maximum altitude or observable hours of every target
    (and the Moon illumination); bottom: one strip per target (observable hours in greens, the requested
    OBS_WINDOW in blues, Moon-unsuitable nights in yellow); the planned window of each project dotted red.
    """
    n = len(items)
    strip_h = 22 * n + 30
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[320, strip_h])
    dates = night_dates(items[0][1]) if items else []
    for req, table, j in items:
        t = req.targets[j]
        label = f'{req.key} {t.display}'
        y = table.max_alt[:, j] if mode == 'alt' else table.hours[:, j]
        custom = [[float(h), float(a), float(m), slot_text(table, grid, i, j)] for i, (h, a, m) in enumerate(zip(table.hours[:, j], table.max_alt[:, j], table.moon))]
        fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', line=dict(width=1.5), name=label, customdata=custom, legendgroup=label,
                                 hovertemplate=label + ' %{x|%Y-%m-%d}: ' + ('%{y:.1f}°' if mode == 'alt' else '%{y:.2f} h') +
                                 '<br>observable %{customdata[0]:.2f} h, max alt %{customdata[1]:.1f}°<br>%{customdata[3]}<br>Moon %{customdata[2]:.0%}<extra></extra>'), row=1, col=1)
    if show_moon and items:
        table = items[0][1]
        fig.add_trace(go.Scatter(x=dates, y=table.moon, mode='lines', line=dict(color='#999', width=1, dash='dot'), name='Moon illumination', yaxis='y3',
                                 hovertemplate='Moon %{y:.0%}<extra></extra>'), row=1, col=1)
        fig.data[-1].update(yaxis='y3')
    labels, cats, hover = [], [], []
    for req, table, j in items:
        t = req.targets[j]
        label = f'{req.key} {t.display}'
        c = phase2.strip_categories(table.hours[:, j], table.requested[:, j], table.moon_ok if show_moon else None)
        labels.append(label)
        cats.append(c)
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h observable, max alt {a:.1f}°' + (', requested' if w else '') + f', Moon {m:.0%}' + ('' if ok else ' (unsuitable)')
                      for d, h, a, w, m, ok in zip(table.dates, table.hours[:, j], table.max_alt[:, j], table.requested[:, j], table.moon, table.moon_ok)])
    if items:
        fig.add_trace(heatmap(dates, labels, np.array(cats), hover), row=2, col=1)
        category_legend(['obs', 'good', 'win_obs', 'win_good', 'win_none'] + (['moon'] if show_moon else []), fig, row=2, col=1)
    shapes = []
    seen = set()
    for req, table, j in items:
        plan = plans.get(req.key)
        if plan is not None and plan.settings.enabled and req.key not in seen:
            seen.add(req.key)
            shapes.append(plan_shape(plan.settings, yref='paper', y0=0, y1=1))
    ref = MIN_ALT if mode == 'alt' else phase2.GOOD_HOURS
    shapes.append(dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=ref, y1=ref, line=dict(color='#888', width=1, dash='dash')))
    if shapes:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=PLAN_COLOR, width=2, dash='dot'), name='planned obsdate_window', hoverinfo='skip'), row=1, col=1)
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=320 + strip_h + 90, margin=dict(l=10, r=10, t=10, b=10), shapes=shapes, hovermode='closest',
                      yaxis=dict(title='max altitude (°)' if mode == 'alt' else 'observable hours (h)', rangemode='tozero'),
                      yaxis3=dict(overlaying='y', side='right', range=[0, 1], tickformat='.0%', showgrid=False, title='Moon'),
                      yaxis2=dict(autorange='reversed', tickfont=dict(size=10)), xaxis=dict(type='date', range=[x0, x1]), xaxis2=dict(type='date', range=[x0, x1]),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def projects_figure(requests: list, tables: dict, plans: dict, conflicts: list, show_moon: bool) -> go.Figure:
    """One strip per project: hours during which any of its targets is above the limit (greens), the planned obsdate_window (blues), conflicts in red."""
    labels, cats, hover = [], [], []
    first = tables[requests[0].key]
    dates = night_dates(first)
    for req in requests:
        table, plan = tables[req.key], plans[req.key]
        s = plan.settings
        window = np.array([s.enabled and s.start <= d <= s.end for d in table.dates], bool)
        c = phase2.strip_categories(table.hours_any, window, table.moon_ok if (show_moon and s.use_moon) else None)
        label = f'{req.key} {req.lead_pi}'
        labels.append(label)
        cats.append(c)
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h with a target up' + (', in the planned window' if w else '') + (f', {p:.2f} h planned' if p > 0 else '') + f', Moon {m:.0%}'
                      for d, h, w, p, m in zip(table.dates, table.hours_any, window, plan.planned_h, table.moon)])
    conflict_row = np.zeros(len(dates), int)
    conflict_hover = [''] * len(dates)
    for c in conflicts:
        i = (c['date'] - phase2.PROGRAM_START).days
        if 0 <= i < len(dates):
            conflict_row[i] = CAT_INDEX['conflict']
            conflict_hover[i] = f"{c['date']}: {', '.join(c['projects'])}; short " + ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items())
    labels.append('conflicts')
    cats.append(conflict_row)
    hover.append(conflict_hover)
    fig = go.Figure(heatmap(dates, labels, np.array(cats), hover))
    category_legend(['obs', 'good', 'win_obs', 'win_good', 'win_none'] + (['moon'] if show_moon else []) + ['conflict'], fig)
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=24 * len(labels) + 110, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type='date', range=[x0, x1]),
                      yaxis=dict(autorange='reversed', tickfont=dict(size=11)), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def night_figure(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid, i: int) -> go.Figure:
    """Time-of-night bars of every planned project on night i: light = a target above the limit (inside its timed window, if any), red = the planned hours as packed."""
    base = datetime(2000, 1, 1) + timedelta(days=grid.start_jd[i] - 2451544.5)
    night_len = (grid.end_jd[i] - grid.start_jd[i]) * 24.0
    fig = go.Figure()
    jobs = [(req.key, float(plans[req.key].planned_h[i]), plans[req.key].intervals[i]) for req in requests if plans[req.key].planned_h[i] > 1e-9]
    got, slots = phase2.pack_night_slots(jobs) if jobs else ({}, {})
    rows = []
    for req in requests:
        plan = plans[req.key]
        if plan.planned_h[i] <= 1e-9:
            continue
        label = f'{req.key} {req.lead_pi}'
        rows.append(label)
        table = tables[req.key]
        for j, t in enumerate(req.targets):
            for lo, hi in table.target_intervals(i, j):
                fig.add_trace(go.Bar(y=[label], x=[(hi - lo) * 3600000], base=[base + timedelta(hours=lo)], orientation='h', marker_color='rgba(44,160,44,0.30)',
                                     hovertext=f'{t.display}: {utc_text(grid, i, lo)}–{utc_text(grid, i, hi)} UT above {MIN_ALT:g}°', hoverinfo='text', showlegend=False))
        for lo, hi in slots.get(req.key, []):
            fig.add_trace(go.Bar(y=[label], x=[(hi - lo) * 3600000], base=[base + timedelta(hours=lo)], orientation='h', marker_color='rgba(214,39,40,0.85)', width=0.45,
                                 hovertext=f'{req.key}: {utc_text(grid, i, lo)}–{utc_text(grid, i, hi)} UT planned', hoverinfo='text', showlegend=False))
        demand, have = plan.planned_h[i], got.get(req.key, 0.0)
        fig.add_annotation(x=base + timedelta(hours=night_len), y=label, xanchor='left', showarrow=False, font=dict(size=11, color='#d62728' if have < demand - 1 / 60 else '#2ca02c'),
                           text=f'{demand:.2f} h planned' + ('' if have >= demand - 1 / 60 else f', only {have:.2f} h fit'))
    if not rows:
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, text='No project is planned on this night.')
    fig.update_layout(height=max(220, 34 * (len(rows) + 2)), margin=dict(l=10, r=130, t=10, b=10), barmode='overlay',
                      xaxis=dict(type='date', title='UTC', range=[base - timedelta(minutes=20), base + timedelta(hours=night_len + 3)], tickformat='%H:%M'),
                      yaxis=dict(autorange='reversed'), showlegend=False)
    return fig


# ------------------------------------------------------------------ page sections
def targets_section(requests: list) -> None:
    st.subheader('Requested targets')
    df = pd.DataFrame(phase2.target_rows(requests))
    st.dataframe(df, hide_index=True, width='stretch', height=min(60 + 36 * len(df), 520),
                 column_config={'RA (°)': st.column_config.NumberColumn(format='%.5f'), 'Dec (°)': st.column_config.NumberColumn(format='%.5f'),
                                'dither_arcsec': st.column_config.NumberColumn(format='%.0f')})
    c1, c2 = st.columns([1, 4])
    c1.download_button('Download table (CSV)', data=df.to_csv(index=False).encode(), file_name='7dt_phase2_targets.csv', mime='text/csv', key='sch_targets_csv')
    c2.caption(f'{len(df)} target sets from {len(requests)} files; one row per set (a target split into several sets gives several rows). '
               'Total exposure = exptime × frames × orders of the mode (specall: 2); with overhead = the overhead model for one visit. dither_arcsec from data/projects.json.')


def projects_table(requests: list, plans: dict) -> pd.DataFrame:
    rows = []
    for req in requests:
        p = plans[req.key]
        s = p.settings
        rows.append({'Project': req.key, 'PI': req.lead_pi, 'Request type': req.observation, 'Mode': req.mode, 'Submode': req.submodes,
                     'Sky brightness': req.sky_background, 'Allocated (h)': req.allocated_hours, 'Requested (h)': round(req.total_h, 1),
                     'Targets': len(req.targets), 'dither_arcsec': req.dither_arcsec, 'Requested window': f'{req.requested_start} to {req.requested_end}' + (' (timed)' if req.timed else ''),
                     'Planned start': s.start.isoformat() if s.enabled else '', 'Planned end': s.end.isoformat() if s.enabled else '',
                     'h / night': s.hours_per_night if s.enabled else None, 'Nights': p.nights, 'Planned (h)': round(p.planned_total, 1), 'Status': p.status,
                     'Issues': len(req.all_issues)})
    return pd.DataFrame(rows)


def select_project(requests: list, df: pd.DataFrame) -> phase2.RequestFile:
    ss = st.session_state
    keys = [r.key for r in requests]
    event = st.dataframe(df, hide_index=True, on_select='rerun', selection_mode='single-row', key='sch_table', width='stretch',
                         column_config={'Allocated (h)': st.column_config.NumberColumn(format='%.1f'), 'h / night': st.column_config.NumberColumn(format='%.1f'),
                                        'dither_arcsec': st.column_config.NumberColumn(format='%.0f')})
    rows = list(getattr(getattr(event, 'selection', None), 'rows', []) or [])
    clicked = keys[rows[0]] if rows and rows[0] < len(keys) else None
    if clicked and clicked != ss.get('sch_last_click'):
        ss['sch_last_click'] = clicked
        ss['sch_project'] = clicked
    if ss.get('sch_project') not in keys:
        ss['sch_project'] = keys[0]
    choice = st.selectbox('Project', keys, format_func=lambda k: f'{k}: {next(r for r in requests if r.key == k).title[:90]}', key='sch_project')
    return next(r for r in requests if r.key == choice)


def settings_widgets(req: phase2.RequestFile, current: phase2.PlanSettings) -> phase2.PlanSettings:
    k = req.key
    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1, 1, 1, 1])
    start = c1.date_input('obsdate_window start', value=current.start, min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, key=f'sch_start_{k}')
    end = c2.date_input('obsdate_window end', value=current.end, min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, key=f'sch_end_{k}')
    hours = c3.number_input('Hours per night', value=float(current.hours_per_night), min_value=0.1, max_value=14.0, step=0.5, key=f'sch_hours_{k}',
                            help='At most this many hours of the project per night (less when its targets are up for a shorter time).')
    nights = c4.number_input('Nights (0 = all)', value=int(current.max_nights), min_value=0, max_value=phase2.PROGRAM_NIGHTS, step=1, key=f'sch_nights_{k}',
                             help='Use only this many of the usable nights, spread evenly over the window. 0: every usable night.')
    moon = c5.checkbox('Moon condition', value=current.use_moon, key=f'sch_moon_{k}', help=f'Only nights whose Moon illumination suits "{req.sky_background}".')
    enabled = c6.checkbox('Scheduled', value=current.enabled, key=f'sch_on_{k}', help='Off: the project is left out of the plan and of the conflict check.')
    if end < start:
        st.error('The window ends before it starts.')
        end = start
    return phase2.PlanSettings(start=start, end=end, hours_per_night=float(hours), use_moon=bool(moon), enabled=bool(enabled), max_nights=int(nights))


def project_section(req: phase2.RequestFile, table: phase2.NightTable, plans: dict) -> None:
    ss = st.session_state
    st.markdown(f'**{req.key}: {req.title}**')
    st.caption(f'PI {req.pi} ({req.emails}) · {req.observation} · {req.mode} {req.submodes} · sky background {req.sky_background} · dither {req.dither_arcsec:g}" · file {req.name}')
    for issue in req.all_issues:
        st.warning(issue)
    current = ss['sch_settings'][req.key]
    new = settings_widgets(req, current)
    if new != current:
        ss['sch_settings'][req.key] = new
        plans[req.key] = phase2.make_plan(req, table, new)
    plan = plans[req.key]
    m = st.columns(6)
    m[0].metric('Allocated', f'{req.allocated_hours:.1f} h' if req.allocated_hours is not None else 'n/a')
    m[1].metric('Requested exposure', f'{req.exposure_h:.1f} h')
    m[2].metric('Requested with overheads', f'{req.total_h:.1f} h')
    m[3].metric('Planned', f'{plan.planned_total:.1f} h')
    m[4].metric('Nights used', f'{plan.nights}')
    m[5].metric('Status', plan.status)
    if req.allocated_hours is not None and req.total_h > req.allocated_hours + 1e-6:
        st.error(f'The request ({req.total_h:.1f} h with overheads) exceeds the allocation ({req.allocated_hours:.1f} h).')


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
    st.caption(f'Top: every chosen target per night; dashed grey line at {MIN_ALT:g}° / {phase2.GOOD_HOURS:g} h; dotted red: the planned obsdate_window of the chosen projects. '
               'Strips: greens = observable hours of the target (light > 0 h, dark ≥ 3 h), blues = the same inside its requested OBS_WINDOW, grey = requested but never above the limit.')


def all_projects_section(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid) -> None:
    st.subheader('All projects')
    conflicts = phase2.night_conflicts(plans, list(grid.dates))
    show_moon = st.checkbox('Mark nights that fail the Moon condition', value=False, key='sch_all_moon')
    st.plotly_chart(projects_figure(requests, tables, plans, conflicts, show_moon), width='stretch', key='sch_projects_fig', config={'displaylogo': False})
    st.caption('Per project and night: greens = hours during which at least one of its targets is above the limit (light > 0 h, dark ≥ 3 h); '
               'blues = the same inside the planned obsdate_window; grey = planned window on a night without a target up.')
    if conflicts:
        st.error(f'{len(conflicts)} night(s) on which the planned projects compete for the same time of night (least-slack packing, one project at a time).')
        rows = [{'Night (local)': c['date'].isoformat(), 'Projects': ', '.join(c['projects']), 'Planned (h)': round(c['demand_h'], 2),
                 'Fits (h)': round(c['scheduled_h'], 2), 'Short': ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items())} for c in conflicts]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(60 + 36 * len(rows), 330))
    else:
        st.success('No night on which the planned projects compete for the same time of night.')
    busy = sum(plans[r.key].planned_h for r in requests)
    nights = (busy > 0).sum()
    st.caption(f'Planned nights: {int(nights)} of {len(grid.dates)}; planned hours in total: {busy.sum():.1f} h.')
    default = conflicts[0]['date'] if conflicts else next((d for d, b in zip(grid.dates, busy) if b > 0), phase2.PROGRAM_START)
    night = st.date_input('Night to inspect (local date of the evening)', value=default, min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, key='sch_night')
    i = (night - phase2.PROGRAM_START).days
    st.caption('Light bars: the project\'s targets above the altitude limit (inside their timed windows, if any). Red: the planned hours, packed one project at a time, least slack first.')
    st.plotly_chart(night_figure(requests, tables, plans, grid, i), width='stretch', key='sch_night_fig', config={'displaylogo': False})


def export_section(requests: list, plans: dict, folder: Path) -> None:
    ss = st.session_state
    st.subheader('Save and export')
    c1, c2, c3, c4 = st.columns(4)
    saved = phase2.load_schedule(folder)
    changed = {k: v for k, v in ss['sch_settings'].items() if saved.get(k) != v}
    if c1.button('Save schedule', type='primary', key='sch_save', help=f'Writes {folder / phase2.SCHEDULE_FILE}'):
        path = phase2.save_schedule(folder, ss['sch_settings'])
        st.caption(f'Saved to {path}')
        changed = {}
    elif changed:
        c1.caption(f'{len(changed)} project(s) changed since the last save.')
    rows = phase2.schedule_rows(requests, plans)
    c2.download_button('Download obsdate_window table (CSV)', data=pd.DataFrame(rows).to_csv(index=False).encode(), file_name='7dt_schedule_targets.csv', mime='text/csv', key='sch_csv')
    payload = {k: v.as_dict() for k, v in ss['sch_settings'].items()}
    c3.download_button('Download schedule (JSON)', data=json.dumps(payload, indent=2).encode(), file_name='7dt_schedule.json', mime='application/json', key='sch_json')
    upload = c4.file_uploader('Load a schedule JSON', type=['json'], key='sch_upload', label_visibility='collapsed')
    if upload is not None and ss.get('sch_uploaded') != upload.file_id:
        try:
            raw = json.loads(upload.getvalue().decode())
            loaded = {str(k): phase2.PlanSettings.from_dict(v) for k, v in raw.items() if isinstance(v, dict) and not str(k).startswith('_')}
        except (ValueError, KeyError) as exc:
            st.error(f'Not a schedule file: {exc}')
        else:
            for k, v in loaded.items():
                if k in ss['sch_settings']:
                    ss['sch_settings'][k] = v
                    for suffix in ('start', 'end', 'hours', 'nights', 'moon', 'on'):
                        ss.pop(f'sch_{suffix}_{k}', None)
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
    if ss.get('sch_signature') != (signature, projects_sig) or 'sch_settings' not in ss:
        saved = phase2.load_schedule(folder)
        ss['sch_settings'] = {req.key: saved.get(req.key) or phase2.default_settings(req, tables[req.key]) for req in requests}
        ss['sch_signature'] = (signature, projects_sig)
    plans = {req.key: phase2.make_plan(req, tables[req.key], ss['sch_settings'][req.key]) for req in requests}
    st.caption(f'{len(requests)} request files in {folder} · programme {phase2.PROGRAM_START} to {phase2.PROGRAM_END} · altitude limit {MIN_ALT:g}° · '
               f'dates are local nights (evening date); times UTC.')

    targets_section(requests)
    st.subheader('Projects')
    st.caption('Tick a row or pick the project below to set its obsdate_window and hours per night.')
    req = select_project(requests, projects_table(requests, plans))
    project_section(req, tables[req.key], plans)
    if ss.get('sch_table_version') != repr([(k, v.as_dict()) for k, v in ss['sch_settings'].items()]):       # the table above was drawn with the old settings
        ss['sch_table_version'] = repr([(k, v.as_dict()) for k, v in ss['sch_settings'].items()])
        st.rerun()
    visibility_section(requests, tables, plans, grid, req)
    all_projects_section(requests, tables, plans, grid)
    export_section(requests, plans, folder)


main()
