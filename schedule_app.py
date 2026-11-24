"""
7DT Scheduling (operator page): the Phase-2 request files of the open-time programme, one project at a
time with its targets' visibility over the programme period, the planned observing window and hours per
night of every project, and the nights on which the planned projects compete for the same time of night.
Run with:  streamlit run schedule_app.py
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sevendt_calc import phase2, trigger
from sevendt_calc import visibility as vis
from ui_common import feedback_note, live_config, season_grid_for, site_key

st.set_page_config(page_title='7DT Scheduling', page_icon='🗓️', layout='wide')

MIN_ALT = 30.0
COLORS = {'window': 'rgba(31,119,180,0.20)', 'vis_light': 'rgba(44,160,44,0.14)', 'vis_dark': 'rgba(44,160,44,0.34)',
          'moon': 'rgba(255,187,0,0.22)', 'plan': 'rgba(214,39,40,0.9)', 'line': '#1f1f1f'}
STATUS_COLORS = {'OK': '#2ca02c', 'off': '#bdbdbd', 'no time requested': '#9e9e9e', 'never observable': '#ff7f0e'}


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
    cfg = live_config()
    return phase2.load_requests(folder, trigger.load_projects(), cfg)


@st.cache_resource(show_spinner=False)
def night_tables(folder: str, signature: tuple, projects_sig: tuple, key: tuple, sun_alt: float) -> dict:
    data = program_data(key, sun_alt)
    return {req.key: phase2.night_table(req, data['grid'], data['moon'], MIN_ALT) for req in load_requests_cached(folder, signature, projects_sig)}


# ------------------------------------------------------------------ helpers
def runs(mask: np.ndarray) -> list:
    """(first, last) index pairs of the True runs of a boolean array."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def vrects(dates: list, mask: np.ndarray, color: str, layer: str = 'below') -> list:
    half = timedelta(hours=12)
    return [dict(type='rect', xref='x', yref='paper', x0=dates[a] - half, x1=dates[b] + half, y0=0, y1=1, fillcolor=color, line=dict(width=0), layer=layer)
            for a, b in runs(mask)]


def plan_rect(settings: phase2.PlanSettings) -> dict:
    return dict(type='rect', xref='x', yref='paper', x0=datetime.combine(settings.start, datetime.min.time()) - timedelta(hours=12),
                x1=datetime.combine(settings.end, datetime.min.time()) + timedelta(hours=12), y0=0, y1=1, fillcolor='rgba(0,0,0,0)',
                line=dict(color=COLORS['plan'], width=2, dash='dot'), layer='above')


def legend_traces(show_moon: bool, mode: str) -> list:
    items = [('OBS_WINDOW (requested)', COLORS['window']),
             ('max altitude ≥ 30°' if mode == 'alt' else 'observable (> 0 h)', COLORS['vis_light']),
             ('transit during the night' if mode == 'alt' else f'≥ {phase2.GOOD_HOURS:g} h observable', COLORS['vis_dark'])]
    if show_moon:
        items.append(('Moon suits the sky background', COLORS['moon']))
    out = [go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=c), name=n, hoverinfo='skip') for n, c in items]
    out.append(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=COLORS['plan'], width=2, dash='dot'), name='planned window', hoverinfo='skip'))
    return out


def utc_text(grid: vis.SeasonGrid, i: int, hours: float) -> str:
    t = datetime(2000, 1, 1) + timedelta(days=grid.start_jd[i] - 2451544.5 + hours / 24.0)
    return t.strftime('%H:%M')


def slot_texts(table: phase2.NightTable, grid: vis.SeasonGrid, j: int) -> list:
    out = []
    for i in range(len(table.dates)):
        parts = [f'{utc_text(grid, i, s)}–{utc_text(grid, i, e)} UT' for s, e in zip(table.starts[i, j], table.ends[i, j]) if not np.isnan(s)]
        out.append(', '.join(parts) if parts else 'below 30° all night')
    return out


def target_figure(table: phase2.NightTable, grid: vis.SeasonGrid, j: int, mode: str, show_moon: bool, settings: phase2.PlanSettings | None, title: str,
                  height: int = 300) -> go.Figure:
    dates = [datetime.combine(d, datetime.min.time()) for d in table.dates]
    y = table.max_alt[:, j] if mode == 'alt' else table.hours[:, j]
    light = (table.max_alt[:, j] >= MIN_ALT) if mode == 'alt' else (table.hours[:, j] > 0)
    dark = table.transit[:, j] if mode == 'alt' else (table.hours[:, j] >= phase2.GOOD_HOURS)
    shapes = vrects(dates, table.requested[:, j], COLORS['window']) + vrects(dates, light, COLORS['vis_light']) + vrects(dates, dark, COLORS['vis_dark'])
    if show_moon and not table.moon_ok.all():
        shapes += vrects(dates, table.moon_ok, COLORS['moon'])
    if settings is not None and settings.enabled:
        shapes.append(plan_rect(settings))
    custom = [[float(h), float(a), float(m), s] for h, a, m, s in zip(table.hours[:, j], table.max_alt[:, j], table.moon, slot_texts(table, grid, j))]
    fig = go.Figure(legend_traces(show_moon, mode))
    fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', line=dict(color=COLORS['line'], width=1.6), name=title, customdata=custom,
                             hovertemplate='%{x|%Y-%m-%d}: ' + ('%{y:.1f}°' if mode == 'alt' else '%{y:.2f} h') +
                             '<br>observable %{customdata[0]:.2f} h, max alt %{customdata[1]:.1f}°<br>%{customdata[3]}<br>Moon %{customdata[2]:.0%}<extra></extra>'))
    ref = MIN_ALT if mode == 'alt' else phase2.GOOD_HOURS
    shapes.append(dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=ref, y1=ref, line=dict(color='#888', width=1, dash='dash')))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), shapes=shapes,
                      yaxis=dict(title='max altitude (°)' if mode == 'alt' else 'observable hours (h)', rangemode='tozero'),
                      xaxis=dict(type='date', range=[dates[0] - timedelta(days=1), dates[-1] + timedelta(days=1)]),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)), hovermode='x unified')
    return fig


def overview_figure(req: phase2.RequestFile, table: phase2.NightTable, mode: str, show_moon: bool, settings: phase2.PlanSettings | None) -> go.Figure:
    dates = [datetime.combine(d, datetime.min.time()) for d in table.dates]
    shapes = vrects(dates, table.requested.any(axis=1), COLORS['window'])
    if show_moon and not table.moon_ok.all():
        shapes += vrects(dates, table.moon_ok, COLORS['moon'])
    if settings is not None and settings.enabled:
        shapes.append(plan_rect(settings))
    fig = go.Figure()
    for j, t in enumerate(req.targets):
        y = table.max_alt[:, j] if mode == 'alt' else table.hours[:, j]
        fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', line=dict(width=1.2), name=t.display,
                                 hovertemplate=t.display + ' %{x|%Y-%m-%d}: ' + ('%{y:.1f}°' if mode == 'alt' else '%{y:.2f} h') + '<extra></extra>'))
    ref = MIN_ALT if mode == 'alt' else phase2.GOOD_HOURS
    shapes.append(dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=ref, y1=ref, line=dict(color='#888', width=1, dash='dash')))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), shapes=shapes, hovermode='closest',
                      yaxis=dict(title='max altitude (°)' if mode == 'alt' else 'observable hours (h)', rangemode='tozero'),
                      xaxis=dict(type='date', range=[dates[0] - timedelta(days=1), dates[-1] + timedelta(days=1)]),
                      legend=dict(font=dict(size=10)), showlegend=len(req.targets) <= 30)
    return fig


def plan_figure(plan: phase2.Plan, table: phase2.NightTable) -> go.Figure:
    dates = [datetime.combine(d, datetime.min.time()) for d in table.dates]
    cum = np.cumsum(plan.planned_h)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=plan.available_h, name='observable (h / night)', marker_color='rgba(44,160,44,0.35)',
                         hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} h observable<extra></extra>'))
    fig.add_trace(go.Bar(x=dates, y=plan.planned_h, name='planned (h / night)', marker_color='rgba(214,39,40,0.8)',
                         hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} h planned<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=cum, name='planned, cumulative (h)', yaxis='y2', mode='lines', line=dict(color='#1f77b4', width=2),
                             hovertemplate='%{x|%Y-%m-%d}: %{y:.1f} h so far<extra></extra>'))
    fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[plan.required_h, plan.required_h], name='requested total (h)', yaxis='y2', mode='lines',
                             line=dict(color='#1f77b4', width=1, dash='dash'), hoverinfo='skip'))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), barmode='overlay', hovermode='x unified',
                      yaxis=dict(title='hours per night', rangemode='tozero'),
                      yaxis2=dict(title='cumulative hours', overlaying='y', side='right', rangemode='tozero', showgrid=False),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
    return fig


def gantt_figure(requests: list, plans: dict, conflicts: list) -> go.Figure:
    fig = go.Figure()
    labels, bases, lengths, colors, texts = [], [], [], [], []
    for req in requests:
        plan = plans[req.key]
        s = plan.settings
        labels.append(f'{req.key} {req.lead_pi}')
        bases.append(datetime.combine(s.start, datetime.min.time()))
        lengths.append(max((s.end - s.start).days + 1, 1) * 86400000)
        color = STATUS_COLORS.get(plan.status, '#d62728')
        colors.append(color if s.enabled else '#bdbdbd')
        texts.append(f'{req.key} {req.lead_pi}<br>{s.start} to {s.end}, {s.hours_per_night:g} h/night on {plan.nights} nights'
                     f'<br>planned {plan.planned_total:.1f} h of {plan.required_h:.1f} h requested: {plan.status}')
    fig.add_trace(go.Bar(y=labels, x=lengths, base=bases, orientation='h', marker_color=colors, text=None, hovertext=texts, hoverinfo='text', name='planned window'))
    if conflicts:
        xs = [datetime.combine(c['date'], datetime.min.time()) + timedelta(hours=12) for c in conflicts]
        fig.add_trace(go.Scatter(x=xs, y=['conflicts'] * len(xs), mode='markers', marker=dict(color='#d62728', size=7, symbol='x'), name='night with a conflict',
                                 hovertext=[f"{c['date']}: {', '.join(c['projects'])}<br>short: " + ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items()) for c in conflicts],
                                 hoverinfo='text'))
    d0, d1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()), datetime.combine(phase2.PROGRAM_END, datetime.min.time())
    fig.update_layout(height=max(260, 28 * (len(labels) + 2)), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type='date', range=[d0, d1 + timedelta(days=1)]),
                      yaxis=dict(autorange='reversed'), showlegend=False)
    return fig


def night_figure(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid, i: int) -> go.Figure:
    """Time-of-night bars of every planned project on night i: light = a target above 30° (inside its timed window, if any), red = the planned hours as packed."""
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
                                     hovertext=f'{t.label}: {utc_text(grid, i, lo)}–{utc_text(grid, i, hi)} UT above {MIN_ALT:g}°', hoverinfo='text', showlegend=False))
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
def projects_table(requests: list, plans: dict) -> pd.DataFrame:
    rows = []
    for req in requests:
        p = plans[req.key]
        s = p.settings
        rows.append({'Project': req.key, 'PI': req.lead_pi, 'Request type': req.observation, 'Mode': req.mode, 'Submode': req.submodes,
                     'Sky brightness': req.sky_background, 'Allocated (h)': req.allocated_hours, 'Requested (h)': round(req.total_h, 1),
                     'Targets': len(req.targets), 'Requested window': f'{req.requested_start} to {req.requested_end}' + (' (timed)' if req.timed else ''),
                     'Planned start': s.start.isoformat() if s.enabled else '', 'Planned end': s.end.isoformat() if s.enabled else '',
                     'h / night': s.hours_per_night if s.enabled else None, 'Nights': p.nights, 'Planned (h)': round(p.planned_total, 1), 'Status': p.status,
                     'Issues': len(req.all_issues)})
    return pd.DataFrame(rows)


def select_project(requests: list, df: pd.DataFrame) -> phase2.RequestFile:
    ss = st.session_state
    keys = [r.key for r in requests]
    event = st.dataframe(df, hide_index=True, on_select='rerun', selection_mode='single-row', key='sch_table', width='stretch',
                         column_config={'Allocated (h)': st.column_config.NumberColumn(format='%.1f'), 'h / night': st.column_config.NumberColumn(format='%.1f')})
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


def targets_dataframe(req: phase2.RequestFile) -> pd.DataFrame:
    rows = []
    for t in req.targets:
        rows.append({'#': t.index, 'Target': t.display, 'Type': t.target_type, 'RA (°)': None if t.ra is None else round(t.ra, 4),
                     'Dec (°)': None if t.dec is None else round(t.dec, 4), 'Mode': t.mode_text, 'Exptime (s)': t.exptime, 'Frames': t.nframes,
                     'Orders': t.n_orders, 'Exposure (h)': round(t.exposure_s / 3600, 2), 'Total (h)': round(t.total_s / 3600, 2),
                     'OBS_WINDOW': t.window_text, 'Note': t.note, 'Issues': '; '.join(t.issues)})
    return pd.DataFrame(rows)


def project_section(req: phase2.RequestFile, table: phase2.NightTable, grid: vis.SeasonGrid, plans: dict) -> None:
    ss = st.session_state
    st.subheader(f'{req.key}: {req.title}')
    st.caption(f'PI {req.pi} ({req.emails}) · {req.observation} · {req.mode} {req.submodes} · sky background {req.sky_background} · file {req.name}')
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
    st.dataframe(targets_dataframe(req), hide_index=True, width='stretch', height=min(60 + 36 * len(req.targets), 420),
                 column_config={'RA (°)': st.column_config.NumberColumn(format='%.4f'), 'Dec (°)': st.column_config.NumberColumn(format='%.4f')})

    st.markdown('**Nightly plan**')
    st.plotly_chart(plan_figure(plan, table), width='stretch', key='sch_plan_fig', config={'displaylogo': False})

    st.markdown('**Visibility of the targets**')
    c1, c2, c3 = st.columns([1.6, 1, 1.4])
    mode = c1.radio('y axis', ['Maximum altitude of the night', 'Observable hours per night'], horizontal=True, key='sch_yaxis')
    mode = 'alt' if mode.startswith('Max') else 'hours'
    show_moon = c2.checkbox('Sky-brightness window', value=True, key='sch_show_moon', help='Nights whose Moon illumination suits the proposal (Dark ≤ 0.2, Gray 0.2–0.8, Bright ≥ 0.8).')
    labels = [t.display for t in req.targets]
    default = labels if len(labels) <= 6 else labels[:3]
    chosen = c3.multiselect('Targets to plot individually', labels, default=default, key=f'sch_targets_{req.key}')
    if len(req.targets) > 1:
        st.plotly_chart(overview_figure(req, table, mode, show_moon, plan.settings), width='stretch', key='sch_overview', config={'displaylogo': False})
    for j, t in enumerate(req.targets):
        if t.display in chosen and t.has_coordinates:
            title = f'{t.display}  ({t.mode_text}, {t.exptime:g} s × {t.nframes} × {t.n_orders}, {t.total_s / 3600:.1f} h)' if t.exptime and t.nframes else t.display
            st.markdown(f'**{title}**' + (f' — {t.window_text}' if len(t.window_text) < 120 else f' — {len(t.windows)} windows'))
            st.plotly_chart(target_figure(table, grid, j, mode, show_moon, plan.settings, t.display), width='stretch', key=f'sch_fig_{req.key}_{j}', config={'displaylogo': False})


def all_projects_section(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid) -> None:
    st.subheader('All projects')
    conflicts = phase2.night_conflicts(plans, list(grid.dates))
    st.plotly_chart(gantt_figure(requests, plans, conflicts), width='stretch', key='sch_gantt', config={'displaylogo': False})
    if conflicts:
        st.error(f'{len(conflicts)} night(s) on which the planned projects compete for the same time of night (least-slack packing, one target at a time).')
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
               f'dates are local nights (evening date); times UTC. Click a project row or pick it below.')

    st.subheader('Projects')
    req = select_project(requests, projects_table(requests, plans))
    project_section(req, tables[req.key], grid, plans)
    # the table above was drawn with the old settings: redraw it when a widget changed something
    if ss.get('sch_table_version') != repr([(k, v.as_dict()) for k, v in ss['sch_settings'].items()]):
        ss['sch_table_version'] = repr([(k, v.as_dict()) for k, v in ss['sch_settings'].items()])
        st.rerun()
    all_projects_section(requests, tables, plans, grid)
    export_section(requests, plans, folder)


main()
