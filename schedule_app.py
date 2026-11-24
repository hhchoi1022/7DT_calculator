"""
7DT Scheduling (operator page): every target requested in the Phase-2 files of the open-time programme in
one editable table (obs_startdate, obs_enddate, frames per epoch, number of epochs; a target set may be split
into parts with their own values), the check that each
target is up long enough inside its range for the entered plan, a load-balanced assumption of which nights
the automatic scheduler would use, visibility of any chosen targets over the programme period, and the
nights on which the assumed epochs compete for the same time of night.
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
from plotly.subplots import make_subplots

from sevendt_calc import phase2, trigger
from sevendt_calc import visibility as vis
from ui_common import feedback_note, live_config, season_grid_for, site_key

st.set_page_config(page_title='7DT Scheduling', page_icon='🗓️', layout='wide')

MIN_ALT = 30.0
WINDOW_COLOR = '#d62728'                  # planned range: a red line through the middle of a strip, epochs as dots
REQUEST_COLOR = '#555555'                 # requested OBS_WINDOW: a thin grey line below the red one
CAT_INDEX = {name: k for k, (name, _, _) in enumerate(phase2.STRIP_CATEGORIES)}
CAT_SCALE = [[k / len(phase2.STRIP_CATEGORIES), c] for k, (_, _, c) in enumerate(phase2.STRIP_CATEGORIES) for _ in (0, 1)]
CAT_SCALE = [[(k // 2 + (k % 2)) / len(phase2.STRIP_CATEGORIES), c] for k, (_, c) in enumerate(CAT_SCALE)]
EDIT_COLUMNS = ['obs_startdate', 'obs_enddate', 'count', 'target_epoch', 'scheduled']
RED, YELLOW = 'background-color: #ffb3b3', 'background-color: #fff3b0'


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


def utc_text(grid: vis.SeasonGrid, i: int, hours: float) -> str:
    t = datetime(2000, 1, 1) + timedelta(days=grid.start_jd[i] - 2451544.5 + hours / 24.0)
    return t.strftime('%H:%M')


def slot_text(table: phase2.NightTable, grid: vis.SeasonGrid, i: int, j: int) -> str:
    parts = [f'{utc_text(grid, i, s)}–{utc_text(grid, i, e)} UT' for s, e in zip(table.starts[i, j], table.ends[i, j]) if not np.isnan(s)]
    return ', '.join(parts) if parts else f'below {MIN_ALT:g}° all night'


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


def category_legend(names: list, fig: go.Figure, row=None, col=None) -> None:
    for name in names:
        _, label, color = phase2.STRIP_CATEGORIES[CAT_INDEX[name]]
        tr = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color, line=dict(color='#999', width=0.5)), name=label, hoverinfo='skip')
        if row:                      # (an if statement, not a conditional expression: Streamlit's magic would st.write() the figure)
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)


def heatmap(dates: list, labels: list, cats: np.ndarray, hover: list) -> go.Heatmap:
    return go.Heatmap(x=dates, y=labels, z=cats, zmin=-0.5, zmax=len(phase2.STRIP_CATEGORIES) - 0.5, colorscale=CAT_SCALE, showscale=False,
                      xgap=0, ygap=2, hovertext=hover, hoverinfo='text')


def line_trace(dates: list, positions: list, masks: list, name: str, color: str, width: float) -> go.Scatter:
    """A horizontal line at the row position (category index, fractional offsets allowed) over the nights of each mask (runs separated by gaps)."""
    xs, ys = [], []
    half = timedelta(hours=12)
    for pos, mask in zip(positions, masks):
        for a, b in runs(np.asarray(mask, bool)):
            xs += [dates[a] - half, dates[b] + half, None]
            ys += [pos, pos, None]
    return go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=width), name=name, hoverinfo='skip', connectgaps=False)


def dot_trace(dates: list, positions: list, masks: list, name: str, hover: list | None = None) -> go.Scatter:
    xs, ys, texts = [], [], []
    for k, (pos, mask) in enumerate(zip(positions, masks)):
        for i in np.flatnonzero(np.asarray(mask, bool)):
            xs.append(dates[i])
            ys.append(pos)
            texts.append(hover[k][i] if hover else '')
    return go.Scatter(x=xs, y=ys, mode='markers', marker=dict(color=WINDOW_COLOR, size=7, symbol='diamond', line=dict(color='white', width=1)), name=name,
                      hovertext=texts, hoverinfo='text' if hover else 'skip')


def visibility_figure(items: list, grid: vis.SeasonGrid, mode: str, show_moon: bool, plans: dict) -> go.Figure:
    """
    items: [(req, table, j)] chosen targets. Top: nightly maximum altitude or observable hours of every target
    (and the Moon illumination); bottom: one strip per target (observable hours in greens, Moon-unsuitable
    nights in yellow), its planned range as a red centre line with the assumed epochs as dots, and the
    requested OBS_WINDOW as a thin grey line below.
    """
    n = len(items)
    strip_h = 26 * n + 30
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
        fig.add_trace(go.Scatter(x=dates, y=table.moon, mode='lines', line=dict(color='#999', width=1, dash='dot'), name='Moon illumination',
                                 hovertemplate='Moon %{y:.0%}<extra></extra>'), row=1, col=1)
        fig.data[-1].update(yaxis='y3')
    labels, cats, hover, ranges, epochs, requested, epoch_hover = [], [], [], [], [], [], []
    for req, table, j in items:
        t = req.targets[j]
        label = f'{req.key} {t.display}'
        parts = [p for p in phase2.plans_of(plans, req, j) if p.settings.enabled]
        n_nights = len(table.dates)
        rng, ep, texts = np.zeros(n_nights, bool), np.zeros(n_nights, bool), [''] * n_nights
        for p in parts:
            rng |= np.array([p.settings.start <= d <= p.settings.end for d in table.dates])
            ep |= p.placed
            for i in np.flatnonzero(p.placed):
                texts[i] = f'{label}: epoch assumed on {table.dates[i]:%Y-%m-%d} ({p.epoch_h:.2f} h' + (f', part {p.part}' if len(parts) > 1 else '') + ')'
        c = phase2.strip_categories(table.hours[:, j], table.moon_ok if show_moon else None)
        labels.append(label)
        cats.append(c)
        ranges.append(rng)
        epochs.append(ep)
        requested.append(table.requested[:, j])
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h observable, max alt {a:.1f}°' + (', requested' if w else '') + f', Moon {m:.0%}' + ('' if ok else ' (unsuitable)')
                      for d, h, a, w, m, ok in zip(table.dates, table.hours[:, j], table.max_alt[:, j], table.requested[:, j], table.moon, table.moon_ok)])
        epoch_hover.append(texts)
    if items:
        fig.add_trace(heatmap(dates, labels, np.array(cats), hover), row=2, col=1)
        fig.add_trace(line_trace(dates, [k + 0.3 for k in range(n)], requested, 'requested OBS_WINDOW', REQUEST_COLOR, 1.5), row=2, col=1)
        fig.add_trace(line_trace(dates, list(range(n)), ranges, 'planned obs_startdate to obs_enddate', WINDOW_COLOR, 4), row=2, col=1)
        fig.add_trace(dot_trace(dates, list(range(n)), epochs, 'assumed epoch', epoch_hover), row=2, col=1)
        category_legend(['obs', 'good'] + (['moon'] if show_moon else []), fig, row=2, col=1)
    ref = MIN_ALT if mode == 'alt' else phase2.GOOD_HOURS
    shapes = [dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=ref, y1=ref, line=dict(color='#888', width=1, dash='dash'))]
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=320 + strip_h + 90, margin=dict(l=10, r=10, t=10, b=10), shapes=shapes, hovermode='closest',
                      yaxis=dict(title='max altitude (°)' if mode == 'alt' else 'observable hours (h)', rangemode='tozero'),
                      yaxis3=dict(overlaying='y', side='right', range=[0, 1], tickformat='.0%', showgrid=False, title='Moon'),
                      yaxis2=dict(autorange='reversed', tickfont=dict(size=10), type='category'), xaxis=dict(type='date', range=[x0, x1]), xaxis2=dict(type='date', range=[x0, x1]),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def projects_figure(requests: list, tables: dict, plans: dict, conflicts: list, show_moon: bool) -> go.Figure:
    """One strip per project: hours during which any of its targets is above the limit (greens), the planned ranges of its targets (red line), assumed epochs (dots), conflicts in red."""
    labels, cats, hover, ranges, epochs, epoch_hover = [], [], [], [], [], []
    dates = night_dates(tables[requests[0].key])
    for req in requests:
        table = tables[req.key]
        n = len(table.dates)
        rng, ep, names = np.zeros(n, bool), np.zeros(n, bool), [[] for _ in range(n)]
        for p in plans.values():
            if p.req is not req or not p.settings.enabled:
                continue
            rng |= np.array([p.settings.start <= d <= p.settings.end for d in table.dates])
            ep |= p.placed
            for i in np.flatnonzero(p.placed):
                names[i].append(f'{p.target.display} ({p.epoch_h:.1f} h)')
        c = phase2.strip_categories(table.hours_any, table.moon_ok if show_moon else None)
        label = f'{req.key} {req.lead_pi}'
        labels.append(label)
        cats.append(c)
        ranges.append(rng)
        epochs.append(ep)
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h with a target up' + (', in a planned range' if w else '') + f', Moon {m:.0%}'
                      for d, h, w, m in zip(table.dates, table.hours_any, rng, table.moon)])
        epoch_hover.append([f'{label} {d:%Y-%m-%d}: ' + ', '.join(nm) for d, nm in zip(table.dates, names)])
    conflict_row = np.zeros(len(dates), int)
    conflict_hover = [''] * len(dates)
    for c in conflicts:
        i = (c['date'] - phase2.PROGRAM_START).days
        if 0 <= i < len(dates):
            conflict_row[i] = CAT_INDEX['conflict']
            conflict_hover[i] = f"{c['date']}: " + ', '.join(c['targets']) + '<br>short: ' + ', '.join(f'{k} {v:.2f} h' for k, v in c['short'].items())
    labels.append('conflicts')
    cats.append(conflict_row)
    hover.append(conflict_hover)
    fig = go.Figure(heatmap(dates, labels, np.array(cats), hover))
    fig.add_trace(line_trace(dates, list(range(len(requests))), ranges, 'planned ranges of the targets', WINDOW_COLOR, 4))
    fig.add_trace(dot_trace(dates, list(range(len(requests))), epochs, 'assumed epoch', epoch_hover))
    category_legend(['obs', 'good'] + (['moon'] if show_moon else []) + ['conflict'], fig)
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=26 * len(labels) + 110, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type='date', range=[x0, x1]),
                      yaxis=dict(autorange='reversed', tickfont=dict(size=11), type='category'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def night_figure(plans: dict, grid: vis.SeasonGrid, i: int) -> go.Figure:
    """Time-of-night bars of every epoch assumed on night i: light = the target above the limit (inside its timed window, if any), red = the epoch as packed."""
    base = datetime(2000, 1, 1) + timedelta(days=grid.start_jd[i] - 2451544.5)
    night_len = (grid.end_jd[i] - grid.start_jd[i]) * 24.0
    fig = go.Figure()
    jobs = phase2.night_jobs(plans, i)
    got, slots = phase2.pack_night_slots(jobs) if jobs else ({}, {})
    rows = []
    for key, demand, intervals in jobs:
        rows.append(key)
        for lo, hi in intervals:
            fig.add_trace(go.Bar(y=[key], x=[(hi - lo) * 3600000], base=[base + timedelta(hours=lo)], orientation='h', marker_color='rgba(44,160,44,0.30)',
                                 hovertext=f'{key}: {utc_text(grid, i, lo)}–{utc_text(grid, i, hi)} UT above {MIN_ALT:g}°', hoverinfo='text', showlegend=False))
        for lo, hi in slots.get(key, []):
            fig.add_trace(go.Bar(y=[key], x=[(hi - lo) * 3600000], base=[base + timedelta(hours=lo)], orientation='h', marker_color='rgba(214,39,40,0.85)', width=0.45,
                                 hovertext=f'{key}: {utc_text(grid, i, lo)}–{utc_text(grid, i, hi)} UT', hoverinfo='text', showlegend=False))
        have = got.get(key, 0.0)
        fig.add_annotation(x=base + timedelta(hours=night_len), y=key, xanchor='left', showarrow=False, font=dict(size=11, color='#d62728' if have < demand - 1 / 60 else '#2ca02c'),
                           text=f'{demand:.2f} h epoch' + ('' if have >= demand - 1 / 60 else f', only {have:.2f} h fit'))
    if not rows:
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, text='No epoch is assumed on this night.')
    fig.update_layout(height=max(220, 30 * (len(rows) + 2)), margin=dict(l=10, r=130, t=10, b=10), barmode='overlay',
                      xaxis=dict(type='date', title='UTC', range=[base - timedelta(minutes=20), base + timedelta(hours=night_len + 3)], tickformat='%H:%M'),
                      yaxis=dict(autorange='reversed', type='category'), showlegend=False)
    return fig


# ------------------------------------------------------------------ the plan table
def base_frame(requests: list, settings: dict) -> pd.DataFrame:
    """One row per part of every target set (settings: target key -> [TargetSettings, ...])."""
    rows = []
    for req in requests:
        for t in req.targets:
            k = phase2.target_key(req, t)
            parts = settings[k]
            for n, s in enumerate(parts, start=1):
                rows.append({'key': phase2.part_key(k, n), 'Project': req.key, 'PI': req.lead_pi, 'Target': t.display, 'Part': f'{n}/{len(parts)}' if len(parts) > 1 else '',
                             'Type': t.target_type, 'Mode': t.mode_text, 'Sky': req.sky_background,
                             'RA (°)': None if t.ra is None else round(t.ra, 4), 'Dec (°)': None if t.dec is None else round(t.dec, 4),
                             'Exptime (s)': t.exptime, 'Requested frames': t.nframes, 'Requested (h)': round(t.total_s / 3600.0, 2), 'dither_arcsec': req.dither_arcsec,
                             'OBS_WINDOW': t.window_text, 'Note': t.note,
                             'obs_startdate': s.start, 'obs_enddate': s.end, 'count': int(s.count), 'target_epoch': int(s.epochs), 'scheduled': bool(s.enabled)})
    return pd.DataFrame(rows)


def _as_date(value, fallback: date) -> date:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return fallback
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return fallback


def settings_from_frame(df: pd.DataFrame, previous: dict) -> dict:
    """The edited rows back into target key -> [TargetSettings, ...] (rows of a target set are consecutive, in part order)."""
    out = {k: [] for k in previous}
    for _, r in df.iterrows():
        tkey, _, n = str(r['key']).rpartition('#')
        n = int(n) - 1
        old = previous[tkey][n] if n < len(previous[tkey]) else previous[tkey][-1]
        try:
            count = int(r['count']) if not pd.isna(r['count']) else old.count
            epochs = int(r['target_epoch']) if not pd.isna(r['target_epoch']) else old.epochs
        except (TypeError, ValueError):
            count, epochs = old.count, old.epochs
        out[tkey].append(phase2.TargetSettings(start=_as_date(r['obs_startdate'], old.start), end=_as_date(r['obs_enddate'], old.end), count=max(count, 1), epochs=max(epochs, 1),
                                               enabled=bool(r['scheduled'])))
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
    tkey = c1.selectbox('Target set to split', [k for k, _ in options], format_func=lambda k: labels[k] + (f' ({len(ss["sch_tsettings"][k])} parts)' if len(ss['sch_tsettings'][k]) > 1 else ''), key='sch_split_target')
    parts = ss['sch_tsettings'][tkey]
    if c2.button('Add a part', key='sch_split_add', help='Adds a part with the same range as the last one; then edit its dates, count and epochs in the table.'):
        last = parts[-1]
        parts.append(phase2.TargetSettings(start=last.start, end=last.end, count=last.count, epochs=1, enabled=last.enabled))
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
    use_moon = c1.checkbox('Apply the Moon condition', value=True, key='sch_use_moon', help='Only nights whose Moon illumination suits the project\'s sky background count (Dark ≤ 0.2, Gray 0.2–0.8, Bright ≥ 0.8).')
    if c2.button('Reset to defaults', key='sch_reset', help='Requested window span; all frames in one epoch when one visit fits into a night of the span, otherwise split into epochs that fit.'):
        ss['sch_tsettings'] = {phase2.target_key(req, t): [phase2.default_target_settings(req, t, tables[req.key], j, use_moon)] for req in requests for j, t in enumerate(req.targets)}
        remount_editor(requests)
    c3.caption('Edit obs_startdate / obs_enddate (local nights, inclusive), count (frames per filter in one epoch) and target_epoch (number of epochs). '
               'One epoch = count frames in every filter of the mode, overheads included; the plan needs target_epoch nights in the range on which the target is above the limit for that long.')
    split_controls(requests)
    column_config = {
        'key': None, 'RA (°)': st.column_config.NumberColumn(format='%.4f'), 'Dec (°)': st.column_config.NumberColumn(format='%.4f'),
        'Exptime (s)': st.column_config.NumberColumn(format='%g'),
        'dither_arcsec': st.column_config.NumberColumn(format='%.0f'), 'Requested (h)': st.column_config.NumberColumn(format='%.2f'),
        'obs_startdate': st.column_config.DateColumn('obs_startdate', min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, format='YYYY-MM-DD', required=True),
        'obs_enddate': st.column_config.DateColumn('obs_enddate', min_value=phase2.PROGRAM_START, max_value=phase2.PROGRAM_END, format='YYYY-MM-DD', required=True),
        'count': st.column_config.NumberColumn('count', min_value=1, max_value=100000, step=1, required=True, help='Frames per filter in one epoch'),
        'target_epoch': st.column_config.NumberColumn('target_epoch', min_value=1, max_value=1000, step=1, required=True, help='Number of epochs (visits)'),
        'scheduled': st.column_config.CheckboxColumn('scheduled', help='Off: left out of the plan and of the conflict check'),
    }
    base = ss['sch_editor_base']
    disabled = [c for c in base.columns if c not in EDIT_COLUMNS]
    edited = st.data_editor(base, key=f"sch_editor_{ss.get('sch_editor_version', 0)}", column_config=column_config, disabled=disabled, hide_index=True,
                            width='stretch', height=min(60 + 36 * len(base), 560), column_order=[c for c in base.columns if c != 'key'])
    ss['sch_tsettings'] = settings_from_frame(edited, ss['sch_tsettings'])
    plans = {}
    for req in requests:
        for j, t in enumerate(req.targets):
            for p in phase2.make_target_plans(req, j, tables[req.key], ss['sch_tsettings'][phase2.target_key(req, t)], use_moon):
                plans[p.key] = p
    phase2.place_epochs(plans, phase2.PROGRAM_NIGHTS)
    check_table(plans)
    return plans, use_moon


def check_table(plans: dict) -> None:
    rows = []
    n_parts = {}
    for p in plans.values():
        n_parts[p.target_key] = max(n_parts.get(p.target_key, 0), p.part)
    for p in plans.values():
        t, s = p.target, p.settings
        rows.append({'Project': p.req.key, 'Target': t.display, 'Part': f'{p.part}/{n_parts[p.target_key]}' if n_parts[p.target_key] > 1 else '', 'Mode': t.mode_text,
                     'Exptime (s)': t.exptime, 'count × epochs': f'{s.count} × {s.epochs} = {p.frames_planned}',
                     'Requested frames': t.nframes, 'Epoch (h)': round(p.epoch_h, 2), 'Planned (h)': round(p.planned_h, 2), 'Requested (h)': round(p.required_h, 2),
                     'Feasible nights': p.n_feasible, 'Epochs placed': p.n_placed, 'Status': p.status, '_flag': 'off' if not s.enabled else ('error' if p.issues else ('warn' if p.warnings else ''))})
    df = pd.DataFrame(rows)
    for c in ('Epoch (h)', 'Planned (h)', 'Requested (h)'):
        df[c] = df[c].astype(float)
    n_err = int((df['_flag'] == 'error').sum())
    n_warn = int((df['_flag'] == 'warn').sum())
    n_on = int((df['_flag'] != 'off').sum())
    msg = f'{n_on} of {len(df)} rows scheduled ({len(n_parts)} target sets); planned {sum(p.planned_h for p in plans.values() if p.settings.enabled):.1f} h in total.'
    if n_err:
        st.error(f'{n_err} row(s) cannot be observed as entered (red). {msg}')
    elif n_warn:
        st.warning(f'Every scheduled row fits; {n_warn} with a remark (yellow). {msg}')
    else:
        st.success(f'Every scheduled row fits. {msg}')
    style = df.drop(columns=['_flag']).style.apply(lambda col: [RED if f == 'error' else (YELLOW if f == 'warn' else '') for f in df['_flag']], axis=0)
    st.dataframe(style, hide_index=True, width='stretch', height=min(60 + 36 * len(df), 420),
                 column_config={'Epoch (h)': st.column_config.NumberColumn(format='%.2f'), 'Planned (h)': st.column_config.NumberColumn(format='%.2f'),
                                'Requested (h)': st.column_config.NumberColumn(format='%.2f'), 'Exptime (s)': st.column_config.NumberColumn(format='%g')})


# ------------------------------------------------------------------ page sections
def projects_table(requests: list, plans: dict) -> pd.DataFrame:
    rows = []
    for req in requests:
        ps = [p for p in plans.values() if p.req is req]
        on = [p for p in ps if p.settings.enabled]
        planned = sum(p.planned_h for p in on)
        bad = sum(1 for p in on if p.issues)
        starts = [p.settings.start for p in on]
        ends = [p.settings.end for p in on]
        rows.append({'Project': req.key, 'PI': req.lead_pi, 'Request type': req.observation, 'Mode': req.mode, 'Submode': req.submodes,
                     'Sky brightness': req.sky_background, 'Allocated (h)': req.allocated_hours, 'Requested (h)': round(req.total_h, 1), 'Planned (h)': round(planned, 1),
                     'Targets': len(req.targets), 'Scheduled': len(on), 'dither_arcsec': req.dither_arcsec,
                     'Planned range': f'{min(starts)} to {max(ends)}' if on else '', 'Epochs': sum(p.n_placed for p in on),
                     'Status': ('off' if not on else (f'{bad} target(s) do not fit' if bad else 'OK')) +
                               (f'; over allocation by {planned - req.allocated_hours:.1f} h' if req.allocated_hours is not None and planned > req.allocated_hours + 1e-6 else ''),
                     'Issues': len(req.all_issues)})
    return pd.DataFrame(rows)


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
    st.caption(f'Top: every chosen target per night; dashed grey line at {MIN_ALT:g}° / {phase2.GOOD_HOURS:g} h. '
               'Strips: greens = observable hours of the target (light > 0 h, dark ≥ 3 h); red centre line = the planned obs_startdate to obs_enddate with the assumed epochs as diamonds; '
               'thin grey line = the requested OBS_WINDOW.')


def all_projects_section(requests: list, tables: dict, plans: dict, grid: vis.SeasonGrid) -> None:
    st.subheader('All projects')
    conflicts = phase2.night_conflicts(plans, list(grid.dates))
    show_moon = st.checkbox('Mark nights that fail the Moon condition', value=False, key='sch_all_moon')
    st.plotly_chart(projects_figure(requests, tables, plans, conflicts, show_moon), width='stretch', key='sch_projects_fig', config={'displaylogo': False})
    st.caption('Per project and night: greens = hours during which at least one of its targets is above the limit (light > 0 h, dark ≥ 3 h); '
               'red centre line = the planned ranges of its targets, diamonds = nights on which an epoch is assumed (automatic scheduling: the most constrained targets first, '
               'each epoch in its own part of the range on the night with the least load).')
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


def export_section(requests: list, plans: dict, folder: Path) -> None:
    ss = st.session_state
    st.subheader('Save and export')
    c1, c2, c3, c4 = st.columns(4)
    saved = phase2.load_schedule(folder)
    changed = {k: v for k, v in ss['sch_tsettings'].items() if saved.get(k) != v}
    if c1.button('Save schedule', type='primary', key='sch_save', help=f'Writes {folder / phase2.SCHEDULE_FILE}'):
        path = phase2.save_schedule(folder, ss['sch_tsettings'])
        st.caption(f'Saved to {path}')
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
    st.subheader('Projects')
    req = select_project(requests, projects_table(requests, plans))
    visibility_section(requests, tables, plans, grid, req)
    all_projects_section(requests, tables, plans, grid)
    export_section(requests, plans, folder)


main()
