"""
Shared pieces of the scheduling pages (operator page schedule_app.py and the read-only viewer plan_app.py):
cached programme data (season grid, Moon, requests, night tables, the saved schedule with its plans) and
the Plotly figures (visibility with strips, per-project strips, one night). No page layout here.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sevendt_calc import phase2, trigger
from sevendt_calc import visibility as vis
from ui_common import live_config, season_grid_for, site_key

MIN_ALT = 30.0
WINDOW_COLOR = '#d62728'                  # planned range: a red line through the middle of a strip, epochs as dots
REQUEST_COLOR = '#555555'                 # requested OBS_WINDOW: a thin grey line below the red one
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


def dot_trace(dates: list, positions: list, masks: list, name: str, hover: list | None = None) -> go.Scatter:
    """Diamonds on the given rows (category labels; numbers would create extra categories) over the nights of each mask."""
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
    items: [(req, table, j)] chosen targets. Top: nightly maximum altitude or observable hours of every target,
    the assumed epochs as diamonds on the curves, the Moon illumination, the entered obs_starttime ..
    obs_endtime as a dotted red outline. Bottom: one strip per chosen target (greens: observable / >= 3 h or
    at transit, yellow: Moon unsuitable for its project), its assumed epochs as diamonds, its entered range
    as a red centre line and its requested OBS_WINDOW as a thin blue line above it.
    """
    strip_h = 26 * len(items) + 30
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[330, strip_h])
    if not items:
        return fig
    first_table = items[0][1]
    dates = night_dates(first_table)
    n_nights = len(dates)
    planned = np.zeros(n_nights, bool)
    epochs = np.zeros(n_nights, bool)
    epoch_names = [[] for _ in range(n_nights)]
    for req, table, j in items:
        t = req.targets[j]
        label = f'{req.key} {t.display}'
        y = table.max_alt[:, j] if mode == 'alt' else table.hours[:, j]
        custom = [[float(h), float(a), float(m), slot_text(table, grid, i, j)] for i, (h, a, m) in enumerate(zip(table.hours[:, j], table.max_alt[:, j], table.moon))]
        fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', line=dict(width=1.5), name=label, customdata=custom, legendgroup=label,
                                 hovertemplate=label + ' %{x|%Y-%m-%d}: ' + ('%{y:.1f}°' if mode == 'alt' else '%{y:.2f} h') +
                                 '<br>observable %{customdata[0]:.2f} h, max alt %{customdata[1]:.1f}°<br>%{customdata[3]}<br>Moon %{customdata[2]:.0%}<extra></extra>'), row=1, col=1)
        parts = [p for p in phase2.plans_of(plans, req, j) if p.settings.enabled]
        placed = np.zeros(n_nights, bool)
        for p in parts:
            placed |= p.placed
            planned |= p.in_range
            for i in np.flatnonzero(p.placed):
                epoch_names[i].append(f'{label} {p.epoch_h:.2f} h' + (f' (part {p.part})' if len(parts) > 1 else ''))
        if placed.any():
            idx = np.flatnonzero(placed)
            fig.add_trace(go.Scatter(x=[dates[i] for i in idx], y=[y[i] for i in idx], mode='markers', marker=dict(symbol='diamond', size=9, color=WINDOW_COLOR, line=dict(color='white', width=1)),
                                     name=f'{label}: assumed epochs', legendgroup=label, showlegend=False,
                                     hovertext=[f'{label}: epoch assumed on {table.dates[i]:%Y-%m-%d}' for i in idx], hoverinfo='text'), row=1, col=1)
        epochs |= placed
    if show_moon:
        fig.add_trace(go.Scatter(x=dates, y=first_table.moon, mode='lines', line=dict(color='#999', width=1, dash='dot'), name='Moon illumination',
                                 hovertemplate='Moon %{y:.0%}<extra></extra>'), row=1, col=1)
        fig.data[-1].update(yaxis='y3')
    # the strips: one row per chosen target
    moon_shown = show_moon
    labels, cats, hover, ranges, requested, epoch_rows, epoch_texts = [], [], [], [], [], [], []
    for req, table, j in items:
        t = req.targets[j]
        label = f'{req.key} {t.display}'
        parts = [p for p in phase2.plans_of(plans, req, j) if p.settings.enabled]
        rng, placed = np.zeros(n_nights, bool), np.zeros(n_nights, bool)
        for p in parts:
            rng |= p.in_range
            placed |= p.placed
        c = np.zeros(n_nights, int)
        if mode == 'alt':
            c[table.max_alt[:, j] >= MIN_ALT] = CAT_INDEX['obs']
            c[table.transit[:, j] & (table.max_alt[:, j] >= MIN_ALT)] = CAT_INDEX['good']
        else:
            c[table.hours[:, j] > 0] = CAT_INDEX['obs']
            c[table.hours[:, j] >= phase2.GOOD_HOURS] = CAT_INDEX['good']
        if moon_shown:
            c[~table.moon_ok] = CAT_INDEX['moon']
        labels.append(label)
        cats.append(c)
        ranges.append(rng)
        requested.append(table.requested[:, j])
        epoch_rows.append(placed)
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h observable, max alt {a:.1f}°' + (', requested OBS_WINDOW' if w else '') + (', in the entered range' if r_ else '')
                      + f', Moon {m:.0%}' + ('' if ok else ' (unsuitable)')
                      for d, h, a, w, r_, m, ok in zip(table.dates, table.hours[:, j], table.max_alt[:, j], table.requested[:, j], rng, table.moon, table.moon_ok)])
        epoch_texts.append([f'{label}: epoch assumed on {table.dates[i]:%Y-%m-%d} (' + ', '.join(f'{p.epoch_h:.2f} h' + (f' part {p.part}' if len(parts) > 1 else '') for p in parts if p.placed[i]) + ')'
                            for i in range(n_nights)])
    fig.add_trace(heatmap(dates, labels, np.array(cats), hover), row=2, col=1)
    fig.add_trace(dot_trace(dates, labels, epoch_rows, 'assumed epoch', epoch_texts), row=2, col=1)
    half = timedelta(hours=12)
    shapes = []
    for k, (rng, req_mask) in enumerate(zip(ranges, requested)):
        for a, b in runs(rng):
            shapes.append(dict(type='line', xref='x2', yref='y2', x0=dates[a] - half, x1=dates[b] + half, y0=k, y1=k, line=dict(color=WINDOW_COLOR, width=4), layer='above'))
        for a, b in runs(req_mask):
            shapes.append(dict(type='line', xref='x2', yref='y2', x0=dates[a] - half, x1=dates[b] + half, y0=k + 0.36, y1=k + 0.36, line=dict(color='#1f77b4', width=2), layer='above'))
    for a, b in runs(planned):
        shapes.append(dict(type='rect', xref='x', yref='paper', x0=dates[a] - half, x1=dates[b] + half, y0=1 - 330 / (330 + strip_h), y1=1, fillcolor='rgba(0,0,0,0)',
                           line=dict(color=WINDOW_COLOR, width=2, dash='dot'), layer='above'))
    ref = MIN_ALT if mode == 'alt' else phase2.GOOD_HOURS
    shapes.append(dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=ref, y1=ref, line=dict(color='#888', width=1, dash='dash')))
    category_legend(['obs', 'good'] + (['moon'] if moon_shown and any(not tb.moon_ok.all() for _, tb, _ in items) else []), fig, row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=WINDOW_COLOR, width=4), name='obs_starttime to obs_endtime', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#1f77b4', width=2), name='OBS_WINDOW (requested)', hoverinfo='skip'), row=2, col=1)
    x0, x1 = datetime.combine(phase2.PROGRAM_START, datetime.min.time()) - timedelta(days=1), datetime.combine(phase2.PROGRAM_END, datetime.min.time()) + timedelta(days=1)
    fig.update_layout(height=330 + strip_h + 100, margin=dict(l=10, r=10, t=10, b=10), shapes=shapes, hovermode='closest',
                      yaxis=dict(title='max altitude (°)' if mode == 'alt' else 'observable hours (h)', rangemode='tozero'),
                      yaxis3=dict(overlaying='y', side='right', range=[0, 1], tickformat='.0%', showgrid=False, title='Moon'),
                      yaxis2=dict(type='category', tickfont=dict(size=10), autorange='reversed'), xaxis=dict(type='date', range=[x0, x1]), xaxis2=dict(type='date', range=[x0, x1]),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)))
    return fig


def projects_figure(requests: list, tables: dict, plans: dict, conflicts: list, show_moon: bool) -> go.Figure:
    """One strip per project: hours during which any of its targets is above the limit (greens), the assumed epochs of its targets as diamonds, conflicts in red."""
    labels, cats, hover, ranges, epochs, epoch_hover = [], [], [], [], [], []
    dates = night_dates(tables[requests[0].key])
    for req in requests:
        table = tables[req.key]
        n = len(table.dates)
        rng, ep, names = np.zeros(n, bool), np.zeros(n, bool), [[] for _ in range(n)]
        for p in plans.values():
            if p.req is not req or not p.settings.enabled:
                continue
            rng |= p.in_range
            ep |= p.placed
            for i in np.flatnonzero(p.placed):
                names[i].append(f'{p.target.display} ({p.epoch_h:.1f} h)')
        c = phase2.strip_categories(table.hours_any, table.moon_ok if show_moon else None)
        label = f'{req.key} {req.lead_pi}'
        labels.append(label)
        cats.append(c)
        ranges.append(rng)
        epochs.append(ep)
        hover.append([f'{label} {d:%Y-%m-%d}: {h:.2f} h with a target up' + (', inside an entered range' if w else '') + f', Moon {m:.0%}'
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
    fig.add_trace(dot_trace(dates, labels[:-1], epochs, 'assumed epoch', epoch_hover))
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




# ------------------------------------------------------------------ the saved schedule and its plans (viewer)
def schedule_signature(folder: str) -> tuple:
    from pathlib import Path
    p = Path(folder) / phase2.SCHEDULE_FILE
    return (str(p), p.stat().st_mtime_ns) if p.is_file() else (str(p), 0)


@st.cache_resource(show_spinner=False)
def saved_plans(folder: str, signature: tuple, projects_sig: tuple, schedule_sig: tuple, key: tuple, sun_alt: float, use_moon: bool = True) -> dict:
    """Every part plan of the schedule saved on disk (defaults for target sets it does not list), epochs placed; keyed by part key."""
    requests = load_requests_cached(folder, signature, projects_sig)
    tables = night_tables(folder, signature, projects_sig, key, sun_alt)
    saved = phase2.load_schedule(folder)
    plans = {}
    for req in requests:
        for j, t in enumerate(req.targets):
            parts = saved.get(phase2.target_key(req, t)) or [phase2.default_target_settings(req, t, tables[req.key], j, use_moon)]
            for p in phase2.make_target_plans(req, j, tables[req.key], parts, use_moon):
                plans[p.key] = p
    phase2.place_epochs(plans, phase2.PROGRAM_NIGHTS)
    return plans


# ------------------------------------------------------------------ tables shared by the pages
def projects_table(requests: list, plans: dict, dates: list) -> pd.DataFrame:
    """Per project: the request, the allocation, what is planned and when the assumed epochs fall."""
    rows = []
    for req in requests:
        ps = [p for p in plans.values() if p.req is req]
        on = [p for p in ps if p.settings.enabled]
        planned = sum(p.planned_h for p in on)
        exposure = sum(p.settings.epochs * p.settings.count * p.target.n_orders * (p.target.exptime or 0) for p in on) / 3600.0
        bad = sum(1 for p in on if p.issues)
        placed = np.zeros(len(dates), bool)
        for p in on:
            placed |= p.placed
        nights = np.flatnonzero(placed)
        over = req.allocated_hours is not None and planned > req.allocated_hours + 0.1          # a few minutes of rounding are not 'over'
        if not on:
            status = 'off'
        elif bad:
            status = f'{bad} row(s) do not fit'
        elif over:
            status = 'over allocation'
        else:
            status = 'OK'
        rows.append({'Project': req.key, 'PI': req.lead_pi, 'Request type': req.observation, 'Mode': req.mode, 'Submode': req.submodes, 'Sky brightness': req.sky_background,
                     'Allocated (h)': req.allocated_hours, 'Requested (h)': round(req.total_h, 1), 'Planned exposure (h)': round(exposure, 1), 'Planned total (h)': round(planned, 1),
                     'Within allocation': '' if req.allocated_hours is None else ('no' if over else 'yes'),
                     'Targets': len(req.targets), 'Rows scheduled': len(on), 'Epochs': int(sum(p.n_placed for p in on)),
                     'First epoch': dates[nights[0]].isoformat() if nights.size else '', 'Last epoch': dates[nights[-1]].isoformat() if nights.size else '',
                     'Nights': int(nights.size), 'dither_arcsec': req.dither_arcsec, 'Status': status, 'File issues': len(req.all_issues)})
    return pd.DataFrame(rows)


