"""Visualisation of an observation mode: which filters each unit observes, and their wavelength coverage."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .config import ObsMode
from .filters import filter_colors
from .photometry import effective_wavelength, get_filter_curve


def plot_obsmode(om: ObsMode, lam_range=(3400.0, 9600.0)):
    """
    Top: transmission curves of the filters of the mode (label = filter, number of units observing it).
    Bottom: the filter sequence of every unit as a coloured grid.
    """
    filters = om.filters
    waves = {f: effective_wavelength(f) for f in filters}
    colors = filter_colors(filters, {f: w for f, w in waves.items() if w})
    mult = om.filter_multiplicity()
    units = om.units
    n_seq = om.n_filters_per_unit

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3.2 + 0.55 * n_seq + 1.6), dpi=110,
                                   gridspec_kw={'height_ratios': [2.2, 0.6 + 0.5 * n_seq]})
    # ---- wavelength coverage
    for f in filters:
        curve = get_filter_curve(f)
        if curve is None:
            continue
        sel = (curve.wavelength >= lam_range[0]) & (curve.wavelength <= lam_range[1])
        w, t = curve.wavelength[sel], curve.response[sel] / curve.response.max()
        ax1.fill_between(w, 0, t, color=colors[f], alpha=0.25, lw=0)
        ax1.plot(w, t, color=colors[f], lw=1.2)
        label = f if mult.get(f, 1) == 1 else f'{f} x{mult[f]}'
        ax1.text(curve.lam_eff, 1.03, label, rotation=90, ha='center', va='bottom', fontsize=7, color=colors[f])
    ax1.set_xlim(*lam_range)
    ax1.set_ylim(0, 1.6)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel('Transmission\n(normalised)', fontsize=9)
    ax1.set_xlabel('Wavelength [A]', fontsize=9)
    ax1.set_title(f'{om.label}: {len(filters)} filters, {len(units)} units, {n_seq} filter(s) per unit', fontsize=11)
    ax1.grid(alpha=0.25)
    # ---- unit x sequence grid
    for i, unit in enumerate(units):
        for j, f in enumerate(om.filters_by_unit[unit]):
            ax2.add_patch(Rectangle((i, j), 1, 1, facecolor=colors.get(f, '#cccccc'), edgecolor='white', lw=1.5, alpha=0.85))
            ax2.text(i + 0.5, j + 0.5, f, ha='center', va='center', fontsize=7, color='black')
    ax2.set_xlim(0, len(units))
    ax2.set_ylim(n_seq, 0)
    ax2.set_xticks(np.arange(len(units)) + 0.5)
    ax2.set_xticklabels(units, fontsize=7)
    ax2.set_yticks(np.arange(n_seq) + 0.5)
    ax2.set_yticklabels([f'{j + 1}' for j in range(n_seq)], fontsize=8)
    ax2.set_ylabel('Order', fontsize=9)
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def plot_obsmode_interactive(om: ObsMode, lam_range=(3400.0, 9600.0), curve_step: int = 16, grid: bool = True):
    """
    Plotly version of plot_obsmode (drawn in the browser): transmission curves on top, the unit x order grid below.
    Shapes and annotations are assigned to the layout in one go (adding them one by one is quadratic in plotly).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from matplotlib.colors import to_hex, to_rgba

    filters = om.filters
    waves = {f: effective_wavelength(f) for f in filters}
    colors = {f: to_hex(c) for f, c in filter_colors(filters, {f: w for f, w in waves.items() if w}).items()}
    mult = om.filter_multiplicity()
    units = om.units
    n_seq = om.n_filters_per_unit
    grid_height = 60 + 34 * max(n_seq, 1) if grid else 0
    top_height = 300
    if grid:
        fig = make_subplots(rows=2, cols=1, row_heights=[top_height, grid_height], vertical_spacing=0.16)
    else:
        fig = go.Figure()
    traces, annotations, shapes = [], [], []
    # ---- wavelength coverage
    for f in filters:
        curve = get_filter_curve(f)
        if curve is None:
            continue
        sel = (curve.wavelength >= lam_range[0]) & (curve.wavelength <= lam_range[1])
        w = np.round(curve.wavelength[sel][::curve_step]).astype(int)
        t = np.round(curve.response[sel][::curve_step] / curve.response.max(), 2)
        r, g, b, _ = to_rgba(colors[f])
        label = f if mult.get(f, 1) == 1 else f'{f} x{mult[f]}'
        traces.append(go.Scatter(x=w, y=t, mode='lines', name=label, showlegend=False, line=dict(color=colors[f], width=1.2),
                                 fill='tozeroy', fillcolor=f'rgba({255 * r:.0f},{255 * g:.0f},{255 * b:.0f},0.25)', xaxis='x', yaxis='y',
                                 hovertemplate=f'<b>{label}</b><br>λ = %{{x:.0f}} Å<br>T = %{{y:.2f}}<extra></extra>'))
        annotations.append(dict(x=curve.lam_eff, y=1.03, xref='x', yref='y', text=label, textangle=-90, showarrow=False,
                                xanchor='center', yanchor='bottom', font=dict(size=9, color=colors[f])))
    # ---- unit x sequence grid
    hx, hy, htext = [], [], []
    for i, unit in enumerate(units if grid else []):
        for j, f in enumerate(om.filters_by_unit[unit]):
            shapes.append(dict(type='rect', xref='x2', yref='y2', x0=i, x1=i + 1, y0=j, y1=j + 1, fillcolor=colors.get(f, '#cccccc'),
                               opacity=0.85, line=dict(color='white', width=1.5)))
            annotations.append(dict(x=i + 0.5, y=j + 0.5, xref='x2', yref='y2', text=f, showarrow=False, font=dict(size=9, color='black')))
            hx.append(i + 0.5); hy.append(j + 0.5); htext.append(f'{unit}: {j + 1}. {f}')
    if hx:
        traces.append(go.Scatter(x=hx, y=hy, mode='markers', marker=dict(size=22, opacity=0), hovertext=htext, hoverinfo='text',
                                 showlegend=False, xaxis='x2', yaxis='y2'))
    fig.add_traces(traces)
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        title=dict(text=f'{om.label}: {len(filters)} filters, {len(units)} units, {n_seq} filter(s) per unit', font=dict(size=13)),
        height=top_height + grid_height + (110 if grid else 60), plot_bgcolor='white', margin=dict(l=60, r=20, t=50, b=40), hovermode='closest',
        xaxis=dict(range=list(lam_range), title=dict(text='Wavelength [Å]', font=dict(size=12)), tickfont=dict(size=10), showgrid=True,
                   gridcolor='#e6e6e6', showline=True, linecolor='#444', mirror=True),
        yaxis=dict(range=[0, 1.6], tickvals=[0, 0.5, 1.0], title=dict(text='Transmission (normalised)', font=dict(size=12)), tickfont=dict(size=10),
                   showgrid=True, gridcolor='#e6e6e6', showline=True, linecolor='#444', mirror=True),
    )
    if grid:
        fig.update_layout(
            xaxis2=dict(range=[0, max(len(units), 1)], tickvals=[i + 0.5 for i in range(len(units))], ticktext=units, tickfont=dict(size=9),
                        showgrid=False, zeroline=False),
            yaxis2=dict(range=[max(n_seq, 1), 0], tickvals=[j + 0.5 for j in range(n_seq)], ticktext=[str(j + 1) for j in range(n_seq)],
                        title=dict(text='Order', font=dict(size=12)), tickfont=dict(size=10), showgrid=False, zeroline=False),
        )
    return fig
