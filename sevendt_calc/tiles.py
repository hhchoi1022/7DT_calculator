"""
(4) 7DS tile matcher.

Tiles (data/final_tiles.txt) are rectangles in the tangent plane of their centre, so their
edges are great circles. Matching is done in the gnomonic (tangent-plane) projection centred on
the target, where great circles are straight lines: point-in-polygon tests and distances to the
tile edge are exact there, at any declination and across RA = 0/360.
"""
from __future__ import annotations

import math
import re
import textwrap
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from shapely.geometry import Point, Polygon

from .config import DATA_DIR
from .targets import Target

TILE_FILE = DATA_DIR / 'final_tiles.txt'
EDGE_TOLERANCE_ARCMIN = 4.0     # a target closer than this to a tile edge gets a warning
DEFAULT_MIN_OVERLAP = 0.0       # radius matching: keep every tile whose area overlaps the circle at all
TILE_HALF_DIAGONAL_DEG = 0.85   # search margin around the target
TILE_ID_PATTERN = re.compile(r'^\s*T\s*(\d{1,5})\s*$', re.IGNORECASE)   # 7DS tile IDs: T00000 ... T25471


def normalise_tile_id(name) -> str | None:
    """'t1234' / 'T01234' -> 'T01234'; None when the name is not a 7DS tile ID."""
    m = TILE_ID_PATTERN.match(str(name))
    return f'T{int(m.group(1)):05d}' if m else None


@lru_cache(maxsize=2)
def tile_centres(path: Path | str = TILE_FILE) -> dict:
    """Tile ID -> (RA, Dec) of the tile centre in degrees."""
    data = np.genfromtxt(path, names=True, dtype=None, encoding='utf-8', usecols=(0, 1, 2))
    return {str(i): (float(r), float(d)) for i, r, d in zip(data['id'], data['ra'], data['dec'])}


def tile_centre(name) -> tuple[float, float] | None:
    """Centre of the 7DS tile named like 'T01234'; None when the name is not a tile ID."""
    tid = normalise_tile_id(name)
    if tid is None:
        return None
    centre = tile_centres().get(tid)
    if centre is None:
        raise ValueError(f"'{name}' looks like a 7DS tile ID but there is no tile {tid} (tiles run from T00000 to T25471)")
    return centre


def gnomonic(ra, dec, ra0, dec0):
    """Tangent-plane coordinates [deg] of (ra, dec) about (ra0, dec0); x grows to the east, y to the north."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    dra = ra - ra0
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.cos(dec) * np.sin(dra) / cos_c
        y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cos_c
    return np.degrees(x), np.degrees(y)


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation [deg] (haversine), array friendly."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


@dataclass
class MatchedTile:
    id: str
    ra: float
    dec: float
    contains_target: bool
    distance_to_edge_arcmin: float | None      # from the target to the nearest edge (inside tiles)
    near_edge: bool
    overlap_fraction: float | None = None      # fraction of the tile area inside the search circle
    circle_fraction: float | None = None       # fraction of the circle area covered by the tile

    def row(self) -> dict:
        return {'tile': self.id, 'ra_center': self.ra, 'dec_center': self.dec, 'contains_target': self.contains_target,
                'distance_to_edge_arcmin': self.distance_to_edge_arcmin, 'near_edge': self.near_edge,
                'tile_fraction_in_circle': self.overlap_fraction, 'circle_fraction_in_tile': self.circle_fraction}


@dataclass
class TileMatch:
    target: Target
    ra: float
    dec: float
    radius_deg: float
    tiles: list
    edge_tolerance_arcmin: float
    min_overlap: float
    nearest_id: str | None = None
    nearest_distance_deg: float | None = None
    polygons: dict = field(default_factory=dict, repr=False)     # nearby tiles, projected, for plotting

    @property
    def containing(self) -> list:
        return [t for t in self.tiles if t.contains_target]

    @property
    def primary(self) -> MatchedTile | None:
        """The tile the target sits deepest inside (or, for radius matching, with the largest overlap)."""
        inside = self.containing
        if inside:
            return max(inside, key=lambda t: t.distance_to_edge_arcmin)
        if self.tiles and self.radius_deg > 0:
            return max(self.tiles, key=lambda t: t.circle_fraction or 0)
        return None

    @property
    def warnings(self) -> list:
        out = []
        if not self.containing:
            if self.nearest_id is not None:
                out.append(f'No 7DS tile contains the target (nearest tile {self.nearest_id}, centre {self.nearest_distance_deg:.2f} deg away)')
            else:
                out.append('No 7DS tile contains the target')
        for t in self.containing:
            if t.near_edge:
                out.append(f'Target is only {t.distance_to_edge_arcmin:.1f} arcmin from the edge of {t.id} (limit {self.edge_tolerance_arcmin:g} arcmin)')
        return out

    @property
    def ok(self) -> bool:
        return not self.warnings

    def table(self) -> list:
        return [t.row() for t in self.tiles]

    def summary(self) -> dict:
        p = self.primary
        return {'target': self.target.label, 'ra_deg': self.ra, 'dec_deg': self.dec, 'radius_deg': self.radius_deg,
                'status': 'OK' if self.ok else 'WARNING', 'primary_tile': p.id if p else '',
                'matched_tiles': ', '.join(t.id for t in self.tiles), 'n_tiles': len(self.tiles),
                'distance_to_edge_arcmin': p.distance_to_edge_arcmin if p else None, 'warnings': '; '.join(self.warnings)}


class TileSet:
    def __init__(self, path: Path | str = TILE_FILE):
        self.path = Path(path)
        data = np.genfromtxt(self.path, names=True, dtype=None, encoding='utf-8')
        self.ids = np.asarray(data['id']).astype(str)
        self._id_list = [str(i) for i in self.ids]
        self.ra = np.asarray(data['ra'], dtype=float)
        self.dec = np.asarray(data['dec'], dtype=float)
        self.corners = np.stack([np.stack([data[f'ra{i}'], data[f'dec{i}']], axis=1) for i in (1, 2, 3, 4)], axis=1).astype(float)
        self._index = {tid: i for i, tid in enumerate(self._id_list)}

    def __len__(self):
        return len(self.ids)

    def nearby(self, ra: float, dec: float, radius_deg: float) -> np.ndarray:
        sep = angular_separation(ra, dec, self.ra, self.dec)
        return np.flatnonzero(sep <= radius_deg + TILE_HALF_DIAGONAL_DEG)

    def projected_polygons(self, indices, ra0: float, dec0: float) -> dict:
        out = {}
        for i in indices:
            x, y = gnomonic(self.corners[i, :, 0], self.corners[i, :, 1], ra0, dec0)
            poly = Polygon(zip(x, y))
            if poly.is_valid and not poly.is_empty:
                out[self._id_list[i]] = poly
        return out

    def match(self, ra: float, dec: float, radius_deg: float = 0.0, edge_tolerance_arcmin: float = EDGE_TOLERANCE_ARCMIN,
              min_overlap: float = DEFAULT_MIN_OVERLAP, target: Target | None = None) -> TileMatch:
        """
        Match a position. radius_deg = 0: the tiles containing the point (with the distance to their
        edges). radius_deg > 0: additionally every tile whose area overlaps the circle by at least
        `min_overlap` of the tile area.
        """
        target = target or Target(ra=ra, dec=dec)
        radius_deg = float(radius_deg or 0.0)
        idx = self.nearby(ra, dec, max(radius_deg, 0.0) + 1.5)
        polys = self.projected_polygons(idx, ra, dec)
        origin = Point(0.0, 0.0)
        circle = origin.buffer(math.degrees(math.tan(math.radians(radius_deg))), quad_segs=64) if radius_deg > 0 else None
        tol_deg = edge_tolerance_arcmin / 60.0
        tiles = []
        for tid, poly in polys.items():
            i = self._index[tid]
            contains = poly.contains(origin) or poly.touches(origin)
            dist = float(poly.exterior.distance(origin)) if contains else None
            overlap = circle_frac = None
            if circle is not None and poly.intersects(circle):
                inter = poly.intersection(circle).area
                overlap, circle_frac = inter / poly.area, inter / circle.area
            keep = contains or (overlap is not None and overlap > 0 and overlap >= min_overlap)
            if not keep:
                continue
            tiles.append(MatchedTile(id=tid, ra=float(self.ra[i]), dec=float(self.dec[i]), contains_target=bool(contains),
                                     distance_to_edge_arcmin=dist * 60.0 if dist is not None else None,
                                     near_edge=bool(contains and dist < tol_deg), overlap_fraction=overlap, circle_fraction=circle_frac))
        tiles.sort(key=lambda t: (not t.contains_target, -(t.distance_to_edge_arcmin or 0), -(t.circle_fraction or 0)))
        sep = angular_separation(ra, dec, self.ra, self.dec)
        j = int(np.argmin(sep))
        nearest_id, nearest_dist = self._id_list[j], float(sep[j])
        return TileMatch(target=target, ra=float(ra), dec=float(dec), radius_deg=radius_deg, tiles=tiles,
                         edge_tolerance_arcmin=edge_tolerance_arcmin, min_overlap=min_overlap,
                         nearest_id=nearest_id, nearest_distance_deg=nearest_dist, polygons=polys)

    def match_targets(self, targets, radius_deg: float = 0.0, **kwargs) -> list:
        return [self.match(t.ra, t.dec, radius_deg=radius_deg, target=t, **kwargs) for t in targets if t.has_coord]


# ------------------------------------------------------------------------ plot
def plot_tile_matches(matches: list, ncols: int = 2, panel_size: float = 5.0):
    """One panel per target: nearby tiles (blue), matched tiles (red, orange when the target is near an edge)."""
    if not matches:
        raise ValueError('No tile matches to plot')
    n = len(matches)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), dpi=110, squeeze=False)
    axes = axes.flatten()
    for ax, m in zip(axes, matches):
        matched = {t.id: t for t in m.tiles}
        extent = max(1.6, m.radius_deg * 1.4 + 0.9)
        for tid, poly in m.polygons.items():
            x, y = poly.exterior.xy
            if tid in matched:
                continue
            ax.fill(x, y, color='tab:blue', alpha=0.12, lw=0)
            ax.plot(x, y, color='tab:blue', lw=0.8, alpha=0.7)
            cx, cy = poly.centroid.x, poly.centroid.y
            if abs(cx) < extent and abs(cy) < extent:
                ax.text(cx, cy, tid, fontsize=6, ha='center', va='center', color='tab:blue', alpha=0.8)
        for tid, t in matched.items():
            poly = m.polygons[tid]
            x, y = poly.exterior.xy
            color = 'orange' if t.near_edge else 'red'
            ax.fill(x, y, color=color, alpha=0.15, lw=0)
            ax.plot(x, y, color=color, lw=2)
            label = tid if t.contains_target else f'{tid}\n{100 * (t.overlap_fraction or 0):.0f} %'
            ax.text(poly.centroid.x, poly.centroid.y, label, fontsize=7, ha='center', va='center', color=color, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        if m.radius_deg > 0:
            ax.add_patch(Circle((0, 0), math.degrees(math.tan(math.radians(m.radius_deg))), fill=False, color='green', lw=1.5, ls='--'))
        ax.plot(0, 0, marker='*', color='green', ms=12, mec='k', mew=0.5)
        ax.set_xlim(extent, -extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel('East offset [deg]', fontsize=8)
        ax.set_ylabel('North offset [deg]', fontsize=8)
        ax.tick_params(labelsize=7)
        status = 'OK' if m.ok else 'WARNING'
        detail = '; '.join(m.warnings) if m.warnings else (f'inside {m.primary.id}' if m.primary else '')
        ax.set_title(f'{m.target.label}  (RA {m.ra:.4f}, Dec {m.dec:+.4f})\n' + textwrap.fill(f'{status}: {detail}', 60),
                     fontsize=8, color='crimson' if m.warnings else 'black')
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig


def plot_tile_match_interactive(m: TileMatch, extent: float | None = None):
    """
    Interactive (plotly) map of one match in the tangent plane about the target: nearby tiles (blue),
    matched tiles (red, orange when the target is near an edge), the search circle and the target.
    Hovering a tile shows its id, centre and overlap.
    """
    import plotly.graph_objects as go

    matched = {t.id: t for t in m.tiles}
    extent = extent or max(1.6, m.radius_deg * 1.4 + 0.9)
    fig = go.Figure()
    tid_to_center = {}
    for tid, poly in m.polygons.items():
        x, y = poly.exterior.xy
        x, y = list(x), list(y)
        cx, cy = poly.centroid.x, poly.centroid.y
        tid_to_center[tid] = (cx, cy)
        t = matched.get(tid)
        if t is None:
            color, fill, width = 'rgba(31,119,180,0.8)', 'rgba(31,119,180,0.12)', 1
        else:
            color = 'rgba(255,140,0,0.95)' if t.near_edge else 'rgba(214,39,40,0.95)'
            fill = 'rgba(255,140,0,0.18)' if t.near_edge else 'rgba(214,39,40,0.18)'
            width = 2.5
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='toself', fillcolor=fill, line=dict(color=color, width=width),
                                 hoveron='fills', name=tid, showlegend=False, text=tid,
                                 hovertemplate=_tile_hover(tid, t) + '<extra></extra>'))
        if t is not None or (abs(cx) < extent and abs(cy) < extent):
            fig.add_annotation(x=cx, y=cy, text=tid if t is None else f'<b>{tid}</b>' + ('' if t.contains_target or t.overlap_fraction is None else f'<br>{100 * t.overlap_fraction:.0f} %'),
                               showarrow=False, font=dict(size=9 if t is None else 10, color='rgba(31,119,180,0.9)' if t is None else ('#cc6a00' if t.near_edge else '#b22222')))
    if m.radius_deg > 0:
        r = math.degrees(math.tan(math.radians(m.radius_deg)))
        ang = np.linspace(0, 2 * np.pi, 181)
        fig.add_trace(go.Scatter(x=r * np.cos(ang), y=r * np.sin(ang), mode='lines', line=dict(color='green', dash='dash', width=1.5),
                                 name=f'radius {m.radius_deg:g}°', hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='star', size=16, color='green', line=dict(color='black', width=0.5)),
                             name=m.target.label, hovertemplate=f'<b>{m.target.label}</b><br>RA {m.ra:.5f}, Dec {m.dec:+.5f}<extra></extra>', showlegend=False))
    status = 'OK' if m.ok else 'WARNING'
    detail = '; '.join(m.warnings) if m.warnings else (f'inside {m.primary.id}' if m.primary else '')
    fig.update_layout(
        title=dict(text=f'{m.target.label} (RA {m.ra:.4f}, Dec {m.dec:+.4f})<br><span style="font-size:12px;color:{"crimson" if m.warnings else "black"}">{status}: {detail}</span>',
                   font=dict(size=14)),
        xaxis=dict(title='East offset [deg]', range=[extent, -extent], zeroline=False, gridcolor='#e6e6e6', constrain='domain'),
        yaxis=dict(title='North offset [deg]', range=[-extent, extent], zeroline=False, gridcolor='#e6e6e6', scaleanchor='x', scaleratio=1),
        plot_bgcolor='white', height=520, margin=dict(l=50, r=20, t=70, b=50), hovermode='closest')
    return fig


def _tile_hover(tid: str, t) -> str:
    if t is None:
        return f'<b>{tid}</b>'
    parts = [f'<b>{tid}</b>', f'centre RA {t.ra:.4f}, Dec {t.dec:+.4f}']
    if t.contains_target:
        parts.append(f'target {t.distance_to_edge_arcmin:.1f} arcmin from the edge')
    if t.overlap_fraction is not None:
        parts.append(f'{100 * t.overlap_fraction:.0f} % of the tile inside the circle, {100 * (t.circle_fraction or 0):.0f} % of the circle in the tile')
    return '<br>'.join(parts)
