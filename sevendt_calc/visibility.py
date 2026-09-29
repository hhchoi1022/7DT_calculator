"""
(1) Visibility of targets from the 7DT site.

For a requested observation time the night containing it (or the next night, when the Sun is
up) is sampled every few minutes: target altitude/azimuth, Moon altitude and separation, Sun
altitude. A target is observable when

    min_alt <= altitude <= max_alt,  Moon separation >= moon_sep,  Sun altitude <= sun_alt

with the limits taken from the scheduler configuration (target.config, nightsession.config).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astroplan import Observer
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time, TimeDelta
from astropy.utils import iers

from .config import Site, load_night_sun_altitude, load_site, load_target_constraints
from .targets import Target

# Do not fail on stale Earth-orientation tables; sub-arcsecond accuracy is irrelevant here.
iers.conf.auto_max_age = None


@dataclass(frozen=True)
class Constraints:
    min_alt: float = 30.0
    max_alt: float = 88.0
    moon_sep: float = 40.0
    sun_alt: float = -18.0

    @classmethod
    def from_config(cls) -> 'Constraints':
        try:
            limits = load_target_constraints()
            sun_alt = load_night_sun_altitude()
        except (OSError, ValueError, KeyError):
            return cls()
        return cls(min_alt=limits['min_alt'], max_alt=limits['max_alt'], moon_sep=limits['moon_sep'], sun_alt=sun_alt)


def make_observer(site: Site | None = None) -> Observer:
    site = site or load_site()
    location = EarthLocation.from_geodetic(lat=site.latitude * u.deg, lon=site.longitude * u.deg, height=site.elevation * u.m)
    return Observer(location=location, name=site.name, timezone=site.timezone)


def as_time(value) -> Time:
    if isinstance(value, Time):
        return value
    return Time(value)


@dataclass
class NightInfo:
    sunset: Time
    sunrise: Time
    evening_twilight: Time     # Sun reaches the observation limit (e.g. -18 deg)
    morning_twilight: Time
    timezone: str

    @property
    def grid_start(self) -> Time:
        return self.sunset - TimeDelta(2 * 3600, format='sec')

    @property
    def grid_end(self) -> Time:
        return self.sunrise + TimeDelta(2 * 3600, format='sec')

    @property
    def local_date(self) -> str:
        """Local calendar date of the evening (the night's label)."""
        return self.sunset.to_datetime(timezone=ZoneInfo(self.timezone)).strftime('%Y-%m-%d')

    def local(self, t: Time) -> str:
        return t.to_datetime(timezone=ZoneInfo(self.timezone)).strftime('%H:%M')

    def utc(self, t: Time) -> str:
        return t.to_datetime().strftime('%H:%M')


def night_for(observer: Observer, time: Time, sun_alt_limit: float = -18.0) -> NightInfo:
    """The night containing `time`, or the coming night when the Sun is up."""
    time = as_time(time)
    if observer.sun_altaz(time).alt.deg < 0:
        sunset = observer.sun_set_time(time, which='previous', horizon=0 * u.deg)
        sunrise = observer.sun_rise_time(time, which='next', horizon=0 * u.deg)
    else:
        sunset = observer.sun_set_time(time, which='next', horizon=0 * u.deg)
        sunrise = observer.sun_rise_time(sunset, which='next', horizon=0 * u.deg)
    evening = observer.sun_set_time(sunset, which='next', horizon=sun_alt_limit * u.deg)
    morning = observer.sun_rise_time(sunrise, which='previous', horizon=sun_alt_limit * u.deg)
    return NightInfo(sunset=sunset, sunrise=sunrise, evening_twilight=evening, morning_twilight=morning,
                     timezone=getattr(observer.timezone, 'key', None) or getattr(observer.timezone, 'zone', None) or str(observer.timezone))


def night_midpoint(observer: Observer, time=None, sun_alt_limit: float = -18.0) -> Time:
    """Middle of the night that contains `time` (default now), or of the coming night in daytime."""
    time = as_time(time) if time is not None else Time.now()
    night = night_for(observer, time, sun_alt_limit=sun_alt_limit)
    return night.evening_twilight + (night.morning_twilight - night.evening_twilight) / 2


def _windows(times: Time, mask: np.ndarray) -> list:
    """Contiguous True runs of `mask` as (start, end) Time pairs."""
    out = []
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return out
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for start, end in zip(edges[::2], edges[1::2]):
        out.append((times[start], times[end - 1]))
    return out


@dataclass
class VisibilityResult:
    target: Target
    obstime: Time
    night: NightInfo
    constraints: Constraints
    times: Time
    alt: np.ndarray
    az: np.ndarray
    moon_alt: np.ndarray
    sun_alt: np.ndarray
    moon_sep: np.ndarray
    observable: np.ndarray
    moon_illumination: float
    at_time: dict
    windows: list
    max_alt: float
    max_alt_time: Time

    @property
    def issues(self) -> list:
        return self.at_time['issues']

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def observable_hours(self) -> float:
        if len(self.times) < 2:
            return 0.0
        step_h = (self.times[1] - self.times[0]).to_value('hour')
        return float(self.observable.sum() * step_h)

    def window_text(self) -> str:
        if not self.windows:
            return 'none'
        parts = []
        for start, end in self.windows:
            parts.append(f'{self.night.utc(start)}-{self.night.utc(end)} UT ({self.night.local(start)}-{self.night.local(end)} local)')
        return '; '.join(parts)

    def summary(self) -> dict:
        a = self.at_time
        return {
            'target': self.target.label,
            'status': 'OK' if self.ok else 'WARNING',
            'altitude_deg': a['alt'], 'azimuth_deg': a['az'],
            'moon_sep_deg': a['moon_sep'], 'sun_alt_deg': a['sun_alt'],
            'moon_illumination': self.moon_illumination,
            'observable_hours': self.observable_hours,
            'windows': self.window_text(),
            'max_alt_deg': self.max_alt,
            'max_alt_time_utc': self.max_alt_time.iso[:16] if self.max_alt_time is not None else '',
            'issues': '; '.join(self.issues),
        }


def _evaluate(alt, moon_sep, sun_alt, c: Constraints) -> list:
    issues = []
    if alt < c.min_alt:
        issues.append(f'altitude {alt:.1f} deg < {c.min_alt:g} deg')
    if alt > c.max_alt:
        issues.append(f'altitude {alt:.1f} deg > {c.max_alt:g} deg (zenith limit)')
    if moon_sep < c.moon_sep:
        issues.append(f'Moon separation {moon_sep:.1f} deg < {c.moon_sep:g} deg')
    if sun_alt > c.sun_alt:
        issues.append(f'not night: Sun altitude {sun_alt:.1f} deg > {c.sun_alt:g} deg')
    return issues


class NightEphemeris:
    """Everything about one night that does not depend on the target (computed once, shared by all targets)."""

    def __init__(self, observer: Observer, obstime: Time, constraints: Constraints, step_minutes: float = 5.0):
        self.observer, self.obstime, self.constraints = observer, obstime, constraints
        self.night = night_for(observer, obstime, sun_alt_limit=constraints.sun_alt)
        n = int(np.floor((self.night.grid_end - self.night.grid_start).to_value('min') / step_minutes)) + 1
        self.times = self.night.grid_start + TimeDelta(np.arange(n) * step_minutes * 60.0, format='sec')
        self.frame = AltAz(obstime=self.times, location=observer.location)
        self.moon = get_body('moon', self.times, location=observer.location).transform_to(self.frame)
        self.sun = get_sun(self.times).transform_to(self.frame)
        self.frame0 = AltAz(obstime=obstime, location=observer.location)
        self.moon0 = get_body('moon', obstime, location=observer.location).transform_to(self.frame0)
        self.sun0 = get_sun(obstime).transform_to(self.frame0)
        self.moon_illumination = float(observer.moon_illumination(obstime))


def compute_visibility(target: Target, obstime, site: Site | None = None,
                       constraints: Constraints | None = None, step_minutes: float = 5.0,
                       observer: Observer | None = None, ephemeris: NightEphemeris | None = None) -> VisibilityResult:
    if not target.has_coord:
        raise ValueError(f'{target.label}: coordinates are required for the visibility calculation')
    obstime = as_time(obstime)
    constraints = constraints or Constraints.from_config()
    observer = observer or make_observer(site)
    eph = ephemeris or NightEphemeris(observer, obstime, constraints, step_minutes)
    night, times, frame, moon, sun = eph.night, eph.times, eph.frame, eph.moon, eph.sun

    coord = target.coord
    t_altaz = coord.transform_to(frame)
    alt, az = t_altaz.alt.deg, t_altaz.az.deg
    moon_alt, sun_alt = moon.alt.deg, sun.alt.deg
    moon_sep = t_altaz.separation(moon).deg
    observable = (alt >= constraints.min_alt) & (alt <= constraints.max_alt) & (moon_sep >= constraints.moon_sep) & (sun_alt <= constraints.sun_alt)

    t0 = coord.transform_to(eph.frame0)
    moon0, sun0 = eph.moon0, eph.sun0
    alt0, az0 = float(t0.alt.deg), float(t0.az.deg)
    sep0, sunalt0 = float(t0.separation(moon0).deg), float(sun0.alt.deg)
    issues = _evaluate(alt0, sep0, sunalt0, constraints)
    at_time = {'alt': alt0, 'az': az0, 'moon_sep': sep0, 'sun_alt': sunalt0,
               'is_night': sunalt0 <= constraints.sun_alt, 'observable': not issues, 'issues': issues}

    night_mask = sun_alt <= constraints.sun_alt
    if night_mask.any():
        idx = int(np.argmax(np.where(night_mask, alt, -np.inf)))
    else:
        idx = int(np.argmax(alt))
    return VisibilityResult(
        target=target, obstime=obstime, night=night, constraints=constraints, times=times,
        alt=alt, az=az, moon_alt=moon_alt, sun_alt=sun_alt, moon_sep=moon_sep, observable=observable,
        moon_illumination=eph.moon_illumination, at_time=at_time,
        windows=_windows(times, observable), max_alt=float(alt[idx]), max_alt_time=times[idx],
    )


def compute_all(targets, obstime, site: Site | None = None, constraints: Constraints | None = None,
                step_minutes: float = 5.0) -> list:
    observer = make_observer(site)
    constraints = constraints or Constraints.from_config()
    eph = NightEphemeris(observer, as_time(obstime), constraints, step_minutes)
    return [compute_visibility(t, obstime, site=site, constraints=constraints, step_minutes=step_minutes, observer=observer, ephemeris=eph)
            for t in targets if t.has_coord]


# ------------------------------------------------------------------------ plot
def _utc_offset_days(night: NightInfo) -> float:
    mid = night.sunset + (night.sunrise - night.sunset) / 2
    offset = mid.to_datetime(timezone=ZoneInfo(night.timezone)).utcoffset() or timedelta(0)
    return offset.total_seconds() / 86400.0


def plot_visibility(results: list, site: Site | None = None, title: str | None = None):
    """Altitude over the night for every result (static matplotlib figure; Moon separation is in the table)."""
    if not results:
        raise ValueError('No visibility results to plot')
    site = site or load_site()
    first = results[0]
    night, c = first.night, first.constraints
    x = mdates.date2num(first.times.datetime)
    sun_alt = first.sun_alt

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
    cmap = plt.get_cmap('tab10')
    ax.fill_between(x, 0, 1, where=sun_alt > 0, transform=ax.get_xaxis_transform(), color='0.75', alpha=0.5, lw=0)
    ax.fill_between(x, 0, 1, where=(sun_alt <= 0) & (sun_alt > c.sun_alt), transform=ax.get_xaxis_transform(), color='0.88', alpha=0.6, lw=0)
    ax.grid(alpha=0.3)
    for i, res in enumerate(results):
        color = cmap(i % 10)
        ax.plot(x, res.alt, color=color, lw=1.2, alpha=0.55)
        ax.plot(x, np.where(res.observable, res.alt, np.nan), color=color, lw=3, label=res.target.label)
    ax.plot(x, first.moon_alt, color='0.35', ls='--', lw=1.2, label=f'Moon ({100 * first.moon_illumination:.0f} % illuminated)')
    ax.axhline(c.min_alt, color='crimson', ls=':', lw=1.2)
    ax.text(x[0], c.min_alt + 1, f'min altitude {c.min_alt:g} deg', color='crimson', fontsize=8, va='bottom')
    ax.axhline(c.max_alt, color='crimson', ls=':', lw=0.8)

    t_obs = mdates.date2num(first.obstime.datetime)
    if x[0] <= t_obs <= x[-1]:
        ax.axvline(t_obs, color='red', lw=1.5, ls='-', alpha=0.8)
        ax.text(t_obs, 88.5, 'requested', color='red', fontsize=7, ha='center', va='top', rotation=90)
    else:
        ax.text(0.5, 0.02, f'requested time {first.obstime.iso[:16]} UT is outside this night (daytime); the coming night is shown',
                transform=ax.transAxes, fontsize=8, color='red', ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    for t, label in ((night.sunset, 'sunset'), (night.evening_twilight, f'Sun {c.sun_alt:g}'),
                     (night.morning_twilight, f'Sun {c.sun_alt:g}'), (night.sunrise, 'sunrise')):
        xt = mdates.date2num(t.datetime)
        ax.axvline(xt, color='k', lw=0.6, ls='--', alpha=0.6)
        ax.text(xt, 88.5, label, fontsize=7, ha='center', va='top', rotation=90, color='0.3')

    ax.set_ylim(0, 90)
    ax.set_ylabel('Altitude [deg]')
    ax.set_xlabel('UTC')
    ax.set_xlim(x[0], x[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.get_xticklabels(), fontsize=8)
    off = _utc_offset_days(night)
    secax = ax.secondary_xaxis('top', functions=(lambda v: v + off, lambda v: v - off))
    secax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    secax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(secax.get_xticklabels(), fontsize=8)
    sign = '+' if off >= 0 else '-'
    hh, mm = divmod(int(round(abs(off) * 24 * 60)), 60)
    secax.set_xlabel(f'Local time ({night.timezone}, UTC{sign}{hh:02d}:{mm:02d})', fontsize=9, labelpad=18)
    if len(results) <= 12:
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    else:
        ax.text(0.99, 0.98, f'{len(results)} targets (see the table for names)', transform=ax.transAxes, fontsize=8, ha='right', va='top')
    ax.set_title(title or f'{site.name} visibility, night of {night.local_date} (local)  |  requested {first.obstime.iso[:16]} UT',
                 fontsize=11, pad=30)
    fig.tight_layout()
    return fig


def plot_visibility_interactive(results: list, site: Site | None = None, mark_time: bool = True):
    """
    Interactive (plotly) altitude plot: hover shows time (UT and local), altitude, azimuth and Moon
    separation of every target; the legend toggles targets; the modebar zooms and saves a PNG.
    """
    import plotly.graph_objects as go

    if not results:
        raise ValueError('No visibility results to plot')
    site = site or load_site()
    first = results[0]
    night, c = first.night, first.constraints
    tz = ZoneInfo(night.timezone)
    times = first.times.datetime                       # naive UTC datetimes
    local = [t.replace(tzinfo=timezone.utc).astimezone(tz).strftime('%H:%M') for t in times]
    sun_alt = first.sun_alt
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig = go.Figure()
    # day / twilight bands
    def bands(mask):
        out, start = [], None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            if (not m or i == len(mask) - 1) and start is not None:
                out.append((times[start], times[i]))
                start = None
        return out
    for t0, t1 in bands(sun_alt > 0):
        fig.add_vrect(x0=t0, x1=t1, fillcolor='#9e9e9e', opacity=0.35, line_width=0, layer='below')
    for t0, t1 in bands((sun_alt <= 0) & (sun_alt > c.sun_alt)):
        fig.add_vrect(x0=t0, x1=t1, fillcolor='#cfcfcf', opacity=0.35, line_width=0, layer='below')

    for i, res in enumerate(results):
        color = palette[i % len(palette)]
        custom = np.column_stack([local, res.az, res.moon_sep])
        hover = ('<b>%{fullData.name}</b><br>%{x|%H:%M} UT (%{customdata[0]} local)<br>altitude %{y:.1f}°'
                 '<br>azimuth %{customdata[1]:.0f}°<br>Moon separation %{customdata[2]:.1f}°<extra></extra>')
        fig.add_trace(go.Scatter(x=times, y=res.alt, mode='lines', name=res.target.label, legendgroup=res.target.label,
                                 line=dict(color=color, width=1.2), opacity=0.5, customdata=custom, hovertemplate=hover))
        # the thick observable segments sit on top of the thin line; only the thin line answers to hover
        fig.add_trace(go.Scatter(x=times, y=np.where(res.observable, res.alt, np.nan), mode='lines', name=res.target.label,
                                 legendgroup=res.target.label, showlegend=False, line=dict(color=color, width=4),
                                 hoverinfo='skip', connectgaps=False))
    fig.add_trace(go.Scatter(x=times, y=first.moon_alt, mode='lines', name=f'Moon ({100 * first.moon_illumination:.0f} % illuminated)',
                             line=dict(color='#555555', width=1.2, dash='dash'),
                             hovertemplate='Moon altitude %{y:.1f}°<extra></extra>'))

    fig.add_hline(y=c.min_alt, line=dict(color='crimson', dash='dot', width=1.2),
                  annotation_text=f'min altitude {c.min_alt:g}°', annotation_position='top left', annotation_font_color='crimson')
    fig.add_hline(y=c.max_alt, line=dict(color='crimson', dash='dot', width=0.8))
    for t, label in ((night.sunset, 'sunset'), (night.evening_twilight, f'Sun {c.sun_alt:g}°'),
                     (night.morning_twilight, f'Sun {c.sun_alt:g}°'), (night.sunrise, 'sunrise')):
        fig.add_vline(x=t.datetime, line=dict(color='black', dash='dash', width=0.6), opacity=0.6,
                      annotation_text=label, annotation_position='top', annotation_font_size=10)
    t_obs = first.obstime.datetime
    if not mark_time:
        pass
    elif times[0] <= t_obs <= times[-1]:
        fig.add_vline(x=t_obs, line=dict(color='red', width=1.5), annotation_text='requested', annotation_position='bottom right',
                      annotation_font_color='red')
    else:
        fig.add_annotation(x=0.5, y=0.03, xref='paper', yref='paper', showarrow=False, font=dict(color='red', size=11),
                           text=f'requested time {first.obstime.iso[:16]} UT is outside this night (daytime); the coming night is shown')

    off = _utc_offset_days(night)
    sign = '+' if off >= 0 else '-'
    hh, mm = divmod(int(round(abs(off) * 24 * 60)), 60)
    # hourly ticks: UT at the bottom, local time at the top (a second x axis sharing the range)
    hours = [t for t in times if t.minute == 0]
    fig.add_trace(go.Scatter(x=[times[0], times[-1]], y=[None, None], xaxis='x2', showlegend=False, hoverinfo='skip'))
    fig.update_layout(
        title=dict(text=f'{site.name} visibility, night of {night.local_date} (local)' + (f'  |  requested {first.obstime.iso[:16]} UT' if mark_time else ''),
                   y=0.985, x=0.0, xanchor='left', font=dict(size=19)),
        height=600, hovermode='closest', hoverdistance=25, margin=dict(l=50, r=20, t=125, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='left', x=0, font=dict(size=11)),
        xaxis=dict(title='UTC', range=[times[0], times[-1]], tickformat='%H:%M', dtick=3600000, showgrid=True, gridcolor='#e6e6e6'),
        xaxis2=dict(overlaying='x', side='top', range=[times[0], times[-1]], matches='x',
                    tickvals=hours, ticktext=[t.replace(tzinfo=timezone.utc).astimezone(tz).strftime('%H:%M') for t in hours],
                    title=f'Local time ({night.timezone}, UTC{sign}{hh:02d}:{mm:02d})', showgrid=False),
        yaxis=dict(title='Altitude [deg]', range=[0, 90], showgrid=True, gridcolor='#e6e6e6'),
        plot_bgcolor='white',
    )
    return fig


# ------------------------------------------------------------ month overview
@dataclass
class MonthlyVisibility:
    """Night-by-night summary for several targets: dates are the local dates of each evening."""
    dates: list                      # datetime.date per night
    timezone: str
    night_start: list                # naive UTC datetimes (evening twilight)
    night_end: list                  # naive UTC datetimes (morning twilight)
    moon_illumination: np.ndarray
    labels: list                     # target labels
    max_alt: np.ndarray              # (n_targets, n_nights) highest altitude during the night
    max_alt_time: list               # (n_targets, n_nights) naive UTC datetimes
    hours: np.ndarray                # (n_targets, n_nights) observable hours
    window_start: list               # (n_targets, n_nights) naive UTC datetime or None
    window_end: list
    constraints: Constraints


@dataclass
class MonthGrid:
    """Target-independent part of the month overview: nights, time grid, sidereal time and Moon positions."""
    dates: list
    night_start: list           # naive UTC datetimes (evening twilight)
    night_end: list             # naive UTC datetimes (morning twilight)
    times_jd: np.ndarray        # fine grid over all nights
    bounds: np.ndarray          # slice bounds per night into times_jd
    lst_deg: np.ndarray         # local sidereal time on the fine grid
    latitude: float
    moon_xyz: np.ndarray        # (n, 3) unit vectors of the topocentric Moon direction on the fine grid
    moon_illumination: np.ndarray
    step_minutes: float
    timezone: str


def month_grid(start_night: date, site: Site | None = None, constraints: Constraints | None = None,
               n_nights: int = 30, step_minutes: float = 10.0, moon_step_minutes: float = 60.0) -> MonthGrid:
    """
    Twilight-to-twilight grids for `n_nights` nights from the night starting on the local date
    `start_night`. Twilights are found with vectorised astroplan calls, the Moon is evaluated hourly
    and interpolated, so this takes a couple of seconds instead of ten.
    """
    site = site or load_site()
    constraints = constraints or Constraints.from_config()
    observer = make_observer(site)
    tz = ZoneInfo(site.timezone)
    dates = [start_night + timedelta(days=i) for i in range(n_nights)]
    noons = Time([datetime.combine(d, dtime(12, 0), tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None) for d in dates])
    evening = observer.sun_set_time(noons, which='next', horizon=constraints.sun_alt * u.deg, n_grid_points=60)
    morning = observer.sun_rise_time(evening, which='next', horizon=constraints.sun_alt * u.deg, n_grid_points=60)
    chunks = []
    for t0, t1 in zip(evening, morning):
        n = max(int(np.floor((t1 - t0).to_value('min') / step_minutes)) + 1, 2)
        chunks.append(t0.jd + np.arange(n) * step_minutes / 1440.0)
    times_jd = np.concatenate(chunks)
    bounds = np.cumsum([0] + [len(c) for c in chunks])
    times = Time(times_jd, format='jd')
    lst = observer.local_sidereal_time(times).deg
    # Moon: topocentric direction every `moon_step_minutes`, interpolated as unit vectors
    coarse = []
    for t0, t1 in zip(evening, morning):
        n = max(int(np.ceil((t1 - t0).to_value('min') / moon_step_minutes)) + 2, 2)
        coarse.append(t0.jd - moon_step_minutes / 1440.0 + np.arange(n) * moon_step_minutes / 1440.0)
    coarse_jd = np.concatenate(coarse)
    moon = get_body('moon', Time(coarse_jd, format='jd'), location=observer.location)
    ra, dec = np.radians(moon.ra.deg), np.radians(moon.dec.deg)
    xyz_c = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    xyz = np.column_stack([np.interp(times_jd, coarse_jd, xyz_c[:, k]) for k in range(3)])
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    illum = np.asarray(observer.moon_illumination(evening + (morning - evening) / 2), dtype=float)
    return MonthGrid(dates=dates, night_start=[t.to_datetime() for t in evening], night_end=[t.to_datetime() for t in morning],
                     times_jd=times_jd, bounds=bounds, lst_deg=lst, latitude=site.latitude, moon_xyz=xyz,
                     moon_illumination=illum, step_minutes=step_minutes, timezone=site.timezone)


def monthly_visibility(targets, start_night: date, site: Site | None = None, constraints: Constraints | None = None,
                       n_nights: int = 30, step_minutes: float = 10.0, grid: MonthGrid | None = None) -> MonthlyVisibility:
    """
    For `n_nights` nights from the night starting on the local date `start_night`: the highest
    altitude each target reaches during the night and the interval in which it is observable.
    Altitudes come from the hour angle (no refraction), accurate to a fraction of a degree.
    """
    site = site or load_site()
    constraints = constraints or Constraints.from_config()
    g = grid or month_grid(start_night, site, constraints, n_nights, step_minutes)
    targets = [t for t in targets if t.has_coord]
    n_nights, n_t = len(g.dates), len(targets)
    lat = np.radians(g.latitude)
    times = Time(g.times_jd, format='jd')
    max_alt = np.full((n_t, n_nights), np.nan)
    hours = np.zeros((n_t, n_nights))
    max_alt_time = [[None] * n_nights for _ in range(n_t)]
    w_start = [[None] * n_nights for _ in range(n_t)]
    w_end = [[None] * n_nights for _ in range(n_t)]
    for k, t in enumerate(targets):
        ra, dec = np.radians(t.ra), np.radians(t.dec)
        ha = np.radians(g.lst_deg) - ra
        alt = np.degrees(np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)))
        xyz = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        sep = np.degrees(np.arccos(np.clip(g.moon_xyz @ xyz, -1, 1)))
        ok = (alt >= constraints.min_alt) & (alt <= constraints.max_alt) & (sep >= constraints.moon_sep)
        for i in range(n_nights):
            sl = slice(g.bounds[i], g.bounds[i + 1])
            a = alt[sl]
            j = int(np.argmax(a))
            max_alt[k, i] = a[j]
            max_alt_time[k][i] = times[sl][j].to_datetime()
            good = np.flatnonzero(ok[sl])
            hours[k, i] = len(good) * g.step_minutes / 60.0
            if len(good):
                w_start[k][i] = times[sl][good[0]].to_datetime()
                w_end[k][i] = times[sl][good[-1]].to_datetime()
    return MonthlyVisibility(dates=list(g.dates), timezone=g.timezone, night_start=list(g.night_start), night_end=list(g.night_end),
                             moon_illumination=g.moon_illumination, labels=[t.label for t in targets],
                             max_alt=max_alt, max_alt_time=max_alt_time, hours=hours, window_start=w_start, window_end=w_end,
                             constraints=constraints)


def plot_monthly_interactive(m: MonthlyVisibility, site: Site | None = None, show_max_alt: bool = True):
    """Over the month: the observable interval per night (and, optionally, the highest altitude of the night on top)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    site = site or load_site()
    tz = ZoneInfo(m.timezone)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    n_nights, n_t = len(m.dates), len(m.labels)
    row_int = 2 if show_max_alt else 1

    def rel_hours(t_utc, d):
        """Hours relative to local midnight at the end of the evening of local date d."""
        midnight = datetime.combine(d + timedelta(days=1), dtime(0, 0), tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)
        return (t_utc - midnight).total_seconds() / 3600.0

    def local_str(t_utc):
        return t_utc.replace(tzinfo=timezone.utc).astimezone(tz).strftime('%H:%M')

    if show_max_alt:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.45, 0.55],
                            subplot_titles=('Highest altitude during the night', 'Observable interval (local time)'))
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Observable interval (local time)',))
    x = [datetime.combine(d, dtime(12, 0)) for d in m.dates]
    # night extent (twilight to twilight) as grey bars
    fig.add_trace(go.Scatter(x=[v for d, s, e in zip(m.dates, m.night_start, m.night_end) for v in (datetime.combine(d, dtime(12, 0)),) * 2 + (None,)],
                             y=[v for d, s, e in zip(m.dates, m.night_start, m.night_end) for v in (rel_hours(s, d), rel_hours(e, d), None)],
                             mode='lines', line=dict(color='#d9d9d9', width=14), name='night (twilight to twilight)',
                             hoverinfo='skip'), row=row_int, col=1)
    for k, label in enumerate(m.labels):
        color = palette[k % len(palette)]
        offset = timedelta(hours=(k - (n_t - 1) / 2) * 3.0)
        custom = [[local_str(m.max_alt_time[k][i]), f'{100 * m.moon_illumination[i]:.0f}'] for i in range(n_nights)]
        if show_max_alt:
            fig.add_trace(go.Scatter(x=x, y=m.max_alt[k], mode='lines+markers', name=label, legendgroup=label,
                                     line=dict(color=color, width=2), marker=dict(size=5), customdata=custom,
                                     hovertemplate='<b>' + label + '</b><br>%{x|%Y-%m-%d}<br>max altitude %{y:.1f}° at %{customdata[0]} local'
                                                   '<br>Moon %{customdata[1]} % illuminated<extra></extra>'), row=1, col=1)
        xs, ys, texts = [], [], []
        for i, d in enumerate(m.dates):
            s, e = m.window_start[k][i], m.window_end[k][i]
            if s is None:
                continue
            xi = datetime.combine(d, dtime(12, 0)) + offset
            xs += [xi, xi, None]
            ys += [rel_hours(s, d), rel_hours(e, d), None]
            txt = f'<b>{label}</b><br>{d:%Y-%m-%d}: {local_str(s)}-{local_str(e)} local ({s:%H:%M}-{e:%H:%M} UT), {m.hours[k, i]:.1f} h'
            texts += [txt, txt, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=label, legendgroup=label, showlegend=not show_max_alt,
                                 line=dict(color=color, width=5), text=texts, hovertemplate='%{text}<extra></extra>',
                                 connectgaps=False), row=row_int, col=1)
    ticks = list(range(-6, 9, 2))
    if show_max_alt:
        fig.add_hline(y=m.constraints.min_alt, line=dict(color='crimson', dash='dot', width=1), row=1, col=1)
        fig.update_yaxes(title_text='Altitude [deg]', range=[0, 90], row=1, col=1, gridcolor='#e6e6e6')
    fig.update_yaxes(title_text='Local time', tickvals=ticks, ticktext=[f'{(v + 24) % 24:02d}:00' for v in ticks], row=row_int, col=1, gridcolor='#e6e6e6')
    fig.update_xaxes(title_text='Night of (local date)', tickformat='%m-%d', dtick=2 * 86400000, row=row_int, col=1, gridcolor='#e6e6e6')
    fig.update_layout(height=700 if show_max_alt else 460, hovermode='closest', plot_bgcolor='white', margin=dict(l=50, r=20, t=60, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=-0.16, xanchor='left', x=0, font=dict(size=11)),
                      title=dict(text=f'{site.name} observable hours, from {m.dates[0]:%Y-%m-%d} to {m.dates[-1] + timedelta(days=1):%Y-%m-%d}', font=dict(size=15)))
    return fig
