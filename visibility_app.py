"""
7DT Visibility: when are the targets observable from the 7DT site?
Run with:  streamlit run visibility_app.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from astropy.time import Time

from sevendt_calc import config
from sevendt_calc import visibility as vis
from sevendt_calc.targets import Target
from ui_common import (feedback_note, live_config, moon_panel, night_date_input, night_midpoint_utc, parse_targets, report_issues,
                       site_key, site_panel, styled_rows, targets_table)

st.set_page_config(page_title='7DT Visibility', page_icon='🌙', layout='wide')

EXAMPLE_ROWS = [
    {'Name': 'NGC0253', 'RA': '00:47:33.12', 'Dec': '-25:17:17.6'},
    {'Name': 'T01022', 'RA': '78.26087', 'Dec': '-71.32075'},
]
SAMPLE_CSV = b'name,ra,dec\nNGC0253,00:47:33.12,-25:17:17.6\nT01022,78.26087,-71.32075\n,15.0542,-28.0406\nM31,00:42:44.3,+41:16:09\n'


@st.cache_data(show_spinner=False, max_entries=32)
def month_grid_cached(key: tuple, limits: tuple, start_night: date):
    """Nights, time grid and Moon positions of the month: independent of the targets, so cached separately."""
    site = config.Site(*key)
    cons = vis.Constraints(min_alt=limits[0], max_alt=limits[1], moon_sep=limits[2], sun_alt=limits[3])
    return vis.month_grid(start_night, site, cons)


@st.cache_data(show_spinner=False, max_entries=64)
def monthly_cached(key: tuple, limits: tuple, start_night: date, coords: tuple):
    site = config.Site(*key)
    cons = vis.Constraints(min_alt=limits[0], max_alt=limits[1], moon_sep=limits[2], sun_alt=limits[3])
    targets = [Target(name=name, ra=ra, dec=dec) for name, ra, dec in coords]
    return vis.monthly_visibility(targets, start_night, site, cons, grid=month_grid_cached(key, limits, start_night))


def night_summary(r: vis.VisibilityResult) -> dict:
    """Status of a target over the whole night (no requested time): observable windows or the reason why not."""
    c = r.constraints
    night = r.sun_alt <= c.sun_alt
    alt_ok = night & (r.alt >= c.min_alt) & (r.alt <= c.max_alt)
    if r.windows:
        status, reason = 'OK', ''
    elif not alt_ok.any():
        status, reason = 'WARNING', f'never above {c.min_alt:g}° during the night'
    elif not (alt_ok & (r.moon_sep >= c.moon_sep)).any():
        status, reason = 'WARNING', f'Moon closer than {c.moon_sep:g}°'
    else:
        status, reason = 'WARNING', 'not observable'
    sep_night = r.moon_sep[night] if night.any() else r.moon_sep
    return {'target': r.target.label, 'status': status, 'observable_hours': round(r.observable_hours, 1),
            'windows': r.window_text(), 'max_alt_deg': round(r.max_alt, 1),
            'max_alt_time_utc': r.max_alt_time.iso[:16] if r.max_alt_time is not None else '',
            'moon_sep_deg': f'{sep_night.min():.0f}-{sep_night.max():.0f}', 'moon_illumination': round(r.moon_illumination, 2),
            'note': reason}


def main():
    feedback_note()
    st.title('7DT Visibility')
    cfg = live_config()
    site, cons = site_panel(cfg)

    st.header('Input target(s)')
    edited = targets_table('vis', EXAMPLE_ROWS, sample_csv=SAMPLE_CSV, csv_help='Columns: name (optional), ra, dec.')
    targets = parse_targets(edited)
    report_issues(targets)
    valid = [t for t in targets if t.has_coord]

    st.header('Input Night')
    c1, c2 = st.columns([1, 1.4], vertical_alignment='center')
    with c1:
        night_date = night_date_input('vis', site)
    with c2:
        moon_panel(site, night_date)
    if not valid:
        st.info('Add at least one target with coordinates.')
        return
    obstime = Time(night_midpoint_utc(site_key(site), night_date))
    with st.spinner('Computing the night...'):
        results = vis.compute_all(valid, obstime, site, cons)
    night = results[0].night
    st.markdown(f'**Night of {night.local_date} (local)**: sunset {night.utc(night.sunset)} UT, '
                f'Sun {cons.sun_alt:g}° at {night.utc(night.evening_twilight)}-{night.utc(night.morning_twilight)} UT, '
                f'sunrise {night.utc(night.sunrise)} UT. Moon illumination {100 * results[0].moon_illumination:.0f} %.')
    df = pd.DataFrame([night_summary(r) for r in results])
    st.dataframe(styled_rows(df, 'status'), hide_index=True, width='stretch')
    st.download_button('Download table (CSV)', df.to_csv(index=False).encode('utf-8'), file_name='visibility.csv', mime='text/csv')
    st.plotly_chart(vis.plot_visibility_interactive(results, site, mark_time=False), width='stretch',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'visibility', 'scale': 2}})
    st.caption('Hover for altitude, azimuth and Moon separation; click legend entries to hide targets; the camera icon saves a PNG.')

    with st.spinner('Computing the month...'):
        monthly = monthly_cached(site_key(site), (cons.min_alt, cons.max_alt, cons.moon_sep, cons.sun_alt), night_date,
                                 tuple((t.label, round(t.ra, 5), round(t.dec, 5)) for t in valid))
    st.plotly_chart(vis.plot_monthly_interactive(monthly, site, show_max_alt=False), width='stretch',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'visibility_month', 'scale': 2}})
    st.caption('Coloured bars: the interval in which each target is observable on that night; grey: the night from evening to morning twilight.')


if __name__ == '__main__':
    main()
