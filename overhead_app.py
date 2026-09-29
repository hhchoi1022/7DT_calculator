"""
7DT Overhead time: total time of one observation (slewing, filter changes, autofocus, readout) for an
observation mode, exposure time and frame count.
Run with:  streamlit run overhead_app.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import streamlit as st

from sevendt_calc import overhead
from ui_common import live_config, obsmode_selector, show_figure

st.set_page_config(page_title='7DT Overhead time', page_icon='⏱️', layout='wide')


def main():
    st.title('7DT Overhead time')
    cfg = live_config()

    c1, c2 = st.columns(2)
    exptime = c1.number_input('Exposure time per frame [s]', 0.0, 7200.0, 100.0, 10.0, key='oh_exp')
    count = int(c2.number_input('Frames per filter', 1, 999, 3, key='oh_cnt'))
    om = obsmode_selector('oh', cfg)
    cfg_oh = overhead.OverheadConfig.load()          # 7DT defaults from data/overhead.json
    if om is None:
        return
    policy = overhead.allowed_autofocus_policies(om)[0]
    st.caption('Autofocus: none for Spec/specall.' if policy == 'none' else f'Autofocus: once per filter of the mode ({om.n_filters_per_unit} per unit).')

    result = overhead.estimate([overhead.ObsRequest('observation', exptime, count, om, policy)], cfg_oh)
    m = st.columns(4)
    m[0].metric('Total observation time', overhead.format_duration(result.total))
    m[1].metric('Open-shutter time', overhead.format_duration(result.science_time))
    m[2].metric('Overhead', overhead.format_duration(result.overhead))
    m[3].metric('Efficiency', f'{100 * result.efficiency:.0f} %')
    m[0].caption(f'{result.total:.0f} s in total')
    st.markdown('\n'.join(f'- {line}' for line in result.summary_lines()))
    st.caption('Counts are those of the slowest unit (all units observe in parallel).')

    show_figure(overhead.plot_timeline(result), 'overhead.png', key='oh_fig')
    with st.expander('Per-unit breakdown'):
        tb = result.targets[0]
        udf = pd.DataFrame([{'unit': u.unit, 'filters': ', '.join(u.filters), 'frames': u.n_frames, 'filter_changes': u.n_filter_changes,
                             'autofocus_runs': u.n_autofocus, 'exposure_s': u.exposure, 'readout_s': u.readout,
                             'filter_change_s': u.filter_change, 'autofocus_s': u.autofocus, 'total_s': u.total} for u in tb.units])
        st.dataframe(udf, hide_index=True, width='stretch')
        st.download_button('Download table (CSV)', udf.to_csv(index=False).encode('utf-8'), file_name='overhead_units.csv', mime='text/csv', key='oh_csv')


if __name__ == '__main__':
    main()
