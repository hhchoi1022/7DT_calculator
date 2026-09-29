"""
7DT Tile matcher: the 7DS survey tiles that contain (or overlap a circle around) each target.
Run with:  streamlit run tiles_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sevendt_calc import tiles
from ui_common import live_config, parse_targets, report_issues, styled_rows, targets_table

st.set_page_config(page_title='7DT Tile matcher', page_icon='🗺️', layout='wide')

EXAMPLE_ROWS = [
    {'Name': 'NGC0253', 'RA': '00:47:33.12', 'Dec': '-25:17:17.6'},
    {'Name': 'T01022', 'RA': '78.26087', 'Dec': '-71.32075'},
]
SAMPLE_CSV = b'name,ra,dec\nNGC0253,00:47:33.12,-25:17:17.6\nT01022,78.26087,-71.32075\n,15.0542,-28.0406\n'


@st.cache_resource(show_spinner='Loading the 7DS tiles...')
def get_tiles() -> tiles.TileSet:
    return tiles.TileSet()


def main():
    st.title('7DT Tile matcher')
    live_config()
    ts = get_tiles()

    st.header('Input target(s)')
    edited = targets_table('tile', EXAMPLE_ROWS, sample_csv=SAMPLE_CSV, csv_help='Columns: name (optional), ra, dec.')
    targets = parse_targets(edited)
    report_issues(targets)
    valid = [t for t in targets if t.has_coord]
    if not valid:
        st.info('Add at least one target with coordinates.')
        return

    same = st.checkbox('Same radius for all targets', value=True, key='tile_same') if len(valid) > 1 else True
    if same:
        radius = st.number_input('Search radius [deg] (0 = point matching)', 0.0, 10.0, 0.0, 0.1, key='tile_radius_all')
        radii = [radius] * len(valid)
    else:
        radii, cols = [], st.columns(min(len(valid), 4))
        for i, t in enumerate(valid):
            radii.append(cols[i % len(cols)].number_input(f'{t.label}: radius [deg]', 0.0, 10.0, 0.0, 0.1, key=f'tile_radius_{t.row}'))
    st.caption(f'7DT defaults: a warning when the target is within {tiles.EDGE_TOLERANCE_ARCMIN:g} arcmin of a tile edge; '
               'with a radius, every tile that overlaps the circle is listed, with the fraction of its area inside the circle.')

    matches = [ts.match(t.ra, t.dec, radius_deg=r, target=t) for t, r in zip(valid, radii)]
    df = pd.DataFrame([m.summary() for m in matches])
    st.dataframe(styled_rows(df.round(3), 'status'), hide_index=True, width='stretch')
    st.download_button('Download table (CSV)', df.to_csv(index=False).encode('utf-8'), file_name='tile_matches.csv', mime='text/csv', key='tile_table_dl')
    with st.expander('Matched tiles in detail'):
        detail = (pd.concat([pd.DataFrame(m.table()).assign(target=m.target.label) for m in matches if m.tiles], ignore_index=True)
                  if any(m.tiles for m in matches) else pd.DataFrame())
        st.dataframe(detail.round(3), hide_index=True, width='stretch')

    cols = st.columns(2 if len(matches) > 1 else 1)
    for i, m in enumerate(matches):
        with cols[i % len(cols)]:
            st.plotly_chart(tiles.plot_tile_match_interactive(m), width='stretch', key=f'tile_map_{m.target.row}',
                            config={'displaylogo': False, 'toImageButtonOptions': {'filename': f'tiles_{m.target.label}', 'scale': 3}})
    st.caption('Hover a tile for its id, centre and overlap; drag to zoom, double-click to reset; the camera icon saves a PNG.')


if __name__ == '__main__':
    main()
