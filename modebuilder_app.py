"""
7DT Mode builder: compose a new Spec / Color observation mode by clicking the filters in the wheel
grid of every unit (or start from an existing / uploaded mode file), visualise it and download it as
a .specmode / .colormode JSON file.
Run with:  streamlit run modebuilder_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sevendt_calc import config, modebuild, modeplot
from ui_common import feedback_note, live_config

st.set_page_config(page_title='7DT Mode builder', page_icon='🧩', layout='wide')

EMPTY = 'Empty'
SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
GRID_KEY = 'mb_grid'
REMOVE_MARK = ' ✕'
SPACER_ROW = ' '
CELL_WIDTH = 72
ROW_HEIGHT = 30
MODE_FILE_TYPES = ['specmode', 'colormode', 'mode', 'json', 'txt']


def _clear():
    st.session_state['mb_seq'] = {}


def _park_column(units: list, clicked: str) -> str:
    """
    After a click the selection is moved to a blank cell of the spacer row instead of being cleared: Streamlit's
    frontend only re-applies a programmatic selection when it differs from the previous one, so the parked column
    alternates (the clicked column, otherwise the next one). Clearing to the same empty state twice would leave the
    old click in the widget and a repeated click on the same cell would be ignored.
    """
    ss = st.session_state
    park = clicked if clicked != ss.get('mb_park') else units[(units.index(clicked) + 1) % len(units)]
    ss['mb_park'] = park
    return park


def apply_click(filtinfo: dict):
    """
    Apply the cell clicked in the previous run. The grid's widget state is already in session state when the
    script starts, so the click is handled before the grid is drawn: one rerun per click, no st.rerun().
    Slot rows add the filter of that slot; Order rows remove that entry from the unit's sequence.
    """
    ss = st.session_state
    state = ss.get(GRID_KEY)
    cells = (state or {}).get('selection', {}).get('cells', []) if isinstance(state, dict) else []
    if not cells:
        return
    r, unit = cells[0]
    units = list(filtinfo)
    n_slots = max((len(f) for f in filtinfo.values()), default=0)
    n_seq = max((len(f) for f in ss['mb_seq'].values()), default=0)       # the table the click was made on
    if unit not in filtinfo or r >= n_slots + n_seq:                      # a parked / spacer cell: nothing to do
        return
    if r < n_slots:
        wheel = filtinfo[unit]
        if r < len(wheel) and not config.is_slot(wheel[r]):
            ss['mb_seq'] = modebuild.add_filter(ss['mb_seq'], unit, wheel[r])
    else:
        ss['mb_seq'] = modebuild.remove_at(ss['mb_seq'], unit, r - n_slots)
    ss['mb_park_next'] = _park_column(units, unit)


def apply_upload():
    """A newly uploaded mode file replaces the sequences, the mode type and the name (handled once per file)."""
    ss = st.session_state
    up = ss.get('mb_upload')
    marker = (up.name, up.size) if up is not None else None
    if up is None or ss.get('mb_upload_marker') == marker:
        return
    ss['mb_upload_marker'] = marker
    try:
        kind, name, seq = modebuild.parse_mode_file(up.getvalue(), up.name)
    except ValueError as exc:
        ss['mb_upload_error'] = f'{up.name}: {exc}'
        return
    ss['mb_seq'], ss['mb_kind'], ss['mb_name'] = seq, kind, name


def grid_table(filtinfo: dict, seq: dict):
    """
    The wheel grid as a table: units across, wheel slots down (cell = the filter in that slot, superscripts = its
    positions in the unit's sequence), one 'Order n' row per position (click = remove that entry), and a blank
    spacer row that holds the parked selection. Returns (frame, styler).
    """
    from matplotlib.colors import to_hex
    from sevendt_calc.filters import filter_colors
    from sevendt_calc.photometry import effective_wavelength
    units = list(filtinfo)
    n_slots = max((len(f) for f in filtinfo.values()), default=0)
    n_seq = max((len(f) for f in seq.values()), default=0)
    used = sorted({f for fs in seq.values() for f in fs})
    colors = {f: to_hex(c) for f, c in filter_colors(used, {f: effective_wavelength(f) for f in used if effective_wavelength(f)}).items()}
    text, css = {}, {}
    for k in range(n_slots):
        row, style = [], []
        for u in units:
            wheel = filtinfo[u]
            if k >= len(wheel) or config.is_slot(wheel[k]):
                row.append(''); style.append('background-color: #f4f4f4')
                continue
            f = wheel[k]
            positions = [i + 1 for i, x in enumerate(seq.get(u, [])) if x == f]
            row.append(f + '·'.join(str(i).translate(SUP) for i in positions))
            style.append('background-color: #ffd9d9; color: #b00020; font-weight: 600' if positions else '')
        text[f'Slot {k + 1}'], css[f'Slot {k + 1}'] = row, style
    for j in range(n_seq):
        row, style = [], []
        for u in units:
            f = seq.get(u, [])[j] if j < len(seq.get(u, [])) else ''
            row.append(f + REMOVE_MARK if f else ''); style.append(f'background-color: {colors.get(f, "#cccccc")}99' if f else 'background-color: #fafafa')
        text[f'Order {j + 1}'], css[f'Order {j + 1}'] = row, style
    text[SPACER_ROW], css[SPACER_ROW] = [''] * len(units), [''] * len(units)
    df = pd.DataFrame(text, index=units).T
    css_df = pd.DataFrame(css, index=units).T
    return df, df.style.apply(lambda _: css_df, axis=None)


def wheel_grid(filtinfo: dict):
    ss = st.session_state
    df, styler = grid_table(filtinfo, ss['mb_seq'])
    park = ss.pop('mb_park_next', None)
    if park is not None:                                   # move the selection to the spacer row (see _park_column)
        ss[GRID_KEY] = {'selection': {'rows': [], 'columns': [], 'cells': [[len(df) - 1, park]]}}
    units = list(filtinfo)
    col_cfg = {u: st.column_config.TextColumn(u, width=CELL_WIDTH) for u in units}
    col_cfg['_index'] = st.column_config.TextColumn('', width=66)
    st.dataframe(styler, on_select='rerun', selection_mode='single-cell', key=GRID_KEY, hide_index=False, row_height=ROW_HEIGHT,
                 height=ROW_HEIGHT * (len(df) + 1) + 3, width='content', column_config=col_cfg)


@st.cache_data(show_spinner=False, max_entries=64)
def mode_figure(kind: str, name: str, table: tuple):
    om = config.ObsMode(mode=kind, submode=name, filters_by_unit={u: list(f) for u, f in table})
    return modeplot.plot_obsmode_interactive(om, grid=False)      # the Order rows of the table already show the sequence


def main():
    feedback_note()
    st.title('7DT Mode builder')
    cfg = live_config()
    filtinfo = cfg['filtinfo']
    ss = st.session_state
    ss.setdefault('mb_seq', {})
    ss.setdefault('mb_kind', 'Spec')
    ss.setdefault('mb_name', 'newmode')
    apply_upload()                     # both read the widget states of the previous run, before the widgets are drawn
    apply_click(filtinfo)

    c1, c2 = st.columns([1, 2])
    kind = c1.radio('Mode type', list(modebuild.KINDS), horizontal=True, key='mb_kind')
    name = c2.text_input('Mode name (file name)', key='mb_name')
    existing = cfg['specmodes'] if kind == 'Spec' else cfg['colormodes']
    t1, t2, t3, t4 = st.columns([1.6, 0.8, 0.8, 1.8], vertical_alignment='bottom')
    template = t1.selectbox('Start from', [EMPTY] + sorted(existing), key=f'mb_template_{kind}')
    if t2.button('Load', width='stretch', key='mb_load'):
        ss['mb_seq'] = {} if template == EMPTY else {u: list(f) for u, f in existing[template].items()}
    t3.button('Clear all', width='stretch', key='mb_clear', on_click=_clear)
    t4.file_uploader('Upload a mode file (.specmode / .colormode)', type=MODE_FILE_TYPES, key='mb_upload')
    if 'mb_upload_error' in ss:
        st.error(ss.pop('mb_upload_error'))

    st.caption('Filter wheels of the units: click a filter to add it to that unit\'s sequence (the superscripts are its positions; '
               'click again to observe it twice). The Order rows show the resulting sequence; click an entry (✕) to remove it.')
    wheel_grid(filtinfo)
    pad = st.checkbox('Pad shorter units by repeating their last filter, so every unit takes the same number of frames', value=False, key='mb_pad')
    seq = modebuild.pad_sequences(ss['mb_seq']) if pad else ss['mb_seq']

    built = modebuild.build_mode(kind, name, seq, filtinfo)
    for msg in built.errors:
        st.error(msg)
    for msg in built.warnings:
        st.warning(msg)
    if not any(built.filters_by_unit.values()):
        return

    om = built.obsmode
    n_frames = sum(len(f) for f in built.filters_by_unit.values())
    st.markdown(f'**{built.file_name}**: {len(om.filters)} distinct filters, {len(om.units)} units observing, '
                f'{om.n_filters_per_unit} filter(s) per unit, {n_frames} unit frames per set.')
    d1, d2 = st.columns([1, 3])
    d1.download_button(f'Download {built.file_name}', built.to_json().encode('utf-8'), file_name=built.file_name, mime='application/json',
                       width='stretch', key='mb_download', disabled=bool(built.errors))
    if built.errors:
        d2.caption('Fix the errors above before downloading.')

    fig = mode_figure(built.kind, built.name, tuple((u, tuple(f)) for u, f in om.filters_by_unit.items()))
    st.plotly_chart(fig, width='stretch', key='mb_fig', config={'displaylogo': False, 'toImageButtonOptions': {'filename': f'obsmode_{built.name}', 'scale': 3}})
    with st.expander('Filters of the mode'):
        st.dataframe(pd.DataFrame(modebuild.filter_summary(built)), hide_index=True, width='stretch')
    with st.expander(f'{built.file_name} (JSON)'):
        st.code(built.to_json(), language='json')


if __name__ == '__main__':
    main()
