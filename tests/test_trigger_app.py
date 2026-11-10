"""Observation request page: the request core (parsing, validation, formatting) and a headless run of the page."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from sevendt_calc import config, trigger


def _cfg():
    return {'filtinfo': config.load_filtinfo(), 'specmodes': config.load_specmodes(), 'colormodes': config.load_colormodes()}


def test_read_table_formats():
    csv = trigger.read_table(b'objname,ra,dec\nA,10.0,-20.0\n', 'a.csv')
    assert list(csv.columns) == ['objname', 'ra', 'dec'] and csv.iloc[0]['ra'] == '10.0'
    ascii_ = trigger.read_table(b'# name   RA    DEC   exposure\nB  01:00:00  +10:00:00  120\n', 'b.txt')
    assert list(ascii_.columns) == ['name', 'RA', 'DEC', 'exposure'] and ascii_.iloc[0]['exposure'] == '120'
    tsv = trigger.read_table('target\tra\tdec\tmystery\nC\t1\t2\tx\n', 'c.tsv')
    mapping, unknown = trigger.recognise_columns(tsv.columns)
    assert mapping == {'target': 'objname', 'ra': 'RA', 'dec': 'De'} and unknown == ['mystery']
    applied = trigger.apply_mapping(tsv, {**mapping, 'mystery': 'count'})
    assert list(applied.columns) == ['objname', 'RA', 'De', 'count'] and applied.iloc[0]['count'] == 'x'


def test_validation_rules():
    ts = trigger.TriggerSettings.load()
    cfg = _cfg()
    defaults = {'obsmode': 'Spec', 'specmode': 'specall', 'exptime': '100', 'count': '3', 'binning': '1', 'gain': '2750'}
    rows = [
        {'objname': 'good', 'RA': '10.5', 'De': '-20.25'},
        {'objname': '', 'RA': 'abc', 'De': '-20'},                                            # missing name, bad RA
        {'objname': 'x' * 101, 'RA': '1', 'De': '2', 'exptime': '200'},                       # too long, exptime above max
        {'objname': 'warn', 'RA': '1', 'De': '2', 'exptime': '60', 'count': '2,2', 'obs_startdate': '2026-11-11T00:00:00', 'obs_enddate': '2026-11-01'},
        {'objname': 'deep', 'RA': '1', 'De': '2', 'obsmode': 'deep', 'filter_': 'r', 'binning': '2', 'gain': '4030'},
        {'objname': 'nomode', 'RA': '1', 'De': '2', 'obsmode': 'Deep'},
        {'objname': 'date', 'RA': '1', 'De': '2', 'obs_startdate': '11/11/2026'},
    ]
    norm, issues = trigger.validate_rows(rows, ts, defaults, cfg)
    by_row = {}
    for it in issues:
        by_row.setdefault(it.row, []).append((it.field, it.level))
    assert 1 not in by_row and norm[0]['exptime_list'] == [100.0, 100.0] and norm[0]['count_list'] == [3, 3]
    assert ('objname', 'error') in by_row[2] and ('RA', 'error') in by_row[2]
    assert ('objname', 'error') in by_row[3] and ('exptime', 'error') in by_row[3]
    assert ('exptime', 'warning') in by_row[4] and ('obs_enddate', 'error') in by_row[4]
    assert 5 not in by_row and norm[4]['obsmode'] == 'Deep' and norm[4]['binning'] == '2' and norm[4]['gain'] == '4030'
    assert ('filter_', 'error') in by_row[6]
    assert ('obs_startdate', 'error') in by_row[7]
    formatted = trigger.format_rows(norm, 'My Project', ts)
    assert formatted[0]['exptime'] == '100,100' and formatted[0]['count'] == '3,3' and formatted[0]['RA'] == '10.500000'
    assert formatted[0]['comment'] == 'My_Project' and formatted[0]['objtype'] == 'Open' and formatted[0]['priority'] == '50'
    assert formatted[4]['specmode'] == 'None' and formatted[4]['filter_'] == 'r' and list(formatted[0]) == list(trigger.SHEET_COLUMNS)
    assert ts.worksheet_name('My Project') == f'{ts.period}_My_Project'


def test_request_modes():
    cfg = _cfg()
    assert trigger.n_orders('Spec', 'specall', cfg) == 2 and trigger.n_orders('Color', 'gri', cfg) == 1 and trigger.n_orders('Single', '', cfg) == 1
    gi = trigger.request_obsmode('Color', 'gi', cfg)
    assert gi is not None and set(gi.filters) == {'g', 'i'} and gi.n_filters_per_unit == 1
    assert trigger.request_obsmode('Deep', 'r', cfg).filter_multiplicity()['r'] == 16
    assert trigger.request_obsmode('Search', '', cfg) is None


def test_trigger_page_renders():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'trigger_app.py'), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Observation request'
    assert any('2 target(s) ready' in s.value for s in at.success)
    at.text_input(key='tr_project').set_value('Test Project').run()
    assert not at.exception, at.exception
    assert any('2610_2708_Test_Project' in m.value for m in at.markdown)
    at.selectbox(key='tr_obsmode').set_value('Deep').run()
    assert not at.exception, at.exception
    assert at.selectbox(key='tr_deepmode').value == 'g'
    assert at.number_input(key='tr_binning').disabled and at.number_input(key='tr_gain').disabled
    assert at.number_input(key='tr_binning').value == 1 and at.number_input(key='tr_gain').value == 2750
    assert [h.value for h in at.header][-1] == 'Request preview'      # the preview sits below the checks
