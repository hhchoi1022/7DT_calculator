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
    m2, u2 = trigger.recognise_columns(['objname', 'ra', 'dec', 'obsmode', 'specmode', 'filter', 'gain'])
    assert set(m2.values()) == {'objname', 'RA', 'De'} and u2 == ['obsmode', 'specmode', 'filter', 'gain']   # mode and fixed values never come from a file
    sample = trigger.read_table(trigger.sample_file(), 'sample.csv')
    assert list(sample.columns) == list(trigger.FILE_COLUMNS) and list(sample['objname']) == ['NGC0253', 'M31']
    survey = trigger.read_table(trigger.sample_file('Survey'), 'survey.csv')
    assert list(survey.columns) == ['objname', 'exptime', 'count', 'obs_startdate', 'obs_enddate'] and list(survey['objname']) == ['T01022', 'T01023']
    applied = trigger.apply_mapping(tsv, {**mapping, 'mystery': 'count'})
    assert list(applied.columns) == ['objname', 'RA', 'De', 'count'] and applied.iloc[0]['count'] == 'x'


def test_validation_rules():
    ts = trigger.TriggerSettings.load()
    cfg = _cfg()
    defaults = {'obsmode': 'Spec', 'specmode': 'specall', 'binning': '1', 'gain': '2750'}
    rows = [
        {'objname': 'good', 'RA': '10.5', 'De': '-20.25', 'exptime': '100', 'count': '3'},
        {'objname': '', 'RA': 'abc', 'De': '-20', 'exptime': '100', 'count': '3'},             # missing name, bad RA
        {'objname': 'x' * 101, 'RA': '1', 'De': '2', 'exptime': '200', 'count': '3'},         # too long, exptime above max
        {'objname': 'warn', 'RA': '1', 'De': '2', 'exptime': '60', 'count': '2', 'obs_startdate': '2026-11-11T00:00:00', 'obs_enddate': '2026-11-01'},
        {'objname': 'deep', 'RA': '1', 'De': '2', 'obsmode': 'deep', 'filter_': 'r', 'exptime': '100', 'count': '3', 'binning': '2', 'gain': '4030'},
        {'objname': 'nomode', 'RA': '1', 'De': '2', 'obsmode': 'Deep', 'exptime': '100', 'count': '3'},
        {'objname': 'date', 'RA': '1', 'De': '2', 'exptime': '100', 'count': '3', 'obs_startdate': '11/11/2026'},
        {'objname': 'noexp', 'RA': '1', 'De': '2'},                                            # exposure time and frames missing
        {'objname': 'list', 'RA': '1', 'De': '2', 'exptime': '100,100', 'count': '3,3'},         # lists are not accepted
        {'objname': 'GRB 260901A', 'RA': '1', 'De': '2', 'exptime': '100', 'count': '3'},        # a space in the name
        {'objname': 'SN_2026abc', 'RA': '1', 'De': '2', 'exptime': '100', 'count': '3'},         # an underscore
        {'objname': 'AT2026-xyz', 'RA': '1', 'De': '2', 'exptime': '100', 'count': '3'},         # '-' is allowed
    ]
    norm, issues = trigger.validate_rows(rows, ts, defaults, cfg)
    by_row = {}
    for it in issues:
        by_row.setdefault(it.row, []).append((it.field, it.level))
    assert 1 not in by_row and norm[0]['exptime_s'] == 100.0 and norm[0]['count_n'] == 3
    assert ('objname', 'error') in by_row[2] and ('RA', 'error') in by_row[2]
    assert ('objname', 'error') in by_row[3] and ('exptime', 'error') in by_row[3]
    assert ('exptime', 'warning') in by_row[4] and ('obs_enddate', 'error') in by_row[4]
    assert 5 not in by_row and norm[4]['obsmode'] == 'Deep' and norm[4]['binning'] == '2' and norm[4]['gain'] == '4030'
    assert ('filter_', 'error') in by_row[6]
    assert ('obs_startdate', 'error') in by_row[7]
    assert ('exptime', 'error') in by_row[8] and ('count', 'error') in by_row[8]
    assert ('exptime', 'error') in by_row[9] and ('count', 'error') in by_row[9]
    assert ('objname', 'error') in by_row[10] and ('objname', 'error') in by_row[11] and 12 not in by_row
    assert trigger.REQUIRED == ('objname', 'RA', 'De', 'exptime', 'count')
    formatted = trigger.format_rows(norm, 'My Project', ts)
    assert formatted[0]['exptime'] == '100' and formatted[0]['count'] == '3' and formatted[0]['RA'] == '10.500000'
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
    assert any('1 target(s) ready' in s.value for s in at.success)
    at.text_input(key='tr_project').set_value('Test Project').run()
    assert not at.exception, at.exception
    assert not any('Worksheet:' in m.value for m in at.markdown)
    assert any(m.label == 'Time allocated' for m in at.metric) or any('Unknown Project' in e.value for e in at.error) or any('no key check' in c.value for c in at.caption)
    at.selectbox(key='tr_obsmode').set_value('Deep').run()
    assert not at.exception, at.exception
    assert at.selectbox(key='tr_deepmode').value == 'g'
    assert any('Binning 1, gain 2750 and priority 50 are fixed' in c.value for c in at.caption)
    headers = [h.value for h in at.header]
    assert headers.index('Request preview') < headers.index('Helper plots') and headers[-1] == 'Helper plots'
    at.radio(key='tr_type').set_value('Survey').run()                   # survey: tile IDs, coordinates from the tiles
    assert not at.exception, at.exception
    assert at.session_state['tr_rows'].columns.tolist() == ['objname', 'exptime', 'count', 'obs_startdate', 'obs_enddate']
    assert list(at.session_state['tr_rows']['objname']) == ['T01022', 'T01023']   # the survey examples replace the target examples
    assert any('2 target(s) ready' in s.value for s in at.success)
    assert any('Overhead details' in m.value for m in at.markdown)


def test_survey_rows_and_projects(tmp_path):
    ts = trigger.TriggerSettings.load()
    cfg = _cfg()
    norm, issues = trigger.validate_rows([{'objname': 't1022', 'exptime': '100', 'count': '3'}, {'objname': 'NGC0253', 'exptime': '100', 'count': '3'}],
                                         ts, {'obsmode': 'Spec', 'specmode': 'specall'}, cfg, survey=True)
    assert norm[0]['objname'] == 'T01022' and abs(norm[0]['RA_deg'] - 78.2609) < 1e-3 and norm[0]['RA'] == '78.260870'
    assert [i.field for i in issues] == ['objname'] and issues[0].row == 2
    om = trigger.request_obsmode('Spec', 'specall', cfg)
    tt = trigger.total_time(norm[:1], om)
    assert tt['n_targets'] == 1 and tt['exposure_s'] == 600 and tt['total_s'] > 600 and tt['lines'] and tt['overhead_s'] > 0
    f = tmp_path / 'projects.json'
    f.write_text('{"_description": "x", "GRB followup": {"key": "abc", "allocated_hours": 30}}')
    projects = trigger.load_projects(f)
    assert projects == {'GRB_followup': {'key': 'abc', 'allocated_hours': 30}}
    assert trigger.check_project('GRB followup', 'abc', projects) == (True, 'Project recognised.', 30.0)
    assert trigger.check_project('GRB followup', 'nope', projects)[0] is False and trigger.check_project('other', '', projects)[0] is False
    assert trigger.check_project('anything', '', None)[0] is True and trigger.load_projects(tmp_path / 'missing.json') is None


def test_ellipse_tile_match():
    from sevendt_calc import tiles
    ts = tiles.TileSet()
    circle = ts.match(10.0, -20.0, radius_deg=1.0)
    ellipse = ts.match(10.0, -20.0, ellipse=(2.0, 0.5, 90.0))
    assert ellipse.ellipse == (2.0, 0.5, 90.0) and ellipse.shape is not None and len(ellipse.tiles) >= 1
    wide = ts.match(10.0, -20.0, ellipse=(3.0, 3.0, 0.0))
    assert len(wide.tiles) > len(circle.tiles)
    fig = tiles.plot_tile_match_interactive(ellipse)
    assert any('ellipse' in (t.name or '') for t in fig.data)
    first = ellipse.tiles[0].id
    fig2 = tiles.plot_tile_match_interactive(ellipse, excluded={first}, added={'T00000'}, clickable=True)
    markers = [t for t in fig2.data if t.name == 'tiles']
    assert len(markers) == 1 and first in list(markers[0].customdata) and fig2.layout.clickmode == 'event+select'
    assert any('excluded' in (a.text or '') for a in fig2.layout.annotations)
