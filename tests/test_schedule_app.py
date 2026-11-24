"""Scheduling page: the Phase-2 request core (parsing, project matching, nights, plans, night packing) and a headless run of the page."""
from datetime import date
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from sevendt_calc import phase2
from sevendt_calc import visibility as vis

SAMPLE = """# comment line
[PROPOSAL]
PROPOSAL_TITLE: A test proposal
PI_NAME: Kim Mi-Ryang, Someone Else
PI_EMAIL: a@b.c
[END PROPOSAL]

[TARGET]
TARGET_TYPE: Target
OBJ_NAME: NGC 253
RA: 00:47:33.1
DEC: -25:17:18
OBS_MODE: Spec
OBS_SUBMODE: specall
EXPTIME_PER_FRAME: 100 s
NUM_FRAMES: 9
OBS_WINDOW: 2026-12-01 to 2026-12-31
OBS_WINDOW: 2027-08-01 to 2027-08-31
NOTE: first line
      continued line

TARGET_TYPE: Survey
OBJ_NAME:
RA:
DEC:
TILE_ID: T01022
OBS_MODE: Color
OBS_SUBMODE: gr
EXPTIME_PER_FRAME: 120
NUM_FRAMES: 3
OBS_WINDOW: 2027-03-04 00:30 to 2027-03-04 08:30
NOTE:
TARGET_TYPE: Target
OBJ_NAME: bad
RA:83.8 deg
DEC: 5.0 deg
OBS_MODE: Spec
OBS_SUBMODE: specred
EXPTIME_PER_FRAME:
NUM_FRAMES: 2
OBS_WINDOW: 2027-01-01 - 2026-12-01
[END TARGET]
"""


def test_parse_request():
    req = phase2.parse_request(SAMPLE, 'x/test.txt')
    assert req.title == 'A test proposal' and req.lead_pi == 'Kim Mi-Ryang' and req.name == 'test.txt' and not req.issues
    assert [t.label for t in req.targets] == ['NGC 253', 'T01022', 'bad'] and [t.display for t in req.targets] == ['NGC 253', 'T01022', 'bad']
    twice = phase2.parse_request(SAMPLE.replace('OBJ_NAME: bad', 'OBJ_NAME: NGC 253'))
    assert [t.display for t in twice.targets] == ['NGC 253 #1', 'T01022', 'NGC 253 #3']
    a, b, c = req.targets
    assert abs(a.ra - 11.888) < 0.01 and abs(a.dec + 25.288) < 0.01 and a.exptime == 100 and a.nframes == 9 and a.note == 'first line continued line'
    assert len(a.windows) == 2 and not a.timed and a.windows[0].start == __import__('datetime').datetime(2026, 12, 1) and a.windows[0].end == __import__('datetime').datetime(2027, 1, 1)
    assert a.window_start == date(2026, 12, 1) and a.window_end == date(2027, 8, 31) and not a.issues
    assert b.target_type == 'Survey' and b.tile_id == 'T01022' and b.has_coordinates and b.mode_text == 'Color/gr' and b.timed and b.windows[0].hours == 8
    assert abs(c.ra - 83.8) < 1e-6 and c.dec == 5.0 and c.exptime is None
    assert any('EXPTIME' in i for i in c.issues) and any('ends before' in i for i in c.issues)
    assert req.requested_start == date(2026, 12, 1) and req.requested_end == phase2.PROGRAM_END


def test_name_forms_and_matching():
    assert phase2._name_forms('Kim Mi-Ryang') & phase2._name_forms('Miryang Kim')
    assert phase2._name_forms('Deokkeun An and the others') & phase2._name_forms('Deokkeun An, Young Sun Lee')
    assert phase2._name_forms('MooKeon Jung') & phase2._name_forms('Moo-Keon Jung')
    assert not (phase2._name_forms('Jiwon Park') & phase2._name_forms('Sangjun Cha'))
    projects = {'26004': {'pi': 'Seo-Won Chang', 'observation': 'Target', 'mode': 'Others'}, '26012': {'pi': 'Seo-Won Chang', 'observation': 'ToO', 'mode': 'Spec'},
                '26001': {'pi': 'Miryang Kim', 'observation': 'Target', 'mode': 'Spec'}}
    req = phase2.parse_request(SAMPLE)
    assert phase2.match_project(req, projects) == '26001'
    too = phase2.parse_request(SAMPLE.replace('Kim Mi-Ryang, Someone Else', 'Seo-Won Chang').replace('NOTE: first line', 'NOTE: ToO. Start within 5 min.'))
    assert phase2.match_project(too, projects) == '26012'
    color = phase2.parse_request(SAMPLE.replace('Kim Mi-Ryang, Someone Else', 'Seo-Won Chang').replace('OBS_MODE: Spec', 'OBS_MODE: Color'))
    assert phase2.match_project(color, projects) == '26004'
    assert phase2.match_project(req, None) == ''


def test_compute_times_and_moon():
    req = phase2.parse_request(SAMPLE)
    phase2.compute_times(req)
    a, b, c = req.targets
    assert a.n_orders == 2 and a.exposure_s == 100 * 9 * 2 and a.total_s > a.exposure_s
    assert b.exposure_s == 120 * 3 and b.total_s >= b.exposure_s and c.total_s == 0
    assert abs(req.exposure_h - (1800 + 360) / 3600) < 1e-9
    f = np.array([0.1, 0.5, 0.9])
    assert phase2.moon_allowed('Dark', f).tolist() == [True, False, False]
    assert phase2.moon_allowed('Dark & Gray', f).tolist() == [True, True, False]
    assert phase2.moon_allowed('Any', f).tolist() == [True, True, True]
    assert phase2.moon_allowed('Bright', f).tolist() == [False, False, True]


def test_pack_night():
    assert phase2.pack_night([('A', 1.0, [(0.0, 1.0)]), ('B', 1.0, [(0.0, 1.0)])]) == {'A': 1.0, 'B': 0.0}
    got, slots = phase2.pack_night_slots([('A', 1.0, [(0.0, 1.0)]), ('B', 1.0, [(0.0, 3.0)])])
    assert got == {'A': 1.0, 'B': 1.0} and slots['A'] == [(0.0, 1.0)] and slots['B'] == [(1.0, 2.0)]
    got = phase2.pack_night([('A', 2.0, [(0.0, 2.0)]), ('B', 1.0, [(1.0, 2.0)]), ('C', 1.0, [(1.5, 4.0)])])
    assert abs(got['A'] + got['B'] - 2.0) < 1e-9 and got['C'] == 1.0            # A and B cannot both fit before 2 h
    assert phase2.merge_intervals([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)] and phase2.interval_hours([(0, 2), (3, 4)]) == 3


def test_nights_and_plan():
    grid = vis.season_grid(date(2026, 12, 1), n_nights=40)
    moon = vis.moon_illumination_nights(grid)
    assert moon.shape == (40,) and 0 <= moon.min() and moon.max() <= 1
    req = phase2.parse_request(SAMPLE)
    phase2.compute_times(req)
    table = phase2.night_table(req, grid, moon)
    assert table.max_alt.shape == (40, 3) and table.hours.shape == (40, 3) and table.starts.shape == (40, 3, 4)
    assert table.requested[:31, 0].all() and not table.requested[31:, 0].any()          # December window of NGC 253
    assert not table.requested[:, 1].any()                                              # the timed window of T01022 is in March
    alt = vis.nightly_max_altitude(grid, [69.0, 11.888], [-25.0, -25.288])               # RA 4.6 h transits near local midnight in early December
    assert alt['transit'][0, 0] and abs(alt['max_alt'][0, 0] - alt['culmination'][0]) < 1e-6 and 0 < alt['at_hours'][0, 0] < 6
    assert not alt['transit'][0, 1] and alt['max_alt'][0, 1] < alt['culmination'][1] and alt['at_hours'][0, 1] == 0   # NGC 253 transits before the night
    assert (alt['max_alt'] <= alt['culmination'][None, :] + 1e-6).all()
    assert abs(table.hours[0, 0] - np.nansum(table.ends[0, 0] - table.starts[0, 0])) < 1e-9
    settings = phase2.default_settings(req, table)
    assert settings.start == date(2026, 12, 1) and settings.enabled and settings.use_moon
    plan = phase2.make_plan(req, table, phase2.PlanSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), hours_per_night=1.0, use_moon=False))
    assert plan.nights == 10 and plan.planned_h[:10].max() <= 1.0 and plan.planned_h[10:].sum() == 0 and plan.planned_total <= 10
    capped = phase2.make_plan(req, table, phase2.PlanSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), hours_per_night=1.0, use_moon=False, max_nights=3))
    assert capped.nights == 3
    off = phase2.make_plan(req, table, phase2.PlanSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), hours_per_night=1.0, enabled=False))
    assert off.nights == 0 and off.status == 'off'
    assert phase2.night_conflicts({'a': plan, 'b': plan}, list(grid.dates)) == []       # two 1 h jobs fit in the same hours above the limit
    full = phase2.make_plan(req, table, phase2.PlanSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), hours_per_night=10.0, use_moon=False))
    conflicts = phase2.night_conflicts({'a': full, 'b': full}, list(grid.dates))
    assert len(conflicts) == 10 and all(c['date'] <= date(2026, 12, 10) and 'b' in c['short'] for c in conflicts)   # the whole availability twice does not
    rows = phase2.schedule_rows([req], {req.key: plan})
    assert len(rows) == 3 and rows[0]['obsdate_window_start'] == '2026-12-01' and rows[0]['hours_per_night'] == 1.0


def test_schedule_file_roundtrip(tmp_path):
    settings = {'26001': phase2.PlanSettings(start=date(2027, 1, 2), end=date(2027, 3, 4), hours_per_night=1.5, use_moon=False, enabled=True, max_nights=4)}
    phase2.save_schedule(tmp_path, settings)
    assert phase2.load_schedule(tmp_path) == settings
    (tmp_path / 'req.txt').write_text(SAMPLE)
    (tmp_path / phase2.MAPPING_FILE).write_text('{"req.txt": "26099"}')
    reqs = phase2.load_requests(tmp_path, {'26099': {'pi': 'Nobody', 'allocated_hours': 5, 'observation': 'Target', 'mode': 'Spec', 'sky_background': 'Dark', 'key': 'x'}})
    assert len(reqs) == 1 and reqs[0].project_id == '26099' and reqs[0].allocated_hours == 5 and 'key' not in reqs[0].project
    assert phase2.folder_signature(tmp_path) and phase2.folder_signature(tmp_path / 'missing') == ()


def test_schedule_page_runs():
    if not phase2.requests_dir().is_dir():
        return
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'schedule_app.py'), default_timeout=300).run()
    assert not at.exception
    assert at.title[0].value == '7DT Scheduling' and at.selectbox[0].value
    assert [m.label for m in at.metric][:3] == ['Allocated', 'Requested exposure', 'Requested with overheads']
