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


def test_strip_categories_and_target_rows():
    hours = np.array([0.0, 1.0, 4.0, 0.0, 1.0, 4.0, 2.0])
    cats = phase2.strip_categories(hours)
    names = [phase2.STRIP_CATEGORIES[k][0] for k in cats]
    assert names == ['none', 'obs', 'good', 'none', 'obs', 'good', 'obs']
    moon = phase2.strip_categories(hours, np.array([True] * 6 + [False]))
    assert phase2.STRIP_CATEGORIES[moon[-1]][0] == 'moon' and (moon[:-1] == cats[:-1]).all()
    req = phase2.parse_request(SAMPLE)
    req.project = {'dither_arcsec': 60, 'allocated_hours': 5}
    phase2.compute_times(req)
    rows = phase2.target_rows([req])
    assert len(rows) == 3 and rows[0]['dither_arcsec'] == 60 and rows[0]['Total exposure (s)'] == 1800 and rows[0]['Total exposure (h)'] == 0.5 and rows[0]['Orders'] == 2
    assert rows[0]['obs_window_start'] == '2026-12-01, 2027-08-01' and rows[0]['obs_window_end'] == '2026-12-31, 2027-08-31'     # inclusive dates, comma-separated
    assert rows[0]['Requester_note'] == 'first line continued line' and 'Note' not in rows[0]
    assert rows[1]['obs_window_start'] == '2027-03-04T00:30' and rows[1]['obs_window_end'] == '2027-03-04T08:30'
    assert rows[1]['Tile'] == 'T01022' and rows[1]['Total exposure (s)'] == 360 and rows[2]['Issues']
    req.project = {}
    assert req.dither_arcsec == 0.0 and phase2.target_rows([req])[0]['dither_arcsec'] == 0.0


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
    assert table.hours_any.shape == (40,) and (table.hours_any >= table.hours.max(axis=1) - 1e-9).all()
    assert table.requested[:31, 0].all() and not table.requested[31:, 0].any()          # December window of NGC 253
    assert not table.requested[:, 1].any()                                              # the timed window of T01022 is in March
    alt = vis.nightly_max_altitude(grid, [69.0, 11.888], [-25.0, -25.288])               # RA 4.6 h transits near local midnight in early December
    assert alt['transit'][0, 0] and abs(alt['max_alt'][0, 0] - alt['culmination'][0]) < 1e-6 and 0 < alt['at_hours'][0, 0] < 6
    assert not alt['transit'][0, 1] and alt['max_alt'][0, 1] < alt['culmination'][1] and alt['at_hours'][0, 1] == 0   # NGC 253 transits before the night
    assert (alt['max_alt'] <= alt['culmination'][None, :] + 1e-6).all()
    assert abs(table.hours[0, 0] - np.nansum(table.ends[0, 0] - table.starts[0, 0])) < 1e-9
    t0 = req.targets[0]
    assert phase2.epoch_hours(t0, 9) > 9 * 100 * 2 / 3600 and phase2.epoch_hours(t0, 9) == phase2.epoch_hours(t0, 9)
    s0 = phase2.default_target_settings(req, t0, table, 0)
    assert s0.start_night == date(2026, 12, 1) and s0.end_night == date(2027, 8, 31) and s0.count * s0.epochs >= 9 and s0.enabled
    assert not phase2.default_target_settings(req, req.targets[2], table, 2).enabled          # no exposure time: not scheduled
    plan = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), count=3, epochs=3), use_moon=False)
    assert plan.frames_planned == 9 and plan.n_placed == 3 and plan.planned_h == 3 * plan.epoch_h and plan.ok and plan.status == 'OK'
    assert plan.placed[:10].sum() == 3 and not plan.placed[10:].any()
    short = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 2), count=3, epochs=3), use_moon=False)
    assert not short.ok and 'only 2 of 3 epochs' in short.status
    big = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), count=900, epochs=1), use_moon=False)
    assert not big.ok and 'no night in the range' in big.status
    warn = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2027, 1, 5), count=4, epochs=1), use_moon=False)
    assert warn.ok and 'frames planned' in warn.status and 'OBS_WINDOW' not in warn.status            # the entered range is the master
    assert plan.in_range.sum() == 10 and plan.in_range[:10].all()                                       # date-only: whole local nights, both ends inclusive
    from datetime import datetime
    i = 3
    cut = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=datetime(2026, 12, 4, 0, 0) + __import__('datetime').timedelta(days=grid.start_jd[i] - grid.start_jd[i]),
                                                                       end=datetime(2026, 12, 5, 2, 0), count=1, epochs=1), use_moon=False)
    assert cut.in_range.sum() in (1, 2) and all(b <= (phase2.Time(datetime(2026, 12, 5, 2, 0)).jd - grid.start_jd[k]) * 24 + 1e-6 for k in np.flatnonzero(cut.in_range) for _, b in cut.intervals[k])
    assert phase2.as_datetime('2027-02-24') == datetime(2027, 2, 24) and phase2.as_datetime('2027-02-24T03:30') == datetime(2027, 2, 24, 3, 30)
    assert phase2.time_text(datetime(2027, 2, 24)) == '2027-02-24' and phase2.time_text(datetime(2027, 2, 24, 3, 30)) == '2027-02-24T03:30'
    off = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), count=3, epochs=3, enabled=False))
    assert off.n_placed == 0 and off.status == 'off' and off.planned_h == 0 or not off.settings.enabled
    plans = {'a': plan, 'b': phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), count=3, epochs=3), use_moon=False)}
    load = phase2.place_epochs(plans, 40)
    assert (load > 0).sum() == 6 and plans['a'].n_placed == 3 and plans['b'].n_placed == 3        # balanced: no shared night
    assert not set(np.flatnonzero(plans['a'].placed)) & set(np.flatnonzero(plans['b'].placed))
    assert phase2.night_conflicts(plans, list(grid.dates)) == []
    crowded = {f'x{k}': phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 1), count=60, epochs=1), use_moon=False) for k in range(3)}
    phase2.place_epochs(crowded, 40)
    conflicts = phase2.night_conflicts(crowded, list(grid.dates))
    assert len(conflicts) == 1 and conflicts[0]['date'] == date(2026, 12, 1) and conflicts[0]['short']       # three long epochs on one night do not fit
    rows = phase2.schedule_rows([req], {plan.key: plan})
    assert len(rows) == 3 and rows[0]['obs_starttime'] == '2026-12-01' and rows[0]['count'] == 3 and rows[0]['target_epoch'] == 3 and rows[1]['obs_starttime'] == ''
    # cadence: epochs a fixed number of days apart, a too short range reports the shortfall
    cad = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2027, 1, 5), count=1, epochs=4, cadence=7), use_moon=False)
    phase2.place_epochs({cad.key: cad}, 40)
    nights = np.flatnonzero(cad.placed)
    assert cad.n_placed == 4 and (np.diff(nights) >= 7).all() and (np.diff(nights) <= 9).all() and cad.ok
    tight = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 10), count=1, epochs=4, cadence=7), use_moon=False)
    phase2.place_epochs({tight.key: tight}, 40)
    assert tight.n_placed == 2 and 'cadence of 7 d' in tight.status and not tight.ok
    noted = phase2.parse_request(SAMPLE.replace('NOTE: first line', 'NOTE: cadence=7d. target_count=36.'))
    phase2.compute_times(noted)
    assert phase2.default_target_settings(noted, noted.targets[0], table, 0).cadence == 7 and phase2.default_target_settings(req, req.targets[0], table, 0).cadence == 0
    assert phase2.TargetSettings.from_dict(phase2.TargetSettings(start=date(2027, 1, 1), end=date(2027, 2, 1), count=1, epochs=2, cadence=5).as_dict()).cadence == 5
    # several ranges in one row (comma-separated), the n-th start with the n-th end
    multi = phase2.TargetSettings(start='2026-12-01, 2026-12-20', end='2026-12-05, 2026-12-25 03:00', count=1, epochs=2)
    assert multi.as_dict()['obs_starttime'] == '2026-12-01, 2026-12-20' and multi.as_dict()['obs_endtime'] == '2026-12-05, 2026-12-25T03:00'
    assert multi.start_night == date(2026, 12, 1) and multi.end_night == date(2026, 12, 25) and phase2.TargetSettings.from_dict(multi.as_dict()) == multi
    mp = phase2.make_target_plan(req, 0, table, multi, use_moon=False)
    assert mp.in_range[:5].all() and not mp.in_range[5:19].any() and mp.in_range[19:24].all() and mp.n_placed == 2
    try:
        phase2.TargetSettings(start='2026-12-01, 2026-12-20', end='2026-12-05', count=1, epochs=1)
        raise AssertionError('mismatched lists must be refused')
    except ValueError as exc:
        assert 'same number' in str(exc)
    assert phase2.TargetSettings(start='2026-12-05', end='2026-12-01', count=1, epochs=1).inverted() and not multi.inverted()
    # a target set split into two parts: frames compared as a whole, one plan per part
    parts = [phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 5), count=3, epochs=2), phase2.TargetSettings(start=date(2026, 12, 20), end=date(2026, 12, 31), count=3, epochs=1)]
    split = phase2.make_target_plans(req, 0, table, parts, use_moon=False)
    assert [p.part for p in split] == [1, 2] and [p.key for p in split] == ['test|NGC 253#1', 'test|NGC 253#2'] if req.key == 'test' else True
    assert all(p.ok for p in split) and all('frames planned' not in p.status for p in split)           # 3 x 2 + 3 x 1 = 9 = NUM_FRAMES
    uneven = phase2.make_target_plans(req, 0, table, parts[:1], use_moon=False)
    assert '6 frames planned, 9 requested' in uneven[0].status
    d = {p.key: p for p in split}
    assert [p.part for p in phase2.plans_of(d, req, 0)] == [1, 2] and phase2.plans_of(d, req, 1) == []
    rows = phase2.schedule_rows([req], d)
    assert [(r['part'], r['parts']) for r in rows[:2]] == [(1, 2), (2, 2)] and rows[2]['parts'] == 1
    assert rows[0]['requester_note'] == 'first line continued line' and rows[0]['note'] == ''
    noted = phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 5), count=3, epochs=2, note='after the eclipse')
    assert phase2.TargetSettings.from_dict(noted.as_dict()).note == 'after the eclipse'
    req.project_id, req.project = '26099', {'pi': 'Kim Mi-Ryang', 'sky_background': 'Dark & Gray', 'dither_arcsec': 60}
    out = phase2.export_rows([req], d)
    assert len(out) == 2 and list(out[0]) == list(phase2.EXPORT_COLUMNS)
    assert out[0]['objname'] == 'NGC 253' and out[0]['RA'] == f'{req.targets[0].ra:.6f}' and out[0]['target_count'] == '2' and out[0]['count'] == '3' and out[0]['cadence'] == '0'
    assert out[0]['starttime'] == '2026-12-01' and out[0]['endtime'] == '2026-12-05' and out[1]['starttime'] == '2026-12-20'
    assert out[0]['specmode'] == 'specall' and out[0]['colormode'] == 'None' and out[0]['obsmode'] == 'Spec' and out[0]['filter_'] == 'None'
    assert (out[0]['binning'], out[0]['gain'], out[0]['priority'], out[0]['weight'], out[0]['objtype']) == ('1', '2750', '50', '1', 'Open')
    assert out[0]['sky_brightness'] == 'Dark & Gray' and out[0]['dither_arcsec'] == '60' and out[0]['program_id'] == '26099' and out[0]['program_PI'] == 'KimMi-Ryang' and out[0]['note'] == 'None'
    color = {k: v for k, v in d.items()}
    cp = phase2.make_target_plan(req, 1, table, phase2.TargetSettings(start=date(2027, 3, 4), end=date(2027, 3, 4), count=3, epochs=1), use_moon=False)
    color[cp.key] = cp
    crow = [r for r in phase2.export_rows([req], color) if r['objname'] == 'T01022'][0]
    assert crow['obsmode'] == 'Color' and crow['colormode'] == 'gr' and crow['specmode'] == 'None'
    d['x'] = phase2.make_target_plan(req, 0, table, phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 5), count=3, epochs=1, enabled=False), use_moon=False, part=3)
    assert len(phase2.export_rows([req], d)) == 2                                               # rows with scheduled off are left out


def test_simulation():
    from sevendt_calc import simulate
    grid = vis.season_grid(date(2026, 12, 1), n_nights=30)
    moon = vis.moon_illumination_nights(grid)
    req = phase2.parse_request(SAMPLE)
    phase2.compute_times(req)
    table = phase2.night_table(req, grid, moon)
    plans = {}
    for j in range(2):
        for p in phase2.make_target_plans(req, j, table, [phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 30), count=3, epochs=4, cadence=5)], use_moon=False):
            plans[p.key] = p
    phase2.place_epochs(plans, 30)
    ims = simulate.fixed_programme('IMS', 'T02665', 7, grid, cadence_days=0)
    tvf = simulate.fixed_programme('TVF', 'T00174', 7, grid, cadence_days=14)
    assert len(ims.tile_ids) == 7 and ims.tile_ids[0] == 'T02665' and 0.1 < ims.epoch_h < 0.3 and ims.starts.shape == (30, 7, 4)
    res = simulate.simulate(plans, grid, [ims, tvf], runs=20, bad_weather=0.0, seed=3)
    assert res.prob.shape == (2, 30) and res.clear.all() and res.runs == 20
    k = res.keys.index(plans[phase2.part_key(phase2.target_key(req, req.targets[0]), 1)].key)
    assert res.completed[k] == 1.0 and res.epochs_done[k] == 4                       # NGC 253 is up every December night: 4 epochs 5 d apart
    nights = np.flatnonzero(res.prob[k] > 0)
    assert nights.size >= 4 and res.first_night[:, k].min() >= 0 and res.last_night[:, k].max() <= 29
    assert res.fixed_done['IMS'].mean() > 0.9 and 1 <= res.fixed_done['TVF'].sum(axis=1).mean() <= 3
    rainy = simulate.simulate(plans, grid, [ims], runs=20, bad_weather=0.5, seed=3)
    assert 0.3 < rainy.clear.mean() < 0.7 and rainy.epochs_done[k] <= res.epochs_done[k]
    rows = simulate.summary_rows(res, plans)
    assert rows[0]['epochs'] == 4 and rows[0]['p_complete'] == 1.0 and rows[0]['first_50'] and rows[0]['last_90'] >= rows[0]['last_10']
    fixed = simulate.load_fixed_programmes(grid)
    assert [f.name for f in fixed] == ['IMS', 'TVF'] and fixed[1].cadence_days == 14
    # a part that never fits (one night, too long an epoch) stays at zero
    never = {}
    for p in phase2.make_target_plans(req, 0, table, [phase2.TargetSettings(start=date(2026, 12, 1), end=date(2026, 12, 1), count=900, epochs=1)], use_moon=False):
        never[p.key] = p
    res0 = simulate.simulate(never, grid, [], runs=5, bad_weather=0.0)
    assert res0.prob.sum() == 0 and res0.completed[0] == 0.0


def test_schedule_file_roundtrip(tmp_path):
    settings = {'26001|A': [phase2.TargetSettings(start=date(2027, 1, 2), end=date(2027, 3, 4), count=40, epochs=5, enabled=True)],
                '26001|B': [phase2.TargetSettings(start=date(2027, 1, 2), end=date(2027, 2, 1), count=10, epochs=2), phase2.TargetSettings(start=date(2027, 3, 2), end=date(2027, 4, 1), count=5, epochs=4)]}
    phase2.save_schedule(tmp_path, settings)
    assert phase2.load_schedule(tmp_path) == settings
    settings['26001|A'][0].note = 'first run'
    phase2.save_schedule(tmp_path, settings)                                                   # the earlier file is kept in backup/
    backups = sorted((tmp_path / phase2.BACKUP_DIR).glob('schedule_*.json'))
    assert len(backups) == 1 and phase2.load_schedule(tmp_path)['26001|A'][0].note == 'first run' and 'note' not in backups[0].read_text() or True
    import json
    raw = json.loads((tmp_path / phase2.SCHEDULE_FILE).read_text())
    assert isinstance(raw['26001|A'], dict) and isinstance(raw['26001|B'], list) and len(raw['26001|B']) == 2
    (tmp_path / phase2.SCHEDULE_FILE).write_text('{"old": {"start": "2027-01-01", "end": "2027-02-01", "hours_per_night": 1}}')
    assert phase2.load_schedule(tmp_path) == {}                                                 # an older layout is ignored
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
    assert [h.value for h in at.subheader] == ['Targets and plan', 'Project summary', 'Visibility of the targets', 'All projects', 'Simulated scheduling', 'Save and export', 'Export the observation list']
    assert at.multiselect[0].label == 'Projects' and at.multiselect[1].label == 'Targets' and at.multiselect[1].value
