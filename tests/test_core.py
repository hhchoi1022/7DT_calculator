"""Sanity tests for the sevendt_calc core (run: pytest tests)."""
import math

import numpy as np
import pytest
from pathlib import Path
from astropy.time import Time

from sevendt_calc import config, etc, overhead, photometry as ph, targets as tg, tiles as tl
from sevendt_calc import visibility as vis


# ---------------------------------------------------------------- photometry
def test_flat_spectrum_gives_same_ab_magnitude_in_every_band():
    flat = ph.flat_spectrum(18.0)
    for name, curve in ph.load_filter_curves().items():
        if curve.lam_min >= flat.lam_min and curve.lam_max <= flat.lam_max:
            assert abs(flat.synthetic_ab_mag(curve) - 18.0) < 1e-4, name


def test_fnu_input_matches_flam_input():
    flat = ph.flat_spectrum(20.0)
    from_fnu = ph.Spectrum.from_arrays(flat.wavelength, flat.fnu, flux_unit='fnu')
    curve = ph.get_filter_curve('m625')
    assert abs(from_fnu.synthetic_ab_mag(curve) - flat.synthetic_ab_mag(curve)) < 1e-6


def test_vega_to_ab_offsets():
    assert ph.to_ab(10.0, 'B') == pytest.approx(9.91)
    assert ph.to_ab(10.0, 'I') == pytest.approx(10.45)
    assert ph.to_ab(10.0, 'r') == 10.0


def test_spectrum_scaled_to_magnitude():
    spec = ph.flat_spectrum(15.0)
    curve = ph.get_filter_curve('V')
    scaled = spec.scaled_to(19.02, curve)
    assert scaled.synthetic_ab_mag(curve) == pytest.approx(19.02, abs=1e-6)


def test_target_uses_spectrum_and_saturation_note():
    t = tg.Target(name='bright', ra=10, dec=-20, mag=10.5, mag_filter='V', spectrum=ph.flat_spectrum(12.0))
    mag, source = t.ab_magnitude('m625')
    assert source == 'spectrum' and mag == pytest.approx(10.52, abs=1e-3)
    assert t.saturation_note(['g']) is not None
    assert tg.Target(name='faint', mag=19, mag_filter='r').saturation_note() is None


# ---------------------------------------------------------------- coordinates
@pytest.mark.parametrize('text', ['165.514', '11:02:03.36', '11h02m03.36s', '11 02 03.36'])
def test_parse_ra_formats(text):
    assert tg.parse_ra(text) == pytest.approx(165.514, abs=1e-3)


@pytest.mark.parametrize('text', ['-30.2031', '-30:12:11', '-30d12m11s', '-30 12 11'])
def test_parse_dec_formats(text):
    assert tg.parse_dec(text) == pytest.approx(-30.2031, abs=1e-3)


def test_missing_fields():
    assert tg.Target(name='x').missing(need_photometry=True) == ['ra', 'dec', 'mag', 'mag_filter']
    assert tg.Target(name='x', ra=1, dec=2, spectrum=ph.flat_spectrum(18)).missing(need_photometry=True) == []


# ---------------------------------------------------------------- configuration
def test_live_config_reads_folder():
    site = config.load_site()
    assert site.latitude < 0 and site.longitude < 0
    assert 'specall' in config.load_specmodes()
    om = config.resolve_obsmode('Spec', 'specall')
    assert om.n_filters_per_unit == 2 and om.is_specall
    deep = config.resolve_obsmode('Deep', 'g,r')
    assert deep.n_filters_per_unit == 2 and len(deep.units) >= 15
    color = config.resolve_obsmode('Color', 'gri')
    assert color.filters == ['g', 'r', 'i']


# ---------------------------------------------------------------- ETC
def test_etc_exptime_and_snr_are_consistent():
    calc = etc.ExposureCalculator(etc.DepthModel())
    cond = etc.conditions_for(moon_phase=0.3, moon_separation=90, hour_since_sunset=4, seeing=1.8)
    target = tg.Target(name='t', ra=10, dec=-20, mag=19.5, mag_filter='r')
    need = calc.exptime(target, ['g', 'm625'], snr=10, count=3, cond=cond)
    for r in need:
        back = calc.snr(target, [r.filter], r.exptime_single, 3, cond)[0]
        assert back.snr_stacked == pytest.approx(10.0)
        assert back.snr_single == pytest.approx(10.0 / math.sqrt(3))
        assert back.mag_err == pytest.approx(1.0857 / 10, rel=1e-3)


def test_etc_interpolates_missing_filters():
    model = etc.DepthModel()
    assert model.parameters('m625')[4] == 'model'
    assert model.parameters('m438')[4] == 'interpolated'
    assert model.parameters('m386')[4] == 'extrapolated'
    assert model.parameters('ginv')[4] == 'unavailable'
    cond = etc.conditions_for()
    d_lo, d, d_hi = (model.depth(f, cond).ul5_ref for f in ('m425', 'm438', 'm450'))
    assert min(d_lo, d_hi) <= d <= max(d_lo, d_hi)


def test_conditions_from_target_and_time():
    cond = etc.conditions_for(ra=11.888, dec=-25.288, obstime=Time('2026-09-23T03:00:00'), seeing=1.5)
    assert 0 <= cond.moon_phase <= 1 and 0 <= cond.moon_separation <= 180 and 0 <= cond.hour_since_sunset < 24
    assert cond.seeing == 1.5 and cond.source == 'target and time'


# ---------------------------------------------------------------- overhead
def test_overhead_specall_two_filters():
    cfg = overhead.OverheadConfig(autofocus=150, filter_change=10, slewing=5, readout=10)
    om = config.resolve_obsmode('Spec', 'specall')
    res = overhead.estimate([overhead.ObsRequest('a', 100, 5, om)], cfg)
    comp, cnt = res.components(), res.counts()
    assert cnt == {'n_frames': 10, 'n_filter_changes': 2, 'n_autofocus': 0, 'n_slews': 1}
    assert comp == {'exposure': 1000, 'readout': 100, 'filter_change': 20, 'autofocus': 0, 'slewing': 5, 'dispatch': 0}
    assert res.total == 1125


def test_overhead_autofocus_rule():
    cfg = overhead.OverheadConfig.load()
    assert cfg.autofocus_modes == ('Spec/specred',)
    om_gri = config.resolve_obsmode('Color', 'gri')
    req = overhead.ObsRequest('a', 60, 3, om_gri, 'per_filter')
    assert req.autofocus_policy == 'none' and req.policy_forced                # only the listed modes autofocus
    assert overhead.estimate([req]).counts()['n_autofocus'] == 0
    deep = overhead.ObsRequest('d', 60, 1, config.resolve_obsmode('Deep', 'g,r,i'))
    assert overhead.estimate([deep]).counts()['n_autofocus'] == 0
    for name, n_expected in (('specall', 0), ('specblue', 0), ('specred', None)):
        om = config.resolve_obsmode('Spec', name)
        res = overhead.estimate([overhead.ObsRequest('a', 100, 5, om, 'per_filter'), overhead.ObsRequest('b', 100, 5, om, 'none')])
        counts = [tb.counts()['n_autofocus'] for tb in res.targets]
        if n_expected == 0:
            assert counts == [0, 0], name
        else:
            assert counts[0] == om.n_filters_per_unit and counts[1] == om.n_filters_per_unit, name   # specred: once per filter, forced on
    custom = overhead.OverheadConfig(autofocus_modes=('Color/gri',))
    assert overhead.allowed_autofocus_policies(om_gri, custom) == ['per_filter']


def test_public_spec_order():
    assert config.load_public_modes()['Spec'] == ['specall', 'specblue', 'specred']


# ---------------------------------------------------------------- visibility
def test_visibility_flags_low_altitude_and_night():
    site = config.load_site()
    cons = vis.Constraints(min_alt=30, max_alt=88, moon_sep=40, sun_alt=-18)
    obstime = Time('2026-09-23T03:00:00')
    res = vis.compute_all([tg.Target(name='south', ra=11.888, dec=-25.288), tg.Target(name='north', ra=10.685, dec=41.269)], obstime, site, cons)
    assert res[0].ok and res[0].windows
    assert not res[1].ok and 'altitude' in res[1].issues[0]
    day = vis.compute_visibility(tg.Target(name='s', ra=11.888, dec=-25.288), Time('2026-09-23T15:00:00'), site, cons)
    assert any('not night' in i for i in day.issues)


# ---------------------------------------------------------------- tiles
@pytest.fixture(scope='module')
def tileset():
    return tl.TileSet()


def test_tile_centre_matches_its_tile(tileset):
    i = 9290
    m = tileset.match(tileset.ra[i], tileset.dec[i])
    assert [t.id for t in m.tiles] == [str(tileset.ids[i])]
    assert m.ok and m.primary.distance_to_edge_arcmin > 20


def test_tile_matching_across_ra_wrap_and_near_pole(tileset):
    assert tileset.match(359.9, -30.5).tiles and tileset.match(0.1, -30.5).tiles
    assert tileset.match(120.0, -89.0).containing


def test_tile_edge_warning_and_outside_footprint(tileset):
    m = tileset.match(1.5, -30.12, edge_tolerance_arcmin=4)
    assert any(t.near_edge for t in m.tiles) and m.warnings
    m = tileset.match(10.0, 45.0)
    assert not m.tiles and 'No 7DS tile' in m.warnings[0] and m.nearest_id


def test_tile_radius_matching(tileset):
    m = tileset.match(11.888, -25.288, radius_deg=1.0, min_overlap=0.2)
    assert len(m.tiles) >= 4 and m.primary.contains_target
    assert all((t.overlap_fraction or 0) >= 0.2 or t.contains_target for t in m.tiles)


# ---------------------------------------------------------------- templates
def test_templates_available_and_ordered():
    from sevendt_calc import templates as tpl
    keys = tpl.template_keys()
    assert keys[0] == tpl.UPLOAD_KEY and keys[1].startswith('Star:') and keys[-1] == tpl.FLAT_KEY
    assert tpl.default_key() == 'Star: G2V'
    for key in keys[1:-1]:
        spec = tpl.load_template(key)
        assert spec.lam_min <= 3000 and spec.lam_max >= 10900, key


def test_template_target_needs_magnitude_and_has_colours():
    t = tg.Target(name='g2v', mag=18.0, mag_filter='r', spectrum_type='Star: G2V')
    assert t.has_photometry and t.uses_template
    g, _ = t.ab_magnitude('g')
    r, src = t.ab_magnitude('r')
    assert src == 'spectrum' and r == pytest.approx(18.0, abs=1e-3) and 0.2 < g - r < 0.7
    assert not tg.Target(name='nomag', spectrum_type='Star: G2V').has_photometry
    assert tg.Target(name='nomag', spectrum_type='Star: G2V').missing(need_coord=False, need_photometry=True) == ['mag', 'mag_filter']


def test_template_redshift_changes_colours():
    z0 = tg.Target(name='q', mag=19.0, mag_filter='r', spectrum_type='QSO (SDSS, Vanden Berk 2001)', redshift=0.0)
    z2 = tg.Target(name='q', mag=19.0, mag_filter='r', spectrum_type='QSO (SDSS, Vanden Berk 2001)', redshift=2.0)
    assert z0.ab_magnitude('m400')[0] != pytest.approx(z2.ab_magnitude('m400')[0], abs=0.01)
    assert 'z = 2' in z2.spectrum_label
    # the magnitude in the reference filter is the observed one at any redshift
    assert z2.ab_magnitude('r')[0] == pytest.approx(z0.ab_magnitude('r')[0], abs=1e-6)


def test_uploaded_spectrum_is_redshifted_too():
    from sevendt_calc.photometry import Spectrum
    import numpy as np
    w = np.arange(3000.0, 11000.0, 5.0)
    rest = Spectrum.from_arrays(w, (w / 5500.0) ** -3, 'flam', source='pl')
    shifted = rest.redshifted(1.0)
    assert shifted.lam_min == pytest.approx(6000.0) and shifted.flux[0] == pytest.approx(rest.flux[0] / 2) and rest.redshifted(0) is rest
    z0 = tg.Target(name='p', mag=19.0, mag_filter='r', spectrum=rest)
    z1 = tg.Target(name='p', mag=19.0, mag_filter='r', spectrum=rest, redshift=1.0)
    assert z1.spectrum.lam_min == pytest.approx(6000.0)
    assert z1.ab_magnitude('r')[0] == pytest.approx(z0.ab_magnitude('r')[0], abs=1e-6)
    assert z1.ab_magnitude('m850')[0] != pytest.approx(z0.ab_magnitude('m850')[0], abs=0.01)
    with pytest.raises(ValueError):
        rest.redshifted(-0.5)


def test_visibility_shared_ephemeris_matches_single(monkeypatch):
    site = config.load_site()
    cons = vis.Constraints(min_alt=30, max_alt=88, moon_sep=40, sun_alt=-18)
    obstime = Time('2026-09-23T03:00:00')
    t = tg.Target(name='s', ra=11.888, dec=-25.288)
    single = vis.compute_visibility(t, obstime, site, cons)
    shared = vis.compute_all([t, tg.Target(name='n', ra=100.0, dec=-10.0)], obstime, site, cons)[0]
    assert np.allclose(single.alt, shared.alt) and np.allclose(single.moon_sep, shared.moon_sep)
    assert single.at_time['moon_sep'] == pytest.approx(shared.at_time['moon_sep'])


def test_tile_id_resolving():
    from sevendt_calc import tiles, targets
    assert tiles.normalise_tile_id('t1234') == 'T01234'
    assert tiles.normalise_tile_id('T 25471') == 'T25471'
    assert tiles.normalise_tile_id('NGC 253') is None and tiles.normalise_tile_id('T123456') is None
    assert targets.resolve_name('T00000') == (0.0, -90.0)
    ra, dec = targets.resolve_name('t25471')
    assert abs(ra - 358.636) < 0.01 and abs(dec - 19.528) < 0.01
    ts = tiles.TileSet()
    i = ts._index['T12345']
    assert targets.resolve_name('T12345') == (float(ts.ra[i]), float(ts.dec[i]))
    try:
        targets.resolve_name('T99999')
    except ValueError as exc:
        assert 'no tile T99999' in str(exc)
    else:
        raise AssertionError('T99999 should not resolve')


def test_public_submodes_filter_and_order():
    available = ['calib_mb', 'specall', 'specblue', 'specnew', 'specred']
    assert config.public_submodes('Spec', available, {'Spec': ['specred', 'specall', 'specblue', 'ghost'], 'Color': None}) == ['specred', 'specall', 'specblue']
    assert config.public_submodes('Spec', available, {'Spec': None, 'Color': None}) == ['specall', 'calib_mb', 'specblue', 'specnew', 'specred']
    assert config.public_submodes('Color', ['ri', 'gri', 'g'], {'Spec': None, 'Color': None}) == ['gri', 'g', 'ri']
    public = config.load_public_modes()                       # the repository file
    assert public['Spec'] == ['specall', 'specblue', 'specred'] and public['Color'] is None
    assert config.submode_options('Spec') == [n for n in public['Spec'] if n in config.load_specmodes()]


def test_config_overlay(tmp_path, monkeypatch):
    overlay = tmp_path / 'overlay'
    (overlay / 'specmode' / '29990101').mkdir(parents=True)
    (overlay / 'filtinfo.dict').write_text('{"7DT01": ["g", "r", "i", "m400", "Slot5", "Slot6", "Slot7", "Slot8", "m769w"]}')
    (overlay / 'specmode.config').write_text('{"SPECMODE_FOLDER": "/home/kds/TCSpy/configuration/specmode/29990101/"}')
    (overlay / 'specmode' / '29990101' / 'only.specmode').write_text('{"7DT01": ["g", "m400"]}')
    monkeypatch.setenv('SEVENDT_CONFIG_OVERLAY', str(overlay))
    assert config.load_filtinfo() == {'7DT01': ['g', 'r', 'i', 'm400', 'Slot5', 'Slot6', 'Slot7', 'Slot8', 'm769w']}
    assert config.specmode_folder() == overlay / 'specmode' / '29990101' and list(config.load_specmodes()) == ['only']
    assert config.load_site().name                      # observer.config is not in the overlay: read from the live folder
    monkeypatch.delenv('SEVENDT_CONFIG_OVERLAY')
    assert '7DT16' in config.load_filtinfo()


def test_remap_path_forms(monkeypatch):
    monkeypatch.delenv('SEVENDT_CONFIG_OVERLAY', raising=False)
    base = config.config_dir()
    for text in ('/home/kds/TCSpy/configuration/specmode/20261013/', '~/TCSpy/configuration/specmode/20261013/',
                 '/home/kds/tcspy/configuration/specmode/20261013', '/opt/TCSpy/configuration/specmode/20261013/'):
        assert config.remap_path(text) == base / 'specmode' / '20261013', text
    assert config.remap_path('/tmp/elsewhere') == Path('/tmp/elsewhere')
    assert config.specmode_folder().is_dir() and 'specall' in config.load_specmodes()


def test_config_dir_from_settings(tmp_path, monkeypatch):
    live = config.config_dir()
    monkeypatch.delenv('SEVENDT_CONFIG_DIR', raising=False)
    assert config.settings()['config_dirs'][0] == '/lyman/data1/7dt/configuration'
    monkeypatch.setattr(config, 'settings', lambda: {'config_dirs': ['/nonexistent/first', str(tmp_path), str(live)]})
    assert config.config_dir() == tmp_path                     # the first candidate that exists
    monkeypatch.setattr(config, 'settings', lambda: {'config_dirs': ['/nonexistent/only']})
    assert config.config_dir() == Path(config.DEFAULT_CONFIG_DIR)
    monkeypatch.setenv('SEVENDT_CONFIG_DIR', str(live))
    assert config.config_dir() == live                         # the environment variable wins
    monkeypatch.setattr(config, 'settings', lambda: {'config_overlay': str(tmp_path)})
    monkeypatch.delenv('SEVENDT_CONFIG_OVERLAY', raising=False)
    assert config.overlay_dir() == tmp_path


def test_observing_season():
    from datetime import date
    site = config.load_site(); cons = vis.Constraints.from_config()
    grid = vis.season_grid(date(2026, 11, 12), site, cons, n_nights=366)
    hours = vis.hours_above(grid, [11.888, 10.685, 80.894], [-25.288, 41.269, -69.756], 30.0)
    assert hours.shape == (366, 3) and 5 < hours[0, 0] < 7.5 and hours[:, 1].max() == 0 and (hours[:, 2] > 0).sum() > 300
    ngc = vis.observing_season(grid, 11.888, -25.288, 30.0)
    assert ngc['observable_tonight'] and ngc['start'] == date(2026, 11, 12) and date(2027, 1, 1) < ngc['end'] < date(2027, 3, 1) and ngc['next_start'] is not None
    m87 = vis.observing_season(grid, 187.706, 12.391, 30.0)
    assert not m87['observable_tonight'] and date(2026, 12, 1) < m87['start'] < date(2027, 1, 31) and m87['end'] > m87['start']
    assert vis.observing_season(grid, 37.95, 89.26, 30.0)['nights_per_year'] == 0
    lmc = vis.observing_season(grid, 80.894, -69.756, 30.0)
    assert lmc['observable_tonight'] and lmc['nights_per_year'] > 300 and lmc['end'] > lmc['start']   # a short gap around the shortest nights
