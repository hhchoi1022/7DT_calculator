"""Headless run of the Streamlit page through every function (run: pytest tests)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FUNCTIONS = ['Visibility', 'Exposure time / SNR', 'Overhead time', '7DS tile matcher']


@pytest.fixture(scope='module')
def app():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'app.py'), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize('function', FUNCTIONS)
def test_every_function_renders_without_exception(app, function):
    app.button(key=f'fn_{function}').click().run()
    assert not app.exception, app.exception
    assert app.title[0].value == '7DT Observation Calculator'


def test_exptime_mode_renders(app):
    app.button(key=f'fn_{FUNCTIONS[1]}').click().run()
    app.radio(key='etc_calc').set_value('Exposure time from SNR').run()
    assert not app.exception, app.exception


def test_editor_state_is_folded_into_the_input_table():
    import app
    base = app.empty_targets_df([
        {'Name': 'A', 'RA': '10', 'Dec': '-20', 'Mag': 19.0, 'Mag filter': 'r', 'Spectrum': 'Star: G2V', 'z': 0.0},
        {'Name': 'B', 'RA': '11', 'Dec': '-21', 'Mag': 18.0, 'Mag filter': 'g', 'Spectrum': 'Star: G2V', 'z': 0.0},
    ])
    state = {'edited_rows': {0: {'Mag': 17.5, 'Spectrum': 'QSO (X-shooter, Selsing 2016)'}},
             'added_rows': [{'Name': 'C', 'RA': '12', 'Dec': '-22'}],
             'deleted_rows': [1]}
    out = app.apply_editor_state(base, state)
    assert list(out['Name']) == ['A', 'C']
    assert out.at[0, 'Mag'] == 17.5 and out.at[0, 'Spectrum'] == 'QSO (X-shooter, Selsing 2016)'
    assert out.at[1, 'Spectrum'] == 'Star: G2V' and out.at[1, 'z'] == 0.0
    targets = app.parse_targets(out)
    assert [t.label for t in targets] == ['A', 'C'] and targets[1].has_coord and not targets[1].has_photometry
