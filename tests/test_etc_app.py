"""Headless run of the stand-alone Exposure time / SNR page."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_etc_page_renders_both_modes():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'etc_app.py'), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Exposure time / SNR'
    at.radio(key='etc_calc').set_value('Exposure time from SNR').run()
    assert not at.exception, at.exception
    at.selectbox(key='etc_type').set_value('Extended source').run()
    assert not at.exception, at.exception
    at.radio(key='etc_sky').set_value('Any/Bright').run()
    assert not at.exception, at.exception
    at.radio(key='etc_seeing_choice').set_value('85%/Poor').run()
    assert not at.exception, at.exception
