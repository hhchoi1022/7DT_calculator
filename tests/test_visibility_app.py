"""Headless run of the stand-alone Visibility page."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_visibility_page_renders():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'visibility_app.py'), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Visibility'
    assert len(at.dataframe) >= 1
