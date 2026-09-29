"""Headless run of the stand-alone Overhead page."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_overhead_page_renders():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'overhead_app.py'), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Overhead time'
    assert any('Filter change' in m.value for m in at.markdown)
