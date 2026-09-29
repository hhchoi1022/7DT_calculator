"""Headless run of the stand-alone Tile matcher page."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_tiles_page_renders():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'tiles_app.py'), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Tile matcher'
    at.number_input(key='tile_radius_all').set_value(1.0).run()
    assert not at.exception, at.exception
