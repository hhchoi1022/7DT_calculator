"""Schedule viewer page: nothing without the project list, a key per project, a pause after too many wrong keys,
a PI sees only their own programme, the admin key everything (driven headlessly with AppTest)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sevendt_calc import keyguard, phase2, trigger

APP = str(Path(__file__).resolve().parent.parent / 'plan_app.py')


def _run(pid: str = '', key: str = '') -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['pl_pid_value'] = pid
    at.session_state['pl_key_value'] = key
    return at.run()


@pytest.fixture
def projects_file(tmp_path, monkeypatch):
    """A temporary data/projects.json with one project and an admin key; the guard starts empty."""
    path = tmp_path / 'projects.json'
    path.write_text(json.dumps({'_admin_key': 'admin-secret', '26007': {'key': 'right-key', 'allocated_hours': 10}}))
    monkeypatch.setattr(trigger, 'PROJECTS_JSON', path)
    monkeypatch.setattr(keyguard, 'GUARD', keyguard.KeyGuard())
    return path


def test_viewer_shows_nothing_without_the_project_list(tmp_path, monkeypatch):
    if not phase2.requests_dir().is_dir():
        return
    monkeypatch.setattr(trigger, 'PROJECTS_JSON', tmp_path / 'absent.json')
    at = _run('26007', 'anything')
    assert not at.exception and not at.subheader and any('not configured' in e.value for e in at.error)


def test_viewer_needs_a_key(projects_file):
    if not phase2.requests_dir().is_dir():
        return
    at = _run()
    assert not at.exception and at.title[0].value == '7DT Schedule' and any('Enter your Project ID' in i.value for i in at.info)
    bad = _run('26007', 'wrong-key')
    assert not bad.exception and any('does not match' in e.value for e in bad.error) and not bad.subheader
    unknown = _run('26999', 'wrong-key')
    assert not unknown.exception and any('Unknown Project ID' in e.value for e in unknown.error)


def test_viewer_pauses_after_too_many_wrong_keys(projects_file, monkeypatch):
    if not phase2.requests_dir().is_dir():
        return
    monkeypatch.setattr(keyguard, 'GUARD', keyguard.KeyGuard(free=2, base=10))
    for i in range(3):                                           # 2 free attempts + the first one that costs
        at = _run('26007', f'wrong-key-{i}')
        assert any('does not match' in e.value for e in at.error)
    paused = _run('26007', 'right-key')                          # even the right key waits now
    assert not paused.exception and any('Too many wrong keys' in e.value for e in paused.error) and not paused.subheader
    assert keyguard.GUARD.failures('pid:26007') == 3
    short = _run('26001', 'abc')                                 # a key still being typed is not counted
    assert any('Unknown Project ID' in e.value for e in short.error) and keyguard.GUARD.failures('pid:26001') == 0


def test_viewer_project_and_admin():
    if not phase2.requests_dir().is_dir() or not trigger.PROJECTS_JSON.is_file():
        return
    raw = json.loads(trigger.PROJECTS_JSON.read_text())
    projects = trigger.load_projects()
    requests = phase2.load_requests(phase2.requests_dir(), projects)
    req = next((r for r in requests if r.project_id and r.project_id in projects and projects[r.project_id].get('key')), None)
    if req is not None:
        at = _run(req.project_id, str(projects[req.project_id]['key']))
        assert not at.exception
        headers = [h.value for h in at.subheader]
        assert 'Visibility of your targets' in headers and 'Your nights among the other programmes' in headers and 'All programmes' not in headers
        assert 'Expected observation dates' in headers and any('simulated runs' in c.value for c in at.caption)
        text = ' '.join(m.value for m in at.markdown)
        others = [r.key for r in requests if r is not req]
        assert req.key in text and not any(o in text for o in others)              # no other programme is named
    admin = raw.get('_admin_key')
    if admin:
        at = _run('', admin)
        assert not at.exception
        headers = [h.value for h in at.subheader]
        assert headers[0] == 'All programmes' and 'One night' in headers
        if req is not None:                                           # admin + Project ID = that PI's view
            at2 = _run(req.project_id, admin)
            headers2 = [h.value for h in at2.subheader]
            assert not at2.exception and 'All programmes' not in headers2 and 'Your nights among the other programmes' in headers2
            assert any('Admin view of project' in c.value for c in at2.caption)
