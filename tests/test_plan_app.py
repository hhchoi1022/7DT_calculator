"""Schedule viewer page: a PI sees only their own programme, the admin key everything (driven headlessly with AppTest)."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from sevendt_calc import phase2, trigger

APP = str(Path(__file__).resolve().parent.parent / 'plan_app.py')


def _run(pid: str = '', key: str = '') -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['pl_pid_value'] = pid
    at.session_state['pl_key_value'] = key
    return at.run()


def test_viewer_needs_a_key():
    if not phase2.requests_dir().is_dir():
        return
    at = _run()
    assert not at.exception and at.title[0].value == '7DT Schedule' and any('Enter your Project ID' in i.value for i in at.info)
    bad = _run('26007', 'wrong-key')
    assert not bad.exception and bad.error and not bad.subheader


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
        text = ' '.join(m.value for m in at.markdown)
        others = [r.key for r in requests if r is not req]
        assert req.key in text and not any(o in text for o in others)              # no other programme is named
    admin = raw.get('_admin_key')
    if admin:
        at = _run('', admin)
        assert not at.exception
        headers = [h.value for h in at.subheader]
        assert headers[0] == 'All programmes' and 'One night' in headers
