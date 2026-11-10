"""Appending request rows to the 7DT request spreadsheet (Google Sheets, service account)."""
from __future__ import annotations

import json
import os
from pathlib import Path

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']


def service_account_key_path() -> Path | None:
    """
    The service-account JSON key, from (in order): GOOGLE_APPLICATION_CREDENTIALS, google_service_account_key
    in data/settings.json, the GOOGLESHEET_AUTH path of TCSpy's googleSheet.config (remapped). None when absent.
    """
    from . import config
    candidates = [os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''), str(config.settings().get('google_service_account_key') or '')]
    try:
        gs = config._read_json(config.config_path('googleSheet.config'))
        candidates.append(str(config.remap_path(gs['GOOGLESHEET_AUTH'])))
    except Exception:
        pass
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser()
    return None


def service_account_email(key_path: Path | str) -> str:
    try:
        return str(json.loads(Path(key_path).read_text()).get('client_email', ''))
    except (OSError, ValueError):
        return ''


def _client(key_path):
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_file(str(key_path), scopes=SCOPES)
    return gspread.authorize(creds)


def append_rows(spreadsheet_url: str, worksheet_title: str, header: list, rows: list, key_path: Path | str) -> dict:
    """
    Append `rows` (dicts keyed by header name) to the worksheet, creating it with the header when it does not
    exist. An existing worksheet keeps its own header order; columns it lacks are added at the end.
    Returns {'worksheet', 'created', 'n_rows', 'url'}.
    """
    import gspread
    gc = _client(key_path)
    sh = gc.open_by_url(spreadsheet_url)
    try:
        ws = sh.worksheet(worksheet_title)
        created = False
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_title, rows=max(200, len(rows) + 20), cols=max(len(header), 26))
        created = True
    existing = [c for c in ws.row_values(1)]
    if not any(existing):
        ws.update(range_name='A1', values=[list(header)])
        columns = list(header)
    else:
        columns = existing + [c for c in header if c not in existing]
        if len(columns) > len(existing):
            ws.update(range_name='A1', values=[columns])
    values = [[str(r.get(c, '')) for c in columns] for r in rows]
    ws.append_rows(values, value_input_option='RAW', table_range='A1')
    return {'worksheet': worksheet_title, 'created': created, 'n_rows': len(values), 'url': f'{sh.url}#gid={ws.id}'}
