# 7DT Observation Calculator

Web calculators for people who request observations with 7DT (the 7-Dimensional Telescope: 16 units,
each with its own filter wheel), plus the operator's scheduling page. Stand-alone Streamlit pages share one core package
(`sevendt_calc/`) and one UI helper module (`ui_common.py`). Every page reads the live TCSpy
configuration of the telescope (site, filters installed in each unit, observation modes) at start-up
and re-reads it every minute, so the pages follow the real instrument state without redeploys.

This README is written so that a coding agent (Claude Code, Codex, ...) can work on the project
without reading everything first. Read **Working rules** and **Gotchas** before changing code.

## Pages

| Page | Entry file | Start script | What it does |
|------|------------|--------------|--------------|
| Visibility | `visibility_app.py` | `./run_visibility.sh` | Altitude / Moon / twilight over one night for a table of targets, plus a month overview of observable hours (Plotly) |
| Exposure time / SNR | `etc_app.py` | `./run_etc.sh` | Empirical 7DT depth model: SNR for an exposure time and frame count, or the total exposure time needed for a target SNR, for every filter of an observation mode; point or extended sources; spectrum templates or uploads (Plotly) |
| Overhead time | `overhead_app.py` | `./run_overhead.sh` | Total observing time of one observation (exposure, readout, filter changes, autofocus, slewing, dispatch) for an observation mode |
| 7DS tile matcher | `tiles_app.py` | `./run_tiles.sh` | Which 7DS survey tiles contain / overlap each target (optionally within a radius), interactive tile maps |
| Observation request | `trigger_app.py` | `./run_trigger.sh` | Target or Survey (7DS tile) requests in a fixed format (table or uploaded file with column assignment), validated against the request rules, the current instrument and the project's allocated time, written to the request spreadsheet as one worksheet per project (`<period>_<Project ID>`); tile matcher dialog (circle / ellipse), visibility and total-time helper plots; an existing worksheet is loaded for editing when the project key matches |
| Mode builder | `modebuilder_app.py` | `./run_modebuilder.sh` | Compose a new Spec / Color observation mode by clicking filters in the wheel grid of every unit, validate it, visualise it and download a `.specmode` / `.colormode` JSON file |
| Scheduling (operator) | `schedule_app.py` | `./run_schedule.sh` | Reads the Phase-2 request files of the open-time programme (one text file per proposal), matches them to the registered projects and lists every requested target set in one editable table: the request (mode, exptime, frames, total with overhead, `dither_arcsec` per project, OBS_WINDOW) plus the operator's `obs_starttime`, `obs_endtime`, `count` (frames per epoch) and `target_epoch` (a target set can be split into parts with their own values); a list of rows that do not fit (not enough nights on which the target is up long enough, frames differing from the request), a per-project summary (requested / planned time against the allocation, first and last assumed epoch), an assumed automatic schedule (epochs placed load-balanced on feasible nights) and the nights on which assumed epochs compete for the same time of night; per project PI / request type / mode / sky background / allocation / planned time; visibility of any chosen targets (several projects at once): curves (max altitude or observable hours) with the assumed epochs as diamonds over one bar per chosen target (observable / at transit or >= 3 h / Moon unsuitable, the entered range and the requested OBS_WINDOW as lines, epochs as diamonds); a per-project, per-night strip chart with the assumed epochs; the schedule is saved next to the request files |

Each page is a separate Streamlit process. Every `run_*.sh` script carries a default port that can be
overridden with `PORT=...`; the ports in use are assigned by the operator and are not part of this
repository. Opening a port on the server needs a human with sudo
(`sudo firewall-cmd --permanent --add-port=NNNN/tcp && sudo firewall-cmd --reload`).

## Running

```bash
# environment: conda env 7dtcalc (Python 3.11, see requirements.txt); ALWAYS with these two variables
export PYTHONNOUSERSITE=1      # the user site-packages carry tcspy pins (numpy 1, old shapely) that break astropy
export MPLBACKEND=Agg
PY=~/anaconda3/envs/7dtcalc/bin/python

PORT=<port> ./run_etc.sh                          # foreground
PORT=<port> nohup ./run_etc.sh > ~/etc_app.log 2>&1 &   # detached

# restart one page (the scripts run `python -m streamlit run <file> --server.port <port> --server.address 0.0.0.0 --server.headless true`)
OLD=$(pgrep -f "[s]treamlit run etc_app.py --server.port <port>" || true); [ -n "$OLD" ] && kill $OLD; sleep 2
PORT=<port> nohup ./run_etc.sh > ~/etc_app.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:<port>/     # 200 when up
```

Do not use `pkill -f streamlit` from an automated shell: the pattern matches the shell itself. Use the
`pgrep -f "[s]treamlit run <file> --server.port <port>"` form above.

Tests (about 2 to 3 minutes; the page tests drive the apps headlessly with `streamlit.testing.v1.AppTest`):

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg ~/anaconda3/envs/7dtcalc/bin/python -m pytest tests -q
PYTHONNOUSERSITE=1 MPLBACKEND=Agg ~/anaconda3/envs/7dtcalc/bin/python -m pytest tests/test_etc_app.py -q   # one page
```

Browser checks: `playwright` with Chromium is installed in the env. Pattern used so far: launch
`chromium`, `goto('http://localhost:<port>/', wait_until='networkidle')`, wait a few seconds, drive
widgets with `get_by_role` / `get_by_test_id('stDataFrame')`, screenshot, inspect the PNG.

## Layout

```
sevendt_calc/            core package, no Streamlit imports
  config.py              live TCSpy configuration: site, filtinfo.dict, spec/color mode files, ObsMode, resolve_obsmode()
  targets.py             Target dataclass, RA/Dec parsing (deg or sexagesimal), resolve_name() (7DS tile ID or Sesame)
  photometry.py          filter transmission curves, AB/Vega conversion, Spectrum, synthetic photometry
  templates.py           spectral template catalogue (data/templates/templates.json)
  filters.py             filter naming, sorting, wavelengths, colours
  etc.py                 depth model, Conditions, ExposureCalculator (SNR / exposure time), matplotlib + Plotly plots
  visibility.py          astroplan-based night ephemeris, visibility windows, month grid, Plotly plots
  overhead.py            overhead model (OverheadConfig from data/overhead.json), estimate(), timeline plot
  tiles.py               7DS tile catalogue (data/final_tiles.txt), tile matching, tile-ID lookup, Plotly maps
  modebuild.py           mode builder core: validation, JSON export in TCSpy layout, padding, add/remove helpers
  trigger.py             observation requests: spreadsheet columns, upload parsing + column aliases, validation, row formatting
  phase2.py              Phase-2 request files: parser, project matching by PI name (or mapping.json), exposure / overhead time per target,
                         per-night tables on the season grid, the operator's plan (window, hours per night) and the night-packing conflict check
  sheets.py              Google Sheets read / append / replace (service account; key from settings.json, GOOGLE_APPLICATION_CREDENTIALS or TCSpy's googleSheet.config)
  components/tile_picker/index.html   browser-side tile map (Plotly.js, plain HTML + JS, no build step): clicks toggle tiles locally,
                         only the selection is sent back (Streamlit component protocol by hand); wrapper ui_common.tile_picker()
  modeplot.py            observation-mode figure (transmission curves + unit x order grid), matplotlib and Plotly
ui_common.py             shared Streamlit pieces: live_config(), target table with CSV upload and name resolving, season_grid_for(),
                         night/moon widgets, obsmode_selector() with the cached mode figure, spectrum upload helpers
*_app.py                 one Streamlit page each (see table above); run_*.sh start them
data/                    settings.json (configuration folder, sheet key, request folder), trigger.json (request page), depth model JSON, overhead.json, public_modes.json, final_tiles.txt, transmission/, templates/, samples
Opentime/<period>/       Phase-2 request files (*.txt), optional mapping.json (file -> Project ID), the saved schedule.json and backup/ (previous versions); git-ignored (PI e-mails)
data/cache/              generated (mode figures); git-ignored
scripts/fetch_templates.py   rebuilds data/templates from public sources (Pickles, Kinney-Calzetti, Nugent, QSO)
tests/                   test_core.py (models) + one AppTest file per page
.streamlit/config.toml   headless, 50 MB uploads, viewer toolbar, light theme
```

## Configuration source

`sevendt_calc.config` reads the first existing folder listed under `config_dirs` in `data/settings.json`
(default `/lyman/data1/7dt/configuration`; add each server's mount path there). The environment variable
`SEVENDT_CONFIG_DIR` overrides it. Files read: `observer.config` (site), `filtinfo.dict` (filters in each unit's wheel; names
like `Slot6` are empty positions), `specmode.config` -> `specmode/<date>/*.specmode`,
`colormode.config` -> `colormode/<date>/*.colormode`, `target.config`, `nightsession.config`.
Paths inside those files are written for the control PC (`/home/kds/TCSpy/configuration/...`) and are
remapped onto the local folder. `config_overlay` in `data/settings.json` (or `SEVENDT_CONFIG_OVERLAY`) may name a folder with the same
layout whose files take precedence (a prepared change not yet applied to the live folder); the run scripts also
source a git-ignored `local.env` for server-local environment variables. The folder is read-only for this project: never write into it, and
never copy its contents into the repo (some files hold credentials). Mode files are JSON objects
`{"7DT01": ["m650", "m769w"], ...}`, one unit per line; a filter may appear several times.

## Models (short)

- **Depth model** (`data/depth_model_*.json`, fitted on 2025 data, 100 s frames, 10 arcsec aperture):
  `UL5_ref = b0 + b1*moon_phase + b2*moon_separation + b3*hours_since_sunset + b4*seeing` per filter;
  `UL5(t) = UL5_ref + 1.25 log10(t/100)`; `SNR_1 = 5 * 10^(-0.4 (m - UL5(t)))`; stacked SNR scales with
  `sqrt(count * units)` where `units` is how many (unit, slot) entries of the mode observe that filter.
  Filters without a fit are interpolated in effective wavelength between medium bands (flagged).
  Extended sources use the surface brightness over the aperture (`SB_OFFSET`) and report SNR per arcsec^2.
  Saturation: point sources brighter than 11 mag; extended limit shifted by the seeing disk.
  Known limits: source shot noise is neglected (a few percent at SNR 10, large at SNR > 100).
- **Visibility**: astroplan Observer at the site from `observer.config`; a night is defined by the Sun
  altitude in `nightsession.config`; observability limits (altitude, Moon separation) from `target.config`.
- **Overhead**: per-unit sequence, units in parallel, the slowest unit counts; autofocus once per filter
  only for the modes in `autofocus_modes` of `data/overhead.json` (currently Spec/specred), none otherwise;
  components in the same file (values from TCSpy logs).
- **Tiles**: gnomonic projection about the target, shapely polygons, every overlapping tile listed,
  warning within 4 arcmin of a tile edge. Tile IDs run `T00000` to `T25471`.
- **Modes**: `Spec` (specmode files), `Color` (colormode files), `Deep` (user filter list observed by every
  unit that has them). `ObsMode.filter_multiplicity()` gives the unit count per filter. The modes offered on the
  ETC and Overhead pages are those listed in `data/public_modes.json` (list order = menu order, first entry =
  default; `null` = every file). The Mode builder always lists every file.

- **Observation requests** (`data/trigger.json`): spreadsheet URL, period prefix of the worksheet name, the
  selectable obsmodes / colormodes / deepmodes (specmodes follow `public_modes.json`), default and maximum
  exposure time (above the maximum = error, any other value than the default = warning), default count,
  binning, gain, the fixed priority and the hidden columns (note, comment = Project ID, weight, is_ToO,
  is_rapidToO, objtype). Writing needs a Google service-account key with editor access to the spreadsheet;
  without one the page still validates and offers the CSV. Column aliases live in `trigger.FIELDS`.
  `data/projects.json` (git-ignored, see `projects.example.json`) maps Project IDs to their submission key and
  allocated hours: with it, a request needs the key and may not exceed the allocation; without it every ID is
  accepted. Survey requests take 7DS tile IDs as object names and fix the coordinates to the tile centres.

- **Scheduling** (`sevendt_calc/phase2.py`): the request folder is `phase2_requests_dir` of `data/settings.json`
  (or `SEVENDT_PHASE2_DIR`). File format: `[PROPOSAL]` block, one `[TARGET] ... [END TARGET]` block with one
  `KEY: value` set per target (blank line between sets; NOTE may continue on indented lines; values may carry
  units). `NUM_FRAMES` is per filter over the whole programme, so the exposure is `exptime x frames x orders`
  (specall: 2 orders); the total adds the overhead model's estimate of one visit. Files are matched to
  `data/projects.json` by the PI name (ToO / mode keywords separate a PI with two projects); `mapping.json`
  overrides. Nights are the 274 local nights 2026-12-01 to 2027-08-31 on the season grid; per target: nightly
  maximum altitude (`vis.nightly_max_altitude`), hours and intervals above 30 deg (`vis.hours_above`,
  `vis.intervals_above`), requested nights (date windows by night date, timed windows by overlap, UTC), Moon
  illumination classes Dark <= 0.2 < Gray <= 0.8 < Bright from the proposal's sky background. The plan is
  per target set (`phase2.TargetSettings`): obs_starttime / obs_endtime (UTC; a date alone means that whole
  local night, both ends inclusive, a time cuts exactly there; several ranges comma-separated, the n-th
  start with the n-th end; the entered ranges are the master, the requested OBS_WINDOW is reference only
  (shown split into obs_window_start / obs_window_end) while timed windows still fix the time of night), count
  (frames per filter in one epoch), target_epoch and cadence (days between epochs, 0 = free; the default
  is read from 'cadence=7d' in the NOTE); a target set may be split into parts, each with its own
  values (`make_target_plans`, part keys `project|target#n`, frames compared with NUM_FRAMES as a whole);
  one epoch lasts `phase2.epoch_hours` (exposure x orders
  plus the overhead model of one visit, cached per mode / exptime / count). `make_target_plan` marks the
  nights of the range (Moon condition optional) on which the target's longest interval above the limit (inside
  its timed window, if any, and inside the range's times) holds one epoch; blocking issues (not enough
  feasible nights, no coordinates) versus warnings (frames differ from NUM_FRAMES). `place_epochs`
  assumes the automatic scheduler: most constrained targets first, each epoch in its own part of the range
  on the night with the least load; with a cadence (and more than one epoch) the epochs are chained that
  many days apart (`_place_with_cadence`: the start from which the most epochs fit with the least load,
  nearest feasible night within a third of the cadence, else the next one) and a chain shorter than
  target_epoch is a blocking issue. The conflict check packs each night pre-emptively, least slack first
  (`phase2.pack_night_slots`), and lists nights where an epoch gets more than a minute less than it needs.
  The schedule is saved as `schedule.json` in the request folder (keys `project|target`, a list when split;
  each entry also carries the operator's `note`); every save first copies the previous file to
  `backup/schedule_<save time>.json`. The PI's NOTE is shown as Requester_note. Strip charts use
  `phase2.strip_categories` (none / observable / >= 3 h / Moon unsuitable / conflict) drawn as one Plotly
  heatmap with a discrete colour scale with the assumed epochs as diamonds on the rows (category labels as y:
  numbers would add rows '0', '1', ...). The visibility figure has two panels: curves with epoch diamonds,
  and one strip per chosen target (lines for the entered range and the OBS_WINDOW are shapes with
  `yref='y2'` and numeric y = category index, which works for shapes but not for traces). `dither_arcsec` comes from `data/projects.json` (per project; 0 when absent).

## Working rules (decisions already made with the telescope operator; keep them)

- Defaults everywhere: exposure time 100 s, 3 frames.
- Pages carry no introductory paragraphs, success banners or long captions. The ETC page shows only the
  note that predictions come from 2025 data with 100 s exposures. Observing-condition values
  (Moon phase, separation, hours since sunset, seeing) are never displayed on the ETC page.
- Plots are Plotly (interactive, drawn in the browser). The one matplotlib figure left is the
  observation-mode figure in `obsmode_selector()`, exported at 300 dpi and cached on disk with a
  warm-up thread. Do not reintroduce server-side PNG rendering for page figures: it was too slow.
- Target tables: columns Index / Name / RA / Dec (extra columns per page), dynamic rows, CSV bulk
  upload with a sample file, a "Resolve names to RA/Dec" button that overwrites existing coordinates
  and also accepts 7DS tile IDs (`T01022`). Example rows are NGC0253 and T01022.
- ETC: single target, no date; sky and seeing come from preset radios (percentile labels), the seeing
  value is hidden except for "Any". Extended sources take a surface brightness.
- Mode builder: units across, wheel slots down, click to append (duplicates allowed), click an Order
  entry (marked with a cross) to remove it, template load and file upload, download in the exact TCSpy
  layout. Filters not installed in the unit are errors, unequal sequence lengths are a warning.
- Streamlit widget keys are prefixed per page (`vis_`, `etc_`, `oh_`, `tile_`, `mb_`, `tr_`, `sch_`).
- Scheduling page: the editable plan table comes first (`st.data_editor` with a key and fixed rows: in
  Streamlit 1.64 such an editor keeps its identity across pure value changes, so the input frame is rebuilt
  every run with live totals (Total exptime, With overhead) without losing edits or scroll; the key is
  bumped only on reset / load / split, which must discard the widget's edits; the request columns are
  disabled, the plan columns editable, the hidden `key` column identifies the row), followed by the
  list of rows that do not fit and the per-project summary; obs_starttime / obs_endtime are text cells (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM`, regex
  validated) because the browser's datetime picker follows the viewer's locale (12-hour clocks cannot
  enter 00); projects table with single-row selection (tick the box) plus a selectbox. The visibility section
  plots the chosen targets of the chosen projects in one figure (no per-target figures, no nightly-plan
  figure).
- New features go into the separate pages and `sevendt_calc/`; there is no combined page any more.

## Gotchas (learned the hard way)

- `PYTHONNOUSERSITE=1` is mandatory; without it astropy fails to import in this account.
- Streamlit 1.64 `st.data_editor` with `num_rows='dynamic'` re-mounts when its data changes; the shared
  table (`ui_common.targets_table`) only re-mounts on add / delete / import / resolve, so edits keep the
  scroll position.
- `st.dataframe(..., on_select='rerun', selection_mode='single-cell')` (mode builder grid): the frontend
  only re-applies a programmatic selection when it differs from the previous one. Clearing to the same
  empty state twice leaves the old click in the widget and a repeated click on the same cell is
  ignored. The grid therefore parks the selection on a blank spacer row, alternating the column
  (`_park_column`). The click is applied at the top of the run from `st.session_state['mb_grid']`,
  before the grid is drawn, so one click costs one script run.
- Streamlit's fixed header swallows clicks in the top ~60 px of the viewport (matters for browser tests
  that scroll a grid to the top). Canvas grids need a hover before the click.
- A selectbox whose option labels change (e.g. a count appended by `format_func`) is a new widget for
  Streamlit even with a key, so its choice resets; keep labels fixed and show such counts elsewhere.
- Streamlit magic: a bare expression statement in a page file that is not a plain call (e.g. a conditional
  expression `a() if x else b()`) is passed to `st.write()`; with a Plotly figure that draws an extra copy
  of the figure each time. Use if statements or assign the result.
- Plotly: never call `fig.add_shape` / `fig.add_annotation` in a loop (quadratic); build lists and assign
  them once in `update_layout`. Downsample curves before sending them to the browser. Legend line
  samples follow the trace width only with `itemsizing='trace'`.
- `st.cache_data` figures / tables are keyed on a `repr()` of every input that matters; arguments
  prefixed with `_` are not hashed.
- The `firewall-cmd`, `sudo` and port changes cannot be run by an agent here; ask the operator.
- `pkill -f` matches the invoking shell; use the `pgrep -f "[s]treamlit ..."` pattern.
- Nugent SN template files are gzip-compressed despite the `.dat` name; the template epochs used are the
  B-band maxima (Ia 19 d, Ib/c 17 d, II-P 11 d). Vega to AB offsets: B -0.09, V +0.02, R +0.21, I +0.45.

## Adding a page

1. Create `<name>_app.py` with `st.set_page_config(...)`, `cfg = live_config()`, page-prefixed widget keys.
2. Put all physics / data handling into `sevendt_calc/`; keep the page file to layout and state.
3. Copy a `run_*.sh` script; the operator assigns the port and opens it in the firewall.
4. Add `tests/test_<name>_app.py` with an `AppTest` smoke test plus core tests in `tests/test_core.py`.
5. Check the page in a headless browser at desktop and phone widths.

## Data provenance

Spectral templates: Pickles (1998) stars O5V to M0V, Kinney-Calzetti (1996) galaxies, Nugent SN
templates at maximum, QSO composites of Vanden Berk et al. (2001) and Selsing et al. (2016); rebuilt by
`scripts/fetch_templates.py`. Transmission curves: copied from the ezphot package. Depth model: fitted on
2025 7DT observations. Tiles: `final_tiles.txt` of the 7DS survey. Overheads: medians from TCSpy logs.
