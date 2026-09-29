# 7DT Observation Calculator

Web calculators for people who request observations with 7DT (the 7-Dimensional Telescope: 16 units,
each with its own filter wheel). Five stand-alone Streamlit pages share one core package
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
| Mode builder | `modebuilder_app.py` | `./run_modebuilder.sh` | Compose a new Spec / Color observation mode by clicking filters in the wheel grid of every unit, validate it, visualise it and download a `.specmode` / `.colormode` JSON file |

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
  modeplot.py            observation-mode figure (transmission curves + unit x order grid), matplotlib and Plotly
ui_common.py             shared Streamlit pieces: live_config(), target table with CSV upload and name resolving,
                         night/moon widgets, obsmode_selector() with the cached mode figure, spectrum upload helpers
*_app.py                 one Streamlit page each (see table above); run_*.sh start them
data/                    depth model JSON, overhead.json, final_tiles.txt, transmission/ curves, templates/, samples
data/cache/              generated (mode figures); git-ignored
scripts/fetch_templates.py   rebuilds data/templates from public sources (Pickles, Kinney-Calzetti, Nugent, QSO)
tests/                   test_core.py (models) + one AppTest file per page
.streamlit/config.toml   headless, 50 MB uploads, viewer toolbar, light theme
```

## Configuration source

`sevendt_calc.config` reads `/lyman/data1/7dt/configuration` (override with the environment variable
`SEVENDT_CONFIG_DIR`): `observer.config` (site), `filtinfo.dict` (filters in each unit's wheel; names
like `Slot6` are empty positions), `specmode.config` -> `specmode/<date>/*.specmode`,
`colormode.config` -> `colormode/<date>/*.colormode`, `target.config`, `nightsession.config`.
Paths inside those files are written for the control PC (`/home/kds/TCSpy/configuration/...`) and are
remapped onto the local folder. The folder is read-only for this project: never write into it, and
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
- **Overhead**: per-unit sequence, units in parallel, the slowest unit counts; autofocus none for
  Spec/specall, otherwise once per filter; components in `data/overhead.json` (values from TCSpy logs).
- **Tiles**: gnomonic projection about the target, shapely polygons, every overlapping tile listed,
  warning within 4 arcmin of a tile edge. Tile IDs run `T00000` to `T25471`.
- **Modes**: `Spec` (specmode files, default `specall`), `Color` (colormode files), `Deep` (user filter list
  observed by every unit that has them). `ObsMode.filter_multiplicity()` gives the unit count per filter.

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
- Streamlit widget keys are prefixed per page (`vis_`, `etc_`, `oh_`, `tile_`, `mb_`).
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
