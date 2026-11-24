#!/usr/bin/env bash
# Start the stand-alone 7DT Mode builder page with the 7dtcalc conda environment.
#   ./run_modebuilder.sh            -> http://<this host>:8513
#   PORT=8600 ./run_modebuilder.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-}"; if [ -z "$PYTHON" ]; then for d in "$HOME/anaconda3" "$HOME/miniconda3"; do if [ -x "$d/envs/7dtcalc/bin/python" ]; then PYTHON="$d/envs/7dtcalc/bin/python"; break; fi; done; fi; PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"   # the 7dtcalc env, wherever conda lives
PORT="${PORT:-8513}"
[ -f "$HERE/local.env" ] && source "$HERE/local.env"     # server-local settings (git-ignored), e.g. SEVENDT_CONFIG_OVERLAY
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run modebuilder_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
