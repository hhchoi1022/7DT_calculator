#!/usr/bin/env bash
# Start the stand-alone 7DT Observation request page with the 7dtcalc conda environment.
#   ./run_trigger.sh            -> http://<this host>:8514
#   PORT=8600 ./run_trigger.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"
PORT="${PORT:-8514}"
[ -f "$HERE/local.env" ] && source "$HERE/local.env"     # server-local settings (git-ignored), e.g. SEVENDT_CONFIG_OVERLAY
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run trigger_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
