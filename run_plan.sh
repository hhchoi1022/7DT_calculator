#!/usr/bin/env bash
# Start the stand-alone 7DT Schedule viewer page (read-only, for PIs) with the 7dtcalc conda environment.
#   ./run_plan.sh            -> http://<this host>:8516
#   PORT=8600 ./run_plan.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"
PORT="${PORT:-8516}"
[ -f "$HERE/local.env" ] && source "$HERE/local.env"     # server-local settings (git-ignored), e.g. SEVENDT_PHASE2_DIR
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run plan_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
