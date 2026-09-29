#!/usr/bin/env bash
# Start the stand-alone 7DT Visibility page with the 7dtcalc conda environment.
#   ./run_visibility.sh            -> http://<this host>:8509
#   PORT=8600 ./run_visibility.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"
PORT="${PORT:-8509}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run visibility_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
