#!/usr/bin/env bash
# Start the stand-alone 7DT Exposure time / SNR page with the 7dtcalc conda environment.
#   ./run_visibility.sh            -> http://<this host>:8510
#   PORT=8600 ./run_visibility.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"
PORT="${PORT:-8510}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run etc_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
