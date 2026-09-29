#!/usr/bin/env bash
# Start the 7DT Observation Calculator (Streamlit) with the 7dtcalc conda environment.
#   ./run_app.sh            -> http://<this host>:8508
#   PORT=8600 ./run_app.sh  -> another port
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/7dtcalc/bin/python}"
PORT="${PORT:-8508}"
export PYTHONNOUSERSITE=1          # keep the user site-packages (tcspy pins) out of this environment
export MPLBACKEND=Agg
cd "$HERE"
exec "$PYTHON" -m streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true "$@"
