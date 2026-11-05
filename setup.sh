#!/bin/bash
# Creates .venv next to this file and installs requirements.txt (macOS / Linux).
#   bash setup.sh            # uses python3
#   PYTHON=python3.12 bash setup.sh
# Windows: py -3.12 -m venv .venv ; .venv\Scripts\python -m pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
PYBIN="${PYTHON:-python3}"
"$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Need Python 3.11+ (3.12 tested); '$PYBIN' is $("$PYBIN" -V 2>&1). Install a newer Python and run: PYTHON=python3.12 bash setup.sh"; exit 1; }
"$PYBIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I - <<'EOF'
import sys, torch
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, training device: {dev}")
if dev == "cpu":
    print("WARNING: no GPU found. Training will be very slow (the segmentation model is heavy); use it only for smoke tests.")
EOF
echo "Setup done. Next: put the LiTS data in data/raw (see docs/RUNNING.md), then: bash run_pipeline.sh check"
