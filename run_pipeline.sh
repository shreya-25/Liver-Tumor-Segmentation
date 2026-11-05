#!/bin/bash
# One-command segmentation pipeline: build fixed-window cache -> train AIM-UNet -> evaluate on the held-out volumes.
# Split: the original rule (train_test_split over the sorted volume names, test_size=0.2, random_state=42):
# 104 train / 27 held-out volumes. The held-out part is both the validation set (best-epoch selection) and the
# evaluation set, so the reported scores are validation-selected, not from an untouched test set.
#   bash run_pipeline.sh check    # verify data + device only
#   bash run_pipeline.sh quick    # few-minute smoke run: 4 train volumes (2 validation), 1 epoch (no evaluation unless EVAL=1)
#   bash run_pipeline.sh full     # all 104 train volumes, LITS_EPOCHS epochs (default 10), then evaluation on the 27 held-out volumes
# Useful overrides (env vars): LITS_EPOCHS, LITS_SEED, LITS_FLIP_AUG=0, LITS_EMPTY_KEEP, EVAL=1, LITS_EVAL_MAX_VOLUMES, PYTHON.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-quick}"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="${PYTHON:-python3}"

# '|| true': with set -e / pipefail an empty glob (ls fails) must not kill the script before the message below
n_vol=$( { ls data/raw/volume-*.nii 2>/dev/null || true; } | wc -l | tr -d ' ')
n_seg=$( { ls data/raw/segmentation-*.nii 2>/dev/null || true; } | wc -l | tr -d ' ')
if [ "$n_vol" -ne 131 ] || [ "$n_seg" -ne 131 ]; then
  echo "Expected 131 volume-*.nii and 131 segmentation-*.nii files in data/raw, found $n_vol / $n_seg."
  ls data/raw/*.nii.gz >/dev/null 2>&1 && echo "Found .nii.gz files: decompress them first (gunzip data/raw/*.gz); the split and the stats are keyed by the .nii names."
  echo "See docs/RUNNING.md, section 'Get the data'."
  exit 1
fi
"$PY" -I -c "
import torch
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('data OK (131 volumes). Training device:', dev)"
[ "$MODE" = check ] && exit 0

# 1) slice cache with fixed HU window + orientation canonicalisation (skips itself if already built)
"$PY" -I "Image Segmentation/prepare_data_v2.py"

export LITS_CACHE_DIR="$PWD/data/cache_v2"
export LITS_EMPTY_KEEP="${LITS_EMPTY_KEEP:-0.1}" LITS_SEED="${LITS_SEED:-1234}"
if [ "$MODE" = quick ]; then
  export LITS_SUBSET_N=4 LITS_SUBSET_VAL_N=2 LITS_EPOCHS=1 LITS_RESULTS_DIR="$PWD/results/quick"
else
  export LITS_EPOCHS="${LITS_EPOCHS:-10}" LITS_RESULTS_DIR="$PWD/results/full"
fi
# 2) train (train part of the original split; the held-out part is the validation set, never trained on)
"$PY" -I "Image Segmentation/train_fast.py"

# 3) evaluate the best-validation checkpoint on all 27 held-out volumes (set LITS_EVAL_MAX_VOLUMES=N for only the
#    first N); the leakage guard checks them against the checkpoint's train list
if [ "$MODE" = full ] || [ "${EVAL:-0}" = 1 ]; then
  ckpt="$LITS_RESULTS_DIR/best.pt"   # train_fast.py writes best.pt (older runs: best_epoch_NNN.pt, raw state_dict)
  [ -f "$ckpt" ] || ckpt=$(ls "$LITS_RESULTS_DIR"/best_epoch_*.pt | sort | tail -1)
  LITS_CKPT="$ckpt" LITS_EVAL_OUT="$LITS_RESULTS_DIR/eval" "$PY" -I "Image Segmentation/eval_fast.py"
  echo "Results: $LITS_RESULTS_DIR/eval/eval.md"
fi
