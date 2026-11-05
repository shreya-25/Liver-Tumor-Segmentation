#!/bin/bash
# RE-RUN DESIGN, not the script that produced the reported numbers. The numbers in docs/baseline_results were
# produced with a stratified patient-level 5-fold split and older train/eval code; that exact code and split are
# archived in docs/baseline_results/archive (see its README). This script runs the same A/B/C comparison on the
# FINAL code, which uses the original split only (train_test_split over the sorted volume names, test_size=0.2,
# random_state=42: 104 train / 27 held-out volumes). The held-out part is both the validation set (checkpoint
# selection) and the evaluation set, as in the original script, so the scores are validation-selected.
# Takes hours (3 trainings + 3 evaluations); its results will not match docs/baseline_results one-to-one.
#   A: v1 cache (per-volume min-max, original orientation), flips on
#   B: v2 cache (fixed HU window, canonical orientation), flips on
#   C: v2 cache, flips off
# Same recipe per arm: 16 train / 6 validation volumes (stratified subset of the original split), 6 epochs, seed 1234.
# Results: results/baseline_original_split/<arm>/ (train log, split_used.json, checkpoints, eval/ with eval.json + eval.md)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="${PYTHON:-python3}"
OUTROOT="$ROOT/results/baseline_original_split"
mkdir -p "$OUTROOT"

export LITS_SUBSET_N=16 LITS_SUBSET_VAL_N=6   # train volumes / validation volumes (stratified, deterministic)
export LITS_EPOCHS=6 LITS_BATCH_SIZE=4 LITS_EMPTY_KEEP=0.1 LITS_SEED=1234

run_arm() {
  local name=$1 cache=$2 flip=$3
  local out="$OUTROOT/$name"
  mkdir -p "$out"
  echo "$(date '+%F %T') $name train start" >> "$OUTROOT/progress.log"
  LITS_CACHE_DIR="$cache" LITS_FLIP_AUG="$flip" LITS_RESULTS_DIR="$out" \
    "$PY" -I "$ROOT/Image Segmentation/train_fast.py" > "$out/stdout.log" 2>&1
  local ckpt
  ckpt="$out/best.pt"   # lowest validation loss so far (train_fast.py now writes best.pt; older code: best_epoch_NNN.pt)
  [ -f "$ckpt" ] || ckpt=$(ls "$out"/best_epoch_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$ckpt" ]; then   # training failed: do not evaluate an empty checkpoint or log a fake success
    echo "$(date '+%F %T') $name FAILED: no checkpoint, see $out/stdout.log" >> "$OUTROOT/progress.log"
    return 1
  fi
  echo "$(date '+%F %T') $name train done, evaluating ${ckpt##*/} on the held-out part" >> "$OUTROOT/progress.log"
  # all 27 held-out volumes; the guard stops the run if one of them is in this arm's train list
  # LITS_BATCH_SIZE=32: the training batch size (4) would also slow inference down
  if LITS_CKPT="$ckpt" LITS_CACHE_DIR="$cache" LITS_EVAL_OUT="$out/eval" LITS_BATCH_SIZE=32 \
    "$PY" -I "$ROOT/Image Segmentation/eval_fast.py" > "$out/eval_stdout.log" 2>&1; then
    echo "$(date '+%F %T') $name eval done" >> "$OUTROOT/progress.log"
  else
    echo "$(date '+%F %T') $name eval FAILED, see $out/eval_stdout.log" >> "$OUTROOT/progress.log"
    return 1
  fi
}

fail=0
run_arm A_v1_minmax_flip   "$ROOT/data/cache"    1 || fail=1
run_arm B_v2_window_flip   "$ROOT/data/cache_v2" 1 || fail=1
run_arm C_v2_window_noflip "$ROOT/data/cache_v2" 0 || fail=1
if [ "$fail" = 0 ]; then echo "$(date '+%F %T') ALL DONE" >> "$OUTROOT/progress.log"
else echo "$(date '+%F %T') FINISHED WITH FAILURES (see lines above)" >> "$OUTROOT/progress.log"; exit 1; fi
