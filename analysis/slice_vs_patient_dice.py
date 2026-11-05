"""Same predictions, two Dice definitions.

(a) the original test script's metric: per-slice Dice with smooth=1e-6, so a slice with no tumor in the label AND
    none predicted scores 1.0, averaged over ALL slices;
(b) patient-level 3D Dice: one Dice over all voxels of a patient (what eval_fast.py and the baseline tables report).

Uses the arm C checkpoint (v2 cache, no flips) of the baseline, the v2 slice cache and the first 8 tumor-bearing
patients of that arm's test fold (as listed in its per_patient.csv). Run from anywhere; paths are relative to the
repository root. Needs local files that are not in git: data/cache_v2 (run prepare_data_v2.py) and the checkpoint.

    .venv/bin/python -I analysis/slice_vs_patient_dice.py

Optional env vars: LITS_CKPT (default results/baseline/C_v2_window_noflip/best_epoch_005.pt),
LITS_CACHE_DIR (default data/cache_v2), LITS_PER_PATIENT_CSV (default docs/baseline_results/C_v2_window_noflip/eval/per_patient.csv),
LITS_DEMO_N (patients, default 8).
Imports AIM_UNet from "Image Segmentation/train_fast.py" (its log goes to a temporary folder, nothing is written to results/).
"""
import os
import sys
import csv
import tempfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["LITS_RESULTS_DIR"] = tempfile.mkdtemp(prefix="slice_vs_patient_dice_")  # train_fast opens a log file on import
import numpy as np
import torch

CKPT = os.environ.get("LITS_CKPT", os.path.join(ROOT, "results", "baseline", "C_v2_window_noflip", "best_epoch_005.pt"))
CACHE = os.environ.get("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache_v2"))
CSV = os.environ.get("LITS_PER_PATIENT_CSV", os.path.join(ROOT, "docs", "baseline_results", "C_v2_window_noflip", "eval", "per_patient.csv"))
N_PAT = int(os.environ.get("LITS_DEMO_N", "8"))
for p, hint in ((CKPT, "checkpoint (not in git; produced by the baseline run, see analysis/run_baseline.sh)"),
                (os.path.join(CACHE, "images.npy"), "slice cache (run: python -I 'Image Segmentation/prepare_data_v2.py')"),
                (CSV, "per-patient table")):
    if not os.path.exists(p):
        sys.exit(f"Missing {os.path.relpath(p, ROOT)}: {hint}")

spec = importlib.util.spec_from_file_location("train_fast", os.path.join(ROOT, "Image Segmentation", "train_fast.py"))
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)
dev = getattr(tf, "device", None) or tf.pick_device()   # older train_fast.py exposed `device`, the current one pick_device()
model = tf.AIM_UNet().to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
model.eval()

names = open(os.path.join(CACHE, "volumes.txt")).read().split("\n")
vi = np.load(os.path.join(CACHE, "vol_index.npy"))
X = np.load(os.path.join(CACHE, "images.npy"), mmap_mode="r")
Y = np.load(os.path.join(CACHE, "masks.npy"), mmap_mode="r")
rows = list(csv.DictReader(open(CSV)))
pats = [r["name"] for r in rows if int(r["n_lesions"]) > 0][:N_PAT]


def dice(p, t, s=1e-6):
    return (2 * (p & t).sum() + s) / (p.sum() + t.sum() + s)


sl = {0: [], 1: [], 2: []}
sl_tumor_slices = []
per_pat = []
n_slices = n_empty = n_empty_ok = 0
for name in pats:
    ids = np.flatnonzero(vi == names.index(name))
    x = torch.from_numpy(np.asarray(X[ids])).float().div(255).unsqueeze(1)
    y = np.asarray(Y[ids])
    pred = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = model(x[i:i + 32].to(dev))
            pred.append(o.float().argmax(1).cpu().numpy().astype(np.uint8))
    pred = np.concatenate(pred)
    for k in range(len(y)):  # (a) the original test script's per-slice metric
        for c in (0, 1, 2):
            sl[c].append(dice(pred[k] == c, y[k] == c))
        if (y[k] == 2).any():
            sl_tumor_slices.append(dice(pred[k] == 2, y[k] == 2))
        n_slices += 1
        n_empty += not (y[k] == 2).any()
        n_empty_ok += (not (y[k] == 2).any()) and (not (pred[k] == 2).any())
    if (y == 2).any():  # (b) patient-level 3D Dice
        per_pat.append((name, dice(pred == 2, y == 2), dice(pred == 1, y == 1)))

print(f"patients {len(pats)} ({len(per_pat)} with tumor), slices {n_slices}, slices with NO GT tumor {n_empty} ({n_empty/n_slices:.1%}); "
      f"of these predicted-empty too: {n_empty_ok} ({n_empty_ok/max(n_empty, 1):.1%})")
print(f"(a) original per-slice-mean Dice over ALL slices : BG {np.mean(sl[0]):.3f}  Liver(label1) {np.mean(sl[1]):.3f}  Tumor {np.mean(sl[2]):.3f}")
print(f"    per-slice Dice on slices that HAVE tumor only  : Tumor {np.mean(sl_tumor_slices):.3f}")
print(f"(b) patient-level 3D Dice (tumor-bearing patients): Tumor {np.mean([p[1] for p in per_pat]):.3f}  Liver(label1) {np.mean([p[2] for p in per_pat]):.3f}")
for n, t, l in per_pat:
    print(f"      {n:16s} tumor3D {t:.3f}  liver3D {l:.3f}")
