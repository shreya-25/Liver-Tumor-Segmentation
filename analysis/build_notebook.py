"""Builds LiTS_Pipeline_Showcase.ipynb (nbformat), optionally executes it and exports a self-contained HTML copy.

Run from the repo root with the notebook environment (see requirements-notebook.txt):

    python -I analysis/build_notebook.py                      # write the (unexecuted) notebook only
    python -I analysis/build_notebook.py --execute --html     # + execute top to bottom (nbclient) + export HTML

Outputs: LiTS_Pipeline_Showcase.ipynb (repo root), docs/showcase/LiTS_Pipeline_Showcase.html (with --html).
The notebook only reads saved files (docs/showcase/*, analysis/output, docs/baseline_results, and optionally results/ and data/cache*). It never trains
or builds a cache unless the RUN_* flags in its first code cell are switched on. If docs/showcase is missing or
older than an arm's eval.json, the notebook re-runs analysis/make_showcase.py itself (about 1-2 minutes).
"""
import os
import sys
import time
import argparse
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "LiTS_Pipeline_Showcase.ipynb")
HTML_PATH = os.path.join(ROOT, "docs", "showcase", "LiTS_Pipeline_Showcase.html")

CELLS = []


def md(text):
    CELLS.append(new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(new_code_cell(text.strip("\n")))


# =====================================================================================================
md(r"""
# Liver tumor segmentation on LiTS: pipeline showcase

A readable tour of this project for someone who has not seen it: the dataset, what was wrong with the first
preprocessing and how it was fixed, a 2D AIM-UNet baseline, a corrected 3D multi-GAN for synthetic tumors, and
what the evidence does and does not support. Everything shown comes from files in this repository.

**Read this first.** The segmentation baseline is deliberately tiny (16 training volumes, 6 epochs, one seed per arm,
validation on 6 volumes, evaluation on 26 test patients). Absolute tumor Dice is still low (the numbers are in section 5). The GAN samples are
blurry and not realistic. This is a tested, working pipeline with honest first numbers, not a state-of-the-art result.

**Provenance of the baseline numbers.** The reported A/B/C numbers were produced by the code archived in `docs/baseline_results/archive/`
(see its README), not by the final code. That run used a stratified patient-level 5-fold manifest: training volumes from folds 2-4
(stratified subset of 16), validation on 6 volumes of fold 0, and evaluation on the whole test fold 1 (26 patients, 23 with tumor), which training never loaded.
The final code uses the project's original split only (sklearn `train_test_split` over the sorted volume names, `test_size=0.2`, `random_state=42`:
104 train / 27 held-out volumes; the held-out part is both validation and test set, so its scores are validation-selected).
The numbers in this notebook therefore do not carry over one-to-one to a run of the final code; `analysis/run_baseline.sh` is the re-run design for the final code.
""")

md(r"""
## 0. Overview and how to use

**Where to run it.** Open the notebook from the repository (or a subfolder). If you copy it elsewhere, set the environment variable
`LITS_ROOT` to the repository path before starting the kernel; the second code cell stops with a clear error otherwise.

**Two modes.**
- **Light mode (default).** Runs top to bottom in a few minutes using only saved files: tables and figures in
  `docs/showcase/`, per-patient results in `docs/baseline_results/`, dataset statistics in `analysis/output/`. A few
  figures are also recomputed live from the slice caches if those exist. No training, no cache build.
- **Heavy mode.** Three switches below (`RUN_PREPARE`, `RUN_TRAIN`, `RUN_GAN`) are `False`. Each heavy cell prints the exact
  command it would run and runs it with `subprocess` only if its flag is `True`. Needs the raw LiTS data (about 50 GB) and a GPU for training.

**Arms of the baseline** (same volumes, same seed, same recipe, only the data pipeline differs):
**A** = old per-volume min-max cache, **B** = fixed-window orientation-corrected cache with flip augmentation,
**C** = as B without flips. An arm without evaluation results would show as *pending* and be picked up on a re-run.

**What is verified here, and what is not.**

| verified (by code in this notebook or the saved unit tests) | not verified |
|---|---|
| Every table number is recomputed from a file; per-patient results agree with the aggregate in `eval.json` | That the gains are general, or due to one specific change: A to B bundles the window and the orientation fix, one seed per arm, 26 test patients |
| Test patients do not overlap with training or validation volumes | Any claim about generalisation beyond this test fold, or equality with a run of the final code (different split, see the provenance note) |
| The 3D GAN passes 24 unit tests (shapes, ranges, losses, resampling, insertion) | That GAN samples are realistic (they are not, visually) or that they help a segmenter (none was trained on them) |
""")

code(r"""
# ---- configuration: switch heavy actions on here (all False = light mode) ----
RUN_PREPARE = False       # build the v2 slice cache from data/raw (needs the raw data; skips itself if already built)
RUN_TRAIN = False         # 1-epoch smoke training run of the segmenter (needs the cache and a GPU)
RUN_GAN = False           # tiny end-to-end GAN smoke run (CPU; minutes)
AUTO_REFRESH = True       # rebuild docs/showcase (about 1-2 min) if it is missing or older than an arm's eval.json
REFRESH_SHOWCASE = False  # force that rebuild now
""")

code(r"""
import os, sys, re, io, json, ast, glob, shlex, zipfile, platform, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

def find_root():
    # Robust repo-root lookup: LITS_ROOT env var, else the current folder or any parent that looks like the repo.
    cands = [Path(os.environ["LITS_ROOT"])] if os.environ.get("LITS_ROOT") else []
    # also try the notebook's own location when the frontend exposes it (VS Code, JupyterLab), then the working directory
    for nbp in (globals().get("__vsc_ipynb_file__"), os.environ.get("JPY_SESSION_NAME")):
        if nbp and Path(nbp).exists():
            cands += [Path(nbp).resolve().parent, *Path(nbp).resolve().parents]
    here = Path.cwd().resolve()
    cands += [here, *here.parents]
    for p in cands:
        if (p / "docs/RUNNING.md").exists() and (p / "Image Segmentation").is_dir():
            return p
    raise RuntimeError("Repo root not found: open the notebook from the repo (or a subfolder) or set LITS_ROOT.")

ROOT = find_root().resolve()   # resolved, so rel() works when the repo is reached through a symlink
SHOW, TAB, FIG = ROOT / "docs" / "showcase", ROOT / "docs" / "showcase" / "tables", ROOT / "docs" / "showcase" / "figures"
BASE, OUTPUT = ROOT / "docs" / "baseline_results", ROOT / "analysis" / "output"   # tracked, path-scrubbed copy of the small baseline result files
V1, V2 = ROOT / "data" / "cache", ROOT / "data" / "cache_v2"
GAN = ROOT / "results" / "gan_long2"   # optional (git-ignored): the GAN run summary is skipped when absent

# short, clear error instead of a raw FileNotFoundError traceback when the shipped baseline files are missing
_need = [BASE / "archive" / "run_baseline_as_run.sh", BASE / "archive" / "split_manifest_used.csv", SHOW / "index.md"] + \
        [BASE / f / g for f in ("A_v1_minmax_flip", "B_v2_window_flip", "C_v2_window_noflip") for g in ("split_used.json", "eval/eval.json", "eval/per_patient.csv")]
_miss = [str(p.relative_to(ROOT)) for p in _need if not p.exists()]
if _miss:
    raise RuntimeError(f"Missing shipped baseline files ({len(_miss)}), e.g. {_miss[0]}. They live in docs/baseline_results/ and docs/showcase/; "
                       "restore them from the repository (the git-ignored results/ folder is not needed).")

# consistent palette (colour-blind safe)
LIVER, TUMOR = "#2a9d8f", "#d62828"
ARMS = {  # key -> (results folder, short label, colour)
    "A": ("A_v1_minmax_flip", "v1 min-max, flips", "#6c757d"),
    "B": ("B_v2_window_flip", "v2 window, flips", "#1d6fb8"),
    "C": ("C_v2_window_noflip", "v2 window, no flips", "#e08e0b"),
}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold"})

def rel(p):
    # repo-relative path for display (never an absolute path)
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name

TABLE_STYLES = [
    {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "12.5px")]},
    {"selector": "th", "props": [("background-color", "#eef1f4"), ("color", "#1b1f23"), ("text-align", "left"),
                                 ("padding", "4px 8px"), ("border-bottom", "1px solid #c9d1d9"), ("vertical-align", "bottom")]},
    {"selector": "td", "props": [("background-color", "#ffffff"), ("color", "#1b1f23"), ("text-align", "left"),
                                 ("padding", "3px 8px"), ("border-bottom", "1px solid #e6e9ed")]},
]

from IPython.display import HTML
_TABLE_N = [0]

def show_df(df, precision=3, formats=None):
    # Render a DataFrame as a clean HTML table: no index, formatted floats, '-' for missing, pending cells in italics.
    # Shown as HTML only (no Styler repr with a memory address in the saved notebook); fixed ids keep re-runs diffable.
    df = df.reset_index(drop=True)
    sty = df.style.hide(axis="index").format(precision=precision, na_rep="-", escape="html")
    if formats:
        sty = sty.format(formats, na_rep="-", escape="html", precision=precision)
    sty = sty.format_index(escape="html", axis=1).set_table_styles(TABLE_STYLES)
    sty = sty.map(lambda v: "color:#8a6d00;font-style:italic" if isinstance(v, str) and v.startswith("pending") else "")
    _TABLE_N[0] += 1
    display(HTML(sty.set_uuid(f"lits{_TABLE_N[0]:02d}").to_html()))

try:
    CAPTIONS = json.load(open(SHOW / "captions.json"))
except Exception:
    CAPTIONS = {}

def caption(text):
    display(Markdown(f"*{text}*"))

def show_fig(key, width=960):
    # Show docs/showcase/figures/<key>_*.png inline with its one-line caption from captions.json.
    hits = sorted(FIG.glob(f"{key}_*.png"))
    if not hits:
        display(Markdown(f"**Figure {key} is not available** (run `analysis/make_showcase.py`)."))
        return
    display(Image(data=hits[0].read_bytes(), format="png", width=width))
    caption(f"Figure {key}. " + CAPTIONS.get(key, ""))

TABLE_FILES = {"T1": "T1_cohort_summary", "T2": "T2_preprocessing_effect", "T3": "T3_baseline_comparison",
               "T3b": "T3b_baseline_by_thickness", "T3c": "T3c_baseline_by_lesion_size", "T3d": "T3d_paired_comparison",
               "T4": "T4_gan_networks", "T4b": "T4b_gan_unit_tests"}

def show_table(key, **kw):
    # Show docs/showcase/tables/<name>.csv as a table with its caption.
    p = TAB / (TABLE_FILES[key] + ".csv")
    if not p.exists():
        display(Markdown(f"**Table {key} is not available** (run `analysis/make_showcase.py`)."))
        return None
    df = pd.read_csv(p, dtype=str).fillna("-")
    show_df(df, **kw)
    caption(f"Table {key}. " + CAPTIONS.get(key, ""))

def run_heavy(title, flag, argv, env=None):
    # Print the exact command; run it only if the flag is True.
    shown = " ".join([f"{k}={shlex.quote(v)}" for k, v in (env or {}).items()] + [shlex.quote(a) for a in argv])
    print(f"[{title}] command (run from the repo root):\n  $ {shown}")
    if not flag:
        print("  -> skipped (flag is False). Set it to True in the configuration cell to run it.")
        return
    argv = [sys.executable if a == "python" else a for a in argv]
    r = subprocess.run(argv, cwd=ROOT, env={**os.environ, **(env or {})})
    print("  -> exit code", r.returncode)
""")

md(r"""
## 1. Environment
""")

code(r"""
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
env_rows = [("Python", platform.python_version()), ("OS", f"{platform.system()} {platform.machine()}"),
            ("torch", torch.__version__), ("numpy", np.__version__), ("pandas", pd.__version__),
            ("matplotlib", matplotlib.__version__), ("training device (cuda > mps > cpu)", device),
            ("repo root", ". (resolved at run time)")]
show_df(pd.DataFrame(env_rows, columns=["component", "version / value"]))
print("Heavy flags:", dict(RUN_PREPARE=RUN_PREPARE, RUN_TRAIN=RUN_TRAIN, RUN_GAN=RUN_GAN))
""")

code(r"""
# Make sure docs/showcase (tables + figures) exists and is current; re-run analysis/make_showcase.py if not.
expected = [TAB / (n + ".csv") for n in TABLE_FILES.values()] + [SHOW / "captions.json", SHOW / "index.md"] + \
           [FIG / f for f in ["F1_cohort_overview.png", "F2_preprocessing_before_after.png", "F3_baseline_per_patient.png",
                              "F4_training_curves.png", "F5_prediction_overlays.png", "F6_recall_by_lesion_size.png",
                              "F7a_gan_real_vs_generated.png", "F7b_gan_insertions.png"]]
missing = [rel(p) for p in expected if not p.exists()]
built = (SHOW / "index.md").stat().st_mtime if (SHOW / "index.md").exists() else 0
newer = [p.parent.parent.name for p in BASE.glob("*/eval/eval.json") if p.stat().st_mtime > built]
if REFRESH_SHOWCASE or missing or (AUTO_REFRESH and newer and (V2 / "meta.csv").exists()):   # a rebuild needs data/; file mtimes of a fresh checkout are arbitrary
    why = "forced" if REFRESH_SHOWCASE else f"missing files: {missing}" if missing else f"newer arm results: {newer}"
    print(f"Rebuilding docs/showcase ({why}); about 1-2 minutes ...")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-I", str(ROOT / "analysis" / "make_showcase.py")], cwd=ROOT,
                       capture_output=True, text=True)
    tail = r.stdout.split("Arm status:")[-1].strip() if "Arm status:" in r.stdout else r.stdout[-400:]
    print(f"exit code {r.returncode} after {time.time() - t0:.0f}s\n" + tail)
    if r.returncode != 0:
        print(r.stderr[-800:].replace(str(ROOT), "."))
    CAPTIONS = json.load(open(SHOW / "captions.json"))
else:
    print("Using the saved docs/showcase outputs (built " + time.strftime("%Y-%m-%d %H:%M", time.localtime(built)) +
          "); they are newer than every arm's eval.json.")
""")

# =====================================================================================================
md(r"""
## 2. The dataset

LiTS: 131 abdominal CT volumes with a liver label and a tumor label. The three properties that shape everything
downstream are small lesions, mixed slice thickness and a header that is sometimes a placeholder.
""")

code(r"""
pv = pd.read_csv(OUTPUT / "per_volume.csv")
pv["group"] = pd.cut(pv["sz_mm"], [0, 1.0, 2.5, 99], labels=["thin (<= 1 mm)", "mid (1-2.5 mm)", "thick (> 2.5 mm)"])
pv["placeholder"] = (pv.sx_mm == 1.0) & (pv.sy_mm == 1.0) & (pv.sz_mm == 1.0)
g = pv.groupby("group", observed=True)
live = pd.DataFrame({
    "volumes": g.size(), "axial slices": g["n_slices"].sum(),
    "slice thickness, median (mm)": g["sz_mm"].median(),
    "volumes with tumor": g["tumor_vox"].apply(lambda s: int((s > 0).sum())),
    "lesions": g["n_tumors"].sum(),
    "tumor burden, median over tumor-bearing (mL)": g.apply(lambda d: d.loc[d.tumor_vox > 0, "tumor_ml"].median()),
}).reset_index().rename(columns={"group": "slice-thickness group"})
total = pd.DataFrame([{"slice-thickness group": "all", "volumes": len(pv), "axial slices": int(pv.n_slices.sum()),
                       "slice thickness, median (mm)": pv.sz_mm.median(), "volumes with tumor": int((pv.tumor_vox > 0).sum()),
                       "lesions": int(pv.n_tumors.sum()), "tumor burden, median over tumor-bearing (mL)": pv.loc[pv.tumor_vox > 0, "tumor_ml"].median()}])
show_df(pd.concat([live, total]), precision=1, formats={"volumes": "{:,.0f}", "axial slices": "{:,.0f}", "lesions": "{:,.0f}", "volumes with tumor": "{:,.0f}"})
caption("Computed live from analysis/output/per_volume.csv. "
        f"{int(pv.placeholder.sum())} volumes (ids 28-47) carry a placeholder 1x1x1 mm header, so their thickness group and lesion sizes in mm are nominal.")
""")

code(r"""
show_table("T1")
""")

code(r"""
show_fig("F1")
""")

md(r"""
**Example slices** (live, from the fixed-window cache `data/cache_v2`): one tumor-bearing volume per thickness group, the axial slice
with the largest tumor area; liver outline teal, tumor outline red.
""")

code(r"""
if (V2 / "images.npy").exists() and (V2 / "meta.csv").exists():
    imgs = np.load(V2 / "images.npy", mmap_mode="r"); msks = np.load(V2 / "masks.npy", mmap_mode="r")
    vidx = np.load(V2 / "vol_index.npy"); names = (V2 / "volumes.txt").read_text().split("\n")
    meta = pd.read_csv(V2 / "meta.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, grp in zip(axes, ["thin", "mid", "thick"]):
        c = meta[(meta.thickness_group == grp) & meta.has_tumor & meta.spacing_reliable].name
        c = sorted(c, key=lambda n: int(re.findall(r"\d+", n)[0]))
        name = c[len(c) // 2]                                  # rule: middle volume (by id) of the group
        sl = np.flatnonzero(vidx == names.index(name))
        m = np.asarray(msks[sl[0]:sl[-1] + 1]); j = int(np.argmax((m == 2).sum((1, 2))))
        ax.imshow(imgs[sl[0] + j], cmap="gray", vmin=0, vmax=255)
        ax.contour(m[j] >= 1, levels=[0.5], colors=LIVER, linewidths=1.2)
        ax.contour(m[j] == 2, levels=[0.5], colors=TUMOR, linewidths=1.2)
        ax.set_title(f"{name.replace('.nii', '')}, {grp}, slice {j}", fontsize=9); ax.axis("off")
    fig.suptitle("Tumors are small and low-contrast even in a fixed -100..250 HU window", fontsize=10, fontweight="bold")
    fig.tight_layout(); plt.show()
    caption("Live from data/cache_v2 (256x256 uint8 slices, canonical orientation). Selection rule: middle tumor-bearing volume by id with a reliable header, per group.")
else:
    print("data/cache_v2 is not present on this machine; example slices skipped (see figure F2 below for the same idea).")
""")

# =====================================================================================================
md(r"""
## 3. Preprocessing: what was wrong and the fix

Two problems in the first (v1) slice cache:
1. **Intensity.** Each volume was min-max scaled on its own. The -3024 HU padding and metal outliers set the scale, so liver and tumor were squeezed into a few uint8 levels.
2. **Orientation.** The raw arrays come in three layouts. The header predicts the anterior-posterior axis (axis 1) in all volumes but not the left-right axis (axis 0): there only volumes 28-47 break the header rule (they say RAS but the liver sits at a low index). With flip augmentation on, a network is shown mirrored anatomy; whether that hurts the AIM-UNet is a hypothesis, untested here (it is what arm C versus B probes).

The fix (`Image Segmentation/prepare_data_v2.py`): one fixed HU window for every volume, and per-volume flips so the liver
sits at the high index of axis 0 and anterior at the high index of axis 1. The rule below is read from the script itself.
""")

code(r"""
path = ROOT / "Image Segmentation" / "prepare_data_v2.py"
src = path.read_text(); tree = ast.parse(src); lines = src.splitlines()
def func_src(name):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)
window_lines = [l for l in lines if re.match(r"(WIN_LO|WIN_HI) =", l)]
proc = func_src("process").splitlines()
inproc = [l.strip() for l in proc if re.search(r"^\s*(flips =|u8 =)", l)]
excerpt = "\n\n".join([
    "# fixed HU window (same for every volume)\n" + "\n".join(window_lines) + "\n# inside process(): clip to the window, scale to uint8, flip axes\n" + "\n".join(inproc),
    func_src("liver_vs_body"), func_src("orientation_class")])
display(Markdown("```python\n" + excerpt + "\n```"))
caption(f"Excerpt of {rel(path)} (extracted with ast; not a copy). The rule uses the ground-truth liver mask, so it is a training-data rule, not an inference rule.")
""")

code(r"""
if (V2 / "meta.csv").exists():
    meta = pd.read_csv(V2 / "meta.csv")
    ct = pd.crosstab(meta.orientation_class, meta.flipped_axes).reset_index().rename(columns={"orientation_class": "raw array class"})
    ct["volumes"] = ct.drop(columns="raw array class").sum(axis=1)
    show_df(ct, formats={c: "{:d}" for c in ct.columns if c != "raw array class"})
    caption("Live from data/cache_v2/meta.csv: raw array class (A/B/C) against the axes flipped to canonical orientation; "
            f"{int((~meta.spacing_reliable).sum())} volumes have a placeholder header (spacing_reliable = False).")
else:
    print("data/cache_v2/meta.csv not present; orientation table skipped.")
""")

code(r"""
show_table("T2")
t2 = pd.read_csv(TAB / "T2_per_volume_cache_stats.csv") if (TAB / "T2_per_volume_cache_stats.csv").exists() else None
if t2 is not None:
    c1, c2 = t2.contrast_v1.median(), t2.contrast_v2.median()
    display(Markdown(f"**Live check.** Median tumor-minus-liver contrast over {t2.contrast_v1.notna().sum()} tumor-bearing volumes: "
                     f"v1 {c1:+.1f} grey levels, v2 {c2:+.1f} grey levels (tumor darker than liver in both): the magnitude is a factor of {abs(c2 / c1):.0f} larger in v2 (of 255 levels). "
                     "The table also shows that the *grey-range* argument for v2 was weaker than expected: almost no v1 volume had its liver mean outside 0.2-0.8."))
""")

code(r"""
show_fig("F2", width=1000)
""")

# =====================================================================================================
md(r"""
## 4. Segmentation model and training recipe

**AIM-UNet (2D).** A U-Net on single 256x256 axial slices (1 input channel, 3 output classes: background, liver, tumor).
Four encoder stages (64, 128, 256, 512 channels, two 3x3 convs each, max-pool), a 1024-channel bottleneck, and decoder
stages that concatenate the upsampled features with an **Inception module** (1x1, 5x5, double 3x3 and pooled branches) applied to the skip connection.

**Loss and sampling.** Cross-entropy (class weights 1 / 2 / 1) plus a soft multi-class Dice loss; Adam, learning rate 1e-4, batch size 4,
bf16 autocast on the accelerator. Slices with tumor are oversampled by the ratio of non-tumor to tumor slices, and most empty slices
are down-weighted (`LITS_EMPTY_KEEP`). Augmentation is a rotation of up to 10 degrees, plus random h/v flips unless `LITS_FLIP_AUG=0`.
The checkpoint with the lowest validation loss is the one evaluated.
""")

code(r"""
import torch.nn as nn
# AIM_UNet straight from train_fast.py's source: execute only its three model classes (importing the module would create a results dir and log file).
tsrc = (ROOT / "Image Segmentation" / "train_fast.py").read_text()
keep = [n for n in ast.parse(tsrc).body if isinstance(n, ast.ClassDef) and n.name in ("InceptionModule", "ConvBlock", "AIM_UNet")]
ns = {"nn": nn, "torch": torch}
exec(compile(ast.Module(keep, []), "train_fast.py", "exec"), ns)
model = ns["AIM_UNet"]().eval()
n_params = sum(p.numel() for p in model.parameters())
rows = [(name, sum(p.numel() for p in m.parameters())) for name, m in model.named_children()
        if sum(p.numel() for p in m.parameters())]
pt = pd.DataFrame(rows, columns=["block", "parameters"])
pt["share of total (%)"] = 100 * pt.parameters / n_params
pt = pd.concat([pt, pd.DataFrame([{"block": "total", "parameters": n_params, "share of total (%)": 100.0}])])
with torch.no_grad():
    out_shape = tuple(model(torch.zeros(1, 1, 256, 256)).shape)
show_df(pt, precision=1, formats={"parameters": "{:,.0f}"})
caption(f"Live parameter count of AIM_UNet from train_fast.py: {n_params:,} parameters; forward pass on a 1x1x256x256 input gives output shape {out_shape}.")
""")

code(r"""
# train_fast.py options: defaults are parsed from the current script, baseline values from the as-run script in the archive
defaults = {}
for var in ["LITS_EPOCHS", "LITS_BATCH_SIZE", "LITS_EMPTY_KEEP", "LITS_SEED", "LITS_FLIP_AUG", "LITS_AMP",
            "LITS_SUBSET_N", "LITS_SUBSET_VAL_N", "LITS_CACHE_DIR"]:
    m = re.search(r'os\.environ\.get\("' + var + r'",\s*([^)]+?)\)', tsrc)
    g = m.group(1) if m else None
    defaults[var] = "unset" if g is None else ("<repo>/" + "/".join(re.findall(r'"([^"]+)"', g)) if "os.path.join" in g else g.strip('" '))
sh = (BASE / "archive" / "run_baseline_as_run.sh").read_text()
base_vals = dict(re.findall(r"(LITS_\w+)=(\S+)", " ".join(l for l in sh.splitlines() if l.startswith("export"))))
base_vals = {k: v.replace('"$ROOT/', "<repo>/").strip('"') for k, v in base_vals.items()}
what = {"LITS_EPOCHS": "training epochs", "LITS_BATCH_SIZE": "slices per batch", "LITS_EMPTY_KEEP": "weight of slices without liver (1 = original)",
        "LITS_SEED": "random seed (unset = unseeded)", "LITS_FLIP_AUG": "h/v flip augmentation (0 = rotation only)",
        "LITS_AMP": "mixed precision", "LITS_SUBSET_N": "training volumes (0 = all; stratified subset of the train part)",
        "LITS_SUBSET_VAL_N": "validation volumes (stratified subset of the held-out part)", "LITS_CACHE_DIR": "slice cache (A: data/cache; B, C: data/cache_v2)"}
opt = pd.DataFrame([(v, defaults[v], base_vals.get(v, "per arm" if v in ("LITS_FLIP_AUG", "LITS_CACHE_DIR") else "default"), what[v]) for v in defaults],
                   columns=["option", "default in train_fast.py", "value in baseline", "meaning"])
show_df(opt)
caption("Defaults parsed from the current Image Segmentation/train_fast.py, baseline values from docs/baseline_results/archive/run_baseline_as_run.sh (the script that produced the reported numbers; "
        "it also set the split-manifest options LITS_SPLIT_MANIFEST / LITS_VAL_FOLD / LITS_TEST_FOLD, which no longer exist in the final code). Learning rate 1e-4 and early-stopping patience 7 are fixed in the script.")
""")

# =====================================================================================================
md(r"""
## 5. Results: the baseline comparison

The **test fold** of the archived manifest-based run (26 patients, 23 with tumor; separate from the validation volumes; see the provenance note at the top, this is not the final code's 104 / 27 split). All arms train on the same 16 volumes
and validate on the same 6. Metrics are per-patient 3D Dice for liver and tumor, lesion recall (a lesion counts as hit by at least one
predicted tumor voxel) and false-positive tumor volume.
""")

code(r"""
def parse_log(path):
    n_done, total = 0, None
    if path.exists():
        txt = path.read_text(errors="replace")
        total = max([int(x) for x in re.findall(r"Epoch \d+/(\d+)", txt)] or [0]) or None
        n_done = len(set(re.findall(r"Epoch (\d+)/\d+\s\S+\sVal Loss", txt)))
    return n_done, total

def load_arm(k):
    folder, label, color = ARMS[k]
    d = BASE / folder
    a = dict(key=k, label=label, color=color, dir=d, pp=None, agg=None)
    try:
        a["agg"] = json.load(open(d / "eval" / "eval.json"))["aggregate"]["overall"]
        a["pp"] = pd.read_csv(d / "eval" / "per_patient.csv")
    except Exception:
        a["agg"] = a["pp"] = None
    n_done, total = parse_log(d / "stdout.log")
    a["status"] = ("evaluated" if a["pp"] is not None else "not started" if n_done == 0
                   else "trained, evaluation pending" if total and n_done >= total else f"training ({n_done}/{total} epochs)")
    return a

ARMDATA = {k: load_arm(k) for k in ARMS}
EVALUATED = [k for k, a in ARMDATA.items() if a["pp"] is not None]
status = pd.DataFrame([(k, a["label"], a["status"] if a["pp"] is not None else "pending: " + a["status"]) for k, a in ARMDATA.items()],
                      columns=["arm", "recipe", "status"])
show_df(status)
""")

code(r"""
# Integrity checks, live: no train/val/test overlap, test set = fold 1 of the archived manifest, per-patient mean = eval.json aggregate.
man = pd.read_csv(BASE / "archive" / "split_manifest_used.csv")
fold1 = set(man[man.fold == 1].name)
checks = []
for k in EVALUATED:
    a = ARMDATA[k]; names = set(a["pp"].name)
    sp = json.load(open(a["dir"] / "split_used.json"))
    overlap = names & (set(sp["train"]) | set(sp["val"]))
    agg_ok = abs(a["pp"].liver_dice.mean() - a["agg"]["liver_dice"]["mean"]) < 1e-9 and abs(a["pp"].tumor_dice.mean() - a["agg"]["tumor_dice"]["mean"]) < 1e-9
    checks.append((k, len(names), "yes" if names == fold1 else "NO", len(overlap), "yes" if agg_ok else "NO"))
if checks:
    show_df(pd.DataFrame(checks, columns=["arm", "test patients", "= fold 1 of archived manifest", "overlap with train/val volumes", "per-patient means = eval.json"]))
else:
    print("No arm evaluated yet.")
""")

code(r"""
show_table("T3")
""")

code(r"""
# Live summary from per_patient.csv: spread over patients, not just the mean.
def ms(s):  return f"{s.mean():.3f} (SD {s.std(ddof=1):.3f})"
def mq(s):  return f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"
rows = []
for k, a in ARMDATA.items():
    base = {"arm": f"{k}: {a['label']}"}
    if a["pp"] is None:
        rows.append({**base, "n patients": "pending: " + a["status"]}); continue
    p = a["pp"]; t = p[p.n_lesions > 0]
    rows.append({**base, "n patients": str(len(p)), "liver Dice, mean (SD over patients)": ms(p.liver_dice), "liver Dice, median [IQR]": mq(p.liver_dice),
                 "tumor-bearing patients": str(len(t)), "tumor Dice, mean (SD)": ms(t.tumor_dice), "tumor Dice, median [IQR]": mq(t.tumor_dice),
                 "patients with tumor Dice > 0": f"{int((t.tumor_dice > 0).sum())} of {len(t)}",
                 "lesions hit / total": f"{int(p.lesions_detected.sum())} / {int(p.n_lesions.sum())}",
                 "false-positive tumor mL, median": f"{p.fp_tumor_ml.median():.2f}"})
show_df(pd.DataFrame(rows))
caption("Live from docs/baseline_results/<arm>/eval/per_patient.csv. SD, median and IQR are over test patients; they are the spread to compare any between-arm difference against.")
""")

code(r"""
show_fig("F3", width=980)
""")

code(r"""
show_fig("F4", width=980)
""")

code(r"""
show_fig("F5", width=980)
""")

code(r"""
show_fig("F6", width=820)
""")

code(r"""
show_table("T3b"); show_table("T3c")
""")

md(r"""
**Paired comparison between arms.** Same test patients, per-patient difference (second arm minus first), bootstrap over patients
(10,000 resamples, seed 12345). It reflects patient sampling only, not training-seed variance. Shown only for arms that are evaluated.
""")

code(r"""
def paired_boot(x, y, n_boot=10000, seed=12345):
    d = (y - x)[~(np.isnan(x) | np.isnan(y))]
    if len(d) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    mean = d[idx].mean(axis=1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(mean, 2.5)), hi=float(np.percentile(mean, 97.5)),
                sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"), up=int((d > 0).sum()), down=int((d < 0).sum()), tie=int((d == 0).sum()))

METRICS = [("liver_dice", "liver Dice"), ("tumor_dice", "tumor Dice"), ("lesion_recall", "lesion recall")]
PAIRS = [("B", "A"), ("C", "A"), ("C", "B")]
PAIRED = {}
rows = []
for k, ref in PAIRS:
    if k not in EVALUATED or ref not in EVALUATED:
        rows.append({"comparison": f"{k} minus {ref}", "metric": "pending (needs both arms evaluated)"}); continue
    pa, pb = ARMDATA[ref]["pp"].set_index("name"), ARMDATA[k]["pp"].set_index("name")
    common = sorted(set(pa.index) & set(pb.index))
    for m, lab in METRICS:
        r = paired_boot(pa.loc[common, m].values.astype(float), pb.loc[common, m].values.astype(float))
        PAIRED[(k, ref, m)] = r
        if r:
            rows.append({"comparison": f"{k} minus {ref}", "metric": lab, "paired patients": r["n"], "mean difference": f"{r['mean']:+.3f}",
                         "95% bootstrap CI": f"[{r['lo']:+.3f}, {r['hi']:+.3f}]", "SD of differences": f"{r['sd']:.3f}",
                         "patients higher / lower / equal": f"{r['up']} / {r['down']} / {r['tie']}"})
show_df(pd.DataFrame(rows))
""")

code(r"""
show_table("T3d")
""")

md(r"""
### What this does and does not show

(Generated from the data above: every statement below is computed from the saved results, none is typed in.)
""")

code(r"""
L = []
EPOCH = {k: re.search(r"best_epoch_(\d+)", json.load(open(ARMDATA[k]["dir"] / "eval" / "eval.json"))["checkpoint"]).group(1).lstrip("0") for k in EVALUATED}
def summ(p):
    t = p[p.n_lesions > 0]
    return dict(liver=p.liver_dice.mean(), tumor=t.tumor_dice.mean(), tmed=t.tumor_dice.median(), n=len(p), nt=len(t),
                hit=int(p.lesions_detected.sum()), les=int(p.n_lesions.sum()), fp=p.fp_tumor_ml.median())
for k in EVALUATED:
    p = ARMDATA[k]["pp"]; v = summ(p); t = p[p.n_lesions > 0]
    L.append(f"- **Arm {k} ({ARMDATA[k]['label']}).** Liver Dice mean {v['liver']:.2f} (SD over patients {p.liver_dice.std(ddof=1):.2f}) / median {p.liver_dice.median():.2f}, "
             f"{int((p.liver_dice < 0.3).sum())} of {v['n']} patients below 0.3; tumor Dice mean {v['tumor']:.2f} / median {v['tmed']:.2f}, "
             f"{int((t.tumor_dice == 0).sum())} of {v['nt']} tumor-bearing patients at exactly 0; {v['hit']} of {v['les']} lesions hit ({v['hit'] / v['les']:.2f}); "
             f"false-positive tumor volume median {v['fp']:.2f} mL.")
pend_pairs = [(k, ref) for k, ref in PAIRS if k not in EVALUATED or ref not in EVALUATED]
for k, ref in pend_pairs:
    pend = sorted(x for x in (k, ref) if x not in EVALUATED)
    L.append(f"- **{k} vs {ref}: pending** ({'arm' if len(pend) == 1 else 'arms'} {', '.join(pend)} not evaluated: " + "; ".join(f"{x}: {ARMDATA[x]['status']}" for x in pend) + ").")
done_pairs = [(k, ref) for k, ref in PAIRS if (k, ref) not in pend_pairs]
if done_pairs:
    parts = []
    for k, ref in done_pairs:
        up = [lab for m, lab in METRICS if PAIRED.get((k, ref, m)) and PAIRED[(k, ref, m)]["lo"] > 0]
        dn = [lab for m, lab in METRICS if PAIRED.get((k, ref, m)) and PAIRED[(k, ref, m)]["hi"] < 0]
        inc = [lab for m, lab in METRICS if PAIRED.get((k, ref, m)) and PAIRED[(k, ref, m)]["lo"] <= 0 <= PAIRED[(k, ref, m)]["hi"]]
        seg = ([f"higher (interval excludes 0) on {', '.join(up)}"] if up else []) + ([f"lower (interval excludes 0) on {', '.join(dn)}"] if dn else []) \
              + ([f"interval includes 0, so not distinguishable here, on {', '.join(inc)}"] if inc else [])
        parts.append(f"**{k} vs {ref}:** " + "; ".join(seg))
    L.append("- **Paired comparisons in one line** (mean per-patient difference, 95% bootstrap interval over patients): " + " | ".join(parts) + ".")
for k, ref in done_pairs:
    for m, lab in METRICS:
        r = PAIRED.get((k, ref, m))
        if not r:
            continue
        if r["lo"] > 0 or r["hi"] < 0:
            s_ = (f"mean paired difference {r['mean']:+.3f} (95% bootstrap CI {r['lo']:+.3f} to {r['hi']:+.3f}) excludes 0; the patient-to-patient SD of the "
                  f"difference is {r['sd']:.3f} and this is one seed on {r['n']} patients, so read it as a single-seed small experiment, not a general result")
        else:
            s_ = (f"mean paired difference {r['mean']:+.3f} (95% bootstrap CI {r['lo']:+.3f} to {r['hi']:+.3f}) includes 0: this experiment cannot distinguish the arms "
                  f"(underpowered: {r['n']} patients, one seed); that is not evidence that they are equal")
        L.append(f"- **{k} vs {ref}, {lab}:** {s_}.")
# direction per thickness group, pooled recall by lesion size (from per_patient.csv and eval.json), absolute level
if "A" in EVALUATED:
    def grp(p):
        out = {}
        for g, d in p.groupby("thickness_group"):
            t = d[d.n_lesions > 0]
            out[g] = (d.liver_dice.mean(), t.tumor_dice.mean(), d.lesions_detected.sum() / max(d.n_lesions.sum(), 1), len(d))
        return out
    ga = grp(ARMDATA["A"]["pp"]); segs = []
    for k in [x for x in EVALUATED if x != "A"]:
        gk = grp(ARMDATA[k]["pp"])
        segs.append(f"arm {k}: " + ", ".join(f"{lab} {sum(gk[g][i] > ga[g][i] for g in ga)}/{len(ga)}" for i, lab in enumerate(["liver Dice mean", "tumor Dice mean", "pooled lesion recall"])))
    if segs:
        L.append("- **Thickness groups in which an arm is above arm A** (groups of " + " / ".join(str(ga[g][3]) for g in ["thin", "mid", "thick"] if g in ga)
                 + " patients, so only the direction is informative): " + "; ".join(segs) + ".")
def small_hits(k):
    les = pd.DataFrame(json.load(open(ARMDATA[k]["dir"] / "eval" / "eval.json"))["lesions"])
    sm = les[les.diameter_mm < 10]
    return int(sm.detected.sum()), len(sm)
if EVALUATED:
    sh_ = {k: small_hits(k) for k in EVALUATED}
    L.append("- **Small lesions (< 10 mm) hit:** " + ", ".join(f"{k} {a_}/{n_}" for k, (a_, n_) in sh_.items())
             + (f"; every arm hits under half of them (best {max(a_ / n_ for a_, n_ in sh_.values()):.2f}), so small lesions stay mostly missed." if all(a_ / n_ < 0.5 for a_, n_ in sh_.values()) else "."))
    best = max(EVALUATED, key=lambda k: ARMDATA[k]["pp"][ARMDATA[k]["pp"].n_lesions > 0].tumor_dice.mean())
    tb = ARMDATA[best]["pp"]; tb = tb[tb.n_lesions > 0]
    L.append(f"- **Absolute level.** The arm with the highest mean tumor Dice (arm {best}) reaches {tb.tumor_dice.mean():.2f} mean / {tb.tumor_dice.median():.2f} median, "
             f"with {int((tb.tumor_dice >= 0.5).sum())} of {len(tb)} tumor-bearing patients at 0.5 or above and {int((tb.tumor_dice == 0).sum())} at exactly 0: tumor Dice is still low in absolute terms.")
if len(EVALUATED) >= 2:
    fps = {k: ARMDATA[k]["pp"].fp_tumor_ml.median() for k in EVALUATED}
    L.append("- **False-positive tumor volume** (median per patient): " + ", ".join(f"{k} {v:.2f} mL" for k, v in fps.items())
             + ". Read it together with lesion recall: an arm that predicts little tumor also produces little false-positive volume, so a low value alone is not a better result.")
if done_pairs:
    L.append("- **Multiple comparisons.** Up to 9 unadjusted 95% intervals (3 pairs x 3 metrics) are reported; an interval that excludes 0 can occur by chance when this many are looked at.")
    L.append("- **What each pair isolates.** B vs A and C vs A change the intensity window and the orientation handling together, so a difference cannot be assigned to either change; only C vs B isolates the flip augmentation.")
L += ["- **One run per arm.** Each arm is a single training run (seed 1234); the intervals cover test-patient sampling only, not seed-to-seed variation. The evaluated epoch is picked per arm on only 6 validation volumes "
      "(" + ", ".join(f"{k} epoch {EPOCH[k]}" for k in EVALUATED) + "), which adds noise to every comparison.",
      "- **Does not show:** anything about other seeds, more training or more data (16 training volumes, 6 epochs); performance outside this one 26-patient test fold; equality with a run of the final code, which uses a different split (provenance note at the top).",
      "- **Does not show:** that any difference between thickness groups or lesion-size bins is reliable (groups hold as few as 3 patients; the Wilson intervals treat lesions as independent although they cluster within patients).",
      "- **Logged validation Dice is inflated:** it is a per-batch mean in which batches with no tumor in label and prediction score 1.0 (see figure F4); only the test-fold numbers above are per-patient 3D Dice."]
display(Markdown("\n".join(L)))
""")

# =====================================================================================================
md(r"""
## 6. Synthetic tumors: a corrected 3D multi-GAN

The idea: train 3D GANs on tumor patches, generate new tumors and insert them into healthy-looking liver tissue of training volumes as augmentation.
The original script (`Syn_Img_Pipeline_Code.py`, kept unchanged) was audited in `Image Synthesis/GAN_Analysis_Report.md`; the audit found it did not train anything.
`Syn_Img_Pipeline_fixed.py` is a rewrite with tests.
""")

code(r"""
show_table("T4")
""")

code(r"""
fix = pd.DataFrame([
    ("Training loop", "Loop body ends in a placeholder comment: no loss, backward or step; saved weights are random", "Two 3D DCGANs, one WGAN-GP, an adversarial aggregator and a supervised mask head are actually trained; losses logged"),
    ("Tensor contract", "Generator emits 80x32x32; discriminators return B*125 / B*7 values; real data scaled into [-3, 1] vs tanh output [-1, 1]", "64^3 patches, one score per sample, one window mapped to [-1, 1]; checked by unit tests"),
    ("Tumor mask", "Unsupervised 3-conv net thresholded; real masks discarded; written into the ground truth as label 2", "Mask head trained on real patch/mask pairs plus healthy negatives; samples are cleaned and rejected if invalid"),
    ("Insertion", "Whole-patch histogram matching and smoothing, healthy patch cut and pasted at different coordinates, labels on non-liver voxels, volumes saved in the [-1, 1] window", "Host-aware insertion inside the liver; augmented volumes saved in HU to a separate folder, never into data/raw"),
    ("Data and split", "Reads a fixed folder (trainOriginal_65), at most 65 volumes", "Reads the LiTS folder; GAN training and insertion hosts come only from the train part of the segmenter's split; placeholder-spacing volumes (28-47) are dropped"),
], columns=["area", "original script (per the audit)", "corrected pipeline"])
show_df(fix)
caption("Static summary of Image Synthesis/GAN_Analysis_Report.md versus the docstring of Image Synthesis/Syn_Img_Pipeline_fixed.py; the audit's claims were not re-run here.")
""")

code(r"""
show_table("T4b")
""")

code(r"""
def npz_shapes(p):
    # array shapes inside an .npz without loading the arrays
    out = {}
    with zipfile.ZipFile(p) as zf:
        for n in zf.namelist():
            with zf.open(n) as f:
                v = np.lib.format.read_magic(f)
                shape = (np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f))[0]
                out[n.replace(".npy", "")] = shape
    return out

if (GAN / "patches.npz").exists():
    ps, ss = npz_shapes(GAN / "patches.npz"), npz_shapes(GAN / "synthetic" / "samples.npz")
    log = (GAN / "pipeline.log").read_text(errors="replace")
    held = re.findall(r"mask head heldout: dice=([\d.]+)", log); trn = re.findall(r"mask head train: dice=([\d.]+)", log)
    hv = re.findall(r"held-out volumes \[([\d, ]+)\]", log)
    tmpman = [m for m in re.findall(r"manifest=(\S+)", log) if m.startswith("/tmp")]   # only tested for; the path is never displayed
    aug = re.findall(r"augmented (\d+)/(\d+) hosts", log)
    show_df(pd.DataFrame([
        ("real tumor patches (64^3)", ps["images"][0]), ("healthy patches", ps["healthy"][0]),
        ("generated samples accepted (after mask-head cleaning)", ss["patches"][0]),
        ("augmented host volumes", f"{aug[-1][0]} of {aug[-1][1]}" if aug else "n/a"),
        ("mask head Dice, train patches (last log line)", trn[-1] if trn else "n/a"),
        ("mask head Dice, held-out volumes (last log line)", (held[-1] + (f" ({len(hv[-1].split(','))} volumes)" if hv else "")) if held else "n/a")], columns=["gan_long2 run (25 epochs, 8 volumes)", "value"]), precision=3)
    caption("Live from results/gan_long2 (patches.npz, synthetic/samples.npz, pipeline.log). The held-out mask-head Dice is far below the train Dice and is measured on only "
            + (f"{len(hv[-1].split(','))} volumes" if hv else "a few volumes") + ": the mask head does not generalise on this little data, so the red outlines on generated patches in F7a are unreliable."
            + (" This run used an earlier version of the GAN script that read a temporary 5-fold split manifest outside the repo (the shipped script takes the train part of the original split), so it cannot be reproduced from the repo as shipped." if tmpman else ""))
else:
    print("results/gan_long2 not present; GAN run summary skipped.")
""")

code(r"""
show_fig("F7a", width=900)
""")

code(r"""
show_fig("F7b", width=900)
""")

md(r"""
**Honest status of the GAN.** The code is correct and tested (shapes, ranges, losses, resampling, insertion). After the short runs done here the generated
patches are blurry blobs, not realistic tumors, the mask head does not generalise to held-out volumes, and the inserted tumors are faint. **No segmenter was trained on
synthetic data**, so there is no evidence that it helps. Treat it as a starting point; realism has to be shown (e.g. a discriminator probe or reader study) before any Dice claim.
""")

# =====================================================================================================
md(r"""
## 7. Findings, limitations, next steps
""")

code(r"""
F = []
n_t = int((pv.tumor_vox > 0).sum()); small = int(pv.n_tumors.sum())
F.append(f"- **Data.** {len(pv)} volumes, {n_t} with tumor. Lesions are small and the headers are unreliable for {int(pv.placeholder.sum())} volumes (placeholder 1 mm spacing).")
if t2 is not None:
    F.append(f"- **Preprocessing.** The fixed-window cache increases the magnitude of the tumor-minus-liver contrast (tumor darker) from {t2.contrast_v1.median():+.1f} to {t2.contrast_v2.median():+.1f} uint8 levels (median over {t2.contrast_v1.notna().sum()} volumes). "
             "That is a statement about the *input*; whether it helps a segmenter is the A/B/C question.")
if EVALUATED:
    bits = []
    for k in EVALUATED:
        a = ARMDATA[k]["pp"]; t = a[a.n_lesions > 0]
        bits.append(f"{k}: liver Dice {a.liver_dice.mean():.2f}, tumor Dice {t.tumor_dice.mean():.2f} (median {t.tumor_dice.median():.2f}), lesions hit {int(a.lesions_detected.sum())}/{int(a.n_lesions.sum())}")
    F.append("- **Baseline (6 epochs on 16 volumes, one seed per arm, 26 test patients).** " + "; ".join(bits) + ".")
pend = [k for k in ARMS if k not in EVALUATED]
F.append("- **Do the data-pipeline changes help?** " + (f"Unanswered: {'arm' if len(pend) == 1 else 'arms'} {', '.join(pend)} {'is' if len(pend) == 1 else 'are'} pending ({'; '.join(ARMDATA[k]['status'] for k in pend)}). Re-run this notebook when they finish."
         if pend else (" ".join(
             f"{k} vs {ref}: " + (", ".join(f"{lab} {'higher' if PAIRED[(k, ref, m)]['lo'] > 0 else 'lower' if PAIRED[(k, ref, m)]['hi'] < 0 else 'not distinguishable'}" for m, lab in METRICS if PAIRED.get((k, ref, m))) + ".")
             for k, ref in PAIRS) + " (Mean paired differences with 95% bootstrap intervals over patients, section 5.) A to B bundles the intensity window and the orientation fix, so a gain cannot be assigned to either; "
             "B to C isolates flip augmentation only. Absolute tumor Dice is still low, the evidence is one seed per arm on 26 patients, and it is not a general claim.")))
F.append("- **GAN.** The corrected pipeline is tested but its samples are blurry and unrealistic after short runs; no downstream benefit is shown.")
display(Markdown("\n".join(F)))
""")

md(r"""
**Limitations**
- Tiny experiment: 16 training and 6 validation volumes, 6 epochs, one training seed per arm, 26 test patients. Bootstrap intervals cover patient sampling only.
- Orientation canonicalisation uses the ground-truth liver mask, so it is a training-data rule; an inference-time rule would need a separate estimator.
- 20 volumes have no true spacing, so lesion sizes in mm and thickness groups are nominal for them.
- Lesion recall counts one predicted voxel as a hit, which is lenient; false-positive volume is reported separately.
- The 2D model sees no through-plane context; whether that limits small-lesion detection is a hypothesis, untested here.

**Next steps**
1. Train longer on all training volumes with several seeds, then repeat the A/B/C comparison and report spreads.
2. Add through-plane context (2.5D or 3D) and a lesion-level metric with a stricter overlap criterion.
3. Train the GAN much longer, check realism independently, and only then test a segmenter trained with the synthetic tumors against the same baseline.
""")

# =====================================================================================================
md(r"""
## 8. Reproduce

Commands are read from `docs/RUNNING.md` (steps 1 to 4). Raw data (about 50 GB) must be placed in `data/raw` first.
""")

code(r"""
sh_text = (ROOT / "docs/RUNNING.md").read_text()
blocks = re.findall(r"^## (\d)\. [^\n]*\n(.*?)(?=^## \d\.|\Z)", sh_text, flags=re.S | re.M)
for num, body in blocks:
    if num in ("1", "3", "4"):
        title = re.search(r"^## " + num + r"\. ([^\n]*)", sh_text, flags=re.M).group(1)
        for b in re.findall(r"```bash\n(.*?)```", body, flags=re.S):
            display(Markdown(f"**Step {num}: {title}**\n\n```bash\n{b.rstrip()}\n```"))
display(Markdown("The 3-arm baseline in section 5 was produced by the as-run script `docs/baseline_results/archive/run_baseline_as_run.sh` together with the archived train / evaluation code and split manifest "
                 "(hours on an Apple M4 Pro; run it in a scratch checkout as described in `docs/baseline_results/archive/README.md`). `analysis/run_baseline.sh` re-runs the same A/B/C comparison on the final code with the original split; "
                 "its numbers will not match section 5 one-to-one. "
                 "Rebuild this showcase with `python -I analysis/make_showcase.py` and this notebook with `python -I analysis/build_notebook.py --execute --html`."))
""")

md(r"""
### Heavy cells (off by default)

Each cell prints its command and runs it only when its flag is `True` (first code cell).
""")

code(r"""
# RUN_PREPARE: build the v2 slice cache (about 2 min for a few volumes, much longer for all 131; skips itself if already built).
run_heavy("prepare", RUN_PREPARE, ["python", "-I", "Image Segmentation/prepare_data_v2.py"])
""")

code(r"""
# RUN_TRAIN: smoke run (4 training volumes, 1 epoch, results/quick). It prepares the cache first if needed.
run_heavy("train (smoke)", RUN_TRAIN, ["bash", "run_pipeline.sh", "quick"], env={"PYTHON": "python"})
print("\nFull recipe, hours (not behind a flag):  bash run_pipeline.sh full    |    3-arm re-run on the final code: bash analysis/run_baseline.sh")
""")

code(r"""
# RUN_GAN: tiny end-to-end GAN smoke run on 4 volumes into results/gan_notebook (CPU; does not touch results/gan_long2).
run_heavy("gan (smoke)", RUN_GAN, ["python", "-I", "Image Synthesis/Syn_Img_Pipeline_fixed.py"],
          env={"LITS_GAN_STAGE": "all", "LITS_GAN_EPOCHS": "1", "LITS_GAN_MAX_VOLUMES": "4", "LITS_GAN_RESULTS_DIR": "results/gan_notebook"})
""")


# =====================================================================================================
def build():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["title"] = "LiTS Pipeline Showcase"
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell-{i:03d}"
    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"wrote {os.path.relpath(NB_PATH, ROOT)} ({len(nb.cells)} cells)")
    return nb


def execute(nb):
    from nbclient import NotebookClient
    t0 = time.time()
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": ROOT}})
    client.execute()
    nbformat.write(nb, NB_PATH)
    print(f"executed in {time.time() - t0:.0f}s -> {os.path.relpath(NB_PATH, ROOT)} ({os.path.getsize(NB_PATH) / 1e6:.1f} MB)")
    return nb


def export_html(nb):
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter(theme="light").from_notebook_node(nb, resources={"metadata": {"name": "LiTS Pipeline Showcase"}})
    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {os.path.relpath(HTML_PATH, ROOT)} ({os.path.getsize(HTML_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute the notebook with nbclient and embed outputs")
    ap.add_argument("--html", action="store_true", help="export docs/showcase/LiTS_Pipeline_Showcase.html (needs --execute for outputs)")
    args = ap.parse_args()
    nb = build()
    if args.execute:
        nb = execute(nb)
    if args.html:
        export_html(nb)
