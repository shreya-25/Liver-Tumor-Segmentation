"""Showcase builder: tables (CSV + Markdown) and figures (PNG) for the liver-tumor project, plus docs/showcase/index.md.

Run from the repo root (any interpreter that has torch, numpy, scipy, nibabel, pandas, matplotlib):

    python -I analysis/make_showcase.py

Everything is read from files under the repo (analysis/output, data/cache, data/cache_v2, docs/baseline_results,
results/gan_long2, data/raw); every number in a table, figure or caption is computed here from those files.
The 3-arm baseline (A, B, C) is read from docs/baseline_results/<arm>/ (an arm without eval/eval.json would be shown as
'pending'). Its numbers come from the ARCHIVED manifest-based run, not from the final code (see PROVENANCE below and
docs/baseline_results/archive/README.md). Output: docs/showcase/{tables,figures,index.md,captions.json}.

Env (optional): SHOWCASE_ONLY=T1,T3,F5 builds only those items (development); SHOWCASE_SKIP_TESTS=1 skips
the unit-test subprocess in T4. Deterministic: fixed seeds, fixed selection rules (documented in the captions).
"""
import os
import re
import sys
import ast
import csv
import json
import time
import shutil
import subprocess
import importlib.util
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "showcase")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
PV_CSV = os.path.join(ROOT, "analysis", "output", "per_volume.csv")
STATS_JSON = os.path.join(ROOT, "analysis", "output", "stats.json")
V1, V2 = os.path.join(ROOT, "data", "cache"), os.path.join(ROOT, "data", "cache_v2")
BASE = os.path.join(ROOT, "docs", "baseline_results")   # tracked, path-scrubbed copy of the small result files
CKPT_BASE = os.path.join(ROOT, "results", "baseline")   # checkpoints (git-ignored, not shipped): only F5 needs them
GAN = os.path.join(ROOT, "results", "gan_long2")
RAW = os.path.join(ROOT, "data", "raw")
TMP = os.path.join(__import__("tempfile").gettempdir(), "showcase_tmp")
os.makedirs(TMP, exist_ok=True)

LIVER, TUMOR = "#2a9d8f", "#d62828"
# arm key -> (results dir, cache dir, short label, colour)
ARMS = {
    "A": ("A_v1_minmax_flip", V1, "v1 min-max, flips", "#6c757d"),
    "B": ("B_v2_window_flip", V2, "v2 window, flips", "#1d6fb8"),
    "C": ("C_v2_window_noflip", V2, "v2 window, no flips", "#e08e0b"),
}
# One provenance note, reused in index.md and the T3 caption: the reported numbers belong to the archived run, not to the final code.
PROVENANCE = (
    "The reported A/B/C numbers were produced by the code archived in `docs/baseline_results/archive/` (see its README), not by the final code. "
    "That run used a stratified patient-level 5-fold manifest: training volumes from folds 2-4 (stratified subset of 16), validation on 6 volumes of fold 0, "
    "and evaluation on the whole test fold 1 (26 patients, 23 with tumor), which training never loaded. "
    "The final code uses the project's original split only (sklearn `train_test_split` over the sorted volume names, `test_size=0.2`, `random_state=42`: "
    "104 train / 27 held-out volumes; the held-out part is both validation and test set, so its scores are validation-selected). "
    "The numbers shown here therefore do not carry over one-to-one to a run of the final code; `analysis/run_baseline.sh` is the re-run design for the final code.")
SIZE_BINS = ["<10 mm", "10-20 mm", ">20 mm"]
GROUPS = ["thin", "mid", "thick"]
ONLY = {s for s in os.environ.get("SHOWCASE_ONLY", "").split(",") if s}

CAPTIONS, TABLE_MD, WROTE, PENDING = {}, {}, [], []
NOTES = []  # honest per-item notes shown in index.md

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
    "font.family": "DejaVu Sans",
})


def want(key):
    return not ONLY or key in ONLY


def note_pending(msg):
    if msg not in PENDING:
        PENDING.append(msg)


# --------------------------------------------------------------------------------------- generic helpers
def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in r.values) + " |")
    return "\n".join(lines)


def save_table(key, name, df, caption):
    os.makedirs(TAB, exist_ok=True)
    base = os.path.join(TAB, f"{key}_{name}")
    df.to_csv(base + ".csv", index=False)
    md = md_table(df)
    open(base + ".md", "w").write(md + "\n")
    TABLE_MD[key] = (md, f"tables/{key}_{name}.csv", f"tables/{key}_{name}.md")
    CAPTIONS[key] = caption
    WROTE.extend([base + ".csv", base + ".md"])


def save_fig(key, name, fig, caption):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, f"{key}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CAPTIONS[key] = caption
    TABLE_MD[key] = (None, f"figures/{key}_{name}.png", None)
    WROTE.append(path)


def vnum(name):
    return int(re.search(r"(\d+)", name).group(1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def thickness_group(sz):
    return "thin" if sz <= 1.0 else "mid" if sz <= 2.5 else "thick"


def load_pv():
    pv = pd.read_csv(PV_CSV)
    pv["thickness_group"] = pv["sz_mm"].map(thickness_group)
    pv["placeholder"] = (pv.sx_mm == 1.0) & (pv.sy_mm == 1.0) & (pv.sz_mm == 1.0)
    return pv


# --------------------------------------------------------------------------------------- baseline readers
def parse_log(path):
    """stdout.log -> {epochs: [{epoch, train, val, dice_bg, dice_liver, dice_tumor}], total, complete}."""
    out = dict(epochs=[], total=None, complete=False)
    if not os.path.exists(path):
        return out
    cur = None
    for line in open(path, errors="replace"):
        m = re.search(r"Epoch (\d+)/(\d+)\s\S+\sTrain Loss:\s*([\d.]+)", line)
        if m:
            cur = dict(epoch=int(m.group(1)), train=float(m.group(3)), val=None, dice_bg=None, dice_liver=None, dice_tumor=None)
            out["epochs"].append(cur)
            out["total"] = int(m.group(2))
        m = re.search(r"Epoch (\d+)/(\d+)\s\S+\sVal Loss:\s*([\d.]+)", line)
        if m and cur and cur["epoch"] == int(m.group(1)):
            cur["val"] = float(m.group(3))
        m = re.search(r"Validation Dice\s\S+\sBackground:\s*([\d.]+),\sLiver:\s*([\d.]+),\sTumor:\s*([\d.]+)", line)
        if m and cur:
            cur["dice_bg"], cur["dice_liver"], cur["dice_tumor"] = (float(m.group(i)) for i in (1, 2, 3))
        if "Training complete" in line:
            out["complete"] = True
    return out


def load_arms():
    """-> {key: dict(label, color, dir, cache, log, eval (parsed json or None), per_patient DataFrame or None,
    ckpt_epoch, status)}"""
    arms = {}
    for k, (d, cache, label, color) in ARMS.items():
        adir = os.path.join(BASE, d)
        a = dict(key=k, label=label, color=color, dir=adir, cache=cache, ev=None, pp=None, ckpt=None, ckpt_epoch=None)
        a["log"] = parse_log(os.path.join(adir, "stdout.log"))
        ej = os.path.join(adir, "eval", "eval.json")
        if os.path.exists(ej):
            try:
                a["ev"] = json.load(open(ej))
            except Exception:  # half-written file while an evaluation is running
                a["ev"] = None
        if a["ev"]:
            pp = pd.DataFrame(a["ev"]["per_patient"]).set_index("name")
            a["pp"] = pp.astype({c: float for c in ("liver_dice", "tumor_dice", "lesion_recall", "fp_tumor_ml")}, errors="ignore")
            a["ckpt"] = a["ev"]["checkpoint"]
            m = re.search(r"best_epoch_(\d+)", a["ckpt"])
            a["ckpt_epoch"] = int(m.group(1)) if m else None
            a["status"] = "evaluated"
        else:
            eps = a["log"]["epochs"]
            if not os.path.isdir(adir) or not eps:
                a["status"] = "not started"
            elif a["log"]["complete"]:
                a["status"] = "trained, evaluation pending"
            else:
                a["status"] = f"training ({len(eps)}/{a['log']['total']} epochs)"
            note_pending(f"arm {k} ({d}): {a['status']}")
        arms[k] = a
    return arms


def done(arms):
    return [k for k, a in arms.items() if a["ev"]]


# =========================================================================================== T1 cohort
def t1_cohort(pv):
    stats = json.load(open(STATS_JSON))
    comps = pd.DataFrame([dict(name=v["name"], **c) for v in stats["per_volume"] for c in v["components"]])
    comps["placeholder"] = comps["name"].isin(pv.loc[pv.placeholder, "name"])
    tb = pv[pv.tumor_ml > 0]
    q = lambda s, qs=(0.05, 0.25, 0.5, 0.75, 0.95): " / ".join(f"{s.quantile(x):.1f}" for x in qs)
    qp = lambda s: " / ".join(f"{100 * s.quantile(x):.2f}" for x in (0.05, 0.25, 0.5, 0.75, 0.95))
    rows = [
        ("Volumes", f"{len(pv)}"),
        ("Axial slices (all volumes)", f"{int(pv.n_slices.sum()):,} (per volume: min {pv.n_slices.min()}, median {pv.n_slices.median():.0f}, max {pv.n_slices.max()})"),
    ]
    for g, lab in zip(GROUPS, ["thin (<= 1 mm)", "mid ((1, 2.5] mm)", "thick (> 2.5 mm)"]):
        s = pv[pv.thickness_group == g]
        rows.append((f"Thickness group {lab}", f"{len(s)} volumes, {int(s.n_slices.sum()):,} slices, slice thickness {s.sz_mm.min():.2f}-{s.sz_mm.max():.2f} mm"))
    rows += [
        ("Volumes with a placeholder 1x1x1 mm header (true spacing unknown)", f"{int(pv.placeholder.sum())} (volume ids {vnum(pv[pv.placeholder].name.iloc[0])}-{vnum(pv[pv.placeholder].name.iloc[-1])}; counted as 'thin')"),
        ("In-plane pixel spacing (mm)", f"{pv.sx_mm.min():.3f}-{pv.sx_mm.max():.3f} (median {pv.sx_mm.median():.3f})"),
        ("Slice thickness (mm)", f"{pv.sz_mm.min():.2f}-{pv.sz_mm.max():.2f} (median {pv.sz_mm.median():.2f})"),
        ("Volumes without tumor", f"{int((pv.tumor_ml <= 0).sum())} of {len(pv)}"),
        ("Volumes with tumor", f"{len(tb)}"),
        ("Lesions (3-D connected tumor components, all volumes)", f"{int(pv.n_tumors.sum())}"),
        ("Lesions per tumor-bearing volume (min / median / max)", f"{int(tb.n_tumors.min())} / {tb.n_tumors.median():.0f} / {int(tb.n_tumors.max())}"),
        ("Tumor burden, mL, tumor-bearing volumes (5 / 25 / 50 / 75 / 95 %)", q(tb.tumor_ml)),
        ("Tumor volume as % of liver volume (5 / 25 / 50 / 75 / 95 %)", qp(tb.tumor_frac_of_liver)),
        ("Lesion equivalent diameter, mm (5 / 25 / 50 / 75 / 95 %)", q(comps.eq_diam_mm)),
        ("Lesions < 10 mm", f"{int((comps.eq_diam_mm < 10).sum())} ({100 * (comps.eq_diam_mm < 10).mean():.0f} %)"),
        ("Median tumor-minus-liver mean HU contrast (tumor-bearing)", f"{tb.contrast_hu.median():+.1f} HU (IQR {tb.contrast_hu.quantile(.25):+.1f} to {tb.contrast_hu.quantile(.75):+.1f})"),
    ]
    df = pd.DataFrame(rows, columns=["item", "value"])
    save_table("T1", "cohort_summary", df,
               f"Cohort summary of the {len(pv)} LiTS training volumes from analysis/output/per_volume.csv (lesion sizes from stats.json); "
               f"diameters of the {int(pv.placeholder.sum())} placeholder-header volumes are nominal.")
    return comps


# =========================================================================================== T2 preprocessing
def cache_volume_stats(cache_dir):
    names = open(os.path.join(cache_dir, "volumes.txt")).read().split("\n")
    vi = np.load(os.path.join(cache_dir, "vol_index.npy"))
    images = np.load(os.path.join(cache_dir, "images.npy"), mmap_mode="r")
    masks = np.load(os.path.join(cache_dir, "masks.npy"), mmap_mode="r")
    starts = np.searchsorted(vi, np.arange(len(names)), "left")
    ends = np.searchsorted(vi, np.arange(len(names)), "right")
    rows = []
    for i, n in enumerate(names):
        im, mk = np.asarray(images[starts[i]:ends[i]]), np.asarray(masks[starts[i]:ends[i]])
        l, t = im[mk == 1], im[mk == 2]
        rows.append(dict(name=n, liver_mean=float(l.mean()) if l.size else np.nan, tumor_mean=float(t.mean()) if t.size else np.nan,
                         liver_px=int(l.size), tumor_px=int(t.size)))
    return pd.DataFrame(rows)


def t2_preproc():
    s1, s2 = cache_volume_stats(V1), cache_volume_stats(V2)
    d = s1.merge(s2, on="name", suffixes=("_v1", "_v2"))
    d["contrast_v1"] = d.tumor_mean_v1 - d.liver_mean_v1
    d["contrast_v2"] = d.tumor_mean_v2 - d.liver_mean_v2
    os.makedirs(TAB, exist_ok=True)
    pth = os.path.join(TAB, "T2_per_volume_cache_stats.csv")
    d.round(3).to_csv(pth, index=False)
    WROTE.append(pth)
    tb = d[(d.tumor_px_v1 > 0) & (d.tumor_px_v2 > 0)]
    rows = []
    for tag in ("v1", "v2"):
        T2_RES[tag] = float(tb[f"contrast_{tag}"].median())
    for tag, lab in (("v1", "v1: per-volume min-max, original orientation"), ("v2", "v2: fixed -100..250 HU window, canonical orientation")):
        c, lm = tb[f"contrast_{tag}"], d[f"liver_mean_{tag}"] / 255.0
        out = ((lm < 0.2) | (lm > 0.8))
        rows.append({"cache": lab, "tumor-bearing volumes": len(tb),
                     "median tumor-minus-liver contrast (uint8 levels)": f"{c.median():+.1f}",
                     "contrast IQR (q25 to q75)": f"{c.quantile(.25):+.1f} to {c.quantile(.75):+.1f}",
                     "median liver mean (0-1 grey)": f"{lm.median():.2f}",
                     "liver-mean IQR (0-1 grey)": f"{lm.quantile(.25):.2f}-{lm.quantile(.75):.2f}",
                     "volumes with liver mean outside 0.2-0.8": f"{int(out.sum())} of {len(d)} ({100 * out.mean():.0f} %)"})
    df = pd.DataFrame(rows)
    save_table("T2", "preprocessing_effect", df,
               "Tumor-minus-liver contrast in uint8 levels (liver = mask 1, tumor = mask 2, per-volume means) in the v1 vs v2 slice cache, over volumes with tumor pixels in both caches; "
               "the grey-range column covers all volumes. Raw HU contrast is similar by construction, v2 only changes the intensity mapping.")
    return d


# =========================================================================================== T3 baseline
def paired_boot(x, y, n_boot=10000, seed=12345):
    """Per-patient differences y - x over pairs where both are defined. -> dict."""
    m = ~(np.isnan(x) | np.isnan(y))
    d = (y - x)[m]
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx]
    med, mean = np.median(bs, axis=1), bs.mean(axis=1)
    return dict(n=len(d), med=float(np.median(d)), mean=float(d.mean()),
                med_ci=(float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))),
                mean_ci=(float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))),
                better=int((d > 0).sum()), worse=int((d < 0).sum()), tied=int((d == 0).sum()))


T2_RES = {}
PAIRED = {}  # (arm, ref, metric) -> result, used by index.md


def t3_baseline(arms):
    ok = done(arms)
    # ---- overall
    rows = []
    for k, a in arms.items():
        base = {"arm": f"{k}: {a['label']}"}
        if not a["ev"]:
            rows.append({**base, "status": f"pending: {a['status']}"})
            continue
        o = a["ev"]["aggregate"]["overall"]
        pl = o["pooled_lesion_recall"]
        eps = a["log"]["epochs"]
        vals = [(e["val"], e["epoch"]) for e in eps if e["val"] is not None]
        bv = min(vals) if vals else None
        ms = lambda d: f"{f3(d['mean'])} / {f3(d['median'])}"
        rows.append({**base, "status": "evaluated", "test patients (liver / tumor-bearing)": f"{o['liver_dice']['n']} / {o['tumor_dice']['n']}",
                     "liver Dice mean / median": ms(o["liver_dice"]), "tumor Dice mean / median": ms(o["tumor_dice"]),
                     "lesion recall, patient mean / median": ms(o["lesion_recall"]),
                     "lesions detected / total (pooled)": f"{pl['detected']} / {pl['lesions']} ({f3(pl['recall'])})",
                     "false-positive tumor mL, median (mean)": f"{f3(o['fp_tumor_ml']['median'], 2)} ({f3(o['fp_tumor_ml']['mean'], 1)})",
                     "best val loss (epoch)": f"{bv[0]:.4f} ({bv[1]})" if bv else "n/a",
                     "evaluated checkpoint epoch": str(a["ckpt_epoch"])})
    df1 = pd.DataFrame(rows).fillna("-")
    save_table("T3", "baseline_comparison", df1,
               "Test fold (26 patients) of the archived manifest-based baseline run (provenance: index.md and notebook section 5): 16 training volumes, 6 epochs, validation on 6 volumes, one seed; mean / median over patients; "
               "tumor Dice and lesion recall average only tumor-bearing patients; lesion recall counts a lesion as hit by >= 1 predicted tumor voxel.")
    # ---- by thickness
    rows = []
    for k, a in arms.items():
        if not a["ev"]:
            rows.append({"arm": f"{k}: {a['label']}", "thickness group": "all", "patients (placeholder spacing)": "pending"})
            continue
        agg, les = a["ev"]["aggregate"]["by_thickness"], pd.DataFrame(a["ev"]["lesions"])
        for g in GROUPS:
            if g not in agg:
                continue
            d = agg[g]
            ls = les[les.thickness_group == g]
            ms = lambda m: f"{f3(d[m]['mean'])} / {f3(d[m]['median'])}"
            rows.append({"arm": f"{k}: {a['label']}", "thickness group": g, "patients (placeholder spacing)": f"{d['patients']} ({d['placeholder_spacing']})",
                         "liver Dice mean / median": ms("liver_dice"), "tumor Dice mean / median": f"{ms('tumor_dice')} (n={d['tumor_dice']['n']})",
                         "lesions detected / total": f"{int(ls.detected.sum())} / {len(ls)} ({f3(d['pooled_lesion_recall'])})",
                         "false-positive tumor mL median": f3(d["fp_tumor_ml"]["median"], 2)})
    save_table("T3b", "baseline_by_thickness", pd.DataFrame(rows).fillna("-"),
               "Same test-fold results split by slice-thickness group; groups hold 17 / 6 / 3 patients, so the size of group differences is not reliable (only the direction is informative).")
    # ---- by lesion size
    rows = []
    for k, a in arms.items():
        if not a["ev"]:
            rows.append({"arm": f"{k}: {a['label']}", "lesion size": "all", "lesions (nominal-mm)": "pending"})
            continue
        for sname in SIZE_BINS:
            d = a["ev"]["aggregate"]["by_lesion_size"][sname]
            lo, hi = wilson(d["detected"], d["lesions"])
            rows.append({"arm": f"{k}: {a['label']}", "lesion size": sname, "lesions (nominal-mm)": f"{d['lesions']} ({d['nominal_mm']})",
                         "detected": str(d["detected"]), "recall": f3(d["recall"]), "Wilson 95 % interval (lesions treated as independent)": f"{f3(lo)}-{f3(hi)}"})
    save_table("T3c", "baseline_by_lesion_size", pd.DataFrame(rows).fillna("-"),
               "Pooled lesion recall by equivalent diameter; the interval treats lesions as independent although they cluster in patients (too narrow); sizes of placeholder-header volumes are nominal.")
    # ---- paired
    metrics = [("liver_dice", "liver Dice"), ("tumor_dice", "tumor Dice"), ("lesion_recall", "lesion recall"), ("fp_tumor_ml", "FP tumor mL")]
    rows, pending_pairs = [], []
    for k, ref in (("B", "A"), ("C", "A"), ("C", "B")):
        if k not in ok or ref not in ok:
            rows.append({"comparison": f"{k} minus {ref}", "metric": "-", "paired patients": "pending (needs both arms evaluated)"})
            continue
        pa, pb = arms[ref]["pp"], arms[k]["pp"]
        common = sorted(set(pa.index) & set(pb.index))
        for m, lab in metrics:
            r = paired_boot(pa.loc[common, m].values.astype(float), pb.loc[common, m].values.astype(float))
            PAIRED[(k, ref, m)] = r
            if r is None:
                continue
            rows.append({"comparison": f"{k} minus {ref}", "metric": lab, "paired patients": r["n"],
                         "median diff": f"{r['med']:+.3f}", "bootstrap 95 % CI of median": f"[{r['med_ci'][0]:+.3f}, {r['med_ci'][1]:+.3f}]",
                         "mean diff": f"{r['mean']:+.3f}", "bootstrap 95 % CI of mean": f"[{r['mean_ci'][0]:+.3f}, {r['mean_ci'][1]:+.3f}]",
                         "patients higher / lower / equal": f"{r['better']} / {r['worse']} / {r['tied']}"})
    dfp = pd.DataFrame(rows).fillna("-")
    save_table("T3d", "paired_comparison", dfp,
               "Per-patient difference (second arm minus first) on the same test patients, bootstrap over patients (10,000 resamples, seed 12345). "
               "CAVEAT: n = 26 test patients, one training seed per arm: the interval reflects patient sampling only, not training-seed variance; intervals are unadjusted for multiple comparisons; "
               "B vs A and C vs A change window and orientation together (only C vs B isolates flips); for FP mL a negative difference is better.")
    print("CAVEAT: n = 26 test patients and a single training seed per arm; paired intervals cover patient sampling only.")


# =========================================================================================== T4 GAN
def load_syn():
    os.environ["LITS_GAN_RESULTS_DIR"] = TMP
    spec = importlib.util.spec_from_file_location("syn_showcase", os.path.join(ROOT, "Image Synthesis", "Syn_Img_Pipeline_fixed.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t4_gan():
    import torch
    syn = load_syn()
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    mg = syn.MultiGAN(cpu)
    P, LD, B = syn.PATCH, syn.LATENT_DIM, 2
    role = {"dc1_g": "DCGAN generator 1", "dc1_d": "DCGAN discriminator 1", "dc2_g": "DCGAN generator 2", "dc2_d": "DCGAN discriminator 2",
            "w_g": "WGAN-GP generator", "w_c": "WGAN-GP critic", "aggr": "Aggregator (fuses the 3 generators)",
            "ag_d": "Aggregator discriminator", "mask": "Mask head (image -> tumor-mask logits)"}
    x = torch.rand(B, 1, P, P, P) * 2 - 1
    z = torch.randn(B, LD, 1, 1, 1)
    rows, tot = [], 0
    with torch.no_grad():
        for name, m in mg.modules().items():
            m.train(False)
            if name.endswith("_g"):
                inp, out = f"z ({B},{LD},1,1,1)", m(z)
            elif name == "aggr":
                inp, out = f"3 x ({B},1,{P},{P},{P})", m(x, x, x)
            else:
                inp, out = f"({B},1,{P},{P},{P})", m(x)
            n = sum(p.numel() for p in m.parameters())
            tot += n
            rows.append({"network": name, "class": type(m).__name__, "role": role[name], "parameters": f"{n:,}", "input": inp, "output shape": str(tuple(out.shape))})
    rows.append({"network": "total", "class": "MultiGAN", "role": "all 9 networks", "parameters": f"{tot:,}", "input": "-", "output shape": "-"})
    save_table("T4", "gan_networks", pd.DataFrame(rows),
               f"Networks of Image Synthesis/Syn_Img_Pipeline_fixed.py instantiated with the module defaults (patch {P}, latent {LD}, base width {syn.WIDTH}), "
               f"parameter counts and output shapes from one forward pass on random input (batch {B}, CPU).")
    # unit tests, really run
    py = os.path.join(ROOT, ".venv", "bin", "python")
    which = py if os.path.exists(py) else sys.executable
    if os.environ.get("SHOWCASE_SKIP_TESTS") == "1":
        res = {"suite": "test_syn_pipeline_fixed.py", "result": "skipped (SHOWCASE_SKIP_TESTS=1)"}
        save_table("T4b", "gan_unit_tests", pd.DataFrame([res]), "Unit tests skipped in this run.")
        return
    env = dict(os.environ, LITS_TEST_SKIP_E2E="1", LITS_GAN_RESULTS_DIR=TMP)
    p = subprocess.run([which, "-I", os.path.join(ROOT, "Image Synthesis", "test_syn_pipeline_fixed.py")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=1500)
    text = p.stdout + "\n" + p.stderr
    m = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
    ok = re.search(r"^OK(.*)$", text, re.M)
    fail = re.search(r"^FAILED \((.*)\)", text, re.M)
    res = {"suite": "Image Synthesis/test_syn_pipeline_fixed.py (end-to-end test skipped: LITS_TEST_SKIP_E2E=1)",
           "tests run": m.group(1) if m else "n/a",
           "result": ("OK" + (ok.group(1) if ok else "")) if (ok and p.returncode == 0) else ("FAILED " + (fail.group(1) if fail else f"exit {p.returncode}"))}
    # no timing / interpreter path columns: they vary per machine and run and would make the table non-reproducible
    save_table("T4b", "gan_unit_tests", pd.DataFrame([res]),
               "Unit-test result parsed from the 'Ran N tests ... OK' line of a real run of the test file; unit tests check shapes, ranges, losses, resampling and insertion logic, not sample realism.")
    print("unit tests:", res["result"], res["tests run"], "tests")


# =========================================================================================== figures
def fig_f1(pv, comps):
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8))
    # (a) thickness
    ax = axs[0, 0]
    for i, g in enumerate(GROUPS):
        s = pv[pv.thickness_group == g]
        rel, ph = (~s.placeholder).sum(), s.placeholder.sum()
        ax.bar(i, rel, color="#457b9d", label="header spacing" if i == 0 else None, width=0.6)
        ax.bar(i, ph, bottom=rel, color="#a8c4d9", hatch="//", edgecolor="white", label="placeholder 1 mm header (ids 28-47)" if i == 0 and ph else None, width=0.6)
        ax.text(i, len(s) + 1, f"{len(s)}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["thin\n(<= 1 mm)", "mid\n(1-2.5 mm)", "thick\n(> 2.5 mm)"])
    ax.set_xlabel("slice-thickness group")
    ax.set_ylabel("volumes (count)")
    ax.set_ylim(0, max(pv.thickness_group.value_counts()) * 1.18)
    ax.legend(loc="upper right")
    ax.set_title(f"(a) Slice thickness: {int((pv.thickness_group == 'thin').sum())} thin, {int((pv.thickness_group == 'mid').sum())} mid, {int((pv.thickness_group == 'thick').sum())} thick volumes")
    # (b) lesion size
    ax = axs[0, 1]
    bins = np.logspace(np.log10(comps.eq_diam_mm.min() * 0.95), np.log10(comps.eq_diam_mm.max() * 1.05), 28)
    ax.hist([comps[~comps.placeholder].eq_diam_mm, comps[comps.placeholder].eq_diam_mm], bins=bins, stacked=True,
            color=[TUMOR, "#f2a6a6"], label=["header spacing", "nominal (placeholder header)"])
    ax.set_xscale("log")
    for xv in (10, 20):
        ax.axvline(xv, color="k", lw=0.8, ls=":")
        ax.text(xv * 1.04, 0.62, f"{xv} mm", transform=ax.get_xaxis_transform(), rotation=90, fontsize=8, ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))
    ax.set_xlabel("lesion equivalent diameter (mm, log scale)")
    ax.set_ylabel("lesions (count)")
    ax.legend(loc="upper right")
    ax.set_title(f"(b) {len(comps)} lesions, median {comps.eq_diam_mm.median():.1f} mm; {100 * (comps.eq_diam_mm < 10).mean():.0f} % are < 10 mm")
    # (c) contrast
    ax = axs[1, 0]
    tb = pv[pv.tumor_ml > 0]
    ax.hist(tb.contrast_hu, bins=22, color="#8d99ae", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(tb.contrast_hu.median(), color=TUMOR, lw=1.8, label=f"median {tb.contrast_hu.median():+.1f} HU")
    ax.set_xlabel("mean tumor HU minus mean liver HU, per volume (HU)")
    ax.set_ylabel("tumor-bearing volumes (count)")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(f"(c) Tumor is {abs(tb.contrast_hu.median()):.0f} HU {'darker' if tb.contrast_hu.median() < 0 else 'brighter'} than liver (median)")
    # (d) class balance
    ax = axs[1, 1]
    vox = (pv.nx * pv.ny * pv.n_slices).sum()
    liver_v, tum_v = pv.liver_vox.sum() / vox * 100, pv.tumor_vox.sum() / vox * 100
    sl = pv.n_slices.sum()
    liver_s, tum_s = pv.slices_liver.sum() / sl * 100, pv.slices_tumor.sum() / sl * 100
    xs = np.arange(2)
    b1 = ax.bar(xs - 0.2, [liver_v, tum_v], 0.38, color=[LIVER, TUMOR], label="% of voxels")
    b2 = ax.bar(xs + 0.2, [liver_s, tum_s], 0.38, color=[LIVER, TUMOR], alpha=0.45, hatch="//", edgecolor="white", label="% of slices containing the class")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12, f"{b.get_height():.2f} %" if b.get_height() < 1 else f"{b.get_height():.1f} %", ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 300)
    ax.set_xticks(xs)
    ax.set_xticklabels(["liver", "tumor"])
    ax.set_xlabel("class")
    ax.set_ylabel("share (%, log scale)")
    h = [plt.Rectangle((0, 0), 1, 1, color="#555"), plt.Rectangle((0, 0), 1, 1, color="#555", alpha=0.45, hatch="//")]
    ax.legend(h, ["% of voxels", "% of slices with the class"], loc="upper right")
    ax.set_title(f"(d) Class balance: tumor is {tum_v:.2f} % of all voxels")
    fig.suptitle(f"LiTS training cohort ({len(pv)} volumes): thin-slice, many small low-contrast lesions, extreme class imbalance", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig("F1", "cohort_overview", fig,
             f"Cohort overview from analysis/output (per_volume.csv, stats.json); {int(pv.placeholder.sum())} volumes carry a placeholder 1 mm header so their lesion diameters and thickness group are nominal; "
             f"voxel shares use the full 512x512 grid including background.")


def pick_f2_volumes(pv, meta):
    tb = pv[pv.tumor_ml > 0].merge(meta[["name", "flipped_axes", "orientation_class"]], on="name")
    tb["id"] = tb.name.map(vnum)
    tb = tb.sort_values("id")
    mid = lambda d: d.iloc[len(d) // 2]
    chosen = []
    ph = tb[(tb.id >= 28) & (tb.id <= 47)]
    chosen.append((mid(ph), "volumes 28-47 (placeholder header)"))
    th = tb[(tb.thickness_group == "thick") & ~tb.name.isin([c[0]["name"] for c in chosen])]
    chosen.append((mid(th), "thick slices"))
    for fl in ("none", "ax0", "ax0+ax1"):
        if len(chosen) >= 4:
            break
        if any(c[0]["flipped_axes"] == fl for c in chosen):
            continue
        cand = tb[(tb.flipped_axes == fl) & ~tb.name.isin([c[0]["name"] for c in chosen]) & ~tb.placeholder]
        if len(cand):
            chosen.append((mid(cand), f"flip class '{fl}'"))
    if len(chosen) < 4:  # fill with the middle of the remaining reliable-header thin volumes
        cand = tb[~tb.name.isin([c[0]["name"] for c in chosen]) & ~tb.placeholder & (tb.thickness_group == "thin")]
        if len(cand):
            chosen.append((mid(cand), "thin slices, reliable header"))
    return chosen[:4]


class Cache:
    def __init__(self, path):
        self.path = path
        self.names = open(os.path.join(path, "volumes.txt")).read().split("\n")
        self.vi = np.load(os.path.join(path, "vol_index.npy"))
        self.images = np.load(os.path.join(path, "images.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(path, "masks.npy"), mmap_mode="r")

    def span(self, name):
        i = self.names.index(name)
        s, e = np.searchsorted(self.vi, i, "left"), np.searchsorted(self.vi, i, "right")
        return int(s), int(e)


def flip_for(arr, flipped_axes):
    for a in (0, 1):
        if f"ax{a}" in flipped_axes.split("+"):
            arr = np.flip(arr, a)
    return arr


def outline(ax, m, color, lw=1.0, ls="-"):
    m = np.asarray(m).astype(float)
    if m.any() and not m.all():
        ax.contour(m, levels=[0.5], colors=color, linewidths=lw, linestyles=ls)


def crop_box(liver2d, pad=14):
    ys, xs = np.nonzero(liver2d)
    if len(ys) == 0:
        return slice(0, 256), slice(0, 256)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, 256), max(xs.min() - pad, 0), min(xs.max() + pad + 1, 256)
    h, w = y1 - y0, x1 - x0
    s = max(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = int(np.clip(cy - s // 2, 0, max(256 - s, 0))), int(np.clip(cx - s // 2, 0, max(256 - s, 0)))
    return slice(y0, min(y0 + s, 256)), slice(x0, min(x0 + s, 256))


def fig_f2(pv):
    meta = pd.read_csv(os.path.join(V2, "meta.csv"))
    c1, c2 = Cache(V1), Cache(V2)
    chosen = pick_f2_volumes(pv, meta)
    fig = plt.figure(figsize=(12.5, 3.6 * len(chosen)))
    gs = fig.add_gridspec(len(chosen), 3, width_ratios=[1, 1, 1.25], wspace=0.12, hspace=0.32)
    for r, (row, why) in enumerate(chosen):
        n, fl = row["name"], row["flipped_axes"]
        s1, e1 = c1.span(n)
        s2, e2 = c2.span(n)
        m2 = np.asarray(c2.masks[s2:e2])
        z = int(np.argmax((m2 == 2).sum(axis=(1, 2))))
        i1, i2 = np.asarray(c1.images[s1 + z]), np.asarray(c2.images[s2 + z])
        k1, k2 = np.asarray(c1.masks[s1 + z]), m2[z]
        i1, k1 = flip_for(i1, fl), flip_for(k1, fl)  # v1 -> canonical orientation for display
        box = crop_box(k2 >= 1)
        for c, (img, mk, ttl) in enumerate([(i1, k1, "v1 min-max (flipped for display)"), (i2, k2, "v2 fixed window -100..250 HU")]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img[box], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
            sub = mk[box]
            outline(ax, sub >= 1, LIVER, 1.2)
            outline(ax, sub == 2, TUMOR, 1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(ttl if r == 0 else "", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{n}\nslice {z}, flips: {fl}\n{why}", fontsize=8.5, rotation=90, labelpad=6)
        ax = fig.add_subplot(gs[r, 2])
        for tag, cache, st, en, ls in (("v1", c1, s1, e1, "--"), ("v2", c2, s2, e2, "-")):
            im, mk = np.asarray(cache.images[st:en]), np.asarray(cache.masks[st:en])
            for cls, col, lab in ((1, LIVER, "liver"), (2, TUMOR, "tumor")):
                v = im[mk == cls]
                if v.size:
                    h, _ = np.histogram(v, bins=64, range=(0, 256), density=True)
                    ax.plot(np.arange(64) * 4 + 2, h, color=col, ls=ls, lw=1.6, label=f"{lab} {tag}")
        ax.set_xlim(0, 255)
        ax.set_xlabel("grey level (uint8)")
        ax.set_ylabel("density")
        if r == 0:
            ax.legend(ncol=1, loc="upper left", fontsize=8)
            ax.set_title("Grey levels inside liver / tumor masks, whole volume")
    t2 = T2_RES.get("v1"), T2_RES.get("v2")
    head = ("Preprocessing before/after: v1 min-max squeezes tumor-liver contrast to a few grey levels, v2 keeps it visible"
            + (f" (median {t2[0]:+.1f} vs {t2[1]:+.1f} levels, T2)" if t2[0] is not None else ""))
    fig.suptitle(head, x=0.01, y=0.93, ha="left", fontsize=11.5, fontweight="bold")
    save_fig("F2", "preprocessing_before_after", fig,
             "Same axial slice (largest tumor area) of 4 tumor-bearing volumes: one from ids 28-47, one thick-slice, plus flip classes not yet covered (rule: middle volume of each candidate list by id); "
             "liver teal, tumor red; v1 slice flipped with the v2 flip rule for display only; right: grey-level density inside masks over the whole volume.")


def beeswarm_x(v, centre, step=0.052, tol=0.02, max_half=0.36):
    """Deterministic horizontal offsets: values within `tol` of each other (e.g. all the exact zeros) are spread
    side by side around `centre`, so tied patients stay visible as separate dots."""
    order = np.argsort(v, kind="stable")
    x = np.full(len(v), float(centre))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] - v[order[i]] <= tol:
            j += 1
        n = j - i + 1
        offs = (np.arange(n) - (n - 1) / 2) * step
        if n > 1 and np.abs(offs).max() > max_half:  # too many ties for one row: shrink the step
            offs = offs * (max_half / np.abs(offs).max())
        x[order[i:j + 1]] = centre + offs
        i = j + 1
    return x


def fig_f3(arms):
    metrics = [("liver_dice", "liver Dice (3-D, per patient)", "liver Dice"), ("tumor_dice", "tumor Dice (3-D, tumor-bearing patients)", "tumor Dice"),
               ("lesion_recall", "lesion recall (fraction of GT lesions hit)", "lesion recall")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
    for ax, (m, ylab, ttl) in zip(axs, metrics):
        for i, (k, a) in enumerate(arms.items()):
            if a["pp"] is None:
                ax.text(i, 0.5, "pending", ha="center", va="center", color="#777", fontsize=10, style="italic")
                continue
            v = a["pp"][m].values.astype(float)
            v = v[~np.isnan(v)]
            ax.bar(i, v.mean(), color=a["color"], alpha=0.4, width=0.8)
            # clip_on=False: dots at exactly 0 sit on the axis line and must not be cut in half
            ax.scatter(beeswarm_x(v, i), v, s=17, color=a["color"], edgecolor="white", linewidth=0.5, zorder=3, clip_on=False)
            ax.hlines(np.median(v), i - 0.4, i + 0.4, color="k", lw=2, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xlim(-0.6, 2.6)
        labs = []
        for k, a in arms.items():
            if a["pp"] is None:
                labs.append(f"{k}: {a['label']}\npending")
            else:
                v = a["pp"][m].values.astype(float)
                v = v[~np.isnan(v)]
                labs.append(f"{k}: {a['label']}\nn={len(v)}, mean {v.mean():.2f}\nmedian {np.median(v):.2f}\nexactly 0: {int((v == 0).sum())} of {len(v)}")
        ax.set_xticklabels(labs, fontsize=7.8)
        ax.set_xlabel("baseline arm (test fold)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(ttl)
    axs[0].set_ylabel("score (0-1)")
    h = [plt.Rectangle((0, 0), 1, 1, color="#888", alpha=0.4), plt.Line2D([], [], color="k", lw=2), plt.Line2D([], [], marker="o", color="#888", ls="", ms=5)]
    fig.legend(h, ["mean (bar)", "median (black line)", "one patient (dot; tied values spread sideways)"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    ok = done(arms)
    parts = []
    for k in ok:
        o = arms[k]["ev"]["aggregate"]["overall"]
        t = arms[k]["pp"]["tumor_dice"].dropna()
        l = arms[k]["pp"]["liver_dice"].dropna()
        parts.append(f"Arm {k}: tumor Dice mean {o['tumor_dice']['mean']:.2f} ({int((t == 0).sum())} of {len(t)} patients exactly 0), "
                     f"liver Dice mean {o['liver_dice']['mean']:.2f} (SD {l.std(ddof=1):.2f}, {int((l < 0.3).sum())} of {len(l)} below 0.3)")
    n_pat = max([len(arms[k]["pp"]) for k in ok] or [0])
    if parts:
        head = f"Per-patient test-fold scores, {n_pat} patients, one seed per arm\n" + "\n".join(parts)
    else:
        head = "Per-patient test-fold scores: no arm evaluated yet"
    fig.text(0.01, 1.0, head, va="bottom", ha="left", fontsize=10.5, fontweight="bold")  # fig.text, not suptitle: tight_layout would reserve a large gap for a suptitle; the tight bbox keeps it in the PNG
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    pend = [k for k in arms if arms[k]["pp"] is None]
    save_fig("F3", "baseline_per_patient", fig,
             "Per-patient test-fold scores per arm (bar = mean, black line = median, dots = patients; patients with equal scores, mostly exact zeros, are spread sideways so each stays visible; "
             "'exactly 0' counts are under each bar); tiny single-seed baseline (16 training volumes, 6 epochs, one training run per arm). "
             "Paired intervals for the arm differences are in table T3d; A to B changes the window and the orientation handling together, only B to C isolates flips"
             + (f"; pending: {', '.join(pend)}" if pend else "") + ".")


def arm_tag(a):
    """' (3/6 epochs, training)' for an arm whose training log is not complete, else ''."""
    eps = a["log"]["epochs"]
    if a["log"]["complete"] or a["ev"] or not eps:
        return ""
    return f" ({len(eps)}/{a['log']['total']} epochs, training)"


def f4_title(arms):
    """Data-derived title from every arm with a complete training log: first-to-last-epoch losses, no adjectives beyond the numbers."""
    parts = []
    for k, a in arms.items():
        eps, ve = a["log"]["epochs"], [e for e in a["log"]["epochs"] if e["val"] is not None]
        if not (a["log"]["complete"] or a["ev"]) or len(ve) < 2:
            continue
        parts.append(f"{k}: train {eps[0]['train']:.2f} to {eps[-1]['train']:.2f}, validation {ve[0]['val']:.2f} to {ve[-1]['val']:.2f} "
                     f"(gap {ve[-1]['val'] - ve[-1]['train']:.2f} at epoch {ve[-1]['epoch']})")
    if not parts:
        return "Training curves (no arm has finished training yet)"
    return "Loss from first to last epoch, 16 training volumes / 6 validation volumes\n" + "\n".join("Arm " + x for x in parts)


def fig_f4(arms):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
    anyd = False
    for k, a in arms.items():
        eps = a["log"]["epochs"]
        if not eps:
            continue
        anyd = True
        ex = [e["epoch"] for e in eps]
        c, tag = a["color"], arm_tag(a)
        axs[0].plot(ex, [e["train"] for e in eps], color=c, ls="--", marker="o", ms=3, label=f"{k} train{tag}")
        ve = [e for e in eps if e["val"] is not None]
        axs[0].plot([e["epoch"] for e in ve], [e["val"] for e in ve], color=c, marker="o", ms=3, label=f"{k} val{tag}")
        dl = [e for e in eps if e["dice_liver"] is not None]
        dt = [e for e in eps if e["dice_tumor"] is not None]
        axs[1].plot([e["epoch"] for e in dl], [e["dice_liver"] for e in dl], color=c, marker="o", ms=3, label=f"{k}{tag}")
        axs[2].plot([e["epoch"] for e in dt], [e["dice_tumor"] for e in dt], color=c, marker="o", ms=3, label=f"{k}{tag}")
        se = a["ckpt_epoch"]
        if se is None and ve:
            se = min(ve, key=lambda e: e["val"])["epoch"]
        sel = next((e for e in eps if e["epoch"] == se), None)
        if a["ev"] and sel and sel["val"] is not None:
            axs[0].scatter([se], [sel["val"]], marker="*", s=170, color=c, edgecolor="k", zorder=5)
            axs[1].scatter([se], [sel["dice_liver"]], marker="*", s=170, color=c, edgecolor="k", zorder=5)
            axs[2].scatter([se], [sel["dice_tumor"]], marker="*", s=170, color=c, edgecolor="k", zorder=5)
    for k, a in arms.items():
        if not a["log"]["epochs"]:
            note_pending(f"F4 curves for arm {k}: no training log yet")
    axs[0].set_title("(a) Loss (CE + Dice)")
    axs[0].set_ylabel("loss (unitless)")
    axs[1].set_title("(b) Validation liver Dice (per-batch mean)")
    axs[1].set_ylabel("Dice (0-1)")
    axs[2].set_title("(c) Validation tumor Dice: inflated, see caption")
    axs[2].set_ylabel("Dice (0-1; empty-vs-empty batches count as 1)")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[0].set_ylim(top=axs[0].get_ylim()[1] + 0.32)  # head-room so the legend never covers a curve
    axs[0].legend(ncol=1, fontsize=7.8, loc="upper right")
    axs[1].legend(loc="lower right", fontsize=7.8)
    axs[2].legend(loc="lower right", fontsize=7.8)
    fig.text(0.01, -0.035, "Star = checkpoint used for the test-fold evaluation (lowest validation loss; evaluated arms only). Arms use different input scaling "
             "(v1 per-volume min-max vs v2 fixed window), so their losses and validation Dice are not directly comparable across arms." + ("" if anyd else " No logs yet."), fontsize=8.3)
    pend = [k for k, a in arms.items() if a["status"] != "evaluated"]
    fig.text(0.01, 1.0, f4_title(arms), va="bottom", ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.99))
    save_fig("F4", "training_curves", fig,
             "Per-epoch train/validation loss and validation Dice parsed from each arm's stdout.log (6 validation volumes, 6 epochs planned); the large train/validation loss gap on 16 volumes is as consistent with overfitting "
             "as with under-training. Losses from different input scalings (arm A vs B/C) are not directly comparable. The logged tumor Dice is a per-batch mean in which batches with no tumor "
             "in label and prediction score 1.0, so it is much higher than the test-fold tumor Dice; it is not comparable to table T3"
             + (f". Not finished (curves partial, marked in the legend): {', '.join(pend)}." if pend else "."))


def load_model_class():
    """AIM_UNet straight from train_fast.py's source, executing only its three model classes (no module side effects)."""
    import torch
    import torch.nn as nn
    src = open(os.path.join(ROOT, "Image Segmentation", "train_fast.py")).read()
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name in ("InceptionModule", "ConvBlock", "AIM_UNet")]
    ns = {"nn": nn, "torch": torch}
    exec(compile(ast.Module(keep, []), "train_fast.py", "exec"), ns)
    return ns["AIM_UNet"]


def dice2d(p, g):
    s = p.sum() + g.sum()
    return np.nan if s == 0 else 2 * (p & g).sum() / s


def fig_f5(arms, pv):
    import torch
    ok = [k for k in done(arms) if arms[k]["ckpt"]]
    if not ok:
        note_pending("F5: no evaluated arm")
        return
    # eval.json stores a scrubbed path; the checkpoints themselves live (git-ignored) in results/baseline/<arm>/
    ok_ck = [k for k in ok if os.path.exists(os.path.join(CKPT_BASE, ARMS[k][0], os.path.basename(arms[k]["ckpt"])))]
    if len(ok_ck) < len(ok):
        print("F5 skipped: checkpoints not found in results/baseline/ (not shipped); the shipped figure is kept")
        return
    ref = "A" if "A" in ok else ok[0]
    pp = arms[ref]["pp"].reset_index().merge(pv[["name", "tumor_ml"]], on="name")
    tb = pp[pp.n_lesions > 0].sort_values(["tumor_ml", "name"], ascending=[False, True])
    det = tb[tb.lesions_detected > 0]
    top2 = list(det.name[:2])
    rest = tb[~tb.name.isin(top2)]
    miss = rest[rest.lesions_detected == 0]
    sel = list(top2)
    ordn = ["largest", "second largest"]
    why = {n: f"{ordn[i]} GT tumor volume among test patients with at least one lesion hit by arm {ref}" for i, n in enumerate(top2)}
    if len(miss):
        n = miss.name.iloc[0]
        sel.append(n)
        why[n] = f"largest tumor volume with no lesion hit by arm {ref}"
    rest = rest[~rest.name.isin(sel)].sort_values(["tumor_ml", "name"])
    if len(rest):
        n = rest.name.iloc[len(rest) // 2]
        sel.append(n)
        why[n] = "median tumor volume of the remaining tumor-bearing patients"
    meta = pd.read_csv(os.path.join(V2, "meta.csv")).set_index("name")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    amp = dev.type != "cpu"
    Net = load_model_class()
    nets = {}
    for k in ok:
        net = Net().to(dev)
        ck = os.path.join(CKPT_BASE, ARMS[k][0], os.path.basename(arms[k]["ckpt"]))
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        net.train(False)
        nets[k] = net
    c2 = Cache(V2)
    caches = {k: (Cache(arms[k]["cache"]) if arms[k]["cache"] != V2 else c2) for k in ok}
    cols = 2 + len(ok)
    weak = []  # (patient, arm, slice liver Dice) where the displayed slice's liver Dice is below 0.5
    fig, axs = plt.subplots(len(sel), cols, figsize=(2.55 * cols + 0.9, 2.65 * len(sel)), squeeze=False)
    for r, n in enumerate(sel):
        fl = meta.loc[n, "flipped_axes"]
        s2, e2 = c2.span(n)
        m2 = np.asarray(c2.masks[s2:e2])
        z = int(np.argmax((m2 == 2).sum(axis=(1, 2))))
        gt, ct = m2[z], np.asarray(c2.images[s2 + z])
        box = crop_box(gt >= 1)
        panels = [("CT (v2 window)", ct, None), ("ground truth", None, gt)]
        for k in ok:
            ca = caches[k]
            s, _ = ca.span(n)
            x = torch.from_numpy(np.array(ca.images[s + z])).to(dev).float()[None, None] / 255.0
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                pr = nets[k](x)
            pred = pr.float().argmax(1)[0].to(torch.uint8).cpu().numpy()
            if arms[k]["cache"] != V2:
                pred = flip_for(pred, fl)
            panels.append((f"{k}: {arms[k]['label']}", None, pred))
        for c, (ttl, img, mk) in enumerate(panels):
            ax = axs[r][c]
            ax.imshow(ct[box], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
            if mk is not None:
                sub = mk[box]
                outline(ax, sub >= 1, LIVER, 1.1)
                outline(ax, sub == 2, TUMOR, 1.3)
                if c >= 2:
                    dt, dl = dice2d(mk == 2, gt == 2), dice2d(mk >= 1, gt >= 1)
                    if dl == dl and dl < 0.5:
                        weak.append((n.replace(".nii", ""), panels[c][0].split(":")[0], dl))
                    ax.text(0.03, 0.03, f"slice Dice liver {f3(dl, 2)} tumor {f3(dt, 2)}", transform=ax.transAxes, color="white", fontsize=7,
                            bbox=dict(facecolor="black", alpha=0.6, pad=1.5, lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(ttl, fontsize=8.5, loc="center")
            if c == 0:
                ax.set_ylabel(f"{n.replace('.nii', '')}\nz={z}, GT tumor {tb.set_index('name').loc[n, 'tumor_ml']:.0f} mL", fontsize=8)
    pend = [k for k in arms if k not in ok]
    h = [plt.Line2D([], [], color=LIVER, lw=2), plt.Line2D([], [], color=TUMOR, lw=2)]
    fig.legend(h, ["liver (label >= 1)", "tumor"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Prediction overlays on the largest GT-tumor slice of held-out patients" + (f" (pending: {', '.join(pend)})" if pend else ""), x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save_fig("F5", "prediction_overlays", fig,
             "Axial slice with the largest GT tumor area for 4 test-fold patients, canonical orientation, each arm run with its own cache and checkpoint (bf16 autocast). Selection rule: " +
             "; ".join(f"{n.replace('.nii', '')} = {why[n]}" for n in sel) +
             f". Selection used arm {ref}'s per-patient outcome, so this is not a comparison between arms and not a random sample. Slice-level Dice is for the displayed slice only; the panel is illustrative, not representative."
             + (" Displayed-slice liver Dice below 0.5 (visible failures, not hidden): " + ", ".join(f"{p} arm {a} ({d:.2f})" for p, a, d in weak) + "." if weak else "")
             + (f" Pending arms omitted: {', '.join(pend)}." if pend else ""))


def fig_f6(arms):
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ok = done(arms)
    n_ok = max(len(ok), 1)
    w = min(0.5, 0.8 / n_ok)
    pend_i = 0
    for k, a in arms.items():
        if not a["ev"]:
            ax.text(0.98, 0.93 - 0.07 * pend_i, f"{k}: pending", color=a["color"], fontsize=9, style="italic", ha="right", transform=ax.transAxes)
            pend_i += 1
            continue
        x0 = np.arange(3) + (ok.index(k) - (n_ok - 1) / 2) * w  # bars centred on the tick when one arm is shown
        for j, s in enumerate(SIZE_BINS):
            d = a["ev"]["aggregate"]["by_lesion_size"][s]
            lo, hi = wilson(d["detected"], d["lesions"])
            ax.bar(x0[j], d["recall"], w * 0.92, color=a["color"], alpha=0.85, label=f"{k}: {a['label']}" if j == 0 else None)
            ax.errorbar(x0[j], d["recall"], yerr=[[d["recall"] - lo], [hi - d["recall"]]], color="k", lw=1, capsize=2)
            ax.text(x0[j] + w * 0.05, hi + 0.015, f"{d['detected']}/{d['lesions']}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xlim(-0.6, 2.6)
    any_k = ok[0] if ok else None
    labs = []
    for s in SIZE_BINS:
        n = arms[any_k]["ev"]["aggregate"]["by_lesion_size"][s]["lesions"] if any_k else 0
        labs.append(f"{s}\n({n} lesions)")
    ax.set_xticklabels(labs)
    ax.set_xlabel("lesion equivalent diameter (approximate, mm)")
    ax.set_ylabel("lesion recall (detected / all GT lesions)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    # title: only the first-vs-last comparison, and only when the Wilson intervals of the two bins do not overlap for every finished arm
    big = {}
    for k in ok:
        bl = arms[k]["ev"]["aggregate"]["by_lesion_size"]
        small, large = bl[SIZE_BINS[0]], bl[SIZE_BINS[-1]]
        big[k] = (small["recall"], large["recall"], wilson(small["detected"], small["lesions"])[1] < wilson(large["detected"], large["lesions"])[0])
    n_pat = max([len(arms[k]["pp"]) for k in ok] or [0])
    if big and all(v[2] for v in big.values()):
        ttl = (f"Lesion recall is higher for {SIZE_BINS[-1]} than for {SIZE_BINS[0]} lesions\n(" + "; ".join(f"arm {k}: {v[1]:.2f} vs {v[0]:.2f}" for k, v in big.items())
               + f")\npooled over {n_pat} test patients; labels = detected/total")
    else:
        ttl = f"Lesion recall by lesion size\npooled over {n_pat} test patients; labels = detected/total"
    ax.set_title(ttl, fontsize=9)
    fig.tight_layout()
    save_fig("F6", "recall_by_lesion_size", fig,
             "Pooled lesion recall by size with Wilson 95 % intervals that treat lesions as independent (they cluster within patients, so the true uncertainty is larger); "
             "only the first-versus-last bin comparison is stated in the title, the middle bin is not claimed to differ; "
             "a lesion counts as detected if any predicted tumor voxel overlaps it; diameters are approximate and nominal for placeholder-header volumes"
             + (f"; pending: {', '.join(k for k in arms if not arms[k]['ev'])}." if any(not a['ev'] for a in arms.values()) else "."))


def gan_note():
    """-> dict with mask-head train / held-out Dice, held-out volume ids, epochs, and whether the run manifest is outside the repo."""
    log = os.path.join(GAN, "pipeline.log")
    r = dict(held=None, train=None, held_vols=None, tmp_manifest=False)
    for line in open(log, errors="replace"):
        m = re.search(r"mask head heldout: dice=([\d.]+)", line)
        if m:
            r["held"] = float(m.group(1))
        m = re.search(r"mask head train: dice=([\d.]+)", line)
        if m:
            r["train"] = float(m.group(1))
        m = re.search(r"held-out volumes \[([\d, ]+)\]", line)
        if m:
            r["held_vols"] = [int(x) for x in m.group(1).split(",")]
        m = re.search(r"manifest=(\S+)", line)
        if m and not os.path.abspath(m.group(1)).startswith(ROOT + os.sep) and not m.group(1).startswith(("analysis/", "results/")):
            r["tmp_manifest"] = True
    r["epochs"] = max([int(m.group(1)) for f in os.listdir(os.path.join(GAN, "figures")) for m in [re.match(r"train_epoch(\d+)", f)] if m] or [0])
    return r


def fig_f7a():
    pz = np.load(os.path.join(GAN, "patches.npz"))
    sm = np.load(os.path.join(GAN, "synthetic", "samples.npz"))
    imgs, msk, meta = pz["images"], pz["masks"], pz["meta"]
    idx = np.linspace(0, len(imgs) - 1, 6).round().astype(int)
    gp, gm = sm["patches"], sm["masks"]
    ncol = max(len(idx), len(gp))
    fig, axs = plt.subplots(2, ncol, figsize=(1.95 * ncol + 0.8, 4.6), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    cases = [int(meta[i][0]) for i in idx]
    seen = {}
    for j, i in enumerate(idx):
        ax = axs[0][j]
        # slice with the largest mask area (mid-slice often misses the tumor); two patches of one case get a / b
        zc = int(np.argmax(msk[i].sum(axis=(0, 1)))) if msk[i].any() else imgs.shape[-1] // 2
        ax.imshow(imgs[i][:, :, zc].astype(np.float32).T, cmap="gray", vmin=-1, vmax=1, origin="lower")
        outline(ax, msk[i][:, :, zc].T, TUMOR, 1.0)
        seen[cases[j]] = seen.get(cases[j], 0) + 1
        tag = "" if cases.count(cases[j]) == 1 else "ab"[seen[cases[j]] - 1]
        ax.set_title(f"case {cases[j]}{tag}, z={zc}", fontsize=8, loc="center")
    for j in range(len(gp)):
        ax = axs[1][j]
        z = int(np.argmax(gm[j].sum(axis=(0, 1)))) if gm[j].any() else gp.shape[-1] // 2
        ax.imshow(gp[j][:, :, z].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
        outline(ax, gm[j][:, :, z].T, TUMOR, 1.0)
        ax.set_title(f"sample {j + 1}, z={z}", fontsize=8, loc="center")
    axs[0][0].text(-0.08, 0.5, f"real tumor patches\n(slice with largest mask area,\n{len(idx)} of {len(imgs)} patches)", transform=axs[0][0].transAxes, rotation=90, ha="right", va="center", fontsize=8.5)
    axs[1][0].text(-0.08, 0.5, f"generated\n({len(gp)} accepted samples)", transform=axs[1][0].transAxes, rotation=90, ha="right", va="center", fontsize=8.5)
    gn = gan_note()
    held, epochs = gn["held"], gn["epochs"]
    fig.suptitle("Generated patches are blurry; realism not verified", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0, 1, 0.93), h_pad=2.5)
    save_fig("F7a", "gan_real_vs_generated", fig,
             f"Real 64^3 tumor patches (1 mm voxels, window -100..250 HU mapped to [-1,1], every ~{len(imgs) // 6}th of {len(imgs)} patches from {len(set(meta[:, 0].tolist()))} cases) next to the {len(gp)} samples the "
             f"gan_long2 run accepted ({epochs} epochs, 8 training volumes); red = tumor mask (real: ground truth, shown on the slice with the largest mask area of each patch, a / b = two patches of one case; generated: mask head, no outline = tumor not on that slice). Generated samples are blurry and realism was not verified"
             + (f". The red outlines on generated patches come from a mask head that does not generalise: Dice {gn['train']:.3f} on train patches vs {held:.3f} on held-out volumes "
                f"({'2' if gn['held_vols'] is None else len(gn['held_vols'])} volumes), so those masks are unreliable."
                if held is not None and gn["train"] is not None else ".")
             + (" The run used an earlier version of the GAN script that read a temporary 5-fold split manifest outside the repo; the shipped script takes the train part of the original split, "
                "so this run cannot be reproduced from the repo as shipped." if gn["tmp_manifest"] else ""))


def fig_f7b():
    import nibabel as nib
    ids = sorted(int(re.search(r"aug_volume-(\d+)", f).group(1)) for f in os.listdir(os.path.join(GAN, "aug")) if f.startswith("aug_volume-"))
    info = []
    for i in ids:
        try:
            hs = np.asanyarray(nib.load(os.path.join(RAW, f"segmentation-{i}.nii")).dataobj)
            ag = np.asanyarray(nib.load(os.path.join(GAN, "aug", f"aug_segmentation-{i}.nii")).dataobj)
        except Exception as e:
            note_pending(f"F7b: case {i} unreadable ({e})")
            continue
        if hs.shape != ag.shape:
            note_pending(f"F7b: case {i} shape mismatch host {hs.shape} vs aug {ag.shape}")
            continue
        new = (ag == 2) & (hs != 2)
        info.append((i, int(new.sum()), new))
    if not info:
        note_pending("F7b: no usable insertion case")
        return
    counts = {i: c for i, c, _ in info}
    chosen = sorted(info, key=lambda t: (-t[1], t[0]))[:2]
    fig, axs = plt.subplots(len(chosen), 3, figsize=(10.2, 3.5 * len(chosen)), squeeze=False)
    drew_existing = False
    for r, (i, cnt, new) in enumerate(chosen):
        hv = nib.load(os.path.join(RAW, f"volume-{i}.nii"))
        host = np.asanyarray(hv.dataobj).astype(np.float32)
        aug = np.asanyarray(nib.load(os.path.join(GAN, "aug", f"aug_volume-{i}.nii")).dataobj).astype(np.float32)
        hseg = np.asanyarray(nib.load(os.path.join(RAW, f"segmentation-{i}.nii")).dataobj)
        z = int(np.argmax(new.sum(axis=(0, 1))))
        xs, ys = np.nonzero(new[:, :, z])
        cx, cy = int(xs.mean()), int(ys.mean())
        hw = 55
        sx = slice(max(cx - hw, 0), min(cx + hw, host.shape[0]))
        sy = slice(max(cy - hw, 0), min(cy + hw, host.shape[1]))
        for c, (img, ttl) in enumerate([(host, "host (before)"), (aug, "augmented (after)"), (aug, "augmented + new label outlined")]):
            ax = axs[r][c]
            ax.imshow(img[sx, sy, z].T, cmap="gray", vmin=-100, vmax=250, origin="lower")
            if c == 2:
                ex = (hseg[sx, sy, z] == 2)
                drew_existing = drew_existing or bool(ex.any() and not ex.all())
                outline(ax, ex.T, TUMOR, 1.0)
                outline(ax, new[sx, sy, z].T, "#ffd60a", 1.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"case {i}, slice {z}: {ttl}", fontsize=8.5, loc="center")
    fig.suptitle("Inserted synthetic tumors: placement and blending check only; realism not verified", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=2.5)
    h, labs = [plt.Line2D([], [], color="#ffd60a", lw=2)], ["new (inserted) tumor label"]
    if drew_existing:  # only list the red outline when one is actually drawn in the shown crops
        h.append(plt.Line2D([], [], color=TUMOR, lw=2))
        labs.append("existing tumor label")
    fig.legend(h, labs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save_fig("F7b", "gan_insertions", fig,
             f"Host before vs after insertion at the slice with the largest new-tumor area (window -100..250 HU, 110 px crop); the 2 cases with the most inserted voxels are shown "
             f"({', '.join(f'case {i}: {c}' for i, c, _ in chosen)}); inserted voxels per host across the {len(info)} cases: min {min(counts.values())}, max {max(counts.values())}. "
             f"Visual placement/blending check only; no segmenter was trained on these volumes and realism is not verified.")


# =========================================================================================== index.md
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"))
METS = (("liver_dice", "liver Dice"), ("tumor_dice", "tumor Dice"), ("lesion_recall", "lesion recall"))


def pair_summary(k, ref):
    """-> (metrics whose mean-difference interval is entirely above 0, entirely below 0, includes 0), as labels."""
    up, down, inc = [], [], []
    for m, lab in METS:
        r = PAIRED.get((k, ref, m))
        if r:
            lo, hi = r["mean_ci"]
            (up if lo > 0 else down if hi < 0 else inc).append(lab)
    return up, down, inc


def verdict_lines(arms):
    """Computed from the evaluated arms only: no outcome is hard-coded, so the text follows the data."""
    ok = done(arms)
    L = []
    for k in ok:
        o = arms[k]["ev"]["aggregate"]["overall"]
        pl = o["pooled_lesion_recall"]
        lv = arms[k]["pp"]["liver_dice"].dropna()
        L.append(f"- Arm {k} ({arms[k]['label']}): liver Dice mean {f3(o['liver_dice']['mean'])} (SD over patients {lv.std(ddof=1):.3f}) / median {f3(o['liver_dice']['median'])}, "
                 f"{int((lv < 0.3).sum())} of {len(lv)} patients below 0.3; "
                 f"tumor Dice mean {f3(o['tumor_dice']['mean'])} / median {f3(o['tumor_dice']['median'])} over {o['tumor_dice']['n']} tumor-bearing patients; "
                 f"{pl['detected']} of {pl['lesions']} lesions hit ({f3(pl['recall'])}); false-positive tumor volume median {f3(o['fp_tumor_ml']['median'], 2)} mL.")
    pairs = [(k, ref) for k, ref in PAIRS if k in ok and ref in ok]
    if pairs:
        parts = []
        for k, ref in pairs:
            up, down, inc = pair_summary(k, ref)
            seg = []
            if up:
                seg.append("higher, interval excludes 0, on " + ", ".join(up))
            if down:
                seg.append("lower, interval excludes 0, on " + ", ".join(down))
            if inc:
                seg.append("interval includes 0 (not distinguishable here) on " + ", ".join(inc))
            parts.append(f"{k} vs {ref}: " + "; ".join(seg))
        L.append("- **Paired comparisons in one line** (mean per-patient difference, 95 % bootstrap interval over patients): " + " | ".join(parts) + ".")
    for k, ref in PAIRS:
        if k in ok and ref in ok:
            for m, lab in METS:
                r = PAIRED.get((k, ref, m))
                if not r:
                    continue
                lo, hi = r["mean_ci"]
                if lo > 0 or hi < 0:
                    s = f"mean paired difference {r['mean']:+.3f} (95 % bootstrap CI {lo:+.3f} to {hi:+.3f}) excludes 0 in this single-seed, {r['n']}-patient experiment"
                else:
                    s = (f"mean paired difference {r['mean']:+.3f} (95 % bootstrap CI {lo:+.3f} to {hi:+.3f}) includes 0: this experiment cannot distinguish the arms "
                         f"(underpowered: {r['n']} patients, one seed); that is not evidence that they are equal")
                L.append(f"- {k} vs {ref}, {lab}: {s}.")
        else:
            L.append(f"- {k} vs {ref}: pending.")
    # direction per thickness group (counts only; the groups hold 17 / 6 / 3 patients)
    if "A" in ok:
        bt = {k: arms[k]["ev"]["aggregate"]["by_thickness"] for k in ok}
        gs = [g for g in GROUPS if g in bt["A"]]
        sizes = " / ".join(str(bt["A"][g]["patients"]) for g in gs)
        segs = []
        for k in [x for x in ok if x != "A"]:
            cnt = []
            for lab, get in (("liver Dice mean", lambda d: d["liver_dice"]["mean"]), ("tumor Dice mean", lambda d: d["tumor_dice"]["mean"]),
                             ("pooled lesion recall", lambda d: d["pooled_lesion_recall"])):
                n = sum(1 for g in gs if g in bt[k] and get(bt[k][g]) > get(bt["A"][g]))
                cnt.append(f"{lab} {n}/{len(gs)}")
            segs.append(f"arm {k}: " + ", ".join(cnt))
        if segs:
            L.append(f"- Thickness groups in which an arm is above arm A (T3b; groups of {sizes} patients, so only the direction is informative): " + "; ".join(segs) + ".")
    # lesion size
    if ok:
        bl = {k: arms[k]["ev"]["aggregate"]["by_lesion_size"] for k in ok}
        L.append("- Lesions hit by size (T3c, detected / total): " + "; ".join(
            f"{s}: " + ", ".join(f"{k} {bl[k][s]['detected']}/{bl[k][s]['lesions']}" for k in ok) for s in SIZE_BINS)
            + (f". Recall for {SIZE_BINS[0]} lesions is below 0.5 in every arm (best {max(bl[k][SIZE_BINS[0]]['recall'] for k in ok):.2f}), so small lesions stay mostly missed."
               if all(bl[k][SIZE_BINS[0]]["recall"] < 0.5 for k in ok) else "."))
        best = max(ok, key=lambda k: arms[k]["ev"]["aggregate"]["overall"]["tumor_dice"]["mean"])
        t = arms[best]["pp"]["tumor_dice"].dropna()
        o = arms[best]["ev"]["aggregate"]["overall"]
        L.append(f"- Absolute level: the arm with the highest mean tumor Dice (arm {best}) reaches {o['tumor_dice']['mean']:.3f} mean / {o['tumor_dice']['median']:.3f} median, "
                 f"with {int((t >= 0.5).sum())} of {len(t)} tumor-bearing patients at 0.5 or above and {int((t == 0).sum())} at exactly 0: tumor Dice is still low in absolute terms.")
    if len(ok) >= 2:
        fps = ", ".join(f"{k} {arms[k]['ev']['aggregate']['overall']['fp_tumor_ml']['median']:.2f} mL" for k in ok)
        td = {k: arms[k]["ev"]["aggregate"]["overall"]["tumor_dice"]["mean"] for k in ok}
        fm = {k: arms[k]["ev"]["aggregate"]["overall"]["fp_tumor_ml"]["median"] for k in ok}
        follows = sorted(ok, key=lambda k: -td[k]) == sorted(ok, key=lambda k: fm[k])
        L.append(f"- False-positive tumor volume (median per patient): {fps}. It " + ("falls as tumor Dice rises across the arms" if follows else "does not follow the tumor Dice ordering of the arms")
                 + " (FP rows in T3d); read it together with lesion recall, because an arm that predicts little tumor also produces little false-positive volume.")
    if any(k in ok and ref in ok for k, ref in PAIRS):
        eps = ", ".join(f"{k} epoch {arms[k]['ckpt_epoch']}" for k in ok if arms[k]["ckpt_epoch"])
        L += ["- Multiple comparisons: up to 9 unadjusted 95 % intervals (3 pairs x 3 metrics) are reported; with that many, an interval excluding 0 is expected by chance now and then.",
              "- What each pair isolates: B vs A and C vs A change the intensity window and the orientation handling together, so a difference cannot be assigned to either change; "
              "only C vs B isolates the flip augmentation.",
              f"- Other limits: one training run per arm (seed 1234), intervals over the test patients only (no seed variance), 16 training volumes and 6 epochs; the epoch evaluated is chosen per arm "
              f"on only 6 validation volumes ({eps}), which adds noise to every comparison."]
    return L


def write_index(arms):
    st = "\n".join(f"| {k} | {a['label']} | {a['status']} |" for k, a in arms.items())
    ok = done(arms)
    tum = ", ".join(f"{k} {arms[k]['ev']['aggregate']['overall']['tumor_dice']['mean']:.3f}" for k in ok)
    rerun = os.path.isdir(os.path.join(ROOT, "results", "baseline_original_split"))
    md = ["# Showcase: liver tumor segmentation on LiTS", "",
          f"Generated by `analysis/make_showcase.py` on {time.strftime('%Y-%m-%d %H:%M')} from files in this repository. Every number comes from a file; captions are in `captions.json`. "
          "Re-run the script after results change.", "",
          "## Read this first", "",
          "- The baseline is tiny: 16 training volumes, 6 epochs, one seed per arm, validation on 6 volumes, evaluation on 26 test patients. "
          f"Tumor Dice is still low in absolute terms (mean over tumor-bearing patients: {tum}).",
          "- Differences between arms are single-seed and patient-level only; they are not evidence about the pipeline in general. The step from A to B changes the intensity window and the orientation "
          "handling together; only B to C isolates one change (flip augmentation).",
          "- GAN samples after the short runs are blurry and not realistic; no segmenter was trained on them. The GAN mask head scores far lower on held-out volumes than on train patches (see F7a), "
          "and that run used an earlier version of the GAN script with a temporary split manifest, so it is not reproducible from the repo as shipped.",
          "- **Provenance of the baseline numbers.** " + PROVENANCE + (" Its results are in `results/baseline_original_split/` and are not shown here." if rerun else " It has not been run, so no results of the final code are shown here."),
          "", "### Baseline arms", "", "| arm | recipe | status |", "|---|---|---|", st, "",
          "### What the data support", ""] + verdict_lines(arms) + ["", "---", ""]
    order = [("T1", "Table T1. Cohort summary"), ("T2", "Table T2. Preprocessing effect (v1 vs v2 cache)"), ("T3", "Table T3. Baseline comparison (test fold)"),
             ("T3b", "Table T3b. Baseline by thickness group"), ("T3c", "Table T3c. Baseline by lesion size"), ("T3d", "Table T3d. Paired comparison"),
             ("T4", "Table T4. GAN pipeline networks"), ("T4b", "Table T4b. GAN unit tests")]
    figs = [("F1", "Figure F1. Cohort overview"), ("F2", "Figure F2. Preprocessing before/after"), ("F3", "Figure F3. Baseline per-patient results"),
            ("F4", "Figure F4. Training curves"), ("F5", "Figure F5. Prediction overlays"), ("F6", "Figure F6. Recall by lesion size"),
            ("F7a", "Figure F7a. GAN real vs generated patches"), ("F7b", "Figure F7b. GAN insertion examples")]
    md.append("## Tables\n")
    for k, t in order:
        if k in TABLE_MD:
            m, csvp, mdp = TABLE_MD[k]
            md += [f"### {t}", "", m, "", f"*{CAPTIONS[k]}*  ([csv]({csvp}), [md]({mdp}))", ""]
    md.append("## Figures\n")
    for k, t in figs:
        if k in TABLE_MD:
            md += [f"### {t}", "", f"![{t}]({TABLE_MD[k][1]})", "", f"*{CAPTIONS[k]}*", ""]
    md += ["## Pending / not available", ""] + ([f"- {p}" for p in PENDING] or ["- nothing pending"]) + [""]
    open(os.path.join(OUT, "index.md"), "w").write("\n".join(md))
    WROTE.append(os.path.join(OUT, "index.md"))


# =========================================================================================== main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    pv = load_pv()
    arms = load_arms()
    steps = [
        ("T1", lambda: t1_cohort(pv)),
        ("T2", t2_preproc),
        ("T3", lambda: t3_baseline(arms)),
        ("T4", t4_gan),
        ("F4", lambda: fig_f4(arms)),
        ("F3", lambda: fig_f3(arms)),
        ("F6", lambda: fig_f6(arms)),
        ("F2", lambda: fig_f2(pv)),
        ("F5", lambda: fig_f5(arms, pv)),
        ("F7a", fig_f7a),
        ("F7b", fig_f7b),
    ]
    comps = None
    for key, fn in steps:
        if not want(key) and not (key == "T1" and want("F1")):
            continue
        t = time.time()
        res = fn()
        if key == "T1":
            comps = res
        print(f"[{key}] done in {time.time() - t:.0f}s")
    if want("F1"):
        if comps is None:
            comps = t1_cohort(pv)
        fig_f1(pv, comps)
    # captions.json / index.md: merge with previous run when only a subset was rebuilt
    cap_path = os.path.join(OUT, "captions.json")
    if ONLY and os.path.exists(cap_path):
        old = json.load(open(cap_path))
        old.update(CAPTIONS)
        CAPTIONS.update(old)
    json.dump(CAPTIONS, open(cap_path, "w"), indent=1, sort_keys=True)
    WROTE.append(cap_path)
    if not ONLY:
        write_index(arms)
    print("\nWrote:")
    for p in WROTE:
        print("  " + os.path.relpath(p, ROOT))
    print("\nArm status:")
    for k, a in arms.items():
        print(f"  {k} {ARMS[k][0]}: {a['status']}")
    print("Pending:", "; ".join(PENDING) if PENDING else "none")
    print(f"Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
