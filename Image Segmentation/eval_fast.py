"""Per-patient evaluation of a train_fast.py checkpoint (best.pt, or an old raw state_dict .pt).

Which volumes (LITS_EVAL_PART):
  test    = the 'test' list of split_used.json next to the checkpoint (train_fast.py with LITS_VAL_N > 0: the held-out
            volumes that were NOT the validation subset; never used for any selection). Default when that list is non-empty.
  val     = the 'val' list of that file (the volumes that picked the epoch / recipe: validation-selected numbers). Without a
            split_used.json with a test list it is the whole held-out part, as in the earlier version of this script.
  heldout = the whole held-out part of the original split (sklearn train_test_split over the sorted volume names,
            test_size=0.2, random_state=42 -> 27 volumes, each one patient), whatever split_used.json says.

Per patient (all slices of the volume, 3D over the 256x256 cache):
  * liver Dice (label >= 1 union, the LiTS convention) and tumor Dice (label 2). Dice is NaN (left out of the
    means) when prediction and ground truth are both empty; tumor Dice is also NaN for patients without GT
    tumor (any false positive there is reported as FP volume instead), so it averages tumor-bearing patients.
  * lesion-wise recall: ground-truth tumor components (6-connected in 3D, scipy default) in the cache mask;
    a lesion counts as detected if >= 1 of its voxels is predicted tumor (any overlap, no Dice/IoU threshold).
  * false-positive tumor volume in mL: predicted-tumor voxels outside the GT tumor, and the number of
    predicted components that touch no GT tumor.
  * 'legacy' per-slice Dice, computed like the original test script (test_aimunet_orig_65.py): per slice and class
    (background, label 1 only, label 2), smooth 1e-6, mean over ALL slices of the evaluated patients. Empty-vs-empty
    slices score 1.0, so it is inflated (especially for tumor); reported in separate columns, never as the main metric.
Aggregates: mean / median over patients, by thickness_group, and by lesion size bin (equivalent diameter
<10, 10-20, >20 mm). Diameter = sphere of the lesion's voxel volume, voxel = (2*sx) x (2*sy) x sz mm taken
from per_volume.csv (512 -> 256 resize doubles the in-plane pixel); approximate because the cache masks are
nearest-resized and volumes 28-47 only have placeholder 1 mm spacing (their mm values are nominal and
they count as 'thin'; the tables show how many patients / lesions come from them).
Thickness group (thin <= 1 mm < mid <= 2.5 mm < thick) comes from sz_mm in per_volume.csv.
Leakage guard: if split_used.json (written by train_fast.py) sits next to the checkpoint, the evaluated volumes
must not overlap ITS train list (checked per checkpoint, so it also covers a LITS_SUBSET_N run and older
checkpoints); for LITS_EVAL_PART=test they must not overlap the validation list either. On overlap the run
stops and names the volumes. Evaluating 'val'/'heldout' volumes that picked the epoch only warns.
The cache is not checked against the one used for training, keep LITS_CACHE_DIR the same.

Env: LITS_CKPT (required), LITS_CACHE_DIR, LITS_EVAL_PART (test|val|heldout), LITS_EVAL_MAX_VOLUMES (default 0 = all;
N = first N volumes of the part in cache order, for smoke tests), LITS_EVAL_OUT (output dir: eval.json,
per_patient.csv, eval.md), LITS_BATCH_SIZE (32), LITS_AMP (auto: as train_fast.py), LITS_DEVICE (auto), LITS_PER_VOLUME.
The model input (1 slice or 2.5D context) is read from the checkpoint's config (old raw state_dicts: from the weights).
LITS_MAX_VOLUMES is a smoke-test option of train_fast.py only and is ignored here.

    LITS_CKPT=results/full/best.pt python eval_fast.py
"""
import os
import csv
import sys
import json
import time
import logging
import importlib.util
import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root; data/ lives here
CACHE_DIR = os.environ.get("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache"))
PER_VOLUME = os.environ.get("LITS_PER_VOLUME", os.path.join(ROOT, "analysis", "output", "per_volume.csv"))
OUT_DIR = os.environ.get("LITS_EVAL_OUT", os.path.join(ROOT, "results", "eval_fast"))
SIZE_BINS = [("<10 mm", 0, 10), ("10-20 mm", 10, 20), (">20 mm", 20, float("inf"))]
LEGACY = ("background", "liver_label1", "tumor")  # classes 0, 1, 2 as in the original test script

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_fast")


def load_train_module():
    """train_fast.py next to this file (model classes, AMP / device rules). Importing it has no side effects."""
    spec = importlib.util.spec_from_file_location("train_fast_for_eval", os.path.join(HERE, "train_fast.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_class():
    return load_train_module().AIM_UNet


def shown(path):
    """Path as written into eval.json / eval.md: relative to the repo root when inside it (no personal paths)."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, ROOT) if ap.startswith(ROOT + os.sep) else path


def nan_to_none(o):
    """eval.json must be strict JSON: NaN (undefined metric) -> null."""
    if isinstance(o, float):
        return None if o != o else o
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [nan_to_none(v) for v in o]
    return o


def dice(p, g):
    s = p.sum() + g.sum()
    return float("nan") if s == 0 else float(2 * (p & g).sum() / s)


def legacy_slice_dice(pred, gt, smooth=1e-6):
    """Per-slice Dice of classes 0 / 1 (label 1 only) / 2 exactly like the original test script's dice_per_class,
    but for every slice of the volume -> {name: (n_slices,) array}. Empty-vs-empty slices score 1.0."""
    out = {}
    for c, name in enumerate(LEGACY):
        p, t = pred == c, gt == c
        inter = (p & t).sum((1, 2)).astype(np.float64)
        union = p.sum((1, 2)).astype(np.float64) + t.sum((1, 2))
        out[name] = (2 * inter + smooth) / (union + smooth)
    return out


@torch.no_grad()
def predict(model, images, autocast, batch_size, device, context=False):
    """images: uint8 (n, H, W) of ONE volume -> uint8 label volume. Context: slices k-1, k, k+1 of this volume,
    clamped at its borders (the same rule as the training gather)."""
    out, n = [], len(images)
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        arr = np.stack([images[np.maximum(idx - 1, 0)], images[idx], images[np.minimum(idx + 1, n - 1)]], 1) if context \
            else images[idx][:, None]
        xb = torch.from_numpy(arr).to(device).float() / 255.
        with autocast():
            logits = model(xb)
        out.append(logits.float().argmax(1).to(torch.uint8).cpu())
    return torch.cat(out).numpy()


def evaluate_volume(pred, gt, vox_mm3):
    liver_p, liver_g, tum_p, tum_g = pred >= 1, gt >= 1, pred == 2, gt == 2
    lab, n = ndi.label(tum_g)  # 3D, 6-connectivity
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hit = np.zeros(n + 1, bool)
    hit[np.unique(lab[tum_p])] = True  # components containing >= 1 predicted tumor voxel
    lesions = [dict(diameter_mm=float((6 * size[k] * vox_mm3 / np.pi) ** (1 / 3)), detected=bool(hit[k + 1])) for k in range(n)]
    plab, pn = ndi.label(tum_p)
    fp_comps = pn - len(np.unique(plab[tum_g & tum_p]))  # predicted components touching no GT tumor
    return dict(liver_dice=dice(liver_p, liver_g), tumor_dice=dice(tum_p, tum_g) if n else float("nan"), n_lesions=n,
                lesions_detected=sum(l["detected"] for l in lesions),
                lesion_recall=sum(l["detected"] for l in lesions) / n if n else float("nan"),
                fp_tumor_ml=float((tum_p & ~tum_g).sum() * vox_mm3 / 1000), fp_components=int(fp_comps)), lesions


def summarize(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return dict(n=len(v), mean=float(v.mean()) if len(v) else float("nan"), median=float(np.median(v)) if len(v) else float("nan"))


def aggregate(rows, lesions):
    metrics = ["liver_dice", "tumor_dice", "lesion_recall", "fp_tumor_ml"]
    agg = {"overall": {m: summarize([r[m] for r in rows]) for m in metrics}}
    agg["overall"]["pooled_lesion_recall"] = dict(lesions=len(lesions), detected=sum(l["detected"] for l in lesions),
                                                  recall=sum(l["detected"] for l in lesions) / len(lesions) if lesions else float("nan"))
    agg["by_thickness"] = {}
    for g in ("thin", "mid", "thick"):
        sub = [r for r in rows if r["thickness_group"] == g]
        if sub:
            les = [l for l in lesions if l["thickness_group"] == g]
            agg["by_thickness"][g] = {"patients": len(sub), "placeholder_spacing": sum(not r["spacing_reliable"] for r in sub),
                                      **{m: summarize([r[m] for r in sub]) for m in metrics},
                                      "lesions": len(les), "pooled_lesion_recall": sum(l["detected"] for l in les) / len(les) if les else float("nan")}
    agg["by_lesion_size"] = {}
    for name, lo, hi in SIZE_BINS:
        les = [l for l in lesions if lo <= l["diameter_mm"] < hi]
        agg["by_lesion_size"][name] = dict(lesions=len(les), nominal_mm=sum(not l["spacing_reliable"] for l in les),
                                           detected=sum(l["detected"] for l in les),
                                           recall=sum(l["detected"] for l in les) / len(les) if les else float("nan"))
    return agg


def legacy_aggregate(rows):
    """Mean over all slices of all evaluated patients (patient means weighted by their slice counts)."""
    n = sum(r["n_slices"] for r in rows)
    out = {k: float(sum(r[f"legacy_dice_{k}"] * r["n_slices"] for r in rows) / n) for k in LEGACY}
    out["n_slices"] = int(n)
    return out


PART_TEXT = {
    "test": "TEST part: the held-out volumes that are not in the validation subset; never used for checkpoint selection, "
            "early stopping or recipe choice.",
    "val": "VALIDATION part: these volumes picked the checkpoint (and the recipe), so the numbers are validation-selected, "
           "not an untouched test.",
    "heldout": "Whole held-out part of the original split (train_test_split(test_size=0.2, random_state=42)); if it was also the "
               "validation set that picked the checkpoint, the numbers are validation-selected.",
}


def markdown(agg, legacy, rows, n_part, ckpt, part):
    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    s = lambda d: f"{f(d['mean'])} / {f(d['median'])}"
    o = agg["overall"]
    md = [f"# eval_fast: {part} part, {len(rows)} of {n_part} patients", "",
          f"Checkpoint: `{shown(ckpt)}`, cache `{shown(CACHE_DIR)}`", "", PART_TEXT[part], "",
          "Placeholder spacing = volumes 28-47 (1x1x1 mm header): counted as thin, their lesion diameters are nominal.", "",
          "Dice is NaN-skipped (both empty; tumor Dice also for tumor-free patients); lesion recall = GT 6-connected components with >= 1 predicted tumor voxel.", "",
          "## Overall (mean / median over patients)", "", "| metric | n patients | mean / median |", "|---|---|---|"]
    for m in ("liver_dice", "tumor_dice", "lesion_recall", "fp_tumor_ml"):
        md.append(f"| {m} | {o[m]['n']} | {s(o[m])} |")
    p = o["pooled_lesion_recall"]
    md += ["", f"Pooled lesion recall: {p['detected']}/{p['lesions']} = {f(p['recall'])}", "",
           "## Legacy per-slice Dice (as the original test script; inflated: empty-vs-empty slices score 1.0)", "",
           f"Mean over all {legacy['n_slices']} slices: background {f(legacy['background'])}, liver (label 1 only) {f(legacy['liver_label1'])}, "
           f"tumor {f(legacy['tumor'])}. Not comparable with the patient-level 3D Dice above.", "",
           "## By thickness group", "",
           "| group | patients (placeholder spacing) | liver Dice | tumor Dice | patient lesion recall | pooled recall (lesions) | FP tumor mL |", "|---|---|---|---|---|---|---|"]
    for g, d in agg["by_thickness"].items():
        md.append(f"| {g} | {d['patients']} ({d['placeholder_spacing']}) | {s(d['liver_dice'])} | {s(d['tumor_dice'])} | {s(d['lesion_recall'])} | "
                  f"{f(d['pooled_lesion_recall'])} ({d['lesions']}) | {s(d['fp_tumor_ml'])} |")
    md += ["", "## By lesion size (equivalent diameter at 256x256 in-plane pixel size, approximate)", "",
           "| size | lesions (nominal mm) | detected | recall |", "|---|---|---|---|"]
    for n, d in agg["by_lesion_size"].items():
        md.append(f"| {n} | {d['lesions']} ({d['nominal_mm']}) | {d['detected']} | {f(d['recall'])} |")
    md += ["", "## Per patient", "", "| volume | thickness | liver Dice | tumor Dice | lesions (detected) | FP tumor mL | FP comps | legacy tumor (slice mean) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['thickness_group']} | {f(r['liver_dice'])} | {f(r['tumor_dice'])} | "
                  f"{r['n_lesions']} ({r['lesions_detected']}) | {r['fp_tumor_ml']:.2f} | {r['fp_components']} | {f(r['legacy_dice_tumor'])} |")
    return "\n".join(md) + "\n"


def resolve_volumes(ckpt, names, part_env):
    """-> (part, ids in cache order, split_used meta | None, path of split_used.json).
    See the module docstring for the three parts."""
    # Same rule as train_fast.py split_volumes(): the held-out part is the "test" side of this split.
    _, held = train_test_split(np.arange(len(names)), test_size=0.2, random_state=42)
    held = sorted(int(i) for i in held)  # cache order = sorted volume names
    used = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "split_used.json")
    meta = None
    if os.path.exists(used):
        with open(used) as fh:
            meta = json.load(fh)
    has_test = bool(meta and meta.get("test"))
    part = (part_env or ("test" if has_test else "val")).lower()
    if part not in PART_TEXT:
        sys.exit(f"LITS_EVAL_PART={part!r}: expected test, val or heldout")
    pos = {n: i for i, n in enumerate(names)}
    if part == "test":
        if not has_test:
            sys.exit(f"LITS_EVAL_PART=test needs a non-empty 'test' list in {shown(used)} (train with LITS_VAL_N > 0)")
        ids = sorted(pos[n] for n in meta["test"])
    elif part == "val" and has_test:
        ids = sorted(pos[n] for n in meta["val"])
    else:  # 'heldout', or 'val' of a run without a test part (val = the held-out part, as before)
        ids = held
    assert ids, "no volumes to evaluate"
    return part, ids, meta, used


def main():
    ckpt = os.environ["LITS_CKPT"]
    max_vols = int(os.environ.get("LITS_EVAL_MAX_VOLUMES", 0))  # 0 = all volumes of the part
    batch_size = int(os.environ.get("LITS_BATCH_SIZE", 32))
    tf = load_train_module()
    device = tf.pick_device(os.environ.get("LITS_DEVICE", "auto").lower())
    cc, bf16_ok = tf.cuda_caps() if device.type == "cuda" else (None, None)
    amp, amp_dtype, _ = tf.resolve_amp(os.environ.get("LITS_AMP", "auto"), device.type, cc, bf16_ok)
    autocast = lambda: torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    with open(os.path.join(CACHE_DIR, "volumes.txt")) as fh:
        names = fh.read().rstrip("\n").split("\n")
    vol_index = np.load(os.path.join(CACHE_DIR, "vol_index.npy"))
    images = np.load(os.path.join(CACHE_DIR, "images.npy"), mmap_mode="r")
    masks = np.load(os.path.join(CACHE_DIR, "masks.npy"), mmap_mode="r")
    with open(PER_VOLUME) as fh:
        pv = {r["name"]: r for r in csv.DictReader(fh)}
    thickness = lambda n: "thin" if float(pv[n]["sz_mm"]) <= 1.0 else "mid" if float(pv[n]["sz_mm"]) <= 2.5 else "thick"
    part, part_ids, meta, used = resolve_volumes(ckpt, names, os.environ.get("LITS_EVAL_PART", ""))
    ids = part_ids[:max_vols] if max_vols else part_ids

    if meta is not None:
        todo = {names[i] for i in ids}
        leak = sorted(set(meta["train"]) & todo)
        # Not a leak for val/heldout, but those volumes picked the epoch (and may come from another split, e.g. the
        # archived baseline): the scores on them are validation-selected, so say it. For 'test' it is an error.
        sel_overlap = sorted(set(meta.get("val", [])) & todo)
        if sel_overlap and part == "test":
            sys.exit(f"Leakage guard: {len(sel_overlap)} test volume(s) are in this checkpoint's VALIDATION list (used for "
                     f"selection): {', '.join(sel_overlap)}. Not evaluating.")
        if sel_overlap:
            logger.warning(f"{len(sel_overlap)} evaluated volume(s) are in this checkpoint's validation list "
                           f"(used to pick the epoch): {', '.join(sel_overlap)}")
        if leak:
            sys.exit(f"Leakage guard: {len(leak)} of the {len(ids)} volumes to evaluate are TRAINING volumes of this "
                     f"checkpoint (train list in {shown(used)}): {', '.join(leak)}. Not evaluating. Checkpoints trained with a "
                     f"different split (e.g. results/baseline/archive) can overlap the original-split held-out part; "
                     f"restrict with LITS_EVAL_MAX_VOLUMES to volumes outside that train list (the guard checks the train "
                     f"list only; overlap with the validation list is only warned about).")

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd, cfg = (ck["model"], ck.get("config") or {}) if isinstance(ck, dict) and "model" in ck else (ck, {})
    in_ch = int(sd["enc1.conv.0.weight"].shape[1])  # 1 = single slice, 3 = 2.5D context
    if cfg.get("in_channels") not in (None, in_ch):
        sys.exit(f"checkpoint config in_channels={cfg['in_channels']} does not match its weights ({in_ch})")
    context = in_ch == 3
    model = tf.AIM_UNet(in_ch)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    logger.info(f"Device {device}, AMP {amp} | {ckpt} (in_channels {in_ch}) on the {part} part: {len(ids)} of {len(part_ids)} volumes, cache {CACHE_DIR}")

    rows, all_lesions, t0 = [], [], time.time()
    for i in ids:
        n = names[i]
        sel = np.flatnonzero(vol_index == i)
        sl = slice(sel[0], sel[-1] + 1)  # slices of one volume are contiguous
        vox = (2 * float(pv[n]["sx_mm"])) * (2 * float(pv[n]["sy_mm"])) * float(pv[n]["sz_mm"])
        pred, gt = predict(model, np.asarray(images[sl]), autocast, batch_size, device, context), np.asarray(masks[sl])
        r, les = evaluate_volume(pred, gt, vox)
        r.update(name=n, thickness_group=thickness(n), n_slices=len(sel),
                 spacing_reliable=not all(float(pv[n][k]) == 1.0 for k in ("sx_mm", "sy_mm", "sz_mm")))
        r.update({f"legacy_dice_{k}": float(v.mean()) for k, v in legacy_slice_dice(pred, gt).items()})
        rows.append(r)
        all_lesions += [{**l, "thickness_group": r["thickness_group"], "name": n, "spacing_reliable": r["spacing_reliable"]} for l in les]
        logger.info(f"{n}: liver Dice {r['liver_dice']:.3f}, tumor Dice {r['tumor_dice']:.3f}, lesions {r['lesions_detected']}/{r['n_lesions']}, "
                    f"FP {r['fp_tumor_ml']:.2f} mL, legacy slice tumor {r['legacy_dice_tumor']:.3f} ({time.time() - t0:.0f}s)")

    agg, legacy = aggregate(rows, all_lesions), legacy_aggregate(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    # "fold" stays (always null) so readers of the older eval.json files keep working; there are no folds any more
    out = dict(checkpoint=shown(ckpt), cache=shown(CACHE_DIR), fold=None,
               eval_part=PART_TEXT[part], part=part, n_held_out=len(part_ids), n_evaluated=len(rows),
               volumes=[r["name"] for r in rows], aggregate=agg, legacy_slice_mean_dice=legacy,
               model_config=dict(in_channels=in_ch, **{k: v for k, v in cfg.items() if k != "in_channels"}), amp=amp,
               per_patient=rows, lesions=all_lesions)
    with open(os.path.join(OUT_DIR, "eval.json"), "w") as fh:
        json.dump(nan_to_none(out), fh, indent=1, allow_nan=False)
    cols = ["name", "thickness_group", "spacing_reliable", "n_slices", "liver_dice", "tumor_dice", "n_lesions", "lesions_detected",
            "lesion_recall", "fp_tumor_ml", "fp_components"] + [f"legacy_dice_{k}" for k in LEGACY]
    with open(os.path.join(OUT_DIR, "per_patient.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    md = markdown(agg, legacy, rows, len(part_ids), ckpt, part)
    with open(os.path.join(OUT_DIR, "eval.md"), "w") as fh:
        fh.write(md)
    logger.info(f"Wrote eval.json, per_patient.csv, eval.md -> {OUT_DIR}\n" + md.split("## Per patient")[0])


if __name__ == "__main__":
    sys.exit(main())
