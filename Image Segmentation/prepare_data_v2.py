"""v2 slice cache: fixed HU window + data-driven orientation canonicalisation + per-volume metadata.

Differences from prepare_data.py (v1):
  * intensity: clip HU to [LITS_WIN_LO, LITS_WIN_HI] (default -100..250) and scale to uint8 with the
    SAME mapping for every volume. v1's per-volume min-max lets the -3024 padding / metal outliers set
    the scale, which squeezes liver and tumor into a few uint8 levels (tumor-liver contrast ~2 levels).
  * orientation: the raw arrays fall into three classes that the NIfTI header does not predict (volumes
    28-47 say RAS but behave like LAS), so as_closest_canonical is not enough. Per volume, the liver
    centroid is compared with the body centroid along array axes 0 and 1 and axes are flipped so that
    every volume ends up RAS-like: liver at the HIGH index of axis 0 (patient right) and anterior at the
    HIGH index of axis 1. The rule uses the liver mask, so it is a training-data rule, not an inference one.
    z needs no rule: all headers are 'S' and the lungs lie at higher z than the liver top in every volume
    that shows both. Set LITS_CANON=0 to keep the raw orientation (class and flips are still reported).
  * meta.csv, one row per volume (see META_COLS): orientation class and flips, thickness group, tumor burden.
    It is metadata for reporting (showcase / notebook orientation table); it does not define any split (that
    is the original sklearn rule in train_fast.py) and does not feed the subset picker (train_fast.py reads
    analysis/output/per_volume.csv).
Slice order, z sampling and the PIL resize are unchanged, so v1 and v2 align slice-for-slice. Flips are
applied AFTER the resize: PIL's nearest resize is not mirror-symmetric, so flipping first would shift
the masks by one voxel against v1.

    python prepare_data_v2.py         # data/raw -> data/cache_v2 (skips if already built with the same settings)

Env vars: LITS_DATA_DIR (default data/raw), LITS_CACHE_DIR (default data/cache_v2), LITS_WIN_LO / LITS_WIN_HI,
LITS_CANON, LITS_FORCE.
Output: images.npy, masks.npy (N, 256, 256 uint8), vol_index.npy, volumes.txt (sorted names), meta.csv, config.json.
LITS_CACHE_DIR is shared with train_fast / eval_fast / prepare_data.py, so a build refuses to overwrite a
directory that holds cache files but no config.json from this script (e.g. the v1 cache). LITS_FORCE=1
overrides that (needed to restart after an interrupted build, which leaves files without config.json).
"""
import os
import sys
import csv
import json
import time
import numpy as np
import nibabel as nib
from PIL import Image
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; data/ lives here
RAW_DIR = os.environ.get("LITS_DATA_DIR", os.path.join(ROOT, "data", "raw"))
CACHE_DIR = os.environ.get("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache_v2"))
WIN_LO = float(os.environ.get("LITS_WIN_LO", -100))
WIN_HI = float(os.environ.get("LITS_WIN_HI", 250))
CANON = os.environ.get("LITS_CANON", "1") != "0"
SIZE = (256, 256)
META_COLS = ["name", "orientation_class", "flipped_axes", "spacing_reliable", "thickness_group", "n_slices",
             "hu_min", "hu_max", "has_tumor", "tumor_ml", "liver_hu_mean", "liver_rel_ax0", "liver_rel_ax1"]


def volume_pairs(folder):
    # Same pairing rule as v1: sorted 'vol*' and 'seg*' file names (-> same volume order).
    names = os.listdir(folder)
    vols = sorted(os.path.join(folder, f) for f in names if 'vol' in f.lower())
    segs = sorted(os.path.join(folder, f) for f in names if 'seg' in f.lower())
    return vols, segs


def liver_vs_body(vol, seg):
    """Liver centroid minus body centroid along array axes 0/1, in units of the body bounding-box extent.
    Body = HU > -300 on a stride-4 subsample (includes the table, as in the report's analysis)."""
    idx = np.argwhere(seg[::2, ::2] >= 1) * np.array([2, 2, 1])
    assert len(idx), "volume has no liver label; orientation cannot be determined"
    liver_c = idx.mean(0)
    nz = np.argwhere(vol[::4, ::4, ::4] > -300) * 4
    body_c, extent = nz.mean(0), np.maximum(nz.max(0) - nz.min(0), 1)
    return ((liver_c - body_c) / extent)[:2]


def orientation_class(rel):
    # A: liver low on axis 0 / anterior low on axis 1; B: low / high; C: high / high (the RAS-like target);
    # D: high / low (not seen in LiTS; would be reported rather than hidden).
    return {(True, True): "A", (True, False): "B", (False, False): "C", (False, True): "D"}[(bool(rel[0] < 0), bool(rel[1] < 0))]


def thickness_group(sz):
    return "thin" if sz <= 1.0 else "mid" if sz <= 2.5 else "thick"


def process(args):
    vp, mp = args
    img = nib.load(vp)  # geometry comes from the IMAGE header only (48-52 have a default mask header)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    raw = np.asarray(img.dataobj).astype(np.float32)
    seg = np.asarray(nib.load(mp).dataobj).astype(np.uint8)
    rel = liver_vs_body(raw, seg)
    cls = orientation_class(rel)
    flips = [a for a in (0, 1) if CANON and rel[a] < 0]  # make liver / anterior sit at the high index

    liver, tumor = seg == 1, seg == 2  # liver = non-tumor liver, as in per_volume.csv
    meta = dict(
        name=os.path.basename(vp), orientation_class=cls, flipped_axes="+".join(f"ax{a}" for a in flips) or "none",
        spacing_reliable=not all(z == 1.0 for z in zooms),  # only 28-47 have an exact 1x1x1 mm header (checked vs per_volume.csv)
        thickness_group=thickness_group(zooms[2]), n_slices=raw.shape[2],
        hu_min=float(raw.min()), hu_max=float(raw.max()), has_tumor=bool(tumor.any()),
        tumor_ml=float(tumor.sum() * np.prod(zooms) / 1000.0),
        liver_hu_mean=float(raw[liver].mean()) if liver.any() else float("nan"),
        liver_rel_ax0=float(rel[0]), liver_rel_ax1=float(rel[1]))

    vol, seg = np.moveaxis(raw, 2, 0), np.moveaxis(seg, 2, 0)  # raw LiTS is (H, W, Z) -> slice along Z
    u8 = np.rint(np.clip((vol - WIN_LO) / (WIN_HI - WIN_LO), 0, 1) * 255).astype(np.uint8)
    imgs = np.stack([np.asarray(Image.fromarray(s).resize(SIZE, Image.BILINEAR)) for s in u8])
    msks = np.stack([np.asarray(Image.fromarray(s).resize(SIZE, Image.NEAREST)) for s in seg])
    for a in flips:  # raw axis a is array axis a+1 of (Z, H, W)
        imgs, msks = np.flip(imgs, a + 1), np.flip(msks, a + 1)
    return np.ascontiguousarray(imgs), np.ascontiguousarray(msks), meta


def main():
    vols, segs = volume_pairs(RAW_DIR)
    assert vols and len(vols) == len(segs), f"found {len(vols)} volumes / {len(segs)} masks in {RAW_DIR}"
    cfg = dict(meta_fmt=2, win_lo=WIN_LO, win_hi=WIN_HI, canon=CANON, volumes=[os.path.basename(v) for v in vols])
    cfg_path = os.path.join(CACHE_DIR, "config.json")
    done = [os.path.join(CACHE_DIR, f) for f in ("images.npy", "masks.npy", "vol_index.npy", "volumes.txt", "meta.csv")]
    if all(os.path.exists(p) for p in done + [cfg_path]) and json.load(open(cfg_path)) == cfg:
        print(f"{CACHE_DIR} already built with window [{WIN_LO}, {WIN_HI}], canon={CANON}; nothing to do")
        return
    if not os.path.exists(cfg_path) and not os.environ.get("LITS_FORCE") and any(os.path.exists(p) for p in done):
        sys.exit(f"{CACHE_DIR} has cache files but no config.json from prepare_data_v2.py; refusing to overwrite "
                 f"(wrong LITS_CACHE_DIR? set LITS_FORCE=1 to rebuild anyway)")
    counts = [nib.load(v).shape[2] for v in vols]
    total = sum(counts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for p in done + [cfg_path]:  # meta.csv / config.json are written last, so a partial build is never taken as done
        if os.path.exists(p):
            os.remove(p)
    images = np.lib.format.open_memmap(os.path.join(CACHE_DIR, "images.npy"), "w+", np.uint8, (total, *SIZE))
    masks = np.lib.format.open_memmap(os.path.join(CACHE_DIR, "masks.npy"), "w+", np.uint8, (total, *SIZE))
    vol_index = np.repeat(np.arange(len(vols), dtype=np.int16), counts)

    t0, pos, rows = time.time(), 0, []
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        for i, (imgs, msks, meta) in enumerate(pool.imap(process, zip(vols, segs))):
            images[pos:pos + len(imgs)], masks[pos:pos + len(msks)] = imgs, msks
            pos += len(imgs)
            rows.append(meta)
            print(f"[{i + 1}/{len(vols)}] {meta['name']}: {len(imgs)} slices, class {meta['orientation_class']}, "
                  f"flip {meta['flipped_axes']} ({time.time() - t0:.0f}s)", flush=True)
    images.flush(), masks.flush()
    assert pos == total
    np.save(os.path.join(CACHE_DIR, "vol_index.npy"), vol_index)
    with open(os.path.join(CACHE_DIR, "volumes.txt"), "w") as f:
        f.write("\n".join(os.path.basename(v) for v in vols))
    with open(os.path.join(CACHE_DIR, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, META_COLS)
        w.writeheader()
        w.writerows(rows)
    json.dump(cfg, open(cfg_path, "w"))
    n_bad = sum(not r["spacing_reliable"] for r in rows)
    print(f"Done: {total} slices -> {CACHE_DIR} in {time.time() - t0:.0f}s; "
          f"classes {dict(sorted({c: sum(r['orientation_class'] == c for r in rows) for c in 'ABCD'}.items()))}, "
          f"{n_bad} volumes with placeholder spacing")


if __name__ == "__main__":
    sys.exit(main())
