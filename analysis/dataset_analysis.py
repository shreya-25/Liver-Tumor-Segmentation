"""Per-volume statistics for the raw LiTS NIfTI files (read-only; no env vars).

    python dataset_analysis.py <raw_dir> <out_dir>

Expects <raw_dir>/volume-N.nii and segmentation-N.nii (N numeric; processed in numeric order).
Writes <out_dir>/per_volume.csv and <out_dir>/stats.json (per-volume stats, lesion components, cohort HU histograms).
Labels: 1 = liver (non-tumor), 2 = tumor. liver_ml / liver_vox count labels >= 1 (tumor included); liver_hu_* use label 1 only.
"""
import os
import sys
import csv
import json
import glob
import time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from multiprocessing import Pool

HU_BINS = np.arange(-1024, 3073, 8)  # 8-HU bins for aggregate histograms


def analyze(args):
    vp, sp = args
    name = os.path.basename(vp)
    v, s = nib.load(vp), nib.load(sp)
    zooms = [float(z) for z in v.header.get_zooms()[:3]]
    vol = np.asarray(v.dataobj)
    seg_raw = np.asarray(s.dataobj)
    labels_raw = np.unique(seg_raw).tolist()
    seg = seg_raw.astype(np.uint8)
    vox_mm3 = zooms[0] * zooms[1] * zooms[2]
    liver_only, tumor = seg == 1, seg == 2
    liver_any = seg >= 1

    samp = vol.ravel()[::37]
    r = dict(
        name=name, shape=list(v.shape), spacing=zooms, axcodes="".join(nib.aff2axcodes(v.affine)),
        vol_dtype=str(v.get_data_dtype()), seg_dtype=str(s.get_data_dtype()), seg_shape_match=bool(v.shape == s.shape),
        labels=labels_raw, n_slices=int(v.shape[2]),
        hu_min=int(vol.min()), hu_max=int(vol.max()), hu_p001=float(np.percentile(samp, 0.1)),
        hu_p999=float(np.percentile(samp, 99.9)), frac_at_min=float((vol == vol.min()).mean()),
    )
    r["liver_ml"] = float(liver_any.sum() * vox_mm3 / 1000)
    r["tumor_ml"] = float(tumor.sum() * vox_mm3 / 1000)
    r["liver_vox"], r["tumor_vox"] = int(liver_any.sum()), int(tumor.sum())
    r["tumor_frac_of_liver"] = r["tumor_ml"] / r["liver_ml"] if r["liver_ml"] else 0.0
    zl = np.flatnonzero(liver_any.any(axis=(0, 1)))
    zt = np.flatnonzero(tumor.any(axis=(0, 1)))
    r["slices_liver"], r["slices_tumor"] = int(len(zl)), int(len(zt))
    if len(zl):
        bb = [np.flatnonzero(liver_any.any(axis=tuple(a for a in range(3) if a != d))) for d in range(3)]
        r["liver_bbox_mm"] = [float((b[-1] - b[0] + 1) * z) for b, z in zip(bb, zooms)]
        r["liver_bbox_vox"] = [int(b[-1] - b[0] + 1) for b in bb]
    lh = vol[liver_only].astype(np.float32) if liver_only.any() else np.zeros(0, np.float32)
    th = vol[tumor].astype(np.float32) if tumor.any() else np.zeros(0, np.float32)
    r["liver_hu_mean"] = float(lh.mean()) if len(lh) else None
    r["liver_hu_std"] = float(lh.std()) if len(lh) else None
    r["tumor_hu_mean"] = float(th.mean()) if len(th) else None
    r["tumor_hu_std"] = float(th.std()) if len(th) else None
    if len(lh) and len(th):
        r["contrast_hu"] = r["tumor_hu_mean"] - r["liver_hu_mean"]
        r["cnr"] = r["contrast_hu"] / max(r["liver_hu_std"], 1e-6)

    comps = []
    if tumor.any():
        lab, _ = ndi.label(tumor)  # 6-connectivity
        sizes = np.bincount(lab.ravel())[1:]
        objs = ndi.find_objects(lab)
        lm = r["liver_hu_mean"] if r["liver_hu_mean"] is not None else np.nan
        for i, (sz, sl) in enumerate(zip(sizes, objs), start=1):
            ext = [float((s_.stop - s_.start) * z) for s_, z in zip(sl, zooms)]
            vmm3 = float(sz) * vox_mm3
            hu = float(vol[sl][lab[sl] == i].mean())
            comps.append(dict(vox=int(sz), ml=vmm3 / 1000, eq_diam_mm=float((6 * vmm3 / np.pi) ** (1 / 3)),
                              max_extent_mm=max(ext), hu_mean=hu, contrast=hu - lm))
    r["n_tumors"] = len(comps)
    r["largest_tumor_share"] = (max(c["vox"] for c in comps) / r["tumor_vox"]) if comps else 0.0
    r["components"] = comps

    hist = dict(
        body=np.histogram(vol.ravel()[::7], HU_BINS)[0].tolist(),
        liver=np.histogram(lh, HU_BINS)[0].tolist(),
        tumor=np.histogram(th, HU_BINS)[0].tolist(),
    )
    return r, hist


def main(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vols = sorted(glob.glob(os.path.join(raw_dir, "volume-*.nii")), key=lambda p: int(p.split("-")[-1].split(".")[0]))
    segs = [p.replace("volume-", "segmentation-") for p in vols]
    t0 = time.time()
    rows, agg = [], {k: np.zeros(len(HU_BINS) - 1, np.int64) for k in ("body", "liver", "tumor")}
    with Pool(8) as pool:
        for i, (r, h) in enumerate(pool.imap(analyze, zip(vols, segs))):
            rows.append(r)
            for k in agg:
                agg[k] += np.array(h[k], np.int64)
            print(f"[{i + 1}/{len(vols)}] {r['name']} {r['shape']} sp={[round(x, 2) for x in r['spacing']]} "
                  f"tumors={r['n_tumors']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(per_volume=rows, hu_bins=HU_BINS.tolist(), hu_hist={k: v.tolist() for k, v in agg.items()}),
              open(os.path.join(out_dir, "stats.json"), "w"))
    flat = [k for k in rows[0] if k not in ("components", "labels", "shape", "spacing", "liver_bbox_mm", "liver_bbox_vox")]
    with open(os.path.join(out_dir, "per_volume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "nx", "ny", "nz", "sx_mm", "sy_mm", "sz_mm", "bbox_x_mm", "bbox_y_mm", "bbox_z_mm"] +
                   [k for k in flat if k != "name"])
        for r in rows:
            bb = r.get("liver_bbox_mm", [None] * 3)
            w.writerow([r["name"], *r["shape"], *[round(x, 4) for x in r["spacing"]], *[None if b is None else round(b, 1) for b in bb]] +
                       [r.get(k) for k in flat if k != "name"])
    print(f"Done {len(rows)} volumes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
