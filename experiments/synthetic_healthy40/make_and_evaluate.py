"""40 synthetic tumour CTs from healthy hosts (CHAOS CT + tumour-free LiTS train-part volumes) and their realism metrics.

Tumours: results/gan_lits130/synthetic/samples_100.npz (generator epoch 36), inserted with insertion_improved (vessel-aware placement,
size control 5 / 10 / 15 mm, HU harmonisation, mass effect, QC; a rejected insertion is retried with another sample / site).
Outputs (this folder): images/synthetic_XX_<host>_r<R>mm.nii.gz (HU int16) + _label.nii.gz (0 bg, 1 liver, 2 tumour), metrics_per_image.csv,
metrics_summary.json, figures/*.png, run.log.

Metrics
  A. synthetic vs its own original healthy scan (same patient, before / after): PSNR and SSIM in the liver window [-100, 250] HU over
     the whole volume, the liver bounding box and the tumour neighbourhood (tumour bbox + 15 mm); voxels changed outside the edit box.
  B. inserted tumours vs REAL LiTS tumours (64^3 patches at 1 mm around each tumour, the same geometry as the GAN training patches):
     tumour HU, tumour-minus-liver contrast, boundary sharpness, texture-feature Frechet distance / MMD, classifier two-sample test
     (C2ST, 0.5 = indistinguishable), SSIM of the central slice to the nearest real tumour; every metric also for a second set of REAL
     tumours from other volumes ("real B"), i.e. the value a perfect synthesis would reach.
"""
import os, sys, csv, json, time, glob
REPO = r"C:\Users\shrey\Downloads\Liver-Tumor-Segmentation-GitHub\Liver-Tumor-Segmentation-GitHub"
OUT = os.environ.get("SYN40_OUT", os.path.join(REPO, "results", "synthetic_healthy40"))
CHAOS = r"C:\Users\shrey\Downloads\Liver Tumor Segmentation 10_26\Dataset\CHAOS_Train_Sets\Train_Sets\NiiFiles"
os.environ.update(LITS_DATA_DIR=os.path.join(REPO, "data", "raw"), LITS_GAN_RESULTS_DIR=OUT, LITS_EVAL_N="64")
sys.path.insert(0, os.path.join(REPO, "Image Synthesis"))
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt
import Syn_Img_Pipeline_fixed as S
import insertion_improved as I
import syn_training_improved as T
import evaluate_synthesis as E

N_IMAGES, SIZES, LO, HI = int(os.environ.get("SYN40_N", 40)), (5.0, 10.0, 15.0), -100.0, 250.0
LITS_HEALTHY_TRAIN = [87, 89, 91, 105, 106, 119]  # tumour-free LiTS volumes of the segmenter's train part (114, 115 are held out)
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
S.setup_logging(OUT)
log = S.logger
rng = np.random.default_rng(2026)


def hosts():
    hs = []
    for f in sorted(glob.glob(os.path.join(CHAOS, "image_*.nii.gz")), key=lambda p: int(p.split("_")[-1].split(".")[0])):
        hs.append(("chaos", int(f.split("_")[-1].split(".")[0]), f, f.replace("image_", "segmentation_")))
    for i in LITS_HEALTHY_TRAIN:
        hs.append(("lits", i, os.path.join(S.DATA_DIR, f"volume-{i}.nii"), os.path.join(S.DATA_DIR, f"segmentation-{i}.nii")))
    return hs


def load(src, vp, sp):
    vol, seg, spacing, aff = S.load_case(vp, sp)
    seg = (seg > 0).astype(np.uint8)  # CHAOS: 1 = liver; LiTS healthy: 1 = liver
    if src == "chaos":
        seg = np.ascontiguousarray(seg[:, :, ::-1])  # CHAOS NIfTI masks are stored with the slice order reversed relative to the image
    return vol, seg, spacing, aff


def liver_fit(vol, seg):
    """Fraction of liver-mask voxels with liver-like HU (0-200); an aligned mask gives > 0.9."""
    v = vol[seg > 0]
    return float(np.mean((v > 0) & (v < 200)))


def win01(x):
    return np.clip((x - LO) / (HI - LO), 0, 1).astype(np.float32)


def bbox(mask, spacing, margin_mm, shape):
    idx = np.argwhere(mask)
    m = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    lo, hi = np.maximum(idx.min(0) - m, 0), np.minimum(idx.max(0) + m + 1, shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def fidelity(orig, aug, liver, tumour, spacing):
    out = {}
    for name, sl in (("volume", tuple(slice(None) for _ in orig.shape)), ("liver_bbox", bbox(liver, spacing, 5, orig.shape)),
                     ("tumour_roi", bbox(tumour, spacing, 15, orig.shape))):
        a, b = win01(orig[sl]), win01(aug[sl])
        out[f"psnr_{name}"] = float(peak_signal_noise_ratio(a, b, data_range=1.0))
        if name != "volume":  # 3D SSIM of a full CT does not fit in memory; the liver box holds every changed voxel
            out[f"ssim_{name}"] = float(structural_similarity(a, b, data_range=1.0, win_size=7))
    return out


def patch_at(hu, seg_t, spacing, centre, size=64):
    """64^3 patch at 1 mm around `centre` (voxel), the geometry of the GAN training patches; returns (unit patch, mask)."""
    half = np.ceil((size / 2 + 2) / np.asarray(spacing)).astype(int)
    lo = np.round(centre).astype(int) - half
    hi = np.round(centre).astype(int) + half
    pad = [(max(0, -l), max(0, h - n)) for l, h, n in zip(lo, hi, hu.shape)]
    sl = tuple(slice(max(l, 0), min(h, n)) for l, h, n in zip(lo, hi, hu.shape))
    c = np.pad(hu[sl], pad, constant_values=-1000.0)
    m = np.pad(seg_t[sl].astype(np.float32), pad)
    z = np.asarray(spacing) / 1.0
    c, m = ndi.zoom(c, z, order=1), ndi.zoom(m, z, order=1) > 0.5
    o = [(s - size) // 2 for s in c.shape]
    sl = tuple(slice(a, a + size) for a in o)
    return S.to_unit(np.clip(c[sl], S.WIN_LO, S.WIN_HI)).astype(np.float32), m[sl]


def sharpness_boundary(x, m):
    """Mean gradient magnitude (HU / mm) on the tumour boundary shell."""
    h = S.from_unit(x)
    g = np.sqrt(sum(np.gradient(h, axis=a) ** 2 for a in range(3)))
    shell = ndi.binary_dilation(m, iterations=1) & ~ndi.binary_erosion(m, iterations=1)
    return float(g[shell].mean()) if shell.any() else np.nan


# ---------------- 1. insertion ----------------
d = S.load_npz(os.path.join(REPO, "results", "gan_lits130", "synthetic", "samples_100.npz"))
samples = {"patches": d["patches"], "masks": d["masks"], "work_mm": float(d["work_mm"]), "radius_mm": d["radius_mm"]}
H = hosts()
rows, syn_patches, syn_masks, shown = [], [], [], []
t0 = time.time()
k, bad, visit = 0, set(), 0
while k < N_IMAGES and visit < 3 * N_IMAGES:  # hosts in turn (every host once, then second images) until N_IMAGES pass QC
    src, hid, vp, sp = H[visit % len(H)]
    visit += 1
    if (src, hid) in bad:
        continue
    r_req = SIZES[k % len(SIZES)]
    vol, liver, spacing, aff = load(src, vp, sp)
    fit = liver_fit(vol, liver)
    if fit < 0.9:
        bad.add((src, hid))
        log.warning(f"{src}-{hid}: only {fit:.0%} of the liver mask has liver HU (mask does not match the image), host excluded")
        rows.append(dict(image=None, host=f"{src}-{hid}", status=f"excluded: liver mask fit {fit:.2f}"))
        continue
    p = I.InsertParams(target_radius_mm=r_req)
    vessels = I.segment_vessels(vol, liver, spacing, p)[0]
    info = None
    for attempt in range(6):  # a QC failure is reverted inside insert_tumours; retry with another sample / site
        aug, seg_new, _, infos = I.insert_tumours(vol, liver, spacing, samples, rng, 1, p, vessels=vessels)
        info = infos[0]
        if info["accepted"]:
            break
    if not info["accepted"]:
        log.warning(f"{src}-{hid}: no insertion passed QC ({info['reason']}), next host")
        rows.append(dict(image=None, host=f"{src}-{hid}", status=f"rejected: {info['reason']}"))
        continue
    k += 1
    name = f"synthetic_{k:02d}_{src}-{hid}_r{r_req:g}mm"
    S.save_nifti(S.to_int16(aug), aff, os.path.join(OUT, "images", name + ".nii.gz"), np.int16)
    S.save_nifti(seg_new, aff, os.path.join(OUT, "images", name + "_label.nii.gz"), np.uint8)
    tum = seg_new == 2
    fid = fidelity(vol, aug, liver > 0, tum, spacing)
    changed = aug.astype(np.int16) != S.to_int16(vol)
    edit = bbox(tum, spacing, 30, vol.shape)
    outside = changed.copy()
    outside[edit] = False
    cen = np.array(ndi.center_of_mass(tum))
    x, m = patch_at(aug, tum, spacing, cen)
    syn_patches.append(x)
    syn_masks.append(m)
    ring = (ndi.distance_transform_edt(~tum, sampling=spacing) <= 8) & ~tum & (liver > 0)
    row = dict(image=k, liver_mask_fit=round(fit, 3), file=name + ".nii.gz", host=f"{src}-{hid}", status="ok", attempts=attempt + 1, spacing_mm="x".join(f"{s:.2f}" for s in spacing),
               radius_requested_mm=r_req, radius_actual_mm=I.qc_row(0, 0, info)["radius_actual_mm"], sample=info.get("sample"), tumour_voxels=int(tum.sum()),
               tumour_hu=float(aug[tum].mean()), liver_ring_hu=float(aug[ring].mean()), contrast_hu=float(aug[tum].mean() - aug[ring].mean()),
               mass_effect=info.get("mass_effect", {}).get("intensity_applied", 0), changed_voxels=int(changed.sum()),
               changed_outside_edit_box=int(outside.sum()), **fid)
    rows.append(row)
    if len(shown) < 8:
        z = int(round(cen[2]))
        shown.append((name, vol[..., z], aug[..., z], tum[..., z], [int(round(c)) for c in cen[:2]]))
    log.info(f"#{k} {name}: r {row['radius_actual_mm']:.1f} mm, contrast {row['contrast_hu']:.1f} HU, PSNR roi {fid['psnr_tumour_roi']:.1f} dB, "
             f"SSIM roi {fid['ssim_tumour_roi']:.3f}, liver SSIM {fid['ssim_liver_bbox']:.4f} ({time.time() - t0:.0f}s)")
    del vol, aug, seg_new, liver, vessels, changed, outside

ok = [r for r in rows if r["status"] == "ok"]
with open(os.path.join(OUT, "metrics_per_image.csv"), "w", newline="") as f:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

# ---------------- 2. realism vs real LiTS tumours ----------------
z = S.load_npz(os.path.join(REPO, "results", "gan_lits130", "patches_all110.npz"))
real, real_m, cids = z["images"].astype(np.float32), z["masks"] > 0, z["meta"][:, 0]
pick = np.random.default_rng(0).choice(len(real), min(600, len(real)), replace=False)
real, real_m, cids = real[pick], real_m[pick], cids[pick]
vols = np.random.default_rng(1).permutation(np.unique(cids))
in_b = np.isin(cids, vols[: len(vols) // 2])
syn, syn_m = np.stack(syn_patches), np.stack(syn_masks)
feat = lambda x: np.stack([E.features(E.hu(p)) for p in x])
f_a, f_b, f_s = feat(real[~in_b]), feat(real[in_b]), feat(syn)
mu, sd = f_a.mean(0), f_a.std(0) + 1e-6
zf = lambda f: (f - mu) / sd
mid_a = E.mid(real[~in_b])


def realism(x, m, f, groups):
    st = T.tumour_statistics(x, m)
    ref = T.tumour_statistics(real[~in_b], real_m[~in_b])
    sm = E.ssim_matrix(E.mid(x), mid_a)
    acc, acc_sd = E.c2st(zf(f_a), zf(f), cids[~in_b], groups)
    return dict(n=int(len(x)), tumour_hu_mean=float(np.mean(st["vox"])), tumour_hu_w1=float(wasserstein_distance(ref["vox"], st["vox"])),
                contrast_hu_mean=float(np.mean(st["contrast"])), contrast_gap_hu=float(abs(np.mean(st["contrast"]) - np.mean(ref["contrast"]))),
                boundary_gradient_hu_per_mm=float(np.nanmean([sharpness_boundary(a, b) for a, b in zip(x, m)])),
                feature_frechet=float(E.frechet(zf(f_a), zf(f))), feature_mmd2=float(E.mmd2(zf(f_a), zf(f))),
                c2st_acc=acc, c2st_std=acc_sd, ssim_to_nearest_real=float(sm.max(1).mean()))


real_row = realism(real[in_b], real_m[in_b], f_b, cids[in_b])
syn_row = realism(syn, syn_m, f_s, None)
ref_a = T.tumour_statistics(real[~in_b], real_m[~in_b])
summary = {
    "images_written": len(ok), "hosts": {"chaos": sum(r["host"].startswith("chaos") for r in ok), "lits": sum(r["host"].startswith("lits") for r in ok)},
    "skipped_or_rejected": [r for r in rows if r["status"] != "ok"],
    "A_vs_original_scan": {k: {"mean": float(np.mean([r[k] for r in ok])), "min": float(np.min([r[k] for r in ok])), "max": float(np.max([r[k] for r in ok]))}
                           for k in ("psnr_volume", "psnr_liver_bbox", "ssim_liver_bbox", "psnr_tumour_roi", "ssim_tumour_roi", "contrast_hu",
                                     "changed_outside_edit_box")},
    "B_vs_real_lits_tumours": {"reference_real_A": dict(n=int((~in_b).sum()), tumour_hu_mean=float(np.mean(ref_a["vox"])),
                                                        contrast_hu_mean=float(np.mean(ref_a["contrast"])),
                                                        boundary_gradient_hu_per_mm=float(np.nanmean([sharpness_boundary(a, b) for a, b in zip(real[~in_b], real_m[~in_b])]))),
                               "real_B (perfect-synthesis baseline)": real_row, "synthetic (inserted)": syn_row},
    "seconds": round(time.time() - t0)}
S.write_json_atomic(os.path.join(OUT, "metrics_summary.json"), summary)

# ---------------- 3. figures ----------------
fig, axs = plt.subplots(len(shown), 4, figsize=(12, 3 * len(shown)), squeeze=False)
for r, (name, a, b, t, (cx, cy)) in enumerate(shown):
    h = 48
    sl = (slice(max(cx - h, 0), cx + h), slice(max(cy - h, 0), cy + h))
    for j, (img, title) in enumerate(((a, "original (healthy)"), (b, "synthetic"), (b, "synthetic + tumour label"), (b - a, "difference"))):
        ax = axs[r, j]
        ax.imshow(img[sl].T, cmap="gray" if j < 3 else "RdBu_r", vmin=LO if j < 3 else -80, vmax=HI if j < 3 else 80, origin="lower")
        if j == 2:
            ax.contour(t[sl].T.astype(float), levels=[0.5], colors="r", linewidths=0.8)
        ax.set_title(f"{name}\n{title}" if j == 0 else title, fontsize=8)
        ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "original_vs_synthetic.png"), dpi=110)
plt.close(fig)
E.plot_montage({"real LiTS tumours": (real[in_b][:10], real_m[in_b][:10]), "inserted synthetic tumours": (syn[:10], syn_m[:10])},
               os.path.join(OUT, "figures", "real_vs_inserted_patches.png"), n=min(10, len(syn)))
print(json.dumps(summary, indent=1, default=str)[:6000], flush=True)
log.info(f"ALL DONE: {len(ok)} images in {(time.time() - t0) / 60:.0f} min")
