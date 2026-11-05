"""Figures fig01-fig08 for the dataset report (no env vars).

    python make_figures.py <raw_dir> <out_dir> <fig_dir>

Reads <out_dir>/stats.json (from dataset_analysis.py) and, read-only, four raw volumes for the example slices (fig08).
Writes <fig_dir>/fig01_geometry.png ... fig08_examples.png.
"""
import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIVER, C_TUMOR, C_BODY, C_GRID = "#2a9d8f", "#d62828", "#6c757d", "#e5e5e5"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": C_GRID, "axes.axisbelow": True, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})


def save(fig, d, name):
    fig.tight_layout()
    fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main(raw_dir, out_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    S = json.load(open(os.path.join(out_dir, "stats.json")))
    rows = S["per_volume"]
    A = lambda k: np.array([r[k] for r in rows if r.get(k) is not None], float)
    sx, sz = (np.array([r["spacing"][i] for r in rows]) for i in (0, 2))  # LiTS in-plane spacing is isotropic (sx == sy), so sy is not plotted
    nz = A("n_slices")

    # 1 geometry
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].hist(sx, bins=20, color=C_BODY); ax[0].set(title="In-plane pixel spacing", xlabel="mm", ylabel="volumes")
    ax[1].hist(sz, bins=np.arange(0, 6.1, 0.25), color=C_BODY); ax[1].set(title="Slice thickness (z spacing)", xlabel="mm")
    ax[2].hist(nz, bins=25, color=C_BODY); ax[2].set(title="Axial slices per volume", xlabel="slices")
    save(fig, fig_dir, "fig01_geometry.png")

    # 2 burden
    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    ax[0].hist(A("liver_ml"), bins=25, color=C_LIVER); ax[0].set(title="Liver volume", xlabel="mL", ylabel="volumes")
    tm = A("tumor_ml"); tm_pos = tm[tm > 0]
    ax[1].hist(np.log10(tm_pos), bins=25, color=C_TUMOR)
    ax[1].set(title=f"Tumor volume (n={len(tm_pos)} with tumor)", xlabel="log10 mL")
    nt = A("n_tumors")
    ax[2].hist(nt, bins=np.arange(0, nt.max() + 2) - 0.5, color=C_TUMOR); ax[2].set(title="Tumor components per volume", xlabel="count")
    ax[3].hist(100 * A("tumor_frac_of_liver"), bins=25, color=C_TUMOR); ax[3].set(title="Tumor volume / liver volume", xlabel="%")
    save(fig, fig_dir, "fig02_burden.png")

    # 3 lesion size + Lorenz
    comps = [c for r in rows for c in r["components"]]
    d = np.array([c["eq_diam_mm"] for c in comps]); ml = np.array([c["ml"] for c in comps])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(np.log10(d), bins=40, color=C_TUMOR)
    for mm in (10, 64):
        ax[0].axvline(np.log10(mm), color="k", ls="--", lw=1); ax[0].text(np.log10(mm), ax[0].get_ylim()[1] * 0.9, f" {mm} mm", fontsize=8)
    ax[0].set(title=f"Lesion equivalent diameter (n={len(d)})", xlabel="log10 mm", ylabel="lesions")
    o = np.argsort(-ml); cum = np.cumsum(ml[o]) / ml.sum()
    ax[1].plot(np.arange(1, len(o) + 1) / len(o) * 100, cum * 100, color=C_TUMOR)
    ax[1].set(title="Few lesions carry most of the tumor volume", xlabel="% of lesions (largest first)", ylabel="% of tumor volume")
    ax[2].hist(A("largest_tumor_share") * 100, bins=20, color=C_TUMOR)
    ax[2].set(title="Share of tumor volume in largest lesion", xlabel="% (per volume)", ylabel="volumes")
    save(fig, fig_dir, "fig03_lesion_size.png")

    # 4 HU distributions
    edges = np.array(S["hu_bins"], float); mid = (edges[:-1] + edges[1:]) / 2
    H = {k: np.array(v, float) for k, v in S["hu_hist"].items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for k, c, lab in (("liver", C_LIVER, "liver (non-tumor)"), ("tumor", C_TUMOR, "tumor")):
        m = (mid >= -200) & (mid <= 250)
        ax[0].plot(mid[m], H[k][m] / H[k].sum() / 8, color=c, label=lab)
        ax[0].fill_between(mid[m], H[k][m] / H[k].sum() / 8, color=c, alpha=0.15)
    ax[0].axvspan(-200, 250, color="none")  # invisible: only extends the autoscaled x-range to the window edges
    ax[0].legend(); ax[0].set(title="HU density inside the [-200, 250] window", xlabel="HU", ylabel="density")
    ax[1].semilogy(mid, np.maximum(H["body"], 1), color=C_BODY)
    ax[1].set(title="Whole-volume HU histogram (log counts)", xlabel="HU")
    ax[1].axvspan(-200, 250, color=C_LIVER, alpha=0.12); ax[1].text(-190, ax[1].get_ylim()[1] * 0.3, "liver window", fontsize=8)
    save(fig, fig_dir, "fig04_hu.png")

    # 5 contrast
    tmn = np.array([r["tumor_hu_mean"] for r in rows if r.get("contrast_hu") is not None])
    lm2 = np.array([r["liver_hu_mean"] for r in rows if r.get("contrast_hu") is not None])
    con, cnr = A("contrast_hu"), A("cnr")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].scatter(lm2, tmn, s=14, color=C_TUMOR, alpha=0.7); lim = [min(lm2.min(), tmn.min()) - 5, max(lm2.max(), tmn.max()) + 5]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set(title="Per-volume mean HU: tumor vs liver", xlabel="liver mean HU", ylabel="tumor mean HU")
    ax[1].hist(con, bins=25, color=C_TUMOR); ax[1].axvline(0, color="k", lw=1)
    ax[1].set(title="Tumor minus liver mean HU (per volume)", xlabel="HU", ylabel="volumes")
    ax[2].hist(cnr, bins=25, color=C_BODY); ax[2].axvline(0, color="k", lw=1); ax[2].set(title="Contrast-to-noise (contrast / liver SD)", xlabel="CNR")
    save(fig, fig_dir, "fig05_contrast.png")

    # 6 HU range per volume (what the segmenter's per-volume min-max sees)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(A("hu_min"), bins=30, color=C_BODY); ax[0].set(title="Per-volume minimum HU", xlabel="HU", ylabel="volumes")
    ax[1].hist(A("hu_max"), bins=30, color=C_BODY); ax[1].set(title="Per-volume maximum HU", xlabel="HU")
    save(fig, fig_dir, "fig06_hu_range.png")

    # 7 class balance
    tot = np.array([np.prod(r["shape"]) for r in rows], float)
    liv = A("liver_vox") - A("tumor_vox"); tum = A("tumor_vox")
    frac = np.array([(tot.sum() - liv.sum() - tum.sum()), liv.sum(), tum.sum()]) / tot.sum() * 100
    sl_l, sl_t = A("slices_liver"), A("slices_tumor")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(["background", "liver", "tumor"], frac, color=[C_BODY, C_LIVER, C_TUMOR]); ax[0].set_yscale("log")
    for i, f in enumerate(frac):
        ax[0].text(i, f * 1.15, f"{f:.2f}%", ha="center", fontsize=8)
    ax[0].set(title="Voxel class balance (all volumes)", ylabel="% of voxels (log)")
    ax[1].bar(["no liver", "liver, no tumor", "tumor"], [nz.sum() - sl_l.sum(), sl_l.sum() - sl_t.sum(), sl_t.sum()] / nz.sum() * 100,
              color=[C_BODY, C_LIVER, C_TUMOR])
    ax[1].set(title="Axial slices by content", ylabel="% of slices")
    save(fig, fig_dir, "fig07_class_balance.png")

    # 8 examples
    withT = [(r["name"], r) for r in rows if r["tumor_ml"] > 0]
    pick = {
        "largest tumor burden": max(withT, key=lambda x: x[1]["tumor_ml"]),
        "median tumor burden": min(withT, key=lambda x: abs(x[1]["tumor_ml"] - np.median([w[1]["tumor_ml"] for w in withT]))),
        "thickest slices": max(withT, key=lambda x: x[1]["spacing"][2]),
        "thinnest slices": min(withT, key=lambda x: x[1]["spacing"][2]),
    }
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, (title, (name, r)) in zip(axs, pick.items()):
        vp = os.path.join(raw_dir, name); sp = vp.replace("volume-", "segmentation-")
        seg = np.asarray(nib.load(sp).dataobj).astype(np.uint8)
        k = int(np.argmax((seg == 2).sum(axis=(0, 1))))
        img = np.asarray(nib.load(vp).dataobj[:, :, k]).astype(np.float32)
        a.imshow(np.rot90(np.clip(img, -150, 250)), cmap="gray", vmin=-150, vmax=250)
        a.contour(np.rot90(seg[:, :, k] >= 1), levels=[0.5], colors=[C_LIVER], linewidths=0.8)
        a.contour(np.rot90(seg[:, :, k] == 2), levels=[0.5], colors=[C_TUMOR], linewidths=0.9)
        a.set_title(f"{title}\n{name.replace('.nii','')}, z={r['spacing'][2]:.1f} mm, {r['tumor_ml']:.0f} mL", fontsize=8)
        a.axis("off"); a.grid(False)
    save(fig, fig_dir, "fig08_examples.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
