"""Evaluation of a finished Syn_Img_Pipeline_fixed.py run (NIfTI source): loss curves per GAN and real-vs-synthetic image metrics.

    LITS_DATA_DIR=... LITS_GAN_RESULTS_DIR=... python evaluate_synthesis.py

Reads <results>/train_metrics.json (loss_history; falls back to parsing pipeline.log), patches.npz, checkpoints/gan_last.pt,
synthetic/samples.npz and aug/. Writes <results>/evaluation/{evaluation.json, *.png}.

Real vs synthetic (tumour patches, 64^3 at 1 mm, compared in HU):
  every generator (dcgan 1, dcgan 2, wgan-gp, aggregator) is sampled N times and compared with the real tumour patches; the
  reference row "real B" is a second set of REAL patches from other volumes (volume-wise split A/B), i.e. the value a perfect
  generator would reach. Metrics: HU histogram Wasserstein-1 distance, radial power-spectrum log distance, gradient magnitude,
  Frechet distance + RBF-MMD^2 on standardised hand-crafted features (histogram, percentiles, gradients, spectrum; NOT an
  Inception FID), C2ST (random-forest real-vs-other accuracy, 5-fold CV; 0.5 = indistinguishable), SSIM of the central axial
  slice (to the nearest real patch = realism / memorisation, mean pairwise inside the set = diversity, high = mode collapse),
  mask-head gate acceptance, tumour volume and tumour-minus-surround contrast (HU).
Augmented volumes: inserted-tumour HU / contrast vs the host liver, compared with the real tumours of the same hosts, and the
  fraction of voxels that changed.
"""
import os
import re
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import Syn_Img_Pipeline_fixed as S

RES = S.RESULTS_DIR
OUT = os.path.join(RES, "evaluation")
N_SYN = int(os.environ.get("LITS_EVAL_N", 256))
GENS = ["dc1", "dc2", "wgan", "aggregator"]
LABEL = {"real_B": "real B (reference)", "dc1": "DCGAN 1", "dc2": "DCGAN 2", "wgan": "WGAN-GP", "aggregator": "Aggregator"}
LOSS_GROUPS = [("DCGAN 1", [("d1", "discriminator"), ("g1", "generator")]),
               ("DCGAN 2", [("d2", "discriminator"), ("g2", "generator")]),
               ("WGAN-GP", [("w_critic", "critic (incl. GP)"), ("w_g", "generator"), ("w_gp", "gradient penalty")]),
               ("Aggregator GAN", [("ag_d", "discriminator"), ("ag_g", "generator (adv.)"), ("ag_l1", "L1 to generator mean")]),
               ("Mask head", [("mask", "BCE + Dice")])]


# -----------------------------
# Loss curves
# -----------------------------
def loss_history():
    m = S.read_json(os.path.join(RES, "train_metrics.json"))
    if m.get("loss_history"):
        return m["loss_history"]
    hist = {}  # fallback: the "epoch k/N (..s) key=val ..." lines of pipeline.log; the last run of an epoch wins
    with open(os.path.join(RES, "pipeline.log")) as f:
        for line in f:
            r = re.search(r"epoch (\d+)/\d+ \((\d+)s\) (.*)$", line)
            if r:
                hist[int(r.group(1))] = dict({k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+(?:e-?\d+)?)", r.group(3))},
                                             epoch=int(r.group(1)), seconds=float(r.group(2)))
    return [hist[k] for k in sorted(hist)]


def plot_losses(hist, path):
    fig, axs = plt.subplots(1, len(LOSS_GROUPS), figsize=(4.2 * len(LOSS_GROUPS), 3.6))
    ep = [h["epoch"] for h in hist]
    for ax, (title, keys) in zip(axs, LOSS_GROUPS):
        for k, lab in keys:
            pts = [(e, h[k]) for e, h in zip(ep, hist) if k in h]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, lw=1.5, label=lab)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("mean loss per epoch")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------
# Image features and metrics
# -----------------------------
def hu(x):
    return S.from_unit(np.asarray(x, np.float32))


def radial_spectrum(x, nb=16):
    """Radially averaged log power spectrum of one 3D patch (mean removed)."""
    p = np.abs(np.fft.fftshift(np.fft.fftn(x - x.mean()))) ** 2
    c = np.array(x.shape) // 2
    r = np.sqrt(sum((np.arange(n) - ci).reshape([-1 if a == ax else 1 for a in range(3)]) ** 2 for ax, (n, ci) in enumerate(zip(x.shape, c))))
    bins = np.linspace(1, x.shape[0] // 2, nb + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], p.ravel()[ok], nb) / np.maximum(np.bincount(idx[ok], minlength=nb), 1)
    return np.log10(s + 1e-6)


def features(h):
    """Hand-crafted per-patch features (HU input): histogram, percentiles, gradient / Laplacian statistics, power spectrum."""
    hist = np.histogram(np.clip(h, S.WIN_LO, S.WIN_HI), bins=24, range=(S.WIN_LO, S.WIN_HI))[0] / h.size
    g = np.sqrt(sum(a ** 2 for a in np.gradient(h)))
    lap = ndi.laplace(h)
    return np.concatenate([hist, np.percentile(h, [5, 25, 50, 75, 95]), [h.mean(), h.std(), g.mean(), g.std(), lap.var() ** 0.5],
                           radial_spectrum(h)])


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    eye = 1e-6 * np.eye(a.shape[1])  # some histogram bins are constant -> singular covariance
    s1, s2 = np.cov(a, rowvar=False) + eye, np.cov(b, rowvar=False) + eye
    cs = linalg.sqrtm(s1 @ s2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * np.real(cs)))


def mmd2(a, b):
    z = np.concatenate([a, b])
    d = ((z[:, None] - z[None]) ** 2).sum(-1)
    k = np.exp(-d / np.median(d[d > 0]))
    n = len(a)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (len(b) * (len(b) - 1)) - 2 * kxy.mean())


def c2st(a, b, ga, gb=None, seed=0):
    """Real-vs-other classifier accuracy. Folds are grouped by volume (ga / gb: volume id per patch; None = independent
    samples, e.g. generated): patches of one volume are strongly correlated, and a plain K-fold would let the classifier
    recognise the volume instead of telling the two sets apart."""
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    ia, ib = rng.permutation(len(a))[:n], rng.permutation(len(b))[:n]
    x = np.concatenate([a[ia], b[ib]])
    y = np.r_[np.zeros(n), np.ones(n)]
    g = np.r_[np.asarray(ga)[ia], np.asarray(gb)[ib] if gb is not None else 10 ** 6 + rng.integers(0, 5, n)]
    acc = cross_val_score(RandomForestClassifier(300, random_state=seed, n_jobs=-1), x, y, groups=g, cv=GroupKFold(5))
    return float(acc.mean()), float(acc.std())


def _ssim_maps(x, y, win=7):
    """Gaussian-window SSIM of 2D slices x (N,H,W) vs y (N,H,W) in the [-1,1] window (data range 2)."""
    g = torch.exp(-(torch.arange(win) - win // 2).float() ** 2 / (2 * 1.5 ** 2))
    k = (g[:, None] * g[None]) / g.sum() ** 2
    k = k[None, None]
    f = lambda t: F.conv2d(t[:, None], k)[:, 0]
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).mean(dim=(1, 2))


def ssim_matrix(a, b, chunk=4096):
    """SSIM of every pair (central axial slices). a (n,P,P), b (m,P,P) -> (n,m)."""
    a, b = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    ii, jj = torch.meshgrid(torch.arange(len(a)), torch.arange(len(b)), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    out = torch.cat([_ssim_maps(a[ii[s:s + chunk]], b[jj[s:s + chunk]]) for s in range(0, len(ii), chunk)])
    return out.view(len(a), len(b)).numpy()


def mid(x):
    return np.ascontiguousarray(np.asarray(x, np.float32)[:, :, :, x.shape[-1] // 2])


def tumour_stats(patches, masks):
    """Tumour volume (voxels = mm^3 at 1 mm) and mean HU inside the tumour minus a 2-10 mm surrounding ring."""
    vol, con = [], []
    for p, m in zip(patches, masks):
        m = m > 0
        if not m.any():
            continue
        d = ndi.distance_transform_edt(~m)
        ring = (d > 2) & (d <= 10)
        h = hu(p)
        vol.append(int(m.sum()))
        con.append(float(h[m].mean() - h[ring].mean()))
    return np.array(vol), np.array(con)


def gate(model, x, min_vox):
    """Mask head + clean_mask on generated patches (x (n,P,P,P) in [-1,1]) -> (masks for accepted, acceptance rate)."""
    acc, masks = 0, np.zeros(x.shape, np.uint8)
    with torch.no_grad():
        model.train(False)
        for s in range(0, len(x), 8):
            prob = torch.sigmoid(model.mask(torch.from_numpy(x[s:s + 8])[:, None].to(model.device)))[:, 0].cpu().numpy()
            for j, p in enumerate(prob):
                m, _ = S.clean_mask(p, min_vox=min_vox)
                if m is not None:
                    masks[s + j], acc = m, acc + 1
    return masks, acc / len(x)


def sample_generators(model, n, seed):
    torch.manual_seed(seed)
    out = {g: [] for g in GENS}
    for s in range(0, n, 8):
        for k, v in model.sample_all(min(8, n - s)).items():
            out[k].append(v[:, 0].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


# -----------------------------
# Augmented volumes
# -----------------------------
def eval_augmented(data_dir, max_cases=None):
    rows = []
    files = sorted(glob.glob(os.path.join(RES, "aug", "aug_volume-*.nii")), key=S.case_id)[:max_cases]
    for f in files:
        cid = S.case_id(f)
        (vp, sp_), = S.pair_cases(data_dir, [cid]).values()
        v0, s0, _, _ = S.load_case(vp, sp_)
        v1 = np.asarray(nib.load(f).dataobj).astype(np.float32)
        s1 = np.rint(np.asarray(nib.load(os.path.join(RES, "aug", f"aug_segmentation-{cid}.nii")).dataobj)).astype(np.uint8)
        new = (s1 == 2) & (s0 != 2)
        if not new.any():
            continue
        liver = s1 == 1
        d = ndi.distance_transform_edt(~new)
        ring = (d > 2) & (d <= 6) & liver  # host liver right around the inserted tumour (voxel units)
        changed = np.abs(v1 - S.to_int16(v0)) > 0
        r = dict(case=cid, new_tumor_vox=int(new.sum()), inserted_hu=float(v1[new].mean()), host_liver_hu=float(v0[s0 == 1].mean()),
                 inserted_minus_ring_hu=float(v1[new].mean() - v1[ring].mean()) if ring.any() else None,
                 changed_vox_frac=float(changed.mean()), changed_outside_liver=int((changed & (s0 == 0)).sum()),
                 labels_changed_outside_new=int(((s1 != s0) & ~new).sum()))
        if (s0 == 2).any():
            real = s0 == 2
            dr = ndi.distance_transform_edt(~real)
            rr = (dr > 2) & (dr <= 6) & (s0 == 1)
            r.update(real_tumor_hu=float(v0[real].mean()), real_minus_ring_hu=float(v0[real].mean() - v0[rr].mean()) if rr.any() else None)
        rows.append(r)
        print(f"aug case {cid}: +{r['new_tumor_vox']} vox, inserted {r['inserted_hu']:.1f} HU, liver {r['host_liver_hu']:.1f} HU", flush=True)
    return rows


# -----------------------------
# Figures
# -----------------------------
def plot_montage(sets, path, n=8):
    rows = list(sets)
    fig, axs = plt.subplots(len(rows), n, figsize=(1.7 * n, 1.8 * len(rows)), squeeze=False)
    for r, k in enumerate(rows):
        x, m = sets[k]
        for j in range(n):
            ax = axs[r][j]
            ax.axis("off")
            z = x.shape[-1] // 2
            ax.imshow(x[j][:, :, z].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
            if m is not None and m[j][:, :, z].any():
                ax.contour(m[j][:, :, z].T.astype(float), levels=[0.5], colors="r", linewidths=0.7)
            if j == 0:
                ax.set_title(k, fontsize=9, loc="left")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_distributions(hu_vox, spectra, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(S.WIN_LO, S.WIN_HI, 71)
    for k, v in hu_vox.items():
        axs[0].hist(np.clip(v, S.WIN_LO, S.WIN_HI), bins=bins, density=True, histtype="step", lw=1.6 if k == "real" else 1.1, label=k)
        axs[1].plot(np.mean(spectra[k], 0), lw=1.6 if k == "real" else 1.1, label=k)
    axs[0].set(title="HU distribution (voxels, window clipped)", xlabel="HU", ylabel="density")
    axs[1].set(title="radial power spectrum", xlabel="frequency bin (low -> high)", ylabel="log10 power")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_bars(table, path):
    keys = [("hu_wasserstein", "HU Wasserstein-1 (HU)"), ("spectrum_log_dist", "spectrum log distance"), ("feature_frechet", "feature Frechet distance"),
            ("c2st_acc", "C2ST accuracy (0.5 ideal)"), ("ssim_to_nearest_real", "SSIM to nearest real"), ("ssim_pairwise_diversity", "pairwise SSIM (lower = diverse)")]
    names = list(table)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.4))
    for ax, (k, t) in zip(axs, keys):
        vals = [table[n].get(k, np.nan) for n in names]
        ax.bar(range(len(names)), vals, color=["#888" if n == "real_B" else "C%d" % i for i, n in enumerate(names)])
        ax.set_xticks(range(len(names)), [LABEL.get(n, n) for n in names], rotation=40, ha="right", fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if k == "c2st_acc":
            ax.axhline(0.5, color="k", ls="--", lw=0.8)
            ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    S.set_seed(S.SEED)
    res = {"results_dir": RES}
    hist = loss_history()
    plot_losses(hist, os.path.join(OUT, "losses_per_gan.png"))
    res["loss_history"] = hist
    print(f"loss curves: {len(hist)} epochs -> losses_per_gan.png", flush=True)

    d = S.load_npz(os.path.join(RES, "patches.npz"))
    imgs, masks, cids = d["images"].astype(np.float32), d["masks"], d["meta"][:, 0]
    work = float(d["work_mm"])
    # volume-wise split of the real patches: A = reference set, B = "perfect generator" baseline
    rng = np.random.default_rng(S.SEED)
    vols = rng.permutation(np.unique(cids))
    b_vols = set(vols[: len(vols) // 2].tolist())
    in_b = np.isin(cids, list(b_vols))
    real, real_m = imgs, masks
    model = S.MultiGAN.load(os.path.join(RES, "checkpoints", "gan_last.pt"), torch.device("cpu"))
    n = min(N_SYN, len(real))
    syn = sample_generators(model, n, S.SEED + 1)
    min_vox = S.vox_at(S.MIN_MASK_VOX, work)
    sets = {"real_B": (imgs[in_b], masks[in_b])}
    gate_rate = {}
    for g in GENS:
        m, gate_rate[g] = gate(model, syn[g], min_vox)
        sets[g] = (syn[g], m)
        print(f"{g}: mask-gate acceptance {gate_rate[g]:.2%}", flush=True)

    feat = lambda x: np.stack([features(hu(p)) for p in x])
    print("features ...", flush=True)
    f_real, f_a = feat(real), None
    f_a = f_real[~in_b]
    mu, sd = f_real.mean(0), f_real.std(0) + 1e-6
    z = lambda f: (f - mu) / sd
    spec = lambda x: np.stack([radial_spectrum(hu(p)) for p in x])
    s_real = spec(real)
    sub = lambda x, k=40000: np.random.default_rng(0).choice(hu(x).ravel(), k, replace=False)
    hu_real = sub(real)
    mid_real, mid_a = mid(real), mid(imgs[~in_b])
    rv, rc = tumour_stats(real, real_m)
    res["real"] = dict(patches=int(len(real)), volumes=int(len(np.unique(cids))), hu_mean=float(hu(real).mean()), hu_std=float(hu(real).std()),
                       tumour_vox_median=float(np.median(rv)), tumour_contrast_hu_mean=float(rc.mean()),
                       ssim_pairwise_diversity=float(np.mean(ssim_matrix(mid_real[:n], mid_real[:n])[np.triu_indices(min(n, len(real)), 1)])))
    table, hu_vox, spectra = {}, {"real": hu_real}, {"real": s_real}
    for k, (x, m) in sets.items():
        # real_B is compared with real A only (otherwise it would be compared with itself); generators with all real patches
        ref_f, ref_mid, ref_hu, ref_s = (f_a, mid_a, sub(imgs[~in_b]), spec(imgs[~in_b])) if k == "real_B" else (f_real, mid_real, hu_real, s_real)
        fx, sx = feat(x), spec(x)
        sm = ssim_matrix(mid(x), ref_mid)
        own = ssim_matrix(mid(x[:n]), mid(x[:n]))
        v, c = tumour_stats(x, m)
        acc, acc_sd = c2st(z(ref_f), z(fx), cids[~in_b], cids[in_b]) if k == "real_B" else c2st(z(ref_f), z(fx), cids)
        table[k] = dict(n=int(len(x)), hu_mean=float(hu(x).mean()), hu_std=float(hu(x).std()),
                        hu_wasserstein=float(wasserstein_distance(ref_hu, sub(x))),
                        spectrum_log_dist=float(np.sqrt(((ref_s.mean(0) - sx.mean(0)) ** 2).mean())),
                        grad_mean=float(fx[:, 24 + 5 + 2].mean()), feature_frechet=frechet(z(ref_f), z(fx)), feature_mmd2=mmd2(z(ref_f), z(fx)),
                        c2st_acc=acc, c2st_std=acc_sd, ssim_to_nearest_real=float(sm.max(1).mean()),
                        ssim_pairwise_diversity=float(own[np.triu_indices(len(own), 1)].mean()),
                        mask_gate_acceptance=gate_rate.get(k, 1.0), tumour_vox_median=float(np.median(v)) if len(v) else None,
                        tumour_contrast_hu_mean=float(c.mean()) if len(c) else None)
        hu_vox[LABEL[k]], spectra[LABEL[k]] = sub(x), sx
        print(f"{k}: " + " ".join(f"{a}={b:.3f}" for a, b in table[k].items() if isinstance(b, float)), flush=True)
    res["real_vs_synthetic"] = table
    res["real_grad_mean"] = float(f_real[:, 24 + 5 + 2].mean())
    plot_montage({"real": (real[:8], real_m[:8]), **{LABEL[g]: sets[g] for g in GENS}}, os.path.join(OUT, "real_vs_synthetic_montage.png"))
    plot_distributions(hu_vox, spectra, os.path.join(OUT, "hu_and_spectrum.png"))
    plot_metric_bars(table, os.path.join(OUT, "metrics_bars.png"))

    if os.path.isdir(os.path.join(RES, "aug")):
        print("augmented volumes ...", flush=True)
        rows = eval_augmented(S.DATA_DIR)
        res["augmented"] = rows
        if rows:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            a = [r["inserted_minus_ring_hu"] for r in rows if r.get("inserted_minus_ring_hu") is not None]
            b = [r["real_minus_ring_hu"] for r in rows if r.get("real_minus_ring_hu") is not None]
            ax.boxplot([b, a], tick_labels=[f"real tumours (n={len(b)})", f"inserted tumours (n={len(a)})"])
            ax.axhline(0, color="k", lw=0.6)
            ax.set(ylabel="tumour mean HU minus surrounding liver (HU)", title="tumour contrast in the augmented hosts")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT, "augmented_contrast.png"), dpi=120)
            plt.close(fig)
    S.write_json_atomic(os.path.join(OUT, "evaluation.json"), res)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
