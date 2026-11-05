"""Training and generation improvements for Syn_Img_Pipeline_fixed.py, ported from the reference synthesis.py.

    python syn_training_improved.py                       # LITS_GAN_STAGE=all | prepare | train | generate (same env vars as the pipeline)
    LITS_GAN_ES_PATIENCE=8 python syn_training_improved.py

Syn_Img_Pipeline_fixed.py is not modified. This module subclasses its MultiGAN and reuses its data preparation, PatchStore, MaskPool,
mask-head evaluation and montages; the insert stage stays in Syn_Img_Pipeline_fixed.py (or insertion_improved.py).

What changes compared with Syn_Img_Pipeline_fixed.py (and where it comes from in synthesis.py):
  1. Differentiable augmentation of every discriminator / critic input, real and fake, in the D and the G step (synthesis.diffaug, after
     DiffAugment and StyleGAN2-ADA). The LiTS train part yields a few hundred tumour patches; without augmentation the discriminators
     memorise them. Flips of the three axes, a 90-degree in-plane rotation, brightness / contrast / noise jitter. No translation and no
     cutout: patches are tumour-centred and a cutout could erase the tumour.
  2. Exponential moving average of the generation path (dc1_g, dc2_g, w_g, aggr) with the warm-up min(decay, (1+n)/(10+n))
     (synthesis.Ema, after PGGAN). Sampling, montages, quality scores and the "modules" of the checkpoint use the averaged weights, so
     Syn_Img_Pipeline_fixed.MultiGAN.load and evaluate_synthesis.py read an improved checkpoint unchanged. The live weights are stored
     next to them ("live_modules") for resume.
  3. WGAN-GP drift term eps * E[C(real)^2], eps = 1e-3 (synthesis.wgan_critic_loss, after PGGAN), keeps the critic output from drifting.
  4. A per-epoch quality score and early stopping (synthesis.quality_score / early_stop_decision): Wasserstein-1 distance of tumour-voxel
     HU, plus the gap in tumour-minus-ring contrast, plus the gap in sharpness (mean neighbouring-voxel difference, HU per voxel), between
     generated patches (aggregator, fixed latents, labelled by the mask head) and real tumour patches of the held-out volumes. Lower is
     better. The best epoch is written to checkpoints/gan_best.pt. With LITS_GAN_ES_PATIENCE > 0 training stops after that many epochs
     without improvement (never before LITS_GAN_ES_MIN_EPOCHS). This is a consistency score, not a measure of realism.
  5. Size-controlled generation (synthesis.SIZE_RANGES_MM): the multi-GAN is unconditional, so size is controlled by accept-until-target
     per size class. Each candidate is labelled by the mask head and cleaned (Syn_Img_Pipeline_fixed.clean_mask); its equivalent
     spherical radius r = (3V / 4 pi)^(1/3) puts it in small (2.5-6 mm], medium (6-14 mm] or large (14-28 mm]. samples.npz gains
     radius_mm and size_class arrays; patches / masks / work_mm keep their meaning, so Syn_Img_Pipeline_fixed.run_insert still reads it.

Env vars on top of the pipeline's: LITS_GAN_DIFFAUG_P (0.5, 0 = off), LITS_GAN_EMA (0.999, 0 = off), LITS_GAN_WGAN_DRIFT (0.001),
LITS_GAN_ES_PATIENCE (0 = off), LITS_GAN_ES_MIN_EPOCHS (10), LITS_GAN_ES_MIN_DELTA (1.0 HU), LITS_GAN_SHARP_WEIGHT (1.0),
LITS_GAN_QUALITY_N (32 generated patches per score), LITS_GAN_STEPS_PER_EPOCH (0 = all batches; N = a fresh random subset of N batches
per epoch, for CPU budgets), LITS_GAN_PER_CLASS (8 samples per size class), LITS_GAN_CLASSES (small,medium,large).
"""
import os
import sys
import time
import math
import random
import contextlib
import logging
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import Syn_Img_Pipeline_fixed as S  # noqa: E402

DIFFAUG_P = float(os.environ.get("LITS_GAN_DIFFAUG_P", 0.5))
EMA_DECAY = float(os.environ.get("LITS_GAN_EMA", 0.999))
WGAN_DRIFT = float(os.environ.get("LITS_GAN_WGAN_DRIFT", 1e-3))
ES_PATIENCE = int(os.environ.get("LITS_GAN_ES_PATIENCE", 0))
ES_MIN_EPOCHS = int(os.environ.get("LITS_GAN_ES_MIN_EPOCHS", 10))
ES_MIN_DELTA = float(os.environ.get("LITS_GAN_ES_MIN_DELTA", 1.0))
SHARP_WEIGHT = float(os.environ.get("LITS_GAN_SHARP_WEIGHT", 1.0))
QUALITY_N = int(os.environ.get("LITS_GAN_QUALITY_N", 32))
STEPS_PER_EPOCH = int(os.environ.get("LITS_GAN_STEPS_PER_EPOCH", 0))  # 0 = every batch; N = a fresh random N batches per epoch
PER_CLASS = int(os.environ.get("LITS_GAN_PER_CLASS", 8))
CLASSES = tuple(c.strip() for c in os.environ.get("LITS_GAN_CLASSES", "small,medium,large").split(",") if c.strip())
RING_MM = (3.0, 8.0)  # surrounding ring of the contrast statistic: distance to the mask in (3, 8] mm
VOXELS_PER_PATCH = 2000  # tumour voxels kept per patch for the pooled HU distribution
QUALITY_SEED = 12345  # fixed latents: every epoch is scored on the same z

#: size classes by equivalent spherical radius (mm), as in synthesis.SIZE_RANGES_MM; small includes its lower edge
SIZE_BINS_MM = {"small": (2.5, 6.0), "medium": (6.0, 14.0), "large": (14.0, 28.0)}

logger = logging.getLogger("gan_fixed")


# -----------------------------
# Differentiable augmentation
# -----------------------------
def diffaug(x, p=DIFFAUG_P):
    """Augment a (B,1,X,Y,Z) batch in [-1,1] before a discriminator; differentiable, so the same call is used in the G step.
    Geometry (each with probability p, one draw for the whole batch): flips of the three axes, a 90-degree rotation in the (X,Y) plane
    (Z is the slice axis of the source volumes and is never rotated into the plane). Intensity (each sample with probability p):
    brightness shift +-0.1, contrast 0.9-1.1 around the sample mean, Gaussian noise 0.02 (all on the [-1,1] scale)."""
    if p <= 0:
        return x
    for ax in (2, 3, 4):
        if random.random() < p:
            x = x.flip(ax)
    if random.random() < p:
        x = torch.rot90(x, random.randint(1, 3), (2, 3))
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    on = lambda: (torch.rand(shape, device=x.device) < p).to(x.dtype)  # noqa: E731
    x = x + on() * (torch.rand(shape, device=x.device) - 0.5) * 0.2
    mean = x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
    x = (x - mean) * (1.0 + on() * (torch.rand(shape, device=x.device) - 0.5) * 0.2) + mean
    return x + on() * 0.02 * torch.randn_like(x)


# -----------------------------
# EMA of the generation path
# -----------------------------
class Ema:
    """Exponential moving average of named modules; swap() puts the averaged weights into the live modules in place (the optimisers
    keep their parameter objects) and restores the live weights afterwards. Integer buffers (BatchNorm num_batches_tracked) are copied."""

    def __init__(self, modules, names, decay):
        self.names, self.decay, self.n = list(names), float(decay), 0
        self.shadow = {k: {n: t.detach().clone() for n, t in modules[k].state_dict().items()} for k in self.names}

    @torch.no_grad()
    def update(self, modules):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k in self.names:
            for n, t in modules[k].state_dict().items():
                if t.dtype.is_floating_point:
                    self.shadow[k][n].mul_(d).add_(t.detach(), alpha=1.0 - d)
                else:
                    self.shadow[k][n].copy_(t)

    @contextlib.contextmanager
    def swap(self, modules):
        live = {k: {n: t.detach().clone() for n, t in modules[k].state_dict().items()} for k in self.names}
        for k in self.names:
            modules[k].load_state_dict(self.shadow[k])
        try:
            yield
        finally:
            for k in self.names:
                modules[k].load_state_dict(live[k])


def _clone_state(module):
    return {n: t.detach().clone() for n, t in module.state_dict().items()}


# -----------------------------
# Improved multi-GAN
# -----------------------------
class ImprovedMultiGAN(S.MultiGAN):
    """S.MultiGAN with DiffAugment on every discriminator input, a WGAN drift term and an EMA of the generation path."""

    EMA_NAMES = ("dc1_g", "dc2_g", "w_g", "aggr")

    def __init__(self, device, ld=S.LATENT_DIM, patch=S.PATCH, width=S.WIDTH, pos_weight=S.POS_WEIGHT,
                 diffaug_p=DIFFAUG_P, ema_decay=EMA_DECAY, wgan_drift=WGAN_DRIFT):
        super().__init__(device, ld, patch, width, pos_weight)
        self.diffaug_p, self.wgan_drift, self.ema_decay = float(diffaug_p), float(wgan_drift), float(ema_decay)
        self.ema = Ema(self.modules(), self.EMA_NAMES, ema_decay) if ema_decay > 0 else None

    def _aug(self, x):
        return diffaug(x, self.diffaug_p)

    def ema_scope(self):
        """Context in which the generation path holds its averaged weights (no-op when the EMA is off)."""
        return self.ema.swap(self.modules()) if self.ema is not None else contextlib.nullcontext()

    # ---- training steps (same structure as S.MultiGAN, discriminator inputs augmented) ----
    def step_dcgan(self, which, real):
        g, d = (self.dc1_g, self.dc1_d) if which == 1 else (self.dc2_g, self.dc2_d)
        og, od = self.opt[f"dc{which}_g"], self.opt[f"dc{which}_d"]
        fake = g(self.z(real.size(0)))
        ld = self._bce(d(self._aug(real)), S.REAL_LABEL) + self._bce(d(self._aug(fake.detach())), 0.0)
        self._step(od, ld)
        lg = self._bce(d(self._aug(fake)), 1.0)
        self._step(og, lg)
        return {f"d{which}": self._v(ld), f"g{which}": self._v(lg)}

    def step_wgan(self, real, update_g):
        fake = self.w_g(self.z(real.size(0))).detach()
        ra, fa = self._aug(real), self._aug(fake)
        gp = S.gradient_penalty(self.w_c, ra, fa)
        score_real = self.w_c(ra)
        drift = self.wgan_drift * (score_real ** 2).mean()
        lc = self.w_c(fa).mean() - score_real.mean() + gp + drift
        self._step(self.opt["w_c"], lc)
        out = {"w_critic": self._v(lc), "w_gp": self._v(gp), "w_drift": self._v(drift)}
        if update_g:
            lg = -self.w_c(self._aug(self.w_g(self.z(real.size(0))))).mean()
            self._step(self.opt["w_g"], lg)
            out["w_g"] = self._v(lg)
        return out

    def step_aggregator(self, real):
        with torch.no_grad():
            z = self.z(real.size(0))
            fakes = [self.dc1_g(z), self.dc2_g(z), self.w_g(z)]
        fused = self.aggr(*fakes)
        ld = self._bce(self.ag_d(self._aug(real)), S.REAL_LABEL) + self._bce(self.ag_d(self._aug(fused.detach())), 0.0)
        self._step(self.opt["ag_d"], ld)
        lg = self._bce(self.ag_d(self._aug(fused)), 1.0)
        l1 = (fused - torch.stack(fakes).mean(0)).abs().mean()
        self._step(self.opt["aggr"], lg + S.AGG_L1 * l1)
        return {"ag_d": self._v(ld), "ag_g": self._v(lg), "ag_l1": self._v(l1)}

    def train_batch(self, real, mask, mask_pool=None):
        out = super().train_batch(real, mask, mask_pool)
        if self.ema is not None:
            self.ema.update(self.modules())
        return out

    # ---- sampling from the averaged weights ----
    def sample_all(self, n):
        with self.ema_scope():
            return super().sample_all(n)

    def sample(self, n, fusion=S.FUSION):
        with self.ema_scope():
            return super().sample(n, fusion)

    @torch.no_grad()
    def sample_fixed(self, n, seed=QUALITY_SEED):
        """Aggregator samples from latents drawn on the CPU with a fixed seed (comparable across epochs and devices)."""
        z = torch.randn(n, self.ld, 1, 1, 1, generator=torch.Generator().manual_seed(int(seed))).to(self.device)
        with self.ema_scope():
            self.train(False)
            try:
                return self.aggr(self.dc1_g(z), self.dc2_g(z), self.w_g(z))
            finally:
                self.train(True)

    # ---- persistence ----
    def save(self, path, epoch=None, elapsed=0.0, with_opt=False, extra=None):
        """"modules" = averaged generation path + live discriminators / mask head (what S.MultiGAN.load reads); "live_modules" = the
        live weights (resume). Atomic like S.MultiGAN.save."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        live = {k: _clone_state(m) for k, m in self.modules().items()}
        with self.ema_scope():
            averaged = {k: _clone_state(m) for k, m in self.modules().items()}
        ck = {"cfg": dict(ld=self.ld, patch=self.patch, width=self.width), "modules": averaged,
              "improved": {"diffaug_p": self.diffaug_p, "ema_decay": self.ema_decay, "wgan_drift": self.wgan_drift,
                           "ema_n": self.ema.n if self.ema is not None else 0}}
        if self.ema is not None:
            ck["live_modules"] = live
        if with_opt:
            ck.update(epoch=epoch, elapsed=elapsed, w_iter=self.w_iter, opt={k: o.state_dict() for k, o in self.opt.items()})
        if extra:
            ck["extra"] = extra
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path, device):
        """Model for sampling: "modules" (the averaged weights) are loaded and the EMA shadow is reset to them."""
        net = super().load(path, device)
        if net.ema is not None:
            net.ema = Ema(net.modules(), cls.EMA_NAMES, net.ema_decay)
        return net

    def load_training_state(self, path):
        """Restore live weights, EMA shadow and optimisers; returns (epoch, elapsed seconds). A plain S.MultiGAN checkpoint also works
        (its weights become both the live and the averaged weights)."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        assert ck["cfg"] == dict(ld=self.ld, patch=self.patch, width=self.width), f"checkpoint config {ck['cfg']} != this run"
        live = ck.get("live_modules", ck["modules"])
        for k, m in self.modules().items():
            m.load_state_dict(live[k])
        if self.ema is not None:
            self.ema = Ema(self.modules(), self.EMA_NAMES, self.ema_decay)
            for k in self.EMA_NAMES:
                self.ema.shadow[k] = {n: t.to(self.device).clone() for n, t in ck["modules"][k].items()}
            self.ema.n = int(ck.get("improved", {}).get("ema_n", 0))
        if "opt" not in ck:
            return 0, 0.0
        for k, o in self.opt.items():
            o.load_state_dict(ck["opt"][k])
        self.w_iter = int(ck.get("w_iter", 0))
        return int(ck["epoch"]), float(ck.get("elapsed", 0.0))


# -----------------------------
# Quality score and early stopping
# -----------------------------
@torch.no_grad()
def mask_prob(model, x, chunk=4):
    """Mask-head probabilities (n,P,P,P) numpy of a (n,1,P,P,P) device batch, in chunks (3D activations are large)."""
    model.train(False)
    try:
        return np.concatenate([torch.sigmoid(model.mask(x[i:i + chunk]))[:, 0].cpu().numpy() for i in range(0, len(x), chunk)])
    finally:
        model.train(True)


def sharpness_hu(hu):
    """Mean absolute difference of neighbouring voxels over the three axes (HU per voxel); a blurred patch scores lower."""
    return float(np.mean([np.abs(np.diff(hu, axis=a)).mean() for a in range(3)]))


def tumour_statistics(patches, masks, work_mm=1.0, seed=0):
    """Pooled tumour-voxel HU (up to VOXELS_PER_PATCH per patch), per-patch contrast (tumour mean minus ring mean, ring 3-8 mm) and
    per-patch sharpness. patches in [-1,1]; patches with an empty mask or ring are skipped."""
    rng = np.random.default_rng(seed)
    vox, contrast, sharp = [], [], []
    for x, m in zip(patches, masks):
        m = np.asarray(m) > 0
        if not m.any():
            continue
        hu = S.from_unit(np.asarray(x, dtype=np.float32))
        d = ndi.distance_transform_edt(~m, sampling=work_mm)
        ring = (d > RING_MM[0]) & (d <= RING_MM[1])
        if not ring.any():
            continue
        t = hu[m]
        vox.append(t if t.size <= VOXELS_PER_PATCH else t[rng.choice(t.size, VOXELS_PER_PATCH, replace=False)])
        contrast.append(float(t.mean() - hu[ring].mean()))
        sharp.append(sharpness_hu(hu))
    return {"vox": np.concatenate(vox) if vox else np.zeros(0, np.float32), "contrast": np.array(contrast), "sharp": np.array(sharp),
            "n": len(contrast)}


def quality_reference(imgs, masks, work_mm=1.0, max_n=64, seed=0):
    """Real statistics the per-epoch score compares against (pass the held-out volumes' tumour patches)."""
    idx = np.arange(len(imgs))
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), max_n, replace=False))
    ref = tumour_statistics(imgs[idx], masks[idx], work_mm, seed)
    assert ref["n"], "quality reference: no real patch with a tumour mask and a surrounding ring"
    return ref


def quality_score(model, ref, n=QUALITY_N, work_mm=1.0, sharp_weight=SHARP_WEIGHT, seed=QUALITY_SEED):
    """Score (HU, lower is better) = W1(tumour-voxel HU) + |contrast gap| + sharp_weight * |sharpness gap| between n aggregator samples
    (fixed latents, EMA weights, mask head + clean_mask) and `ref`. score is None when no sample passes the mask gate."""
    x = model.sample_fixed(n, seed)
    min_vox = S.vox_at(S.MIN_MASK_VOX, work_mm)
    pats, msks = [], []
    for xi, pi in zip(x[:, 0].cpu().numpy(), mask_prob(model, x)):
        m, _ = S.clean_mask(pi, min_vox=min_vox)
        if m is not None:
            pats.append(xi)
            msks.append(m)
    out = {"score": None, "hu_w1": None, "contrast_err": None, "sharp_err": None, "gate_accept": len(pats) / max(n, 1)}
    gen = tumour_statistics(pats, msks, work_mm, seed)
    if not gen["n"]:
        return out
    w1 = float(wasserstein_distance(gen["vox"], ref["vox"]))
    ce = abs(float(gen["contrast"].mean()) - float(ref["contrast"].mean()))
    se = abs(float(gen["sharp"].mean()) - float(ref["sharp"].mean()))
    out.update(score=w1 + ce + sharp_weight * se, hu_w1=w1, contrast_err=ce, sharp_err=se, contrast_hu=float(gen["contrast"].mean()))
    return out


def early_stop_decision(scores, patience, min_epochs, min_delta):
    """(best epoch, stop?) for per-epoch scores (index 0 = epoch 1), lower is better. A score is a new best only when it beats the best
    by more than min_delta; None / non-finite scores never are. Stop when >= min_epochs epochs ran and the last `patience` brought no
    new best. Pure function of the scores, so a resumed run takes the same decision (same as synthesis.early_stop_decision)."""
    best_epoch, best = None, float("inf")
    for epoch, s in enumerate(scores, start=1):
        if s is not None and np.isfinite(s) and s < best - min_delta:
            best_epoch, best = epoch, float(s)
    stop = patience > 0 and best_epoch is not None and len(scores) >= min_epochs and len(scores) - best_epoch >= patience
    return best_epoch, stop


# -----------------------------
# Training loop
# -----------------------------
def train_improved(patches_path, epochs=S.EPOCHS, device=None, results_dir=S.RESULTS_DIR, resume=None, max_seconds=None,
                   patience=ES_PATIENCE, min_epochs=ES_MIN_EPOCHS, min_delta=ES_MIN_DELTA, width=None, quality_n=QUALITY_N,
                   batch_size=None, sample_every=S.SAMPLE_EVERY, steps_per_epoch=STEPS_PER_EPOCH):
    """S.train with ImprovedMultiGAN, a quality score every epoch, checkpoints/gan_best.pt and early stopping (patience > 0).
    Writes checkpoints/gan_last.pt (every S.CKPT_EVERY epochs and at the end), train_metrics.json (incl. quality_history) and
    figures/train_epoch*.png like S.train. Returns the model; its live weights are those of the last epoch, gan_best.pt holds the best."""
    device = device or S.pick_device()
    resume = S.RESUME if resume is None else resume
    max_seconds = S.MAX_SECONDS if max_seconds is None else max_seconds
    S.setup_backends(device)
    d = S.load_npz(patches_path)
    imgs, masks, healthy = d["images"], d["masks"], d["healthy"]
    t_cid = d["meta"][:, 0]
    h_cid = d["healthy_cid"] if "healthy_cid" in d.files else np.full(len(healthy), -1)
    work = float(d["work_mm"]) if "work_mm" in d.files else 1.0
    patch = imgs.shape[-1]
    S.assert_unit_range(imgs, "training patches")
    n = len(imgs)
    bs = min(batch_size or S.BATCH_SIZE, n)
    rng = np.random.default_rng(S.SEED)
    held = S.split_holdout(np.concatenate([t_cid, h_cid]), rng)
    t_tr, h_tr = ~np.isin(t_cid, list(held)), ~np.isin(h_cid, list(held))
    t_tr_i, h_tr_i = np.flatnonzero(t_tr), np.flatnonzero(h_tr)
    assert len(t_tr_i), "no tumour patches left for mask-head training"
    ref_idx = np.flatnonzero(~t_tr) if held and (~t_tr).any() else np.arange(n)  # held-out volumes when there are any
    ref = quality_reference(imgs[ref_idx].astype(np.float32), masks[ref_idx], work)
    model = ImprovedMultiGAN(device, patch=patch, width=width or S.WIDTH)
    model.check_range = False
    min_mask_vox = S.vox_at(S.MIN_MASK_VOX, work)
    ckpt = os.path.join(results_dir, "checkpoints", "gan_last.pt")
    best_ckpt = os.path.join(results_dir, "checkpoints", "gan_best.pt")
    metrics_path = os.path.join(results_dir, "train_metrics.json")
    start, elapsed0 = 0, 0.0
    if resume and os.path.exists(ckpt):
        start, elapsed0 = model.load_training_state(ckpt)
        rng = np.random.default_rng([S.SEED, start])
        logger.info(f"resumed from {ckpt}: epoch {start}, {elapsed0:.0f}s of training already spent")
    store = S.PatchStore({"images": imgs, "masks": masks, "healthy": healthy}, device)
    logger.info(f"[improved] device {device}; {n} tumour / {len(healthy)} healthy patches {patch}^3 at {work} mm; batch {bs}; "
                f"epochs {epochs}; diffaug p={model.diffaug_p} ema={model.ema_decay} drift={model.wgan_drift}; quality reference "
                f"{ref['n']} real patches ({'held-out volumes' if len(ref_idx) < n else 'all volumes'}); early stopping "
                f"{'off' if patience <= 0 else f'patience {patience}, min epochs {min_epochs}, min delta {min_delta} HU'}")
    pool = S.MaskPool(store, t_tr_i, h_tr_i, bs, rng, device)
    metrics = S.read_json(metrics_path) if resume and os.path.exists(metrics_path) else {}
    metrics.update(patches=int(n), healthy_patches=int(len(healthy)), patch=int(patch), work_mm=work, epochs_target=int(epochs),
                   held_out_volumes=[int(v) for v in sorted(held)], device=str(device), improved=True)
    qhist = [h for h in metrics.get("quality_history", []) if h["epoch"] <= start]
    best_epoch, stop = early_stop_decision([h["score"] for h in qhist], patience, min_epochs, min_delta)
    steps = n // bs
    assert steps > 0, f"batch size {bs} larger than the patch set"
    if steps_per_epoch and steps_per_epoch < steps:  # the permutation below draws a fresh random subset every epoch
        logger.info(f"{steps_per_epoch} of {steps} batches per epoch ({100 * steps_per_epoch / steps:.0f}% of the patches)")
        steps = steps_per_epoch
    t0, ep, stopped = time.time(), start, ("early" if stop else ("max" if start >= epochs else None))
    model.sync = False
    for ep in range(start + 1, epochs + 1):
        if stop:
            break
        perm = torch.as_tensor(rng.permutation(n)[:steps * bs], dtype=torch.long).to(device if store.on_device else "cpu")
        pool.refill(steps * S.MASK_STEPS)
        sums, cnts = {}, {}
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            for k, v in model.train_batch(store.gather("images", idx), store.gather("masks", idx), pool).items():
                sums[k], cnts[k] = sums.get(k, 0.0) + v, cnts.get(k, 0) + 1
        means = {k: float(v) / cnts[k] for k, v in sums.items()}
        assert all(np.isfinite(v) for v in means.values()), "non-finite loss"
        model.sync = True
        q = quality_score(model, ref, quality_n, work)
        qhist.append(dict(q, epoch=ep))
        best_epoch, stop = early_stop_decision([h["score"] for h in qhist], patience, min_epochs, min_delta)
        spent = elapsed0 + time.time() - t0
        qs = "none" if q["score"] is None else f"{q['score']:.1f}"
        logger.info(f"epoch {ep}/{epochs} ({spent:.0f}s) " + " ".join(f"{k}={v:.3f}" for k, v in means.items())
                    + f" | quality {qs} HU (gate {q['gate_accept']:.2f}), best epoch {best_epoch}")
        if best_epoch == ep:
            model.save(best_ckpt, epoch=ep, elapsed=spent, extra={"quality": {k: v for k, v in q.items() if v is not None}})
        if ep % sample_every == 0 or ep == epochs or stop:
            for tag, ev in (("train", S.eval_mask_head(model, imgs[t_tr_i], masks[t_tr_i], healthy[h_tr_i], min_vox=min_mask_vox)),
                            ("heldout", S.eval_mask_head(model, imgs[~t_tr], masks[~t_tr], healthy[~h_tr], n=64, min_vox=min_mask_vox) if held else {})):
                if ev:
                    logger.info(f"mask head {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))
                    metrics[f"mask_head_{tag}"] = dict(ev, epoch=ep)
            s = {k: v[:, 0].cpu().numpy() for k, v in model.sample_all(4).items()}
            pm = mask_prob(model, torch.from_numpy(s["aggregator"])[:, None].to(device)) > 0.5
            S.save_montage(os.path.join(results_dir, "figures", f"train_epoch{ep:03d}.png"),
                           [("real", imgs[:4].astype(np.float32), masks[:4] > 0), ("dcgan 1 (EMA)", s["dc1"], None),
                            ("dcgan 2 (EMA)", s["dc2"], None), ("wgan-gp (EMA)", s["wgan"], None), ("aggregator + mask head", s["aggregator"], pm)])
        model.sync = False
        out_of_time = bool(max_seconds) and spent >= max_seconds and ep < epochs
        if ep % max(S.CKPT_EVERY, 1) == 0 or ep == epochs or out_of_time or stop:
            model.save(ckpt, epoch=ep, elapsed=spent, with_opt=True)
        history = [h for h in metrics.get("loss_history", []) if h["epoch"] < ep]
        metrics.update(epochs_done=ep, train_seconds=spent, last_losses=means, loss_history=history + [dict(means, epoch=ep, seconds=spent)],
                       quality_history=qhist, best_epoch=best_epoch)
        S.write_json_atomic(metrics_path, metrics)
        if out_of_time:
            logger.warning(f"GAN time budget {max_seconds:.0f}s reached after epoch {ep}/{epochs}: stopping early")
            stopped = "time"
            break
        if stop:
            logger.info(f"early stopping after epoch {ep}: no better quality score for {patience} epochs (best epoch {best_epoch})")
            stopped = "early"
    model.sync = True
    metrics.update(epochs_done=ep, stopped=stopped or "max", best_epoch=best_epoch)
    S.write_json_atomic(metrics_path, metrics)
    return model


# -----------------------------
# Size-controlled generation
# -----------------------------
def equivalent_radius_mm(mask, spacing=1.0):
    """Radius (mm) of the sphere with the volume of the binary mask (voxel count times voxel volume)."""
    v = float(np.count_nonzero(mask)) * float(np.prod(np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))))
    return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)


def size_class(radius_mm, bins=None):
    """Size class of an equivalent radius (small includes its lower edge), or None outside every class."""
    for name, (lo, hi) in (bins or SIZE_BINS_MM).items():
        if (lo <= radius_mm <= hi) if name == "small" else (lo < radius_mm <= hi):
            return name
    return None


def generate_by_size(model, n_per_class=PER_CLASS, classes=CLASSES, fusion=S.FUSION, work_mm=1.0, max_attempts=None, batch=8, stats=None):
    """Accept-until-target per size class: sample (EMA weights for an ImprovedMultiGAN), label with the mask head, clean the mask
    (threshold, largest component, minimum size, away from the border), measure its equivalent radius at work_mm and keep it when
    its class still needs samples. Returns {class: (patches (n,P,P,P) float32, masks uint8, radii_mm float32)}; a class can come out
    short when the generator rarely produces that size (see stats['rejects'] / stats['radius_hist_mm'])."""
    unknown = [c for c in classes if c not in SIZE_BINS_MM]
    assert not unknown, f"unknown size classes {unknown}; known: {list(SIZE_BINS_MM)}"
    max_attempts = max_attempts or 100 * n_per_class * len(classes)
    min_vox = S.vox_at(S.MIN_MASK_VOX, work_mm)
    bins = {c: SIZE_BINS_MM[c] for c in classes}
    keep = {c: ([], [], []) for c in classes}
    rejects, radii_seen, tried = {}, [], 0
    while any(len(keep[c][0]) < n_per_class for c in classes) and tried < max_attempts:
        x = model.sample(batch, fusion)
        for xi, pi in zip(x[:, 0].cpu().numpy(), mask_prob(model, x)):
            tried += 1
            m, why = S.clean_mask(pi, min_vox=min_vox)
            if m is None:
                rejects[why] = rejects.get(why, 0) + 1
                continue
            r = equivalent_radius_mm(m, work_mm)
            radii_seen.append(r)
            c = size_class(r, bins)
            if c is None:
                rejects["size_out_of_range"] = rejects.get("size_out_of_range", 0) + 1
            elif len(keep[c][0]) >= n_per_class:
                rejects[f"{c}_full"] = rejects.get(f"{c}_full", 0) + 1
            else:
                S.assert_unit_range(xi, "generated patch")
                keep[c][0].append(xi)
                keep[c][1].append(m)
                keep[c][2].append(r)
    out = {}
    for c in classes:
        p, m, r = keep[c]
        e = (0,) + (model.patch,) * 3
        out[c] = (np.stack(p).astype(np.float32) if p else np.zeros(e, np.float32), np.stack(m).astype(np.uint8) if m else np.zeros(e, np.uint8),
                  np.asarray(r, dtype=np.float32))
    hist, edges = np.histogram(radii_seen, bins=np.arange(0, 32, 2)) if radii_seen else (np.zeros(15, int), np.arange(0, 32, 2))
    st = dict(tried=tried, accepted={c: int(len(out[c][0])) for c in classes}, rejects=rejects,
              radius_hist_mm={"edges": edges.tolist(), "counts": hist.tolist()})
    logger.info(f"size-controlled generation ({fusion}): {st['accepted']} in {tried} attempts; rejects {rejects}")
    if stats is not None:
        stats.update(st)
    return out


def save_samples(path, by_size, work_mm=1.0):
    """samples.npz readable by S.run_insert (patches, masks, work_mm) plus radius_mm and size_class per sample."""
    cls = [c for c in by_size for _ in range(len(by_size[c][0]))]
    nonempty = [by_size[c] for c in by_size if len(by_size[c][0])]
    if nonempty:
        p, m, r = (np.concatenate([v[i] for v in nonempty]) for i in range(3))
    else:
        any_c = next(iter(by_size.values()))
        p, m, r = any_c
    S.save_npz_atomic(path, patches=p, masks=m, radius_mm=r.astype(np.float32), size_class=np.array(cls, dtype="U8"), work_mm=np.float32(work_mm))
    return path


# -----------------------------
# Entry point
# -----------------------------
def main():
    stage = os.environ.get("LITS_GAN_STAGE", "all")
    assert stage in ("all", "prepare", "train", "generate"), f"{stage}: the insert stage lives in Syn_Img_Pipeline_fixed.py / insertion_improved.py"
    assert S.GAN_SOURCE == "nifti", "syn_training_improved.py runs the NIfTI source; for the cache source use the patches.npz its prepare stage writes"
    S.check_patch(S.PATCH)
    S.setup_logging(S.RESULTS_DIR)
    S.set_seed(S.SEED)
    device = S.pick_device()
    patches = os.path.join(S.RESULTS_DIR, "patches.npz")
    samples = os.path.join(S.RESULTS_DIR, "synthetic", "samples.npz")
    if stage in ("all", "prepare"):
        ids = S.split_cases(restrict=S.parse_volume_list(S.VOLUME_LIST) if S.VOLUME_LIST else None)
        S.prepare_patches(ids, patches, source=S.NiftiSource(S.DATA_DIR, S.WORK_MM))
    if stage in ("all", "train"):
        train_improved(patches, S.EPOCHS, device)
    if stage in ("all", "generate"):
        ck = os.path.join(S.RESULTS_DIR, "checkpoints", "gan_best.pt")
        ck = ck if os.path.exists(ck) else os.path.join(S.RESULTS_DIR, "checkpoints", "gan_last.pt")
        model = ImprovedMultiGAN.load(ck, device)  # "modules" are the averaged weights already
        logger.info(f"generating from {ck}")
        st = {}
        by_size = generate_by_size(model, PER_CLASS, CLASSES, S.FUSION, S.WORK_MM, S.GEN_ATTEMPTS or None, stats=st)
        save_samples(samples, by_size, S.WORK_MM)
        S.write_json_atomic(os.path.join(S.RESULTS_DIR, "synthetic", "generation_stats.json"), st)
        rows = [(f"{c} (r {r.min():.1f}-{r.max():.1f} mm)", p[:8], m[:8] > 0) for c, (p, m, r) in by_size.items() if len(p)]
        if rows:
            S.save_montage(os.path.join(S.RESULTS_DIR, "figures", "generated_by_size.png"), rows)
    logger.info("done")


if __name__ == "__main__":
    main()
