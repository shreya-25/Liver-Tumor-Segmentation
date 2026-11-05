"""Improved host insertion for Syn_Img_Pipeline_fixed.py, ported from the reference insertion.py (Hu et al., CVPR 2023 style).

    python insertion_improved.py                          # NIfTI hosts of the train part, samples from <results>/synthetic/samples.npz
    LITS_INS_SIZE=large LITS_INS_N=2 python insertion_improved.py
    LITS_INS_NO_VESSELS=1 LITS_INS_NO_MASS=1 python insertion_improved.py      # ablations

Syn_Img_Pipeline_fixed.insert_one picks a random site in the liver eroded by the tumour radius, shifts the patch mean onto the host
liver mean and blends it with a feathered alpha. It does not know about vessels, keeps whatever size the generator produced, has no
mass effect and no automatic quality control. This module keeps its blending (feathered alpha, host-matched high-frequency noise,
volume-preserving mask resampling) and adds, from insertion.py:

  1. Vessel mask: CT smoothed inside the liver (G(ct*liver)/G(liver), sigma 1.5 mm), robust parenchyma mean (eroded liver, sigma
     clipping), vessels = parenchyma voxels brighter than mean + 15 HU, components < 0.05 ml dropped (insertion.segment_vessels).
  2. Vessel-aware placement: a centre is a candidate when its depth in the liver > R + 3 mm, its vessel distance > R + 3 mm and its
     distance to existing tumours > R + LITS_GAN_EXCLUDE_MM (R = equivalent radius on the host grid); then the REAL resampled mask is
     checked at that centre (every tumour voxel keeps the liver and vessel margins, insertion.mask_clearance_ok), up to 40 draws.
  3. Physical size control: the generated patch is scaled isotropically so that its mask has the requested equivalent radius
     (small 5 / medium 10 / large 15 mm, or any value), then resampled onto the host spacing in one step.
  4. HU harmonisation from a 3-10 mm host shell around the placed tumour that excludes vessels and tumours, against a robust ring mean
     of the generated patch that excludes clipped air (insertion.harmonize_hu, mode offset); alpha is 0 on vessel voxels outside the
     tumour, so vessels keep their contrast.
  5. Mass effect: the paper's local scaling warp gamma' = [1 - (1 - gamma/gamma_max)^2 I/100] gamma, gamma_max = 1.3 R, I = 0 / 20 / 30
     for small / medium / large; the numerical Jacobian of the sampling map must stay >= 0.2 (I is lowered in steps of 5 otherwise).
     CT warped with a cubic spline clipped to the local range, liver and vessels by nearest neighbour, tumour labels with a smoothed
     indicator so that sub-voxel surface shifts are not lost (insertion.apply_mass_effect).
  6. Optional capsule ring (additive HU peak per contrast phase, off by default: LiTS does not record the phase).
  7. Quality control per insertion: tumour in liver, vessel overlap, radius vs request (expansion by the warp included), finite HU in
     range, tumour-to-parenchyma contrast and tumour std within bounds, one connected component, valid warp. A failed insertion is
     reverted (on_fail="reject") or kept and flagged (on_fail="flag"). Every insertion, accepted or not, is a row of qc_report.csv.

Conventions follow Syn_Img_Pipeline_fixed.py: host arrays are (X,Y,Z) in array-voxel space with per-axis spacing from the header zooms;
HU float32 while working, int16 on disk; labels 0 background, 1 liver, 2 tumour. Every write is confined to a local edit box, which is
backed up before the insertion, so voxels outside it are bit-identical to the host.

Env vars: LITS_INS_SIZE (small|medium|large|<radius mm>|keep, default medium), LITS_INS_N (tumours per host, default
LITS_GAN_INSERTIONS), LITS_INS_NO_VESSELS, LITS_INS_NO_MASS, LITS_INS_CAPSULE, LITS_INS_PHASE (default unknown),
LITS_INS_ON_FAIL (reject|flag), LITS_INS_OUT (default <results>/aug_improved), LITS_INS_MAX_HOSTS (0 = all).
"""
import os
import sys
import csv
import json
import math
import logging
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import Syn_Img_Pipeline_fixed as S  # noqa: E402

SIZES_MM = {"small": 5.0, "medium": 10.0, "large": 15.0}
MASS_INTENSITY = {"small": 0.0, "medium": 20.0, "large": 30.0}
CAPSULE_HU = {"non_contrast": 0.0, "arterial": 60.0, "portal_venous": 120.0, "delayed": 80.0, "unknown": 60.0}
NEW = 3  # temporary label of the tumour being inserted (turned into 2 at the end)
STRUCT26 = np.ones((3, 3, 3), bool)

logger = logging.getLogger("gan_fixed")


@dataclass
class InsertParams:
    """Every setting of one insertion (lengths in mm, intensities in HU)."""
    target_radius_mm: float = None  # None keeps the generated size
    min_scale: float = 0.5  # bounds of the isotropic size rescale of a generated patch
    max_scale: float = 2.0
    patch_radius_tol_mm: float = 1.5  # samples within this of the best radius match are equally likely (needs samples radius_mm)
    vessels: bool = True
    vessel_sigma_mm: float = 1.5
    vessel_offset_hu: float = 15.0
    vessel_erode_mm: float = 5.0
    vessel_min_ml: float = 0.05
    margin_liver_mm: float = 3.0
    margin_vessel_mm: float = 3.0
    exclude_mm: float = S.EXCLUDE_MM
    fit_attempts: int = 40
    feather_mm: float = S.FEATHER_MM
    host_shell_mm: tuple = (3.0, 10.0)
    gen_ring_mm: tuple = (2.0, 10.0)
    mass_effect: bool = True
    mass_intensity: float = None  # None: by size class of the radius (MASS_INTENSITY)
    gamma_factor: float = 1.3
    min_jacobian: float = 0.2
    max_displacement_mm: float = 12.0
    intensity_step: float = 5.0
    capsule: bool = False
    capsule_sigma_mm: float = 0.8
    capsule_inner_mm: float = -1.0
    capsule_outer_mm: float = 1.5
    on_fail: str = "reject"  # reject | flag
    radius_tol: float = 0.25
    max_outside_liver_frac: float = 0.0
    max_vessel_overlap_frac: float = 0.02
    contrast_hu_bounds: tuple = (-120.0, 80.0)
    tumor_std_hu_bounds: tuple = (2.0, 80.0)

    def __post_init__(self):
        assert self.on_fail in ("reject", "flag"), self.on_fail
        assert 0 < self.min_scale <= 1 <= self.max_scale
        assert self.capsule_outer_mm > self.capsule_inner_mm and self.capsule_sigma_mm > 0
        assert 0 < self.min_jacobian <= 1 and self.intensity_step > 0


def size_class_of(radius_mm):
    return "small" if radius_mm <= 6.0 else ("medium" if radius_mm <= 14.0 else "large")


def equivalent_radius_mm(mask, spacing):
    v = float(np.count_nonzero(mask)) * float(np.prod(spacing))
    return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)


def _bbox(mask, pad=(0, 0, 0)):
    idx = [np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    if any(len(i) == 0 for i in idx):
        return None
    return tuple(slice(max(int(i[0]) - int(p), 0), min(int(i[-1]) + 1 + int(p), n)) for i, p, n in zip(idx, pad, mask.shape))


def _robust_mean(v, k=2.5, n_iter=3):
    v = np.asarray(v, np.float64)
    for _ in range(n_iter):
        mu, sd = v.mean(), v.std()
        keep = np.abs(v - mu) <= k * sd
        if sd == 0 or keep.all() or keep.sum() < 5:
            break
        v = v[keep]
    return float(v.mean())


# -----------------------------
# 1. Vessels
# -----------------------------
def sigma_clipped(values, k=3.0, n_iter=10):
    """Mean and std after iterative k-sigma clipping started from median / MAD (std corrected for the truncation)."""
    v = np.asarray(values, np.float64)
    c, s = float(np.median(v)), 1.4826 * float(np.median(np.abs(v - np.median(v)))) or float(v.std())
    keep = np.abs(v - c) <= k * s if s > 0 else np.ones(v.shape, bool)
    for _ in range(n_iter):
        mu, sd = v[keep].mean(), v[keep].std()
        new = np.abs(v - mu) <= k * sd if sd > 0 else keep
        if not new.any() or np.array_equal(new, keep):
            break
        keep = new
    phi = math.exp(-0.5 * k * k) / math.sqrt(2 * math.pi)
    factor = math.sqrt(max(1.0 - 2.0 * k * phi / math.erf(k / math.sqrt(2.0)), 1e-12))
    return float(v[keep].mean()), float(v[keep].std()) / factor


def segment_vessels(hu, seg, spacing, p=None):
    """Bool (X,Y,Z) vessel mask inside the liver parenchyma (label 1) and its statistics; see the module docstring (item 1)."""
    p = p or InsertParams()
    sp = np.asarray(spacing, np.float64)
    par = seg == 1
    full = np.zeros(seg.shape, bool)
    if not par.any():
        return full, {"source": "empty_liver"}
    sig = p.vessel_sigma_mm / sp
    box = _bbox(par, np.ceil(4 * sig).astype(int) + 1)
    c, m = hu[box].astype(np.float32), par[box]
    num = ndi.gaussian_filter(np.where(m, c, 0).astype(np.float32), sig, mode="constant")
    den = ndi.gaussian_filter(m.astype(np.float32), sig, mode="constant")
    sm = np.where(den > 1e-3, num / np.maximum(den, 1e-3), c)
    depth = ndi.distance_transform_edt(np.pad(m, 1), sampling=sp)[1:-1, 1:-1, 1:-1]
    radius, region = p.vessel_erode_mm, depth > p.vessel_erode_mm
    while region.sum() < 200 and radius > 0.1:
        radius *= 0.5
        region = depth > radius
    if region.sum() < 200:
        region = m
    mean, std = sigma_clipped(sm[region])
    thr = mean + p.vessel_offset_hu
    cand = (sm > thr) & m
    lab, n = ndi.label(cand, structure=STRUCT26)
    removed = 0
    if n:
        ml = np.bincount(lab.ravel())[1:] * float(np.prod(sp)) / 1000.0
        keep = np.concatenate([[False], ml >= p.vessel_min_ml])
        removed = int((~keep[1:]).sum())
        cand = keep[lab]
    full[box] = cand
    liver_ml = float(par.sum()) * float(np.prod(sp)) / 1000.0
    vol_ml = float(cand.sum()) * float(np.prod(sp)) / 1000.0
    return full, {"source": "threshold", "parenchyma_mean_hu": mean, "parenchyma_std_hu": std, "threshold_hu": thr, "volume_ml": vol_ml,
                  "fraction_of_liver": vol_ml / liver_ml if liver_ml else 0.0, "components_removed": removed}


# -----------------------------
# 2. Placement maps
# -----------------------------
class HostMaps:
    """Depth in the liver, distance to vessels and distance to existing tumours (mm), on the liver bbox padded by one voxel (array
    borders count as outside the liver). Index maps with full-volume voxel indices through `origin`."""

    def __init__(self, seg, vessels, spacing):
        sp = tuple(float(s) for s in spacing)
        box = _bbox(seg > 0)
        assert box is not None, "host has no liver label"
        self.origin = np.array([s.start for s in box]) - 1
        liver = np.pad(seg[box] > 0, 1)
        self.depth = ndi.distance_transform_edt(liver, sampling=sp).astype(np.float32)
        tum = np.pad(seg[box] >= 2, 1)
        self.tdist = ndi.distance_transform_edt(~tum, sampling=sp).astype(np.float32) if tum.any() else np.full(liver.shape, np.inf, np.float32)
        ves = np.pad(vessels[box], 1) if vessels is not None else np.zeros_like(liver)
        self.vdist = ndi.distance_transform_edt(~ves, sampling=sp).astype(np.float32) if ves.any() else np.full(liver.shape, np.inf, np.float32)

    def stats(self, idx_full):
        """(depth, vessel distance, tumour distance) minima over full-volume voxel indices (n,3); depth 0 outside the maps."""
        loc = np.asarray(idx_full) - self.origin
        if np.any(loc < 0) or np.any(loc >= np.array(self.depth.shape)):
            return 0.0, 0.0, 0.0
        ix = tuple(loc.T)
        return float(self.depth[ix].min()), float(self.vdist[ix].min()), float(self.tdist[ix].min())


def candidate_centres(maps, radius_mm, cen, patch_shape, vol_shape, p):
    """Full-volume voxel indices of valid centres for a sphere of radius_mm (strict margins, window inside the volume), plus the voxel
    counts left after each criterion (to name the one that emptied the map)."""
    ok = maps.depth > radius_mm + p.margin_liver_mm
    counts = {"depth": int(ok.sum())}
    ok &= maps.vdist > radius_mm + p.margin_vessel_mm
    counts["vessel"] = int(ok.sum())
    ok &= maps.tdist > radius_mm + p.exclude_mm
    counts["tumour_gap"] = int(ok.sum())
    c = np.argwhere(ok) + maps.origin
    start = c - cen
    inside = np.all(start >= 0, 1) & np.all(start + np.asarray(patch_shape) <= np.asarray(vol_shape), 1)
    counts["window"] = int(inside.sum())
    return c[inside], counts


# -----------------------------
# 3. Size control
# -----------------------------
def prepare_patch(patch, mask, spacing, work_mm, target_radius_mm, p):
    """Scale the generated patch isotropically to target_radius_mm (clipped to [min_scale, max_scale]) and resample it to the host spacing
    in one step: S.resample_patch_to_host is told that a generator voxel measures work_mm * scale, so the physical size changes by
    `scale` and the mask volume is preserved exactly. Returns (patch_rs in [-1,1], mask_rs bool, scale)."""
    r0 = equivalent_radius_mm(mask, (work_mm,) * 3)
    scale = 1.0 if not target_radius_mm else float(np.clip(target_radius_mm / max(r0, 1e-6), p.min_scale, p.max_scale))
    img, m = S.resample_patch_to_host(patch, mask, spacing, work_mm * scale)
    return img, m, scale


def pick_sample(samples, target_radius_mm, rng, p):
    """Index of a sample: random, or (target and samples['radius_mm']) among the samples nearest to the target radius."""
    n = len(samples["patches"])
    r = samples.get("radius_mm")
    if target_radius_mm and r is not None and len(r) == n and n:
        err = np.abs(np.asarray(r, np.float64) - target_radius_mm)
        cand = np.flatnonzero(err <= err.min() + p.patch_radius_tol_mm + 1e-9)
        return int(cand[rng.integers(len(cand))])
    return int(rng.integers(n))


# -----------------------------
# 4. Blending with vessel-aware harmonisation
# -----------------------------
def blend_at(hu, seg, vessels, spacing, patch_rs, m_rs, start, rng, p):
    """Blend patch_rs into float32 `hu` IN PLACE with its window starting at `start`; the tumour is written as label NEW. Blending as in
    S.insert_one, with the host reference from a vessel-free shell and alpha 0 on vessel voxels outside the tumour. None if the mask
    has no liver voxel there."""
    sp = np.asarray(spacing, np.float64)
    ps = np.array(patch_rs.shape)
    win = tuple(slice(int(s), int(s + n)) for s, n in zip(start, ps))
    hw, sw, vw = hu[win].astype(np.float32), seg[win], vessels[win]
    liver_w = sw == 1
    m_eff = m_rs & liver_w
    if not m_eff.any():
        return None
    dil = ndi.distance_transform_edt(~m_eff, sampling=sp) <= 1.0
    alpha = ndi.gaussian_filter(dil.astype(np.float32), p.feather_mm / sp, mode="constant")
    alpha = np.maximum(alpha, m_eff) * S.border_taper(tuple(ps), sp) * liver_w
    alpha[vw & ~m_eff] = 0.0
    alpha[alpha < 0.01] = 0.0
    support = alpha > 0
    # harmonisation (offset mode): host shell around the placed tumour (no vessels, no tumours) vs a robust generated ring without air
    d = ndi.distance_transform_edt(~m_eff, sampling=sp)
    shell = (d >= p.host_shell_mm[0]) & (d <= p.host_shell_mm[1]) & liver_w & ~vw
    host_ref = _robust_mean(hw[shell]) if shell.sum() >= 30 else float(hw[liver_w & ~support].mean() if (liver_w & ~support).sum() >= 100 else hw[liver_w].mean())
    gen = S.from_unit(patch_rs)
    dg = ndi.distance_transform_edt(~m_rs, sampling=sp)
    ring = (dg > p.gen_ring_mm[0]) & (dg <= p.gen_ring_mm[1])
    ring_soft = ring & (gen > S.WIN_LO + 5.0)  # generated values at the window floor are air padding / clipped, not parenchyma
    ring = ring_soft if ring_soft.sum() >= 20 else (ring if ring.sum() >= 20 else ~m_rs)
    gen_ref = _robust_mean(gen[ring])
    shift = host_ref - gen_ref
    gen = gen + shift
    # host-matched high-frequency noise (same as S.insert_one)
    pad = np.ceil(24.0 / sp).astype(int)
    nsl = tuple(slice(max(0, w.start - q), min(n, w.stop + q)) for w, q, n in zip(win, pad, seg.shape))
    sig = S.liver_noise_std(hu, seg, spacing, nsl)
    sig = 0.0 if sig is None else sig
    sg = S.NOISE_SIGMA_MM / sp
    noise = ndi.gaussian_filter(rng.standard_normal(hw.shape).astype(np.float32), S.NOISE_PRE_MM / sp)
    noise = noise - ndi.gaussian_filter(noise, sg)
    noise *= sig / max(float(noise.std()), 1e-6)
    host_lp = ndi.gaussian_filter(hw, sg, mode="nearest")
    out = ((1 - alpha) * host_lp + alpha * ndi.gaussian_filter(gen, sg, mode="nearest")
           + np.sqrt(1 - alpha) * (hw - host_lp) + np.sqrt(alpha) * noise)
    hu[win][support] = out[support]
    sw[m_eff] = NEW
    return {"window": win, "support": support, "alpha": alpha, "mask": m_eff, "shift_hu": float(shift), "host_ref_hu": host_ref,
            "gen_ref_hu": gen_ref, "noise_std": float(sig), "n_label": int(m_eff.sum())}


# -----------------------------
# 5. Mass effect
# -----------------------------
def _ratio(gamma, gmax, intensity):
    u = np.clip(gamma / gmax, 0.0, 1.0)
    return 1.0 - (1.0 - u) ** 2 * (intensity / 100.0)


def radial_scale(gamma, gamma_max, intensity):
    """Paper formula gamma' = [1 - (1 - gamma/gamma_max)^2 I/100] gamma for gamma <= gamma_max, identity beyond."""
    g = np.asarray(gamma, np.float64)
    return np.where(g <= gamma_max, _ratio(g, gamma_max, intensity) * g, g)


def expected_radius_after_warp(radius_mm, gamma_max_mm, intensity):
    """Output radius of a sphere of input radius radius_mm (solves gamma'(g) = r by bisection; identity for I <= 0)."""
    k = min(max(float(intensity), 0.0) / 100.0, 0.99)
    if k <= 0 or radius_mm <= 0 or radius_mm >= gamma_max_mm:
        return float(radius_mm)
    lo, hi = float(radius_mm), float(gamma_max_mm)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid * (1 - (1 - mid / gamma_max_mm) ** 2 * k) < radius_mm else (lo, mid)
    return 0.5 * (lo + hi)


def compute_roi(shape, spacing, centre, reach_mm, pad_vox=6):
    half = reach_mm / np.asarray(spacing, np.float64)
    c = np.asarray(centre, np.float64)
    lo = np.maximum(np.floor(c - half).astype(int) - pad_vox, 0)
    hi = np.minimum(np.ceil(c + half).astype(int) + pad_vox + 1, np.asarray(shape))
    return tuple(slice(int(a), int(max(b, a))) for a, b in zip(lo, hi))


def sampling_jacobian(src, spacing):
    """Numerical determinant of d S / d x (both in mm) of a (3, rx, ry, rz) sampling map."""
    sp = np.asarray(spacing, np.float64)
    s_mm = src * sp[:, None, None, None]
    j = np.zeros((3, 3) + src.shape[1:])
    for i in range(3):
        for k in range(3):
            j[i, k] = np.gradient(s_mm[i], sp[k], axis=k) if src.shape[1 + k] >= 2 else float(i == k)
    a, b, c = j[0]
    d, e, f = j[1]
    g, h, k_ = j[2]
    return a * (e * k_ - f * h) - b * (d * k_ - f * g) + c * (d * h - e * g)


def _warp_indicator(ind, coords, moved):
    """Sub-voxel warp of a binary indicator: Gaussian-smoothed (1 voxel), resampled linearly, thresholded at the level that keeps its
    voxel count (insertion._warp_tumor_labels)."""
    k = int(ind.sum())
    if k == 0:
        return np.zeros(int(moved.sum()), bool)
    f = ndi.gaussian_filter(ind.astype(np.float32), 1.0, mode="nearest")
    flat = f.ravel()
    tau = float(np.clip(np.partition(flat, flat.size - k)[flat.size - k], 0.05, 0.95))
    return ndi.map_coordinates(f, coords[:, moved], order=1, mode="nearest") >= tau


def mass_effect(hu, seg, vessels, spacing, centre, radius_mm, intensity, p):
    """Local scaling warp around `centre` (float voxel index) IN PLACE on hu (cubic, clipped to the local range), seg (liver by nearest
    neighbour, tumour labels 2 and NEW by the smoothed indicator) and vessels (nearest). Only voxels with gamma < gamma_max change.
    Returns a dict (applied, intensity_requested / applied, gamma_max_mm, jacobian_min, max_displacement_mm, roi, reductions)."""
    sp = np.asarray(spacing, np.float64)
    gmax = p.gamma_factor * float(radius_mm)
    res = {"applied": False, "intensity_requested": float(intensity), "intensity_applied": 0.0, "gamma_max_mm": gmax, "jacobian_min": 1.0,
           "max_displacement_mm": 0.0, "roi": None, "reductions": []}
    if not p.mass_effect or intensity <= 0 or radius_mm <= 0:
        return res
    c = np.asarray(centre, np.float64)
    roi = compute_roi(hu.shape, sp, c, gmax)
    grid = np.stack(np.meshgrid(*[np.arange(s.start, s.stop, dtype=np.float64) for s in roi], indexing="ij"))
    rel = grid - c[:, None, None, None]
    gamma = np.sqrt(((rel * sp[:, None, None, None]) ** 2).sum(0))
    inside = gamma < gmax
    cur, src = float(intensity), None
    while cur > 0:
        cand = np.where(inside[None], c[:, None, None, None] + rel * _ratio(gamma, gmax, cur)[None], grid)
        jac = float(sampling_jacobian(cand, sp).min())
        disp = float(np.sqrt((((cand - grid) * sp[:, None, None, None]) ** 2).sum(0)).max())
        if np.isfinite(jac) and jac >= p.min_jacobian and disp <= p.max_displacement_mm:
            src = cand
            break
        res["reductions"].append({"intensity": cur, "jacobian_min": jac, "max_displacement_mm": disp})
        cur = max(cur - p.intensity_step, 0.0)
    if src is None:
        return res
    local = src - np.array([s.start for s in roi], np.float64)[:, None, None, None]
    moved = np.any(src != grid, axis=0)
    sub = hu[roi]
    near = tuple(np.clip(np.rint(local[i, moved]).astype(np.intp), 0, sub.shape[i] - 1) for i in range(3))
    lo_l = ndi.minimum_filter(sub, size=3, mode="nearest")[near]
    hi_l = ndi.maximum_filter(sub, size=3, mode="nearest")[near]
    vals = np.clip(ndi.map_coordinates(sub.astype(np.float64), local[:, moved], order=3, mode="nearest"), lo_l, hi_l)
    s_sub, v_sub = seg[roi], vessels[roi]
    liver_new = ndi.map_coordinates((s_sub > 0).astype(np.uint8), local[:, moved], order=0, mode="nearest") > 0
    tum = {lab: _warp_indicator(s_sub == lab, local, moved) for lab in (2, NEW)}
    lab_new = liver_new.astype(np.uint8)
    for lab in (2, NEW):
        lab_new[tum[lab]] = lab
    ves_new = ndi.map_coordinates(v_sub.astype(np.uint8), local[:, moved], order=0, mode="nearest") > 0
    sub[moved] = vals.astype(sub.dtype)
    s_sub[moved] = lab_new
    v_sub[moved] = ves_new & (lab_new == 1)
    res.update(applied=True, intensity_applied=cur, jacobian_min=jac, max_displacement_mm=disp, roi=roi)
    return res


# -----------------------------
# 6. Capsule
# -----------------------------
def signed_distance_mm(mask, spacing):
    """Signed distance (mm) to the voxel-face surface of a binary mask, negative inside (box distance, insertion.signed_distance_mm)."""
    sp = np.asarray(spacing, np.float64)
    m = np.pad(np.asarray(mask) > 0, 1)
    grid = np.indices(m.shape).astype(np.float64)

    def box(binary):
        _, idx = ndi.distance_transform_edt(binary, sampling=sp, return_indices=True)
        diff = np.abs(grid - idx) * sp[:, None, None, None] - 0.5 * sp[:, None, None, None]
        return np.sqrt((np.maximum(diff, 0.0) ** 2).sum(0))

    sd = np.where(m, -box(m), box(~m))
    return sd[1:-1, 1:-1, 1:-1].astype(np.float32)


def apply_capsule(hu, seg, spacing, phase, p):
    """Add the capsule ring around label NEW (additive peak CAPSULE_HU[phase], Gaussian profile zero at the band edges), inside the
    liver only, IN PLACE. Returns the peak HU (0 when nothing was added)."""
    peak = CAPSULE_HU.get(phase, CAPSULE_HU["unknown"])
    tm = seg == NEW
    if not p.capsule or peak == 0 or not tm.any():
        return 0.0
    sp = np.asarray(spacing, np.float64)
    reach = max(abs(p.capsule_inner_mm), abs(p.capsule_outer_mm))
    box = _bbox(tm, np.ceil(reach / sp).astype(int) + 2)
    sd = signed_distance_mm(tm[box], sp)
    lo, hi, sig = p.capsule_inner_mm, p.capsule_outer_mm, p.capsule_sigma_mm
    c0, w = 0.5 * (lo + hi), 0.5 * (hi - lo)
    g, g_edge = np.exp(-0.5 * ((sd - c0) / sig) ** 2), math.exp(-0.5 * (w / sig) ** 2)
    prof = np.where((sd > lo) & (sd < hi), np.maximum((g - g_edge) / (1 - g_edge), 0), 0)
    delta = (peak * prof).astype(np.float32)
    delta[~(seg[box] > 0)] = 0
    hu[box] += delta
    return float(peak)


# -----------------------------
# 7. Quality control
# -----------------------------
def _chk(passed, value, threshold, message):
    return {"passed": bool(passed), "value": value, "threshold": threshold, "message": message}


def qc_insertion(hu_b, seg_b, ves_b, hu_a, seg_a, spacing, expected_radius_mm, me, p, host_range):
    """Checks of one insertion inside its edit box (_b: backup before, _a: after; the new tumour is label NEW in seg_a)."""
    sp = np.asarray(spacing, np.float64)
    tum = seg_a == NEW
    n = int(tum.sum())
    checks = {"nonempty": _chk(n > 0, n, 1, f"{n} new tumour voxels")}
    if n:
        slack = me["max_displacement_mm"] if me["applied"] else 0.0
        host = seg_b > 0
        permitted = ndi.distance_transform_edt(~host, sampling=sp) <= slack if slack > 0 else host
        frac_out = float((tum & ~permitted).sum()) / n
        checks["tumor_in_liver"] = _chk(frac_out <= p.max_outside_liver_frac, frac_out, p.max_outside_liver_frac,
                                        f"{100 * frac_out:.2f}% of the tumour outside the host liver (+{slack:.1f} mm warp slack)")
        if p.vessels:
            ov = float((tum & ves_b).sum()) / n
            checks["vessel_overlap"] = _chk(ov <= p.max_vessel_overlap_frac, ov, p.max_vessel_overlap_frac, f"{100 * ov:.2f}% in vessels")
        r = equivalent_radius_mm(tum, sp)
        rel = abs(r - expected_radius_mm) / max(expected_radius_mm, 1e-6)
        checks["radius"] = _chk(rel <= p.radius_tol, {"actual_mm": r, "expected_mm": expected_radius_mm}, p.radius_tol,
                                f"equivalent radius {r:.2f} mm, expected {expected_radius_mm:.2f} mm")
        vals = hu_a[tum].astype(np.float64)
        d = ndi.distance_transform_edt(~tum, sampling=sp)
        shell = (d >= p.host_shell_mm[0]) & (d <= p.host_shell_mm[1]) & (seg_b == 1) & ~ves_b
        ref = float(hu_b[shell].mean()) if shell.sum() >= 30 else float(hu_b[seg_b == 1].mean())
        contrast, std = float(vals.mean() - ref), float(vals.std())
        lo, hi = p.contrast_hu_bounds
        checks["contrast"] = _chk(lo <= contrast <= hi, contrast, list(p.contrast_hu_bounds), f"tumour minus parenchyma {contrast:.1f} HU")
        lo, hi = p.tumor_std_hu_bounds
        checks["texture_std"] = _chk(lo <= std <= hi, std, list(p.tumor_std_hu_bounds), f"tumour HU std {std:.1f}")
        ncomp = int(ndi.label(tum, structure=STRUCT26)[1])
        checks["components"] = _chk(ncomp == 1, ncomp, 1, f"{ncomp} connected component(s)")
    fin = bool(np.isfinite(hu_a).all())
    lo_r, hi_r = min(host_range[0], -1024.0), max(host_range[1], 3071.0)
    in_range = fin and float(hu_a.min()) >= lo_r and float(hu_a.max()) <= hi_r
    checks["finite_range"] = _chk(in_range, [float(np.nanmin(hu_a)), float(np.nanmax(hu_a))], [lo_r, hi_r], "HU finite and in range")
    if me["applied"]:
        ok = me["jacobian_min"] >= p.min_jacobian and me["max_displacement_mm"] <= p.max_displacement_mm
        checks["displacement"] = _chk(ok, {"jacobian_min": me["jacobian_min"], "max_displacement_mm": me["max_displacement_mm"]},
                                      {"min_jacobian": p.min_jacobian, "max_displacement_mm": p.max_displacement_mm}, "warp valid")
    reasons = [k for k, v in checks.items() if not v["passed"]]
    return {"passed": not reasons, "reasons": reasons, "checks": checks}


# -----------------------------
# One insertion, several insertions, a host run
# -----------------------------
def insert_one_improved(hu, seg, vessels, spacing, samples, rng, p, phase="unknown", host_range=None):
    """One tumour into float32 hu / uint8 seg / bool vessels IN PLACE (all writes inside an edit box that is restored when QC rejects).
    Returns an info dict with accepted (bool), reason, qc and the insertion details."""
    sp = np.asarray(spacing, np.float64)
    host_range = host_range or (float(hu.min()), float(hu.max()))
    k = pick_sample(samples, p.target_radius_mm, rng, p)
    patch_rs, m_rs, scale = prepare_patch(samples["patches"][k], samples["masks"][k], spacing, samples["work_mm"], p.target_radius_mm, p)
    info = {"accepted": False, "reason": "", "sample": k, "scale": scale}
    if not m_rs.any():
        return dict(info, reason="mask_vanished")
    r_eq = equivalent_radius_mm(m_rs, sp)
    info["resampled_radius_mm"] = r_eq
    cen = np.rint(np.argwhere(m_rs).mean(0)).astype(int)
    m_idx = np.argwhere(m_rs)
    maps = HostMaps(seg, vessels if p.vessels else None, sp)
    cands, counts = candidate_centres(maps, r_eq, cen, m_rs.shape, seg.shape, p)
    info["candidate_counts"] = counts
    if not len(cands):
        empty = next((key for key in ("depth", "vessel", "tumour_gap", "window") if counts[key] == 0), "window")
        return dict(info, reason=f"no_site_{empty}")
    centre = None
    for _ in range(p.fit_attempts):
        c = cands[rng.integers(len(cands))]
        depth, vdist, tdist = maps.stats(m_idx + (c - cen))
        if depth >= p.margin_liver_mm and vdist >= p.margin_vessel_mm and tdist >= p.exclude_mm:
            centre, info["surface_clearance_mm"] = c, {"liver": depth, "vessel": vdist, "tumour": tdist}
            break
    if centre is None:
        return dict(info, reason="shape_clearance")
    start = centre - cen
    radius = p.target_radius_mm or r_eq
    intensity = p.mass_intensity if p.mass_intensity is not None else MASS_INTENSITY[size_class_of(radius)]
    gmax = p.gamma_factor * radius
    win = tuple(slice(int(s), int(s + n)) for s, n in zip(start, m_rs.shape))
    reach = gmax + max(abs(p.capsule_inner_mm), abs(p.capsule_outer_mm)) + 2.0
    roi = compute_roi(seg.shape, sp, start + np.argwhere(m_rs).mean(0), reach, pad_vox=8)
    box = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(win, roi))
    hu_b, seg_b, ves_b = hu[box].copy(), seg[box].copy(), vessels[box].copy()
    bl = blend_at(hu, seg, vessels, sp, patch_rs, m_rs, start, rng, p)
    if bl is None:
        return dict(info, reason="no_liver_voxels")
    warp_centre = start + np.argwhere(bl["mask"]).mean(0)  # centroid of the placed tumour (float voxel index)
    me = mass_effect(hu, seg, vessels, sp, warp_centre, radius, intensity, p)
    cap = apply_capsule(hu, seg, sp, phase, p)
    r_pre = equivalent_radius_mm(bl["mask"], sp)
    expected = expected_radius_after_warp(p.target_radius_mm or r_pre, gmax, me["intensity_applied"]) if me["applied"] else (p.target_radius_mm or r_pre)
    qc = qc_insertion(hu_b, seg_b, ves_b, hu[box], seg[box], sp, expected, me, p, host_range)
    accepted = qc["passed"] or p.on_fail == "flag"
    if accepted:
        sub = seg[box]
        n_new = int((sub == NEW).sum())
        sub[sub == NEW] = 2
    else:
        hu[box], seg[box], vessels[box] = hu_b, seg_b, ves_b
        n_new = 0
    info.update(accepted=accepted, flagged=accepted and not qc["passed"], reason="" if qc["passed"] else "qc:" + ";".join(qc["reasons"]),
                qc=qc, window=win, box=box, centre=centre, radius_requested_mm=p.target_radius_mm, radius_pre_warp_mm=r_pre,
                radius_expected_mm=expected, shift_hu=bl["shift_hu"], host_ref_hu=bl["host_ref_hu"], gen_ref_hu=bl["gen_ref_hu"],
                noise_std=bl["noise_std"], n_label=n_new, mass_effect={k_: v for k_, v in me.items() if k_ != "roi"}, capsule_hu=cap)
    return info


def insert_tumours(hu, seg, spacing, samples, rng, n=1, params=None, phase="unknown", vessels=None):
    """Up to n insertions into copies of a host. samples: dict with patches (k,P,P,P) in [-1,1], masks, work_mm and optionally
    radius_mm (an npz written by syn_training_improved.save_samples or Syn_Img_Pipeline_fixed). Returns (hu float32, seg uint8,
    vessels bool (warped), infos: one dict per attempted insertion)."""
    p = params or InsertParams()
    samples = dict(samples)
    samples["work_mm"] = float(samples.get("work_mm", 1.0))
    assert len(samples["patches"]), "no generated samples"
    out_hu, out_seg = hu.astype(np.float32).copy(), seg.astype(np.uint8).copy()
    if vessels is None:
        vessels = segment_vessels(out_hu, out_seg, spacing, p)[0] if p.vessels else np.zeros(seg.shape, bool)
    vessels = np.asarray(vessels, bool).copy()
    host_range = (float(out_hu.min()), float(out_hu.max()))
    infos = []
    for _ in range(n):
        info = insert_one_improved(out_hu, out_seg, vessels, spacing, samples, rng, p, phase, host_range)
        infos.append(info)
        logger.info(f"insertion: {'accepted' if info['accepted'] else 'rejected'}{' (flagged)' if info.get('flagged') else ''}"
                    f"{' ' + info['reason'] if info['reason'] else ''}; r {info.get('resampled_radius_mm', float('nan')):.1f} mm, "
                    f"mass effect I={info.get('mass_effect', {}).get('intensity_applied', 0):g}")
    return out_hu, out_seg, vessels, infos


QC_COLUMNS = ("case", "attempt", "accepted", "flagged", "reason", "sample", "scale", "radius_requested_mm", "resampled_radius_mm",
              "radius_expected_mm", "radius_actual_mm", "centre_x", "centre_y", "centre_z", "vessel_clearance_mm", "liver_clearance_mm",
              "mass_intensity", "jacobian_min", "max_displacement_mm", "shift_hu", "contrast_hu", "n_label")


def qc_row(cid, j, info):
    qc = info.get("qc", {}).get("checks", {})
    c = info.get("centre")
    cl = info.get("surface_clearance_mm", {})
    me = info.get("mass_effect", {})
    return {"case": cid, "attempt": j, "accepted": info["accepted"], "flagged": info.get("flagged", False), "reason": info["reason"],
            "sample": info.get("sample"), "scale": info.get("scale"), "radius_requested_mm": info.get("radius_requested_mm"),
            "resampled_radius_mm": info.get("resampled_radius_mm"), "radius_expected_mm": info.get("radius_expected_mm"),
            "radius_actual_mm": (qc.get("radius", {}).get("value") or {}).get("actual_mm"),
            "centre_x": None if c is None else int(c[0]), "centre_y": None if c is None else int(c[1]), "centre_z": None if c is None else int(c[2]),
            "vessel_clearance_mm": cl.get("vessel"), "liver_clearance_mm": cl.get("liver"), "mass_intensity": me.get("intensity_applied"),
            "jacobian_min": me.get("jacobian_min"), "max_displacement_mm": me.get("max_displacement_mm"), "shift_hu": info.get("shift_hu"),
            "contrast_hu": qc.get("contrast", {}).get("value"), "n_label": info.get("n_label", 0)}


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.4g}"
    return str(v)


def run_insert_improved(samples_path, ids, out_dir, data_dir=S.DATA_DIR, n=S.INSERTIONS, params=None, rng=None, phase="unknown", fig_path=None):
    """Insert into the NIfTI hosts `ids` (the train part only: S.split_cases), write aug_volume-<id>.nii (int16),
    aug_segmentation-<id>.nii (uint8), qc_report.csv, insert_summary.json and a host / augmented / labels montage."""
    p = params or InsertParams()
    rng = rng or np.random.default_rng(S.SEED)
    assert not os.path.abspath(out_dir).startswith(os.path.abspath(data_dir)), "never write augmented files into the raw data dir"
    d = S.load_npz(samples_path)
    samples = {"patches": d["patches"], "masks": d["masks"], "work_mm": float(d["work_mm"]) if "work_mm" in d.files else 1.0,
               "radius_mm": d["radius_mm"] if "radius_mm" in d.files else None}
    if not len(samples["patches"]):
        logger.error("no generated samples: nothing to insert")
        return {"hosts_written": 0}
    os.makedirs(out_dir, exist_ok=True)
    cases, rows, shown, written = S.pair_cases(data_dir, ids), [], [], 0
    for cid in ids:
        vol, seg, sp, aff = S.load_case(*cases[cid])
        hu, seg_new, _, infos = insert_tumours(vol, seg, sp, samples, rng, n, p, phase)
        rows += [qc_row(cid, j, i) for j, i in enumerate(infos)]
        ok = [i for i in infos if i["accepted"]]
        if not ok:
            logger.warning(f"case {cid}: no tumour accepted ({[i['reason'] for i in infos]}), not written")
            continue
        S.save_nifti(S.to_int16(hu), aff, os.path.join(out_dir, f"aug_volume-{cid}.nii"), np.int16)
        S.save_nifti(seg_new, aff, os.path.join(out_dir, f"aug_segmentation-{cid}.nii"), np.uint8)
        written += 1
        if len(shown) < 4:
            shown.append((cid, S.to_int16(vol), S.to_int16(hu), seg, seg_new, ok[0], sp))
    with open(os.path.join(out_dir, "qc_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(QC_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in QC_COLUMNS})
    reasons = {}
    for r in rows:
        if r["reason"]:
            for x in r["reason"].replace("qc:", "").split(";"):
                reasons[x] = reasons.get(x, 0) + 1
    summary = {"hosts": len(ids), "hosts_written": written, "attempts": len(rows), "accepted": sum(r["accepted"] for r in rows),
               "flagged": sum(r["flagged"] for r in rows), "reasons": reasons, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
               "phase": phase}
    S.write_json_atomic(os.path.join(out_dir, "insert_summary.json"), summary)
    if shown:
        S.save_insert_montage(fig_path or os.path.join(out_dir, "insertions.png"), shown)
    logger.info(f"improved insertion: {summary['accepted']}/{summary['attempts']} accepted, {written}/{len(ids)} hosts written -> {out_dir}; "
                f"reasons {reasons}")
    return summary


# -----------------------------
# Synthetic stand-ins (tests, notebook)
# -----------------------------
def toy_host(shape=(160, 160, 48), spacing=(0.8, 0.8, 2.5), seed=0):
    """Synthetic host: air, body (40 HU), ellipsoidal liver (110 HU, correlated noise), four bright tubular vessels (190 HU, label 1 like
    LiTS) and one existing tumour (50 HU, label 2). Returns (hu float32, seg uint8, spacing, true vessel mask)."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing, np.float64)
    g = np.stack(np.meshgrid(*[(np.arange(n) - n / 2) * s for n, s in zip(shape, sp)], indexing="ij"))
    body = (g[0] / 62) ** 2 + (g[1] / 58) ** 2 <= 1
    liver = (g[0] / 55) ** 2 + (g[1] / 46) ** 2 + (g[2] / 50) ** 2 <= 1
    hu = np.where(body, 40.0, -1000.0)
    hu = np.where(liver, 110.0, hu)
    ves = np.zeros(shape, bool)
    for a, b, r in ((-20, -15, 3.0), (25, 10, 2.5), (0, 30, 2.0), (-35, 20, 2.0)):
        ves |= ((g[0] - a) ** 2 + (g[1] - b) ** 2 <= r * r) & liver
    hu = np.where(ves, 190.0, hu)
    tum = ((g[0] - 30) ** 2 + (g[1] + 25) ** 2 + (g[2] - 10) ** 2 <= 8.0 ** 2) & liver
    hu = np.where(tum, 50.0, hu)
    noise = ndi.gaussian_filter(rng.standard_normal(shape), 0.8 / sp)
    hu = hu + noise / noise.std() * 10.0
    seg = np.where(liver, 1, 0).astype(np.uint8)
    seg[tum] = 2
    return hu.astype(np.float32), seg, tuple(float(s) for s in sp), ves


def toy_patch(p=64, radius_mm=10.0, work_mm=1.0, seed=0, tumour_hu=50.0, liver_hu=110.0):
    """Lobulated synthetic tumour patch in [-1,1] with its mask (a stand-in for a generated sample; insertion does not care where the
    patch came from)."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[(np.arange(p) - (p - 1) / 2) * work_mm] * 3, indexing="ij"))
    rad = np.sqrt((g ** 2).sum(0))
    bump = ndi.gaussian_filter(rng.standard_normal((p,) * 3), 4.0 / work_mm)
    bump /= np.abs(bump).max() + 1e-8
    mask = rad <= radius_mm * (1 + 0.15 * bump)
    lab, n = ndi.label(mask, structure=STRUCT26)
    if n > 1:
        mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    tex = ndi.gaussian_filter(rng.standard_normal((p,) * 3), 1.0)
    hu = np.where(mask, tumour_hu, liver_hu) + tex / tex.std() * 8.0
    return S.to_unit(hu.astype(np.float32)), mask


# -----------------------------
# Entry point
# -----------------------------
def params_from_env():
    size = os.environ.get("LITS_INS_SIZE", "medium").strip().lower()
    target = None if size in ("keep", "none", "") else (SIZES_MM[size] if size in SIZES_MM else float(size))
    return InsertParams(target_radius_mm=target, vessels=os.environ.get("LITS_INS_NO_VESSELS", "0") != "1",
                        mass_effect=os.environ.get("LITS_INS_NO_MASS", "0") != "1", capsule=os.environ.get("LITS_INS_CAPSULE", "0") == "1",
                        on_fail=os.environ.get("LITS_INS_ON_FAIL", "reject"))


def main():
    assert S.GAN_SOURCE == "nifti", "insertion_improved.py writes NIfTI hosts; the cache source keeps using Syn_Img_Pipeline_fixed.py"
    S.setup_logging(S.RESULTS_DIR)
    S.set_seed(S.SEED)
    p = params_from_env()
    ids = S.split_cases(restrict=S.parse_volume_list(S.VOLUME_LIST) if S.VOLUME_LIST else None)
    max_hosts = int(os.environ.get("LITS_INS_MAX_HOSTS", 0))
    ids = ids[:max_hosts] if max_hosts else ids
    out = os.environ.get("LITS_INS_OUT", os.path.join(S.RESULTS_DIR, "aug_improved"))
    logger.info(f"improved insertion: {len(ids)} hosts, target radius {p.target_radius_mm} mm, vessels {p.vessels}, mass effect {p.mass_effect}, "
                f"capsule {p.capsule}, on_fail {p.on_fail}")
    run_insert_improved(os.path.join(S.RESULTS_DIR, "synthetic", "samples.npz"), ids, out, n=int(os.environ.get("LITS_INS_N", S.INSERTIONS)),
                        params=p, phase=os.environ.get("LITS_INS_PHASE", "unknown"))


if __name__ == "__main__":
    main()
