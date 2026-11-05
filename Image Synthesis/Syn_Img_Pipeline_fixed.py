"""Corrected multi-GAN liver-tumour synthesis pipeline (replaces Syn_Img_Pipeline_Code.py, which is kept as is).

    python Syn_Img_Pipeline_fixed.py          # all stages; LITS_GAN_STAGE=<stage> runs one (it reads the earlier stages' files from the results dir)
    python test_syn_pipeline_fixed.py         # unit / smoke tests

Stages (LITS_GAN_STAGE = all | prepare | train | generate | insert):
  prepare   per-volume: crop liver bbox -> resample to 1 mm -> fixed HU window -> [-1,1] -> pad once with air (-1)
            -> tumour patches sampled per connected component (+ healthy patches for evaluation)
  train     two 3D DCGANs, one 3D WGAN-GP, an Aggregator (adversarial, generators detached, L1-anchored to the
            generator mean) and a supervised mask head (BCE+Dice on real patch/mask pairs plus healthy patches as
            all-zero negatives; whole volumes are held out for its evaluation); losses logged, PNG montages, checkpoints
  generate  LITS_GAN_FUSION=aggregator|mixture ('mixture' = one random generator per sample, no aggregator);
            the mask head labels each sample, the mask is cleaned (threshold 0.5, largest component, min voxels,
            not touching the patch border) and invalid samples are rejected
  insert    host-aware insertion into train-part volumes only; augmented files go to <results>/aug (HU int16)
            (cache source: slabs of augmented cache slices -> <results>/aug_slabs.npz + gan_summary.json, see below)

Renames vs the original: StyleTransfer3D -> MaskHead3D (it is now a supervised segmenter, not a style net).
Generator contract: z (B,ld,1,1,1) -> ConvT(ld,8b,seed,1,0) -> seed^3 -> four k4/s2/p1 layers -> PATCH^3 with
seed = PATCH/16, so PATCH must be a multiple of 16 (16, 32, 48, 64, ...; 64 is the default and uses the usual 4^3 seed).
Discriminators end in a patch map that is averaged to ONE score per sample (B,).
Orientation: patches/insertion work in array-voxel space with per-axis spacing from the header zooms; the
raw LiTS arrays have mixed axcodes (LAS/RAS/LPS), so an inserted tumour may be mirrored relative to anatomy
(irrelevant for roughly isotropic blobs).
Device: LITS_DEVICE=auto|cpu|mps|cuda. auto = cuda if present, otherwise cpu: measured on an M4 Pro (batch 4, 64^3, width 32)
one full training batch took ~1.4 s on cpu and ~2.9 s on mps (measured twice, machine not idle), so mps is opt-in (LITS_DEVICE=mps).
PYTORCH_ENABLE_MPS_FALLBACK=1 is set so an op missing on MPS runs on CPU; conv3d / conv_transpose3d and the gradient-penalty
double backward run natively on mps.

Split: the segmenter's original rule, sklearn train_test_split over the lexicographically sorted volume file names of
LITS_DATA_DIR (test_size=0.2, random_state=42; 104 train / 27 held-out of the 131 LiTS volumes). GAN training AND insertion
hosts come only from the train part, so the segmenter's held-out volumes (its validation and test set) never reach the GAN.
Volumes whose image header zooms are exactly 1.0 x 1.0 x 1.0 (LiTS volumes 28-47: placeholder spacing, every mm-based step would
be wrong) are detected from the NIfTI header, logged and dropped from the GAN as well.

Cache source (Colab: no raw NIfTI, only the uint8 slice cache of prepare_data_v2.py): LITS_GAN_SOURCE=cache. A volume is then
rebuilt from the cache: HU-like float = u8/255*(hi-lo)+lo (the exact inverse of the cache window, so x = u8/255*2-1 is the GAN's
[-1,1] window), slice axis moved last -> (256,256,Z), spacing (nx/256*sx, ny/256*sy, sz) from per_volume.csv, canonical orientation as
cached. Everything else (patch sampling per component, networks, mask gate, feathered insertion) is the NIfTI code, run at the
working voxel size LITS_GAN_SPACING_MM (1.0 for nifti, 1.5 for cache; every mm-based constant is converted, at 1.0 they are unchanged).
LITS_GAN_VOLUME_LIST (json list of volume names, or {"volumes": [...]}) restricts GAN patch extraction AND insertion hosts; it is
intersected with the original-split train part and minus placeholder-spacing volumes, so the 27 held-out volumes are never loaded
(CacheSource.load refuses them). The insert stage then writes <results>/aug_slabs.npz (one slab = the z range of an insertion
window, merged when two windows of one host overlap in z; images/masks are the cache slices of the AUGMENTED host, unchanged voxels
bit-identical) and gan_summary.json. aug_slabs.npz keys: images uint8 (M,256,256), masks uint8 {0,1,2}, slab_id int32 (M,), host_vol (S,)
names, z0 (S,), new_tumor_vox (S,), n_slices (S,); the same file with zero slabs is written when nothing was accepted. It keeps inserting until LITS_GAN_NUM_AUG slabs are accepted or LITS_GAN_MAX_ATTEMPTS host
visits are used. Resume after a disconnect: LITS_GAN_RESUME=1 (train continues from checkpoints/gan_last.pt incl. optimiser state,
finished prepare / generate / insert outputs are kept); LITS_GAN_MAX_SECONDS stops training after the epoch that crosses the budget.
GAN steps run fp32 (WGAN-GP double backward), without per-step host syncs (losses stay on the device until the epoch end; the patch
set lives on the device as fp16).

Config via env vars (defaults at the top of the module): LITS_DATA_DIR, LITS_GAN_RESULTS_DIR, LITS_GAN_STAGE, LITS_DEVICE,
LITS_WIN_LO/HI, LITS_GAN_PATCH, LITS_GAN_LATENT, LITS_GAN_WIDTH, LITS_GAN_BATCH, LITS_GAN_EPOCHS, LITS_GAN_SEED,
LITS_GAN_MAX_VOLUMES, LITS_GAN_MAX_SLICE_MM, LITS_GAN_CAP_COMP, LITS_GAN_CAP_VOL, LITS_GAN_CAP_HEALTHY, LITS_GAN_JITTER,
LITS_GAN_MIN_TUMOR_VOX, LITS_GAN_POS_WEIGHT, LITS_GAN_MASK_STEPS, LITS_GAN_HOLDOUT, LITS_GAN_AGG_L1, LITS_GAN_SAMPLE_EVERY,
LITS_GAN_NUM_SAMPLES, LITS_GAN_FUSION, LITS_GAN_MIN_MASK_VOX, LITS_GAN_EDGE_MARGIN, LITS_GAN_INSERTIONS, LITS_GAN_FEATHER_MM,
LITS_GAN_EXCLUDE_MM; cache source: LITS_GAN_SOURCE, LITS_CACHE_DIR, LITS_PER_VOLUME, LITS_GAN_SPACING_MM, LITS_GAN_VOLUME_LIST,
LITS_GAN_NUM_AUG, LITS_GAN_MAX_ATTEMPTS, LITS_GAN_GEN_ATTEMPTS, LITS_GAN_RESUME, LITS_GAN_MAX_SECONDS, LITS_GAN_DATA_MB, LITS_GAN_CKPT_EVERY, LITS_GAN_FORCE.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede the torch import
import re
import csv
import glob
import json
import time
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; data/ lives here
DATA_DIR = os.environ.get("LITS_DATA_DIR", os.path.join(ROOT, "data", "raw"))
RESULTS_DIR = os.environ.get("LITS_GAN_RESULTS_DIR", os.path.join(ROOT, "results", "gan_fixed"))
WIN_LO = float(os.environ.get("LITS_WIN_LO", -100))  # one window shared with the segmenter pipeline
WIN_HI = float(os.environ.get("LITS_WIN_HI", 250))
PATCH = int(os.environ.get("LITS_GAN_PATCH", 64))
LATENT_DIM = int(os.environ.get("LITS_GAN_LATENT", 64))
WIDTH = int(os.environ.get("LITS_GAN_WIDTH", 32))  # base channel count of every GAN net
BATCH_SIZE = int(os.environ.get("LITS_GAN_BATCH", 4))
EPOCHS = int(os.environ.get("LITS_GAN_EPOCHS", 50))
MAX_VOLUMES = int(os.environ.get("LITS_GAN_MAX_VOLUMES", 0))  # 0 = all usable volumes of the train part
NUM_SAMPLES = int(os.environ.get("LITS_GAN_NUM_SAMPLES", 16))
INSERTIONS = int(os.environ.get("LITS_GAN_INSERTIONS", 1))
FUSION = os.environ.get("LITS_GAN_FUSION", "aggregator")
SEED = int(os.environ.get("LITS_GAN_SEED", 42))
MAX_SLICE_MM = float(os.environ.get("LITS_GAN_MAX_SLICE_MM", 2.5))  # GAN training skips thicker volumes (0 = keep all)
CAP_COMP = int(os.environ.get("LITS_GAN_CAP_COMP", 4))  # patches per tumour component
CAP_VOL = int(os.environ.get("LITS_GAN_CAP_VOL", 24))  # tumour patches per volume
CAP_HEALTHY = int(os.environ.get("LITS_GAN_CAP_HEALTHY", 4))
JITTER = int(os.environ.get("LITS_GAN_JITTER", 4))  # voxels (1 mm) of random centre jitter
MIN_TUMOR_VOX = int(os.environ.get("LITS_GAN_MIN_TUMOR_VOX", 30))  # components smaller than this (1 mm voxels) are ignored
POS_WEIGHT = float(os.environ.get("LITS_GAN_POS_WEIGHT", 5))  # patches are tumour-centred (~10% positive), not 0.5%
MIN_MASK_VOX = int(os.environ.get("LITS_GAN_MIN_MASK_VOX", 100))
EDGE_MARGIN = int(os.environ.get("LITS_GAN_EDGE_MARGIN", 4))  # generated mask must stay this far from the patch border
MASK_STEPS = int(os.environ.get("LITS_GAN_MASK_STEPS", 1))  # mask-head updates per GAN batch (it learns slower than the GANs move)
HOLDOUT = float(os.environ.get("LITS_GAN_HOLDOUT", 0.2))  # fraction of volumes held out of mask-head training (eval only)
AGG_L1 = float(os.environ.get("LITS_GAN_AGG_L1", 5.0))  # anchor of the aggregator output to the generator mean (stops tanh saturation)
SAMPLE_EVERY = int(os.environ.get("LITS_GAN_SAMPLE_EVERY", 5))
FEATHER_MM = float(os.environ.get("LITS_GAN_FEATHER_MM", 2.5))  # alpha gaussian sigma (2-3 voxels at 1 mm)
EXCLUDE_MM = float(os.environ.get("LITS_GAN_EXCLUDE_MM", 5))  # keep-out margin around existing tumours
GAN_SOURCE = os.environ.get("LITS_GAN_SOURCE", "nifti").lower()  # nifti | cache
CACHE_DIR = os.environ.get("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache_v2"))
PER_VOLUME = os.environ.get("LITS_PER_VOLUME", os.path.join(ROOT, "analysis", "output", "per_volume.csv"))
WORK_MM = float(os.environ.get("LITS_GAN_SPACING_MM", 1.5 if GAN_SOURCE == "cache" else 1.0))  # working voxel size of patches / generator
VOLUME_LIST = os.environ.get("LITS_GAN_VOLUME_LIST", "")
NUM_AUG = int(os.environ.get("LITS_GAN_NUM_AUG", 32))  # cache source: target number of accepted slabs
MAX_ATTEMPTS = int(os.environ.get("LITS_GAN_MAX_ATTEMPTS", 0))  # host visits in the insert stage; 0 = 5 x NUM_AUG
GEN_ATTEMPTS = int(os.environ.get("LITS_GAN_GEN_ATTEMPTS", 0))  # generated candidates tried by the mask gate; 0 = 50 x pool size
RESUME = os.environ.get("LITS_GAN_RESUME", "0") == "1"
MAX_SECONDS = float(os.environ.get("LITS_GAN_MAX_SECONDS", 0))  # 0 = no time limit on GAN training
DATA_MB = float(os.environ.get("LITS_GAN_DATA_MB", 3072))  # patch set kept on the device up to this size, else pinned host memory
CKPT_EVERY = int(os.environ.get("LITS_GAN_CKPT_EVERY", 5))  # gan_last.pt (~240 MB with optimiser state at width 32) every N epochs, always at the end
FORCE = os.environ.get("LITS_GAN_FORCE", "0") == "1"
NOISE_SIGMA_MM = 1.5  # high-pass scale used to estimate / re-inject CT noise
NOISE_PRE_MM = 0.8  # pre-smoothing of the white noise (host high-pass lag-1 autocorrelation measured 0.3-0.76, white gives 0)
REAL_LABEL = 0.9  # one-sided label smoothing for DCGAN discriminators
N_CRITIC, GP_LAMBDA = 5, 10.0

logger = logging.getLogger("gan_fixed")


def pick_device():
    """LITS_DEVICE=auto -> cuda if present, else cpu (mps is opt-in, see docstring). An explicit cuda/mps request that this
    machine cannot serve is an error, not a silent fall back to a 100x slower cpu run."""
    want = os.environ.get("LITS_DEVICE", "auto").lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(want)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LITS_DEVICE=cuda but torch.cuda.is_available() is False (Colab: Runtime > Change runtime type > GPU)")
    if dev.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("LITS_DEVICE=mps but the mps backend is not available")
    return dev


def setup_backends(device):
    """fp32 everywhere for the GAN (WGAN-GP double backward): TF32 off; cuDNN autotune on cuda (fixed shapes)."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"seeds: random={seed} numpy={seed} torch={seed}")


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(results_dir, "pipeline.log"), mode="a"), logging.StreamHandler()],
        force=True)


def check_patch(p):
    assert p >= 16 and p % 16 == 0, f"PATCH must be a multiple of 16, got {p}"


# -----------------------------
# Normalisation, patch extraction, I/O
# -----------------------------
def to_unit(hu, lo=WIN_LO, hi=WIN_HI):
    """HU -> [-1,1] with one fixed window; air (<= lo) maps to -1."""
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) * 2.0 - 1.0


def from_unit(x, lo=WIN_LO, hi=WIN_HI):
    return (x + 1.0) * 0.5 * (hi - lo) + lo


def assert_unit_range(x, name="data", tol=1e-3):
    lo, hi = float(x.min()), float(x.max())
    assert lo >= -1 - tol and hi <= 1 + tol, f"{name} outside [-1,1]: [{lo:.3f},{hi:.3f}] (double normalisation?)"


def pad_air(vol, size, value):
    """Pad once by size//2 on every side; patch for centre c is then padded[c:c+size] (no low-edge shift)."""
    return np.pad(vol, size // 2, mode="constant", constant_values=value)


def extract_patch(padded, centre, size):
    c = [int(v) for v in centre]
    return padded[c[0]:c[0] + size, c[1]:c[1] + size, c[2]:c[2] + size]


def case_id(name):
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    assert m, f"no case id in {name}"
    return int(m.group(1))


def pair_cases(data_dir, ids=None):
    """{case id: (volume path, segmentation path)}, paired by id (not by sorted position)."""
    vols = {case_id(p): p for p in glob.glob(os.path.join(data_dir, "volume-*.nii*"))}
    segs = {case_id(p): p for p in glob.glob(os.path.join(data_dir, "segmentation-*.nii*"))}
    if ids is None:
        assert set(vols) == set(segs), f"unpaired ids: {sorted(set(vols) ^ set(segs))}"
        ids = sorted(vols)
    for i in ids:
        assert i in vols and i in segs, f"case {i}: volume/segmentation pair missing in {data_dir}"
    return {i: (vols[i], segs[i]) for i in ids}


def sorted_volume_names(data_dir):
    """Volume file names in the segmenter's order (lexicographic: volume-0, volume-1, volume-10, ...)."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "volume-*.nii*")))


def train_part_ids(data_dir):
    """Case ids of the segmenter's train part: the original train_test_split(test_size=0.2, random_state=42) over the sorted names."""
    names = sorted_volume_names(data_dir)
    assert names, f"no volume-*.nii* in {data_dir}"
    train_idx, _ = train_test_split(range(len(names)), test_size=0.2, random_state=42)
    return sorted(case_id(names[i]) for i in train_idx)


def placeholder_spacing(vpath):
    """True when the image header zooms are exactly 1.0 x 1.0 x 1.0 (LiTS volumes 28-47); header only, no voxel data is read."""
    return tuple(float(z) for z in nib.load(vpath).header.get_zooms()[:3]) == (1.0, 1.0, 1.0)


def parse_volume_list(path):
    """Case ids named by a json file: a list of names ('volume-5.nii', 'volume-5' or 5), {"volumes": [...]}, or a dict keyed by volume names."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if isinstance(obj.get("volumes"), list):
            obj = obj["volumes"]
        elif all(re.match(r"volume-\d+", str(k)) for k in obj):
            obj = list(obj)
        else:
            raise ValueError(f"{path}: expected a list of volume names, {{'volumes': [...]}} or a dict keyed by volume names")
    assert isinstance(obj, list) and obj, f"{path}: empty volume list"
    return {case_id(str(v)) for v in obj}


def split_cases(data_dir=None, max_volumes=None, restrict=None):
    """Case ids for GAN training and insertion hosts: the train part of the original split (the held-out part never
    reaches the GAN), minus volumes with placeholder header spacing (makes every mm-based step wrong).
    restrict: optional set of case ids (LITS_GAN_VOLUME_LIST); the result is its intersection with the above."""
    data_dir = data_dir or DATA_DIR
    max_volumes = MAX_VOLUMES if max_volumes is None else max_volumes
    train = train_part_ids(data_dir)
    paths = {case_id(p): p for p in glob.glob(os.path.join(data_dir, "volume-*.nii*"))}
    cand = train if restrict is None else [i for i in train if i in restrict]
    bad = [i for i in cand if placeholder_spacing(paths[i])]
    if bad:
        logger.info(f"dropping {len(bad)} train-part volumes with placeholder 1x1x1 header spacing: {bad}")
    ids = [i for i in cand if i not in bad]
    if restrict is not None:
        logger.info(f"volume list: {len(restrict)} named, {len(ids)} usable (in the train part, real spacing)")
    assert ids, f"no usable train-part volumes in {data_dir}"
    return ids[:max_volumes] if max_volumes else ids


class NiftiSource:
    """Raw LiTS NIfTI volumes (the original path). load -> (HU float32 (X,Y,Z), seg uint8, spacing mm, affine)."""
    kind = "nifti"

    def __init__(self, data_dir=None, work_mm=1.0):
        self.data_dir, self.work_mm, self._cases = data_dir or DATA_DIR, work_mm, None

    def split_cases(self, restrict=None, max_volumes=None):
        return split_cases(self.data_dir, max_volumes, restrict)

    def prepare(self, ids):
        self._cases = pair_cases(self.data_dir, ids)

    def load(self, cid):
        return load_case(*self._cases[cid])


class CacheSource:
    """The uint8 slice cache of prepare_data_v2.py as a volume source (no NIfTI needed). A volume is rebuilt as
    HU-like float u8/255*(hi-lo)+lo (exact inverse of the cache window), slice axis last -> (256,256,Z), labels uint8,
    spacing (nx/256*sx, ny/256*sy, sz) from per_volume.csv (cache = 2x in-plane downsample of the 512x512 raw slices).
    Axis order and canonical orientation are the cache's own (array axes 0/1 of a slice, not mirrored back)."""
    kind = "cache"

    def __init__(self, cache_dir=None, per_volume=None, work_mm=None):
        self.cache_dir, self.work_mm = cache_dir or CACHE_DIR, WORK_MM if work_mm is None else work_mm
        cd = self.cache_dir
        with open(os.path.join(cd, "config.json")) as f:
            cfg = json.load(f)
        self.lo, self.hi = float(cfg["win_lo"]), float(cfg["win_hi"])
        assert (self.lo, self.hi) == (WIN_LO, WIN_HI), f"cache window [{self.lo},{self.hi}] != LITS_WIN_LO/HI [{WIN_LO},{WIN_HI}]"
        with open(os.path.join(cd, "volumes.txt")) as f:
            self.names = f.read().split()
        self.vol_index = np.load(os.path.join(cd, "vol_index.npy"))
        assert len(self.names) == int(self.vol_index.max()) + 1 and (np.diff(self.vol_index) >= 0).all(), "bad vol_index"
        cnt = np.bincount(self.vol_index, minlength=len(self.names))
        self.start = np.concatenate([[0], np.cumsum(cnt)])
        self.images = np.load(os.path.join(cd, "images.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(cd, "masks.npy"), mmap_mode="r")
        assert len(self.images) == len(self.masks) == len(self.vol_index), "cache arrays disagree in length"
        self.cid_to_idx = {case_id(n): i for i, n in enumerate(self.names)}
        with open(per_volume or PER_VOLUME, newline="") as f:
            self.per_volume = {r["name"]: r for r in csv.DictReader(f)}
        missing = [n for n in self.names if n not in self.per_volume]
        assert not missing, f"per_volume.csv has no row for {missing[:5]} (spacing source)"
        tr, _ = train_test_split(range(len(self.names)), test_size=0.2, random_state=42)  # the segmenter's rule on volumes.txt order
        self.train_ids = sorted(case_id(self.names[i]) for i in tr)

    def raw_spacing(self, cid):
        r = self.per_volume[self.names[self.cid_to_idx[cid]]]
        return float(r["sx_mm"]), float(r["sy_mm"]), float(r["sz_mm"])

    def spacing(self, cid):
        r = self.per_volume[self.names[self.cid_to_idx[cid]]]
        sx, sy, sz = self.raw_spacing(cid)
        return (float(r["nx"]) / self.images.shape[1] * sx, float(r["ny"]) / self.images.shape[2] * sy, sz)

    def placeholder(self, cid):
        return self.raw_spacing(cid) == (1.0, 1.0, 1.0)  # LiTS 28-47: header carries 1x1x1 although the true spacing is unknown

    def train_part_ids(self):
        return list(self.train_ids)

    def split_cases(self, restrict=None, max_volumes=None):
        max_volumes = MAX_VOLUMES if max_volumes is None else max_volumes
        cand = self.train_ids if restrict is None else [i for i in self.train_ids if i in restrict]
        bad = [i for i in cand if self.placeholder(i)]
        if bad:
            logger.info(f"dropping {len(bad)} train-part volumes with placeholder 1x1x1 header spacing: {bad}")
        ids = [i for i in cand if i not in bad]
        if restrict is not None:
            unknown = sorted(restrict - set(self.cid_to_idx))
            held = sorted((restrict & set(self.cid_to_idx)) - set(self.train_ids))
            logger.info(f"volume list: {len(restrict)} named, {len(ids)} usable; ignored: {len(held)} held-out part {held}, "
                        f"{len(bad)} placeholder, {len(unknown)} unknown {unknown}")
        assert ids, f"no usable train-part volumes in the cache {self.cache_dir}"
        return ids[:max_volumes] if max_volumes else ids

    def prepare(self, ids):
        pass

    def load_u8(self, cid):
        """Cache slices of one TRAIN-part volume: (images (Z,256,256) uint8, masks (Z,256,256) uint8), copied out of the memmap."""
        assert cid in self.train_ids, f"volume {cid} is not in the original-split train part (held-out volumes are never loaded by the GAN)"
        i = self.cid_to_idx[cid]
        a, b = int(self.start[i]), int(self.start[i + 1])
        return np.array(self.images[a:b]), np.array(self.masks[a:b])

    def to_hu(self, u8):
        return u8.astype(np.float32) * np.float32((self.hi - self.lo) / 255.0) + np.float32(self.lo)

    def load(self, cid):
        img, msk = self.load_u8(cid)
        return self.to_hu(np.moveaxis(img, 0, 2)), np.ascontiguousarray(np.moveaxis(msk, 0, 2)), self.spacing(cid), None

    def name(self, cid):
        return self.names[self.cid_to_idx[cid]]


def make_source(kind=None):
    kind = kind or GAN_SOURCE
    assert kind in ("nifti", "cache"), f"LITS_GAN_SOURCE must be nifti or cache, got {kind}"
    return CacheSource() if kind == "cache" else NiftiSource(work_mm=WORK_MM)


def load_case(vpath, spath):
    vimg, simg = nib.load(vpath), nib.load(spath)
    vol = np.asarray(vimg.dataobj).astype(np.float32)
    seg = np.rint(np.asarray(simg.dataobj)).astype(np.uint8)
    assert vol.shape == seg.shape, f"shape mismatch {vol.shape} vs {seg.shape}"
    return vol, seg, tuple(float(z) for z in vimg.header.get_zooms()[:3]), vimg.affine


def save_nifti(arr, affine, path, dtype):
    """Fresh header from the host affine (a reused header could quantise values on disk)."""
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype), affine)
    img.set_data_dtype(dtype)
    nib.save(img, path)


def save_npz_atomic(path, **arrays):
    """np.savez to a temp file in the same directory, then os.replace: a disconnect never leaves a half-written file under `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


class _Npz(dict):
    @property
    def files(self):
        return list(self)


def load_npz(path):
    """All arrays of an .npz read eagerly and the file closed: a lazy np.load keeps the handle open, and on Windows an open
    file cannot be replaced (save_npz_atomic) or deleted. The result is a dict with an NpzFile-like .files."""
    with np.load(path) as z:
        return _Npz({k: z[k] for k in z.files})


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def to_int16(hu):
    return np.clip(np.rint(hu), -32768, 32767).astype(np.int16)


def liver_crop(seg, spacing, margin_mm):
    """Slices of the liver bbox plus a margin; None for an empty mask."""
    objs = ndi.find_objects((seg > 0).astype(np.int8))
    if not objs:
        return None
    return tuple(slice(max(0, s.start - int(np.ceil(margin_mm / sp)) - 1), min(n, s.stop + int(np.ceil(margin_mm / sp)) + 1))
                 for s, sp, n in zip(objs[0], spacing, seg.shape))


def resample_iso(arr, spacing, order, work_mm=1.0):
    """To isotropic work_mm voxels (1 mm for the NIfTI path)."""
    return ndi.zoom(arr, np.asarray(spacing) / work_mm, order=order, mode="nearest")


# -----------------------------
# Patch sampling
# -----------------------------
def sample_centres(tumor, rng, cap_comp=CAP_COMP, cap_vol=CAP_VOL, jitter=JITTER, min_vox=MIN_TUMOR_VOX):
    """Centres per connected tumour component (cap per component and per volume, random jitter).
    Round-robin over components so small tumours always get a patch before large ones get their k-th."""
    lab, _ = ndi.label(tumor, structure=np.ones((3, 3, 3)))
    per_comp = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        coords = np.argwhere(lab[sl] == i) + np.array([s.start for s in sl])
        if len(coords) < min_vox:
            continue
        pick = coords[rng.choice(len(coords), min(cap_comp, len(coords)), replace=False)]
        pick = pick + rng.integers(-jitter, jitter + 1, size=pick.shape)
        per_comp.append((i, np.clip(pick, 0, np.array(tumor.shape) - 1)))
    order = rng.permutation(len(per_comp))
    out = []
    for k in range(cap_comp):
        for j in order:
            comp, pick = per_comp[j]
            if k < len(pick) and len(out) < cap_vol:
                out.append((pick[k], comp))
    return out


def sample_healthy(seg_rs, seg_pad, rng, size=PATCH, cap=CAP_HEALTHY):
    """Liver-centred patches that contain no tumour label (negatives / evaluation for the mask head, never for GAN training)."""
    liver = np.argwhere(seg_rs == 1)
    out = []
    if len(liver) == 0:
        return out
    for c in liver[rng.choice(len(liver), min(len(liver), cap * 10), replace=False)]:
        if (extract_patch(seg_pad, c, size) == 2).sum() == 0:
            out.append(c)
        if len(out) >= cap:
            break
    return out


def vox_at(n_vox_1mm, work_mm):
    """A voxel count defined at 1 mm voxels (volume ~ n mm^3) converted to the working voxel size (unchanged at 1.0)."""
    return max(1, int(round(n_vox_1mm / work_mm ** 3)))


def prepare_patches(ids, out_path, data_dir=DATA_DIR, size=PATCH, rng=None, source=None):
    check_patch(size)
    rng = rng or np.random.default_rng(SEED)
    source = source or NiftiSource(data_dir)
    work = source.work_mm
    source.prepare(ids)
    min_vox, jitter = vox_at(MIN_TUMOR_VOX, work), max(1, int(round(JITTER / work)))  # mm-defined constants at the working voxel size
    imgs, masks, healthy, meta, healthy_cid = [], [], [], [], []
    for cid in ids:
        vol, seg, sp, _ = source.load(cid)
        if MAX_SLICE_MM and max(sp) > MAX_SLICE_MM:
            logger.info(f"case {cid}: slice thickness {max(sp):.1f} mm > {MAX_SLICE_MM}, skipped for GAN training")
            continue
        sl = liver_crop(seg, sp, (size / 2 + EDGE_MARGIN) * work)  # crop first, then resample (no whole-volume zoom)
        if sl is None:
            logger.warning(f"case {cid}: empty segmentation, skipped")
            continue
        vol_rs = to_unit(resample_iso(vol[sl], sp, 1, work))
        seg_rs = resample_iso(seg[sl], sp, 0, work)
        vol_pad, seg_pad = pad_air(vol_rs, size, -1.0), pad_air(seg_rs, size, 0)
        cs = sample_centres(seg_rs == 2, rng, jitter=jitter, min_vox=min_vox)
        for c, comp in cs:
            imgs.append(extract_patch(vol_pad, c, size))
            masks.append(extract_patch(seg_pad, c, size) == 2)
            meta.append((cid, comp))
        for c in sample_healthy(seg_rs, seg_pad, rng, size):
            healthy.append(extract_patch(vol_pad, c, size))
            healthy_cid.append(cid)
        logger.info(f"case {cid}: {len(cs)} tumour patches ({len({k for _, k in cs})} components), "
                    f"healthy so far {len(healthy)}")
    assert imgs, "no tumour patches extracted (every listed volume was too thick, had no tumour component above the size limit, or no liver)"
    imgs, masks = np.stack(imgs).astype(np.float16), np.stack(masks).astype(np.uint8)
    healthy = np.stack(healthy).astype(np.float16) if healthy else np.zeros((0, size, size, size), np.float16)
    assert_unit_range(imgs, "patches")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_npz_atomic(out_path, images=imgs, masks=masks, healthy=healthy, meta=np.array(meta), healthy_cid=np.array(healthy_cid, dtype=int),
                    work_mm=np.float32(work), ids=np.array(ids, dtype=int))
    logger.info(f"saved {len(imgs)} tumour / {len(healthy)} healthy patches ({size}^3 at {work} mm) -> {out_path}")
    return out_path


# -----------------------------
# Networks
# -----------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class DCGAN3DGenerator(nn.Module):
    def __init__(self, ld=LATENT_DIM, patch=PATCH, base=WIDTH):
        super().__init__()
        check_patch(patch)
        b, self.ld, self.patch = base, ld, patch
        # conv bias is dropped before BatchNorm (its gradient is exactly zero there)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(ld, b * 8, patch // 16, 1, 0, bias=False), nn.BatchNorm3d(b * 8), nn.ReLU(True),
            nn.ConvTranspose3d(b * 8, b * 4, 4, 2, 1, bias=False), nn.BatchNorm3d(b * 4), nn.ReLU(True),
            nn.ConvTranspose3d(b * 4, b * 2, 4, 2, 1, bias=False), nn.BatchNorm3d(b * 2), nn.ReLU(True),
            nn.ConvTranspose3d(b * 2, b, 4, 2, 1, bias=False), nn.BatchNorm3d(b), nn.ReLU(True),
            nn.ConvTranspose3d(b, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        assert z.shape[1:] == (self.ld, 1, 1, 1), f"z must be (B,{self.ld},1,1,1), got {tuple(z.shape)}"
        out = self.net(z)
        assert out.shape[2:] == (self.patch,) * 3, f"generator emitted {tuple(out.shape)}"
        return out


class WGAN3DGenerator(DCGAN3DGenerator):  # same architecture, different loss/optimiser
    pass


class _Disc3D(nn.Module):
    """Strided conv stack -> patch map -> mean = one score per sample (B,). No Sigmoid: scores are logits."""

    def __init__(self, norm, base=WIDTH):
        super().__init__()
        b = base

        def block(i, o):
            if norm == "bn":
                return [nn.Conv3d(i, o, 4, 2, 1, bias=False), nn.BatchNorm3d(o), nn.LeakyReLU(0.2, True)]
            return [nn.Conv3d(i, o, 4, 2, 1), nn.GroupNorm(1, o), nn.LeakyReLU(0.2, True)]  # per-sample norm (WGAN-GP)

        self.net = nn.Sequential(nn.Conv3d(1, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                 *block(b, b * 2), *block(b * 2, b * 4), *block(b * 4, b * 8),
                                 nn.Conv3d(b * 8, 1, 3, 1, 1, bias=norm == "bn"))  # critic: output bias cancels in the Wasserstein loss

    def forward(self, x):
        return self.net(x).mean(dim=(1, 2, 3, 4))


class DCGAN3DDiscriminator(_Disc3D):
    def __init__(self, base=WIDTH):
        super().__init__("bn", base)


class WGAN3DCritic(_Disc3D):
    def __init__(self, base=WIDTH):
        super().__init__("gn", base)


class Aggregator3DDiscriminator(_Disc3D):
    def __init__(self, base=WIDTH):
        super().__init__("bn", base)


class Aggregator3D(nn.Module):
    """Fuses three generator outputs (B,1,P,P,P each) into one; residual on the channel mean so it starts near averaging."""

    def __init__(self, base=8):
        super().__init__()
        b = base
        self.net = nn.Sequential(
            nn.Conv3d(3, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, 1, 3, 1, 1))

    def forward(self, a, b, c):
        x = torch.cat([a, b, c], 1)
        return torch.tanh(self.net(x) + x.mean(1, keepdim=True))


class MaskHead3D(nn.Module):
    """Supervised image -> tumour-mask logits (replaces StyleTransfer3D). Runs at half resolution with dilated
    convs (large receptive field, 8x cheaper than full resolution) and upsamples back to the input grid."""

    def __init__(self, base=16):
        super().__init__()
        b = base

        def blk(i, o, s=1, d=1):
            return [nn.Conv3d(i, o, 3, s, d, dilation=d), nn.GroupNorm(4, o), nn.ReLU(True)]

        self.enc = nn.Sequential(*blk(1, b, s=2), *blk(b, b, d=2), *blk(b, b, d=4), *blk(b, b))
        self.out = nn.Conv3d(b, 1, 3, 1, 1)

    def forward(self, x):
        h = F.interpolate(self.enc(x), size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.out(h)


def gradient_penalty(critic, real, fake, lam=GP_LAMBDA):
    """WGAN-GP on per-sample scalar scores; with a GroupNorm critic samples do not interact."""
    a = torch.rand(real.size(0), 1, 1, 1, 1, device=real.device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    score = critic(x)
    assert score.shape == (real.size(0),), f"critic must return (B,), got {tuple(score.shape)}"
    g, = torch.autograd.grad(score.sum(), x, create_graph=True)
    return lam * ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3, 4))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + eps)).mean()


class MultiGAN:
    """Two DCGANs, one WGAN-GP, an Aggregator (+ its discriminator) and the supervised mask head."""

    def __init__(self, device, ld=LATENT_DIM, patch=PATCH, width=WIDTH, pos_weight=POS_WEIGHT):
        self.device, self.ld, self.patch, self.width = device, ld, patch, width
        self.dc1_g, self.dc1_d = DCGAN3DGenerator(ld, patch, width), DCGAN3DDiscriminator(width)
        self.dc2_g, self.dc2_d = DCGAN3DGenerator(ld, patch, width), DCGAN3DDiscriminator(width)
        self.w_g, self.w_c = WGAN3DGenerator(ld, patch, width), WGAN3DCritic(width)
        self.aggr, self.ag_d = Aggregator3D(), Aggregator3DDiscriminator(width)
        self.mask = MaskHead3D()
        for m in (self.dc1_g, self.dc1_d, self.dc2_g, self.dc2_d, self.w_g, self.w_c, self.ag_d):
            m.apply(weights_init)
        for m in self.modules().values():
            m.to(device)
        dc = dict(lr=2e-4, betas=(0.5, 0.999))
        wg = dict(lr=1e-4, betas=(0.0, 0.9))
        self.opt = {
            "dc1_g": optim.Adam(self.dc1_g.parameters(), **dc), "dc1_d": optim.Adam(self.dc1_d.parameters(), **dc),
            "dc2_g": optim.Adam(self.dc2_g.parameters(), **dc), "dc2_d": optim.Adam(self.dc2_d.parameters(), **dc),
            "w_g": optim.Adam(self.w_g.parameters(), **wg), "w_c": optim.Adam(self.w_c.parameters(), **wg),
            "aggr": optim.Adam(self.aggr.parameters(), **dc), "ag_d": optim.Adam(self.ag_d.parameters(), **dc),
            "mask": optim.Adam(self.mask.parameters(), lr=1e-3)}
        self.pos_weight = torch.tensor(pos_weight, device=device)
        self.w_iter = 0
        self.sync = True  # True: step_* return python floats (.item(), one host sync each); False: detached device tensors (train loop)
        self.check_range = True  # per-batch [-1,1] assertion needs a host sync; train() validates the whole set once and turns it off

    def _v(self, t):
        return t.item() if self.sync else t.detach()

    def modules(self):
        return {"dc1_g": self.dc1_g, "dc1_d": self.dc1_d, "dc2_g": self.dc2_g, "dc2_d": self.dc2_d, "w_g": self.w_g,
                "w_c": self.w_c, "aggr": self.aggr, "ag_d": self.ag_d, "mask": self.mask}

    def train(self, mode=True):
        for m in self.modules().values():
            m.train(mode)

    def z(self, n):
        return torch.randn(n, self.ld, 1, 1, 1, device=self.device)

    @staticmethod
    def _bce(logits, target):
        return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))

    @staticmethod
    def _step(opt, loss):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def step_dcgan(self, which, real):
        g, d = (self.dc1_g, self.dc1_d) if which == 1 else (self.dc2_g, self.dc2_d)
        og, od = self.opt[f"dc{which}_g"], self.opt[f"dc{which}_d"]
        fake = g(self.z(real.size(0)))
        ld = self._bce(d(real), REAL_LABEL) + self._bce(d(fake.detach()), 0.0)
        self._step(od, ld)
        lg = self._bce(d(fake), 1.0)
        self._step(og, lg)
        return {f"d{which}": self._v(ld), f"g{which}": self._v(lg)}

    def step_wgan(self, real, update_g):
        fake = self.w_g(self.z(real.size(0))).detach()
        gp = gradient_penalty(self.w_c, real, fake)
        lc = self.w_c(fake).mean() - self.w_c(real).mean() + gp
        self._step(self.opt["w_c"], lc)
        out = {"w_critic": self._v(lc), "w_gp": self._v(gp)}
        if update_g:  # generator moves once every N_CRITIC critic steps
            lg = -self.w_c(self.w_g(self.z(real.size(0)))).mean()
            self._step(self.opt["w_g"], lg)
            out["w_g"] = self._v(lg)
        return out

    def step_aggregator(self, real):
        with torch.no_grad():  # generators detached; one shared z so the three outputs are views of one latent
            z = self.z(real.size(0))
            fakes = [self.dc1_g(z), self.dc2_g(z), self.w_g(z)]
        fused = self.aggr(*fakes)
        ld = self._bce(self.ag_d(real), REAL_LABEL) + self._bce(self.ag_d(fused.detach()), 0.0)
        self._step(self.opt["ag_d"], ld)
        lg = self._bce(self.ag_d(fused), 1.0)
        l1 = (fused - torch.stack(fakes).mean(0)).abs().mean()  # without it the discriminator wins and the fusion drives voxels to +-1
        self._step(self.opt["aggr"], lg + AGG_L1 * l1)
        return {"ag_d": self._v(ld), "ag_g": self._v(lg), "ag_l1": self._v(l1)}

    def step_mask(self, real, mask):
        logits = self.mask(real)
        bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=self.pos_weight)
        loss = bce + dice_loss(logits, mask)
        self._step(self.opt["mask"], loss)
        return {"mask": self._v(loss)}

    def train_batch(self, real, mask, mask_pool=None):
        """mask_pool() -> (images, masks) for the mask head (tumour + healthy negatives); default reuses (real, mask)."""
        if self.check_range:
            assert_unit_range(real, "real batch")
        self.w_iter += 1
        out = {}
        out.update(self.step_dcgan(1, real))
        out.update(self.step_dcgan(2, real))
        out.update(self.step_wgan(real, self.w_iter % N_CRITIC == 0))
        out.update(self.step_aggregator(real))
        for _ in range(MASK_STEPS):
            mx, mm = mask_pool() if mask_pool else (real, mask)
            out.update(self.step_mask(mx, mm))
        return out

    @torch.no_grad()
    def sample_all(self, n):
        self.train(False)
        z = self.z(n)
        out = {"dc1": self.dc1_g(z), "dc2": self.dc2_g(z), "wgan": self.w_g(z)}
        out["aggregator"] = self.aggr(out["dc1"], out["dc2"], out["wgan"])
        self.train(True)
        return out

    @torch.no_grad()
    def sample(self, n, fusion=FUSION):
        assert fusion in ("aggregator", "mixture"), fusion
        self.train(False)
        if fusion == "aggregator":
            z = self.z(n)
            out = self.aggr(self.dc1_g(z), self.dc2_g(z), self.w_g(z))
        else:  # one generator per output, no aggregator
            outs = torch.stack([g(self.z(n)) for g in (self.dc1_g, self.dc2_g, self.w_g)])
            pick = torch.randint(0, 3, (n,), device=self.device)
            out = outs[pick, torch.arange(n, device=self.device)]
        self.train(True)
        return out

    def save(self, path, epoch=None, elapsed=0.0, with_opt=False):
        """Atomic (temp file + replace). with_opt adds optimiser states, epoch and elapsed seconds for LITS_GAN_RESUME."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ck = {"cfg": dict(ld=self.ld, patch=self.patch, width=self.width),
              "modules": {k: m.state_dict() for k, m in self.modules().items()}}
        if with_opt:
            ck.update(epoch=epoch, elapsed=elapsed, w_iter=self.w_iter, opt={k: o.state_dict() for k, o in self.opt.items()})
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path, device):
        ck = torch.load(path, map_location=device, weights_only=True)
        net = cls(device, **ck["cfg"])
        for k, m in net.modules().items():
            m.load_state_dict(ck["modules"][k])
        return net

    def load_training_state(self, path):
        """Restore modules + optimisers from a with_opt checkpoint; returns (epoch, elapsed seconds), or (0, 0.0) if it has no optimiser state."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        assert ck["cfg"] == dict(ld=self.ld, patch=self.patch, width=self.width), f"checkpoint config {ck['cfg']} != this run"
        for k, m in self.modules().items():
            m.load_state_dict(ck["modules"][k])
        if "opt" not in ck:
            return 0, 0.0
        for k, o in self.opt.items():
            o.load_state_dict(ck["opt"][k])
        self.w_iter = int(ck.get("w_iter", 0))
        return int(ck["epoch"]), float(ck.get("elapsed", 0.0))


# -----------------------------
# Visualisation (files only, never plt.show)
# -----------------------------
def save_montage(path, rows):
    """rows: [(label, patches (n,P,P,P) in [-1,1], masks (n,P,P,P) or None)]; shows the axial slice with most mask."""
    n = max(len(a) for _, a, _ in rows)
    fig, axs = plt.subplots(len(rows), n, figsize=(1.8 * n, 1.9 * len(rows)), squeeze=False)
    for r, (label, arr, msk) in enumerate(rows):
        for j in range(n):
            ax = axs[r][j]
            ax.axis("off")
            if j >= len(arr):
                continue
            z = int(np.argmax(msk[j].sum(axis=(0, 1)))) if msk is not None and msk[j].any() else arr.shape[-1] // 2
            ax.imshow(arr[j][:, :, z].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
            if msk is not None and msk[j][:, :, z].any():
                ax.contour(msk[j][:, :, z].T.astype(float), levels=[0.5], colors="r", linewidths=0.8)
            if j == 0:
                ax.set_title(label, fontsize=8, loc="left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_insert_montage(path, cases):
    """cases: [(cid, host_hu, aug_hu, seg_old, seg_new, info, spacing)]: host | augmented | augmented + labels."""
    fig, axs = plt.subplots(len(cases), 3, figsize=(9, 3.1 * len(cases)), squeeze=False)
    for r, (cid, host, aug, s0, s1, info, sp) in enumerate(cases):
        w = info["window"]
        z = int(np.clip(info["centre"][2], 0, host.shape[2] - 1))
        x = slice(max(0, w[0].start - 12), min(host.shape[0], w[0].stop + 12))
        y = slice(max(0, w[1].start - 12), min(host.shape[1], w[1].stop + 12))
        for c, (img, title) in enumerate([(host, "host"), (aug, "augmented"), (aug, "augmented + labels")]):
            ax = axs[r][c]
            ax.imshow(img[x, y, z].T, cmap="gray", vmin=WIN_LO, vmax=WIN_HI, origin="lower", aspect=sp[1] / sp[0])
            if c == 2:
                o, n = s0[x, y, z] == 2, s1[x, y, z] == 2
                for t, col in ((o, "lime"), (n & ~o, "red")):  # existing tumours green, inserted voxels red
                    t = t.T.astype(float)
                    if t.any():
                        ax.contour(t, levels=[0.5], colors=col, linewidths=0.9)
            ax.set_title(f"case {cid} z={z}: {title}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -----------------------------
# Training
# -----------------------------
@torch.no_grad()
def eval_mask_head(model, imgs, masks, healthy, n=32, min_vox=MIN_MASK_VOX):
    """Dice on tumour patches and false-positive behaviour on tumour-free (healthy) patches."""
    model.train(False)
    dev = model.device

    def pred(a):  # chunks of 4 keep the 3D activations small
        return np.concatenate([torch.sigmoid(model.mask(torch.from_numpy(a[i:i + 4].astype(np.float32))[:, None].to(dev)))[:, 0].cpu().numpy() > 0.5
                               for i in range(0, len(a), 4)])

    out = {}
    if len(imgs):
        p, t = pred(imgs[:n]), masks[:n] > 0
        out["dice"] = float(np.mean([2 * (a & b).sum() / max(a.sum() + b.sum(), 1) for a, b in zip(p, t)]))
    if len(healthy):
        h = pred(healthy[:n])
        out["healthy_fp_vox_frac"] = float(h.mean())
        out["healthy_fp_patches"] = float(np.mean([x.sum() >= min_vox for x in h]))
    model.train(True)
    return out


def split_holdout(cids, rng, frac=HOLDOUT):
    """Whole volumes held out of mask-head training (patches of one volume are strongly correlated)."""
    cids = np.unique(cids)
    if len(cids) < 2:
        return set()
    return set(rng.choice(cids, min(len(cids) - 1, max(1, int(round(frac * len(cids))))), replace=False).tolist())


class PatchStore:
    """Patch arrays for the training loop. Up to limit_mb they live on the device (fp16 images, uint8 masks): a batch is a
    device-side gather with no host transfer and no sync. Larger sets stay in host memory (pinned on cuda) and each batch is
    copied with non_blocking=True."""

    def __init__(self, arrays, device, limit_mb=DATA_MB):
        self.device = device
        self.t = {k: torch.from_numpy(np.ascontiguousarray(a)) for k, a in arrays.items()}
        mb = sum(t.numel() * t.element_size() for t in self.t.values()) / 2 ** 20
        self.on_device = device.type != "cpu" and mb <= limit_mb
        if self.on_device:
            self.t = {k: t.to(device) for k, t in self.t.items()}
        elif device.type == "cuda":
            self.t = {k: t.pin_memory() for k, t in self.t.items()}
        self.mb = mb

    def gather(self, key, idx):
        """idx: int64 tensor (on the device when on_device, else on the host). Returns float32 (B,1,P,P,P) on the device."""
        x = self.t[key][idx]
        if not self.on_device and self.device.type != "cpu":
            x = x.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else x.to(self.device)
        return x.float()[:, None]


class MaskPool:
    """mask_pool() callable for MultiGAN.train_batch: half tumour, half healthy (all-zero mask) samples of the non-held-out
    volumes. Index tables are drawn on the host once per refill and moved to the device in one copy (no per-step transfer)."""

    def __init__(self, store, t_idx, h_idx, bs, rng, device):
        self.store, self.t_idx, self.h_idx, self.bs, self.rng, self.device = store, t_idx, h_idx, bs, rng, device
        self.nh = bs // 2 if len(h_idx) else 0
        self.c = self.n = 0

    def refill(self, n):
        ti = self.rng.choice(self.t_idx, (n, self.bs - self.nh))
        hi = self.rng.choice(self.h_idx, (n, self.nh)) if self.nh else np.zeros((n, 0), int)
        mv = lambda a: torch.as_tensor(a, dtype=torch.long).to(self.device if self.store.on_device else "cpu")
        self.ti, self.hi, self.c, self.n = mv(ti), mv(hi), 0, n

    def __call__(self):
        if self.c >= self.n:
            self.refill(max(self.n, 64))
        ti, hi = self.ti[self.c], self.hi[self.c]
        self.c += 1
        x = self.store.gather("images", ti)
        m = self.store.gather("masks", ti)
        if self.nh:
            x = torch.cat([x, self.store.gather("healthy", hi)])
            m = torch.cat([m, torch.zeros((self.nh,) + tuple(m.shape[1:]), dtype=m.dtype, device=m.device)])
        return x, m


def train(patches_path, epochs=EPOCHS, device=None, results_dir=RESULTS_DIR, resume=None, max_seconds=None):
    """Train the GANs + mask head. resume (default LITS_GAN_RESUME): continue from checkpoints/gan_last.pt (optimisers, epoch,
    elapsed time); a finished run (epoch >= epochs) returns at once. max_seconds (default LITS_GAN_MAX_SECONDS): stop after the
    epoch that crosses the budget. Writes checkpoints/gan_last.pt after every epoch and train_metrics.json at every evaluation."""
    device = device or pick_device()
    resume = RESUME if resume is None else resume
    max_seconds = MAX_SECONDS if max_seconds is None else max_seconds
    setup_backends(device)
    d = load_npz(patches_path)
    imgs, masks, healthy = d["images"], d["masks"], d["healthy"]
    t_cid = d["meta"][:, 0]
    h_cid = d["healthy_cid"] if "healthy_cid" in d.files else np.full(len(healthy), -1)
    work = float(d["work_mm"]) if "work_mm" in d.files else 1.0
    patch = imgs.shape[-1]
    assert_unit_range(imgs, "training patches")  # whole set once: the per-batch check (a host sync) is off below
    n, bs = len(imgs), min(BATCH_SIZE, len(imgs))
    rng = np.random.default_rng(SEED)
    # mask head: trained on tumour patches + healthy patches (all-zero masks) of the non-held-out volumes, evaluated on the held-out ones
    held = split_holdout(np.concatenate([t_cid, h_cid]), rng)
    t_tr, h_tr = ~np.isin(t_cid, list(held)), ~np.isin(h_cid, list(held))
    t_tr_i, h_tr_i = np.flatnonzero(t_tr), np.flatnonzero(h_tr)
    assert len(t_tr_i), "no tumour patches left for mask-head training"
    model = MultiGAN(device, patch=patch)
    model.check_range = False
    min_mask_vox = vox_at(MIN_MASK_VOX, work)
    ckpt = os.path.join(results_dir, "checkpoints", "gan_last.pt")
    start, elapsed0 = 0, 0.0
    if resume and os.path.exists(ckpt):
        start, elapsed0 = model.load_training_state(ckpt)
        rng = np.random.default_rng([SEED, start])  # the held-out split above is the same as in the first run (same seed, same data)
        logger.info(f"resumed from {ckpt}: epoch {start}, {elapsed0:.0f}s of training already spent")
    store = PatchStore({"images": imgs, "masks": masks, "healthy": healthy}, device)
    logger.info(f"device {device}; {n} tumour / {len(healthy)} healthy patches {patch}^3 at {work} mm; batch {bs}; epochs {epochs}; "
                f"patch set {store.mb:.0f} MB {'on the device' if store.on_device else 'in host memory'}")
    logger.info(f"mask head: train {len(t_tr_i)} tumour + {len(h_tr_i)} healthy patches; held-out volumes {sorted(held)} "
                f"({int((~t_tr).sum())} tumour + {int((~h_tr).sum())} healthy patches)")
    if not held:
        logger.warning("only one volume: no held-out set, mask-head 'heldout_*' metrics are not available")
    pool = MaskPool(store, t_tr_i, h_tr_i, bs, rng, device)
    metrics_path = os.path.join(results_dir, "train_metrics.json")
    metrics = read_json(metrics_path) if resume and os.path.exists(metrics_path) else {}
    metrics.update(patches=int(n), healthy_patches=int(len(healthy)), patch=int(patch), work_mm=work, epochs_target=int(epochs),
                   held_out_volumes=[int(v) for v in sorted(held)], device=str(device))
    steps = n // bs  # drop the partial last batch (BatchNorm)
    t0, ep, stopped = time.time(), start, "max" if start >= epochs else None
    model.sync = False  # losses stay on the device until the epoch end
    for ep in range(start + 1, epochs + 1):
        perm = torch.as_tensor(rng.permutation(n)[:steps * bs], dtype=torch.long).to(device if store.on_device else "cpu")
        pool.refill(steps * MASK_STEPS)
        sums, cnts = {}, {}
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            real, msk = store.gather("images", idx), store.gather("masks", idx)
            for k, v in model.train_batch(real, msk, pool).items():
                sums[k], cnts[k] = sums.get(k, 0.0) + v, cnts.get(k, 0) + 1  # w_g is only produced every N_CRITIC-th batch
        means = {k: float(v) / cnts[k] for k, v in sums.items()}  # the one host sync of the epoch
        msg = " ".join(f"{k}={v:.3f}" for k, v in means.items())
        assert all(np.isfinite(v) for v in means.values()), "non-finite loss"
        spent = elapsed0 + time.time() - t0
        logger.info(f"epoch {ep}/{epochs} ({spent:.0f}s) {msg}")
        if ep % SAMPLE_EVERY == 0 or ep == epochs:
            model.sync = True
            for tag, ev in (("train", eval_mask_head(model, imgs[t_tr_i], masks[t_tr_i], healthy[h_tr_i], min_vox=min_mask_vox)),
                            ("heldout", eval_mask_head(model, imgs[~t_tr], masks[~t_tr], healthy[~h_tr], n=64, min_vox=min_mask_vox) if held else {})):
                if ev:
                    logger.info(f"mask head {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))
                    metrics[f"mask_head_{tag}"] = dict(ev, epoch=ep)
            model.sync = False
            s = {k: v[:, 0].cpu().numpy() for k, v in model.sample_all(4).items()}
            with torch.no_grad():
                model.train(False)
                pm = (torch.sigmoid(model.mask(torch.from_numpy(s["aggregator"])[:, None].to(device)))[:, 0] > 0.5).cpu().numpy()
                model.train(True)
            save_montage(os.path.join(results_dir, "figures", f"train_epoch{ep:03d}.png"),
                         [("real", imgs[:4].astype(np.float32), masks[:4] > 0), ("dcgan 1", s["dc1"], None),
                          ("dcgan 2", s["dc2"], None), ("wgan-gp", s["wgan"], None),
                          ("aggregator + mask head", s["aggregator"], pm)])
        out_of_time = bool(max_seconds) and spent >= max_seconds and ep < epochs
        if ep % max(CKPT_EVERY, 1) == 0 or ep == epochs or out_of_time:  # resume loses at most CKPT_EVERY-1 epochs
            model.save(ckpt, epoch=ep, elapsed=spent, with_opt=True)
        history = [h for h in metrics.get("loss_history", []) if h["epoch"] < ep]  # resume: drop epochs that are re-run
        metrics.update(epochs_done=ep, train_seconds=spent, last_losses=means, loss_history=history + [dict(means, epoch=ep, seconds=spent)])
        write_json_atomic(metrics_path, metrics)
        if out_of_time:
            logger.warning(f"GAN time budget {max_seconds:.0f}s reached after epoch {ep}/{epochs}: stopping early")
            stopped = "time"
            break
    model.sync = True
    metrics.update(epochs_done=ep, stopped=stopped or "max")
    write_json_atomic(metrics_path, metrics)
    return model


# -----------------------------
# Generation
# -----------------------------
def clean_mask(prob, min_vox=MIN_MASK_VOX, edge=EDGE_MARGIN):
    """Threshold 0.5, keep the largest component. Returns (mask or None, reject reason)."""
    lab, n = ndi.label(prob > 0.5, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None, "empty"
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if m.sum() < min_vox:
        return None, "too_small"
    if any(m.take(range(edge), axis=a).any() or m.take(range(m.shape[a] - edge, m.shape[a]), axis=a).any() for a in range(3)):
        return None, "touches_border"
    return m, ""


def generate_samples(model, num=NUM_SAMPLES, fusion=FUSION, max_attempts=None, work_mm=1.0, stats=None):
    """Accept-until-target: sample, label with the mask head, gate, until `num` samples pass or max_attempts candidates were tried.
    min mask size is defined at 1 mm voxels (MIN_MASK_VOX) and converted to work_mm. stats (dict) receives tried/accepted/rejects."""
    max_attempts = max_attempts or num * 50
    min_vox = vox_at(MIN_MASK_VOX, work_mm)
    patches, masks, rejects, tried = [], [], {}, 0
    while len(patches) < num and tried < max_attempts:
        x = model.sample(min(8, num), fusion)
        with torch.no_grad():
            model.train(False)
            prob = torch.sigmoid(model.mask(x))[:, 0].cpu().numpy()
            model.train(True)
        for xi, pi in zip(x[:, 0].cpu().numpy(), prob):
            tried += 1
            m, why = clean_mask(pi, min_vox=min_vox)
            if m is None:
                rejects[why] = rejects.get(why, 0) + 1
            elif len(patches) < num:
                assert_unit_range(xi, "generated patch")
                patches.append(xi)
                masks.append(m)
    logger.info(f"generated {len(patches)}/{num} samples ({fusion}) in {tried} attempts; rejects {rejects}")
    if stats is not None:
        stats.update(tried=tried, accepted=len(patches), rejects=rejects)
    if not patches:
        e = (0,) + (model.patch,) * 3
        return np.zeros(e, np.float32), np.zeros(e, np.uint8)
    return np.stack(patches).astype(np.float32), np.stack(masks).astype(np.uint8)


# -----------------------------
# Host-aware insertion
# -----------------------------
def resample_patch_to_host(patch, mask, spacing, work_mm=1.0):
    """work_mm generated patch + mask -> host voxel spacing. Exact target shape with grid_mode (no corner-alignment stretch),
    box low-pass before downsampling; the mask is thresholded so that its physical volume is preserved
    (a plain order-0 zoom of a thin mask changes its volume by up to 50 % on 5 mm slices). Empty mask = vanished."""
    sp = np.asarray(spacing, dtype=np.float64)
    r = sp / work_mm  # host voxel size in generator voxels (== sp at 1 mm)
    shape = np.maximum(np.rint(np.asarray(patch.shape) / r).astype(int), 1)
    zoom = shape / np.asarray(patch.shape)

    def down(a):
        for ax, s_ in enumerate(r):
            if s_ > 1.0:  # anti-alias: average over the host voxel footprint
                a = ndi.uniform_filter1d(a, size=int(s_) | 1, axis=ax, mode="nearest")
        return ndi.zoom(a, zoom, order=1, mode="nearest", grid_mode=True)

    img = down(patch.astype(np.float32))
    f = down(mask.astype(np.float32))
    k = int(round(mask.sum() * work_mm ** 3 / np.prod(sp)))  # voxels the mask should occupy on the host grid
    if k < 1:
        return img, np.zeros(shape, bool)
    c = np.asarray(ndi.center_of_mass(f))
    dist = np.sqrt(sum(((np.arange(n) - ci) * s_).reshape([-1 if a == ax else 1 for a in range(3)]) ** 2
                       for ax, (n, ci, s_) in enumerate(zip(shape, c, sp))))
    score = f - 1e-5 * dist / max(float(dist.max()), 1e-6)  # exactly k voxels: ties go to the voxels nearest the centroid
    thr = float(np.partition(score.ravel(), -k)[-k])
    return img, (score >= thr) & (f > 1e-3)


def border_taper(shape, spacing, taper_mm=4.0):
    """0 on the window border rising to 1 over taper_mm, so alpha vanishes at the window edge."""
    w = np.ones(shape, np.float32)
    for ax, (n, sp) in enumerate(zip(shape, spacing)):
        d = np.minimum(np.arange(n), np.arange(n)[::-1]) * sp
        s = [1, 1, 1]
        s[ax] = n
        w = w * np.clip(d / taper_mm, 0, 1).reshape(s)
    return w


def liver_noise_std(hu, seg, spacing, sl):
    """Robust std of the high-passed HU over eroded liver voxels (CT noise level); None if too few voxels."""
    x = hu[sl].astype(np.float32)
    lm = ndi.binary_erosion(seg[sl] == 1, iterations=2)
    if lm.sum() < 200:
        return None
    hp = (x - ndi.gaussian_filter(x, NOISE_SIGMA_MM / np.asarray(spacing), mode="nearest"))[lm]
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def _reject(why, key, msg):
    logger.info(f"insert skipped: {msg}")
    if why is not None:
        why[key] = why.get(key, 0) + 1


def insert_one(hu, seg, spacing, patch, mask, rng, feather_mm=FEATHER_MM, exclude_mm=EXCLUDE_MM, work_mm=1.0, why=None):
    """Insert one generated tumour into int16 (or float32) HU `hu` / uint8 `seg` IN PLACE. Returns an info dict, or None (reason
    logged, and counted in the `why` dict when given). Only voxels with alpha > 0 inside a local window change; the label is
    (mask & liver) written as 2. A float32 `hu` keeps the blended values unrounded (cache source: written back to uint8 once)."""
    sp = np.asarray(spacing, dtype=np.float64)
    patch_rs, m_rs = resample_patch_to_host(patch, mask, spacing, work_mm)
    if not m_rs.any():
        _reject(why, "mask_vanished", "mask vanished on resampling to host spacing")
        return None
    ps = np.array(patch_rs.shape)
    coords = np.argwhere(mask)
    r_mm = float(np.linalg.norm((coords - coords.mean(0)) * work_mm, axis=1).max()) + work_mm  # circumscribed radius (mm)
    cen = np.rint(np.argwhere(m_rs).mean(0)).astype(int)  # mask centroid inside the resampled window
    # candidate centres: liver eroded by the tumour radius, away from existing tumours, window fully inside the volume
    box = ndi.find_objects((seg > 0).astype(np.int8))
    if not box:
        _reject(why, "host_no_labels", "host has no labels")
        return None
    box = box[0]
    off = np.array([s.start for s in box])
    liver = np.pad(seg[box] == 1, 1)
    ok = ndi.distance_transform_edt(liver, sampling=sp)[1:-1, 1:-1, 1:-1] >= r_mm
    tum = seg[box] == 2
    if tum.any():
        ok &= ndi.distance_transform_edt(~tum, sampling=sp) >= r_mm + exclude_mm
    lo, hi = cen - off, np.array(seg.shape) - ps + cen - off  # valid centre range in crop coordinates (inclusive)
    inside = np.zeros_like(ok)
    sub = tuple(slice(max(l, 0), min(h + 1, n)) for l, h, n in zip(lo, hi, ok.shape))
    inside[sub] = ok[sub]
    cand = np.argwhere(inside)
    if len(cand) == 0:
        _reject(why, "no_site", f"no valid site (tumour radius {r_mm:.1f} mm)")
        return None
    p = cand[rng.integers(len(cand))] + off
    start = p - cen
    win = tuple(slice(int(s), int(s + n)) for s, n in zip(start, ps))
    hw, sw = hu[win].astype(np.float32), seg[win]  # sw is a view: labels are written through it
    liver_w = sw == 1
    m_eff = m_rs & liver_w
    if not m_eff.any():
        _reject(why, "no_liver_voxels", "mask has no liver voxels")
        return None
    # feathered alpha: gaussian of the dilated mask on the host grid, 1 inside the mask, 0 on the window border
    dil = ndi.distance_transform_edt(~m_eff, sampling=sp) <= 1.0  # 1 mm dilation (a voxel dilation would add a whole 5 mm slice)
    alpha = ndi.gaussian_filter(dil.astype(np.float32), feather_mm / sp, mode="constant")
    alpha = np.maximum(alpha, m_eff) * border_taper(tuple(ps), sp) * liver_w  # liver only: no background, no existing tumour
    alpha[alpha < 0.01] = 0.0
    support = alpha > 0
    # masked intensity transfer: shift so the generated non-tumour mean (ring around the tumour) matches the host liver
    gen = from_unit(patch_rs)
    d = ndi.distance_transform_edt(~m_rs, sampling=sp)
    ring = (d > 2.0) & (d <= 10.0)
    ring = ring if ring.sum() >= 20 else ~m_rs
    local = liver_w & ~support
    host_ref = float(hw[local].mean() if local.sum() >= 100 else hw[liver_w].mean())
    shift = host_ref - float(gen[ring].mean())
    gen = gen + shift
    # host-matched high-frequency noise inside alpha (variance-preserving mix of host and new noise)
    pad = np.ceil(24.0 / sp).astype(int)
    nsl = tuple(slice(max(0, w.start - q), min(n, w.stop + q)) for w, q, n in zip(win, pad, seg.shape))
    sig = liver_noise_std(hu, seg, spacing, nsl)
    if sig is None:
        sig = liver_noise_std(hu, seg, spacing, tuple(slice(s.start, s.stop) for s in box))
    sig = 0.0 if sig is None else sig
    sg = NOISE_SIGMA_MM / sp
    noise = ndi.gaussian_filter(rng.standard_normal(hw.shape).astype(np.float32), NOISE_PRE_MM / sp)  # CT noise is spatially correlated
    noise = noise - ndi.gaussian_filter(noise, sg)
    noise *= sig / max(float(noise.std()), 1e-6)
    host_lp = ndi.gaussian_filter(hw, sg, mode="nearest")
    out = ((1 - alpha) * host_lp + alpha * ndi.gaussian_filter(gen, sg, mode="nearest")
           + np.sqrt(1 - alpha) * (hw - host_lp) + np.sqrt(alpha) * noise)
    hu[win][support] = to_int16(out[support]) if np.issubdtype(hu.dtype, np.integer) else out[support].astype(hu.dtype)
    new = m_eff & liver_w
    sw[new] = 2
    return {"window": win, "support": support, "alpha": alpha, "mask": m_eff, "centre": p, "radius_mm": r_mm,
            "shift_hu": shift, "noise_std": sig, "n_label": int(new.sum()), "host_ref_hu": host_ref}


def insert_tumors(hu, seg, spacing, samples, rng, n=INSERTIONS, work_mm=1.0, why=None, keep_float=False):
    """Up to n non-overlapping insertions (each placement sees the labels of the previous ones).
    Returns (hu_int16, seg_uint8, infos); infos is empty when no site was valid. keep_float: return float32 HU, not rounded to int16."""
    out_hu = hu.astype(np.float32).copy() if keep_float else to_int16(hu)
    out_seg, infos = seg.astype(np.uint8).copy(), []
    patches, masks = samples
    for _ in range(n):
        k = int(rng.integers(len(patches)))
        info = insert_one(out_hu, out_seg, spacing, patches[k], masks[k], rng, work_mm=work_mm, why=why)
        if info is not None:
            infos.append(info)
    return out_hu, out_seg, infos


def run_insert(samples_path, ids, out_dir, fig_path, data_dir=DATA_DIR, n=INSERTIONS, rng=None):
    rng = rng or np.random.default_rng(SEED)
    assert not os.path.abspath(out_dir).startswith(os.path.abspath(data_dir)), "never write augmented files into the raw data dir"
    d = load_npz(samples_path)
    samples = (d["patches"], d["masks"])
    work = float(d["work_mm"]) if "work_mm" in d.files else 1.0
    if not len(samples[0]):
        logger.error("no valid generated samples (all rejected by the mask gate): nothing to insert")
        return 0
    cases, shown, done = pair_cases(data_dir, ids), [], 0
    os.makedirs(out_dir, exist_ok=True)
    for cid in ids:
        vol, seg, sp, aff = load_case(*cases[cid])
        hu, seg_new, infos = insert_tumors(vol, seg, sp, samples, rng, n, work_mm=work)
        if not infos:
            logger.warning(f"case {cid}: no tumour inserted (no valid site), not written")
            continue
        save_nifti(hu, aff, os.path.join(out_dir, f"aug_volume-{cid}.nii"), np.int16)
        save_nifti(seg_new, aff, os.path.join(out_dir, f"aug_segmentation-{cid}.nii"), np.uint8)
        logger.info(f"case {cid}: inserted {len(infos)}, +{sum(i['n_label'] for i in infos)} tumour voxels, "
                    f"shift {[round(i['shift_hu'], 1) for i in infos]} HU, noise std {[round(i['noise_std'], 1) for i in infos]}")
        done += 1
        if len(shown) < 4:
            shown.append((cid, to_int16(vol), hu, seg, seg_new, infos[0], sp))
    if shown:
        save_insert_montage(fig_path, shown)
    logger.info(f"augmented {done}/{len(ids)} hosts -> {out_dir}")
    return done


# -----------------------------
# Cache source: slabs of augmented cache slices
# -----------------------------
def accept_until_target(attempt, target, max_attempts):
    """Call attempt(i) (-> list of accepted items, empty = rejected) until `target` items are accepted or max_attempts calls were
    made. The list is cut at target. Returns (items, attempts used)."""
    out, tried = [], 0
    while len(out) < target and tried < max_attempts:
        out.extend(attempt(tried))
        tried += 1
    return out[:target], tried


def quantise_window(hu, lo=WIN_LO, hi=WIN_HI):
    """HU -> the cache's uint8 window mapping (rounded, clipped); inverse of CacheSource.to_hu up to the 1.37 HU step."""
    return np.clip(np.rint((hu - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def host_slabs(source, cid, samples, rng, per_host=1, work_mm=1.0, why=None):
    """Insert up to per_host tumours into ONE cache host and cut slabs: the z range of each insertion window (windows of this host
    that overlap in z are merged, so no slice appears twice). Slab images/masks are the cache slices of the AUGMENTED host: voxels
    that the insertion did not change keep the host's uint8 value bit for bit, only changed voxels are re-quantised.
    Returns a list of dicts (images (n,256,256) u8, masks, host_vol, z0, new_tumor_vox, n_slices, preview); [] if nothing fit."""
    img, msk = source.load_u8(cid)  # (Z,256,256)
    hu, seg = source.to_hu(np.moveaxis(img, 0, 2)), np.moveaxis(msk, 0, 2)  # (256,256,Z): the layout the NIfTI code works in
    aug_hu, seg_new, infos = insert_tumors(hu, seg, source.spacing(cid), samples, rng, per_host, work_mm, why, keep_float=True)
    spans = []
    for a, b in sorted((i["window"][2].start, i["window"][2].stop) for i in infos):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    out = []
    for z0, z1 in spans:
        a, h = aug_hu[:, :, z0:z1], hu[:, :, z0:z1]
        changed = a != h  # insert_one only assigns inside alpha support, everything else is a bit-identical copy of the host
        u8 = img[z0:z1].copy()
        np.moveaxis(u8, 0, 2)[changed] = quantise_window(a[changed], source.lo, source.hi)
        m_new = np.ascontiguousarray(np.moveaxis(seg_new[:, :, z0:z1], 2, 0))
        grown = (m_new == 2) & (msk[z0:z1] != 2)
        k = int(np.argmax(grown.sum(axis=(1, 2))))
        out.append(dict(images=u8, masks=m_new, host_vol=source.name(cid), z0=int(z0), n_slices=int(z1 - z0), new_tumor_vox=int(grown.sum()),
                        preview=(img[z0 + k].copy(), u8[k].copy(), msk[z0 + k].copy(), m_new[k].copy(), int(z0 + k))))
    return out


def pack_slabs(slabs, size=256):
    """list of host_slabs dicts -> the aug_slabs.npz arrays (schema in the module docstring); also valid for zero slabs."""
    if not slabs:
        return dict(images=np.zeros((0, size, size), np.uint8), masks=np.zeros((0, size, size), np.uint8), slab_id=np.zeros(0, np.int32),
                    host_vol=np.array([], dtype="<U16"), z0=np.zeros(0, np.int32), new_tumor_vox=np.zeros(0, np.int64), n_slices=np.zeros(0, np.int32))
    n = np.array([s["n_slices"] for s in slabs], np.int32)
    return dict(images=np.concatenate([s["images"] for s in slabs]), masks=np.concatenate([s["masks"] for s in slabs]),
                slab_id=np.repeat(np.arange(len(slabs), dtype=np.int32), n), host_vol=np.array([s["host_vol"] for s in slabs]),
                z0=np.array([s["z0"] for s in slabs], np.int32), new_tumor_vox=np.array([s["new_tumor_vox"] for s in slabs], np.int64), n_slices=n)


def save_slab_montage(path, previews, crop=48):
    """previews: [(title, host u8 slice, augmented u8 slice, old mask, new mask)]: host | augmented | augmented + labels
    (existing tumours green, inserted voxels red). Shown in the cache's own orientation, cropped around the new tumour."""
    fig, axs = plt.subplots(len(previews), 3, figsize=(8.4, 2.9 * len(previews)), squeeze=False)
    for r, (title, h, a, m0, m1) in enumerate(previews):
        grown = (m1 == 2) & (m0 != 2)
        cy, cx = (np.rint(np.argwhere(grown).mean(0)).astype(int) if grown.any() else (np.array(h.shape) // 2))
        ys, xs = slice(max(0, cy - crop), min(h.shape[0], cy + crop)), slice(max(0, cx - crop), min(h.shape[1], cx + crop))
        for c, (img, t) in enumerate([(h, "host (before)"), (a, "augmented (after)"), (a, "augmented + labels")]):
            ax = axs[r][c]
            ax.imshow(img[ys, xs], cmap="gray", vmin=0, vmax=255)
            if c == 2:
                for t_, col in (((m0 == 2)[ys, xs], "lime"), (grown[ys, xs], "red")):
                    if t_.any():
                        ax.contour(t_.astype(float), levels=[0.5], colors=col, linewidths=0.9)
            ax.set_title(f"{title}: {t}" if c == 0 else t, fontsize=8, loc="left")
            ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_insert_slabs(source, hosts, samples, out_path, target=NUM_AUG, per_host=INSERTIONS, max_attempts=None, rng=None,
                     work_mm=1.0, fig_path=None):
    """Cache-source insert stage: visit hosts (shuffled, cycling) and insert until `target` slabs are accepted or `max_attempts`
    host visits are used; writes aug_slabs.npz atomically. Returns a stats dict (accepted, attempts, reasons, ...)."""
    rng = rng or np.random.default_rng(SEED)
    max_attempts = max_attempts or 5 * target
    hosts = [int(h) for h in hosts]
    assert hosts, "no insertion hosts"
    assert not set(hosts) - set(source.train_part_ids()), "hosts must come from the original-split train part"
    why, t0, state = {}, time.time(), {"order": []}
    if not len(samples[0]):
        logger.error("no valid generated samples (all rejected by the mask gate): nothing to insert")
        max_attempts = 0

    def attempt(i):
        if not state["order"]:
            state["order"] = [hosts[j] for j in rng.permutation(len(hosts))]  # a new shuffled pass over the hosts once all were visited
        cid = state["order"].pop()
        slabs = host_slabs(source, cid, samples, rng, per_host, work_mm, why)
        logger.info(f"attempt {i + 1}/{max_attempts} host {source.name(cid)}: " +
                    (f"{len(slabs)} slab(s), {[s['n_slices'] for s in slabs]} slices, +{sum(s['new_tumor_vox'] for s in slabs)} tumour voxels" if slabs
                     else "rejected"))
        return slabs

    slabs, tried = accept_until_target(attempt, target, max_attempts)
    save_npz_atomic(out_path, **pack_slabs(slabs))
    if fig_path and slabs:
        pick = np.linspace(0, len(slabs) - 1, min(6, len(slabs))).round().astype(int)
        save_slab_montage(fig_path, [(f"{slabs[k]['host_vol'][:-4]} z={slabs[k]['preview'][4]}",) + tuple(slabs[k]["preview"][:4]) for k in pick])
    st = dict(accepted=len(slabs), target=target, attempts=tried, max_attempts=max_attempts, insert_rejected=why, hosts=len(hosts),
              slab_slices=int(sum(s["n_slices"] for s in slabs)), new_tumor_vox=int(sum(s["new_tumor_vox"] for s in slabs)),
              hosts_used=sorted({s["host_vol"] for s in slabs}), seconds=time.time() - t0)
    if len(slabs) < target:
        logger.warning(f"only {len(slabs)}/{target} slabs accepted in {tried} host visits (reasons {why})")
    logger.info(f"wrote {len(slabs)} slabs ({st['slab_slices']} slices, +{st['new_tumor_vox']} tumour voxels) -> {out_path}")
    return st


def record_stage(results_dir, stage, **kv):
    """Merge per-stage numbers into stage_metrics.json (kept across resumes) and return the whole dict."""
    path = os.path.join(results_dir, "stage_metrics.json")
    cur = read_json(path) if os.path.exists(path) else {}
    cur.setdefault(stage, {}).update(kv)
    write_json_atomic(path, cur)
    return cur


def write_gan_summary(results_dir, extra=None):
    """gan_summary.json: {patches, epochs, accepted, rejected_reasons, mask_head_train_dice, mask_head_heldout_dice, volumes_used, seconds, ...}
    built from stage_metrics.json; rejected_reasons is flat: gate_<reason> (mask gate) and insert_<reason> (insertion)."""
    m = read_json(os.path.join(results_dir, "stage_metrics.json"))
    pre, tr, gen, ins = (m.get(k, {}) for k in ("prepare", "train", "generate", "insert"))
    rej = {f"gate_{k}": v for k, v in gen.get("rejects", {}).items()}
    rej.update({f"insert_{k}": v for k, v in ins.get("insert_rejected", {}).items()})
    by_stage = {k: round(v.get("seconds", 0.0), 1) for k, v in m.items() if isinstance(v, dict) and "seconds" in v}
    mh = lambda tag: (tr.get(f"mask_head_{tag}") or {}).get("dice")
    out = dict(source=GAN_SOURCE, work_mm=pre.get("work_mm"), patch=tr.get("patch", pre.get("patch")), patches=tr.get("patches", pre.get("patches")),
               healthy_patches=tr.get("healthy_patches", pre.get("healthy_patches")), epochs=tr.get("epochs_done"), epochs_target=tr.get("epochs_target"),
               train_stopped=tr.get("stopped"), accepted=ins.get("accepted", 0), target=ins.get("target"), attempts=ins.get("attempts"),
               rejected_reasons=rej, gate_tried=gen.get("tried"), gate_accepted=gen.get("accepted"),
               mask_head_train_dice=mh("train"), mask_head_heldout_dice=mh("heldout"),
               mask_head_train=tr.get("mask_head_train"), mask_head_heldout=tr.get("mask_head_heldout"),
               volumes_listed=pre.get("volumes_listed"), volumes_used=pre.get("volumes_used"), volumes_used_names=pre.get("volumes_used_names"),
               hosts=ins.get("hosts"), slab_slices=ins.get("slab_slices"), new_tumor_vox=ins.get("new_tumor_vox"),
               seconds=round(sum(by_stage.values()), 1), seconds_by_stage=by_stage)
    out.update(extra or {})
    write_json_atomic(os.path.join(results_dir, "gan_summary.json"), out)
    f = lambda v: "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"GAN_SUMMARY slabs {out['accepted']}/{f(out['target'])} (visits {f(out['attempts'])}) | patches {f(out['patches'])} from {f(out['volumes_used'])}/"
          f"{f(out['volumes_listed'])} volumes | epochs {f(out['epochs'])} | mask head dice train {f(out['mask_head_train_dice'])} held-out "
          f"{f(out['mask_head_heldout_dice'])} | gate {f(out['gate_accepted'])}/{f(out['gate_tried'])} | rejected {rej} | {out['seconds']}s "
          f"{by_stage} -> {os.path.join(results_dir, 'gan_summary.json')}", flush=True)
    return out


def main_cache(stage, device):
    """Stages prepare/train/generate/insert for LITS_GAN_SOURCE=cache. Every stage is idempotent under LITS_GAN_RESUME=1."""
    source = make_source("cache")
    restrict = parse_volume_list(VOLUME_LIST) if VOLUME_LIST else None
    ids = source.split_cases(restrict)
    logger.info(f"cache {source.cache_dir}: {len(ids)} usable train-part volumes (original split) at {WORK_MM} mm: {ids[:12]}{'...' if len(ids) > 12 else ''}")
    patches = os.path.join(RESULTS_DIR, "patches.npz")
    samples = os.path.join(RESULTS_DIR, "synthetic", "samples.npz")
    slabs_path = os.path.join(RESULTS_DIR, "aug_slabs.npz")
    pool_size = int(os.environ["LITS_GAN_NUM_SAMPLES"]) if "LITS_GAN_NUM_SAMPLES" in os.environ else int(np.clip(NUM_AUG, 16, 256))
    skip = lambda path: RESUME and not FORCE and os.path.exists(path)
    try:
        if stage in ("all", "prepare"):
            t0 = time.time()
            same = False
            if skip(patches):
                z = load_npz(patches)
                same = "ids" in z.files and z["ids"].tolist() == ids and z["images"].shape[-1] == PATCH and float(z["work_mm"]) == WORK_MM
                logger.info(f"patches.npz exists ({'same volume list, kept' if same else 'different settings, rebuilt'})")
            if not same:
                prepare_patches(ids, patches, size=PATCH, source=source)
            z = load_npz(patches)
            used = sorted({int(c) for c in z["meta"][:, 0]})
            record_stage(RESULTS_DIR, "prepare", patches=int(len(z["images"])), healthy_patches=int(len(z["healthy"])), patch=PATCH, work_mm=WORK_MM,
                         volumes_listed=len(ids), volumes_used=len(used), volumes_used_names=[source.name(c) for c in used],
                         **({} if same else {"seconds": time.time() - t0}))  # a kept file keeps the seconds of the run that built it
        if stage in ("all", "train"):
            t0 = time.time()
            train(patches, EPOCHS, device)
            tm = read_json(os.path.join(RESULTS_DIR, "train_metrics.json"))
            record_stage(RESULTS_DIR, "train", seconds=tm.get("train_seconds", time.time() - t0), **tm)
        if stage in ("all", "generate"):
            if skip(samples):
                logger.info(f"{samples} exists, kept")
            else:
                t0 = time.time()
                model = MultiGAN.load(os.path.join(RESULTS_DIR, "checkpoints", "gan_last.pt"), device)
                st = {}
                p, m = generate_samples(model, pool_size, FUSION, GEN_ATTEMPTS or None, work_mm=WORK_MM, stats=st)
                save_npz_atomic(samples, patches=p, masks=m, work_mm=np.float32(WORK_MM))
                if len(p):
                    save_montage(os.path.join(RESULTS_DIR, "figures", "generated.png"), [(f"generated ({FUSION}) + mask head", p[:8], m[:8] > 0)])
                record_stage(RESULTS_DIR, "generate", seconds=time.time() - t0, pool_size=pool_size, **st)
        if stage in ("all", "insert"):
            if skip(slabs_path) and os.path.exists(os.path.join(RESULTS_DIR, "gan_summary.json")):
                logger.info(f"{slabs_path} exists, kept")
            else:
                d = load_npz(samples)
                work = float(d["work_mm"]) if "work_mm" in d.files else WORK_MM
                st = run_insert_slabs(source, ids, (d["patches"], d["masks"]), slabs_path, NUM_AUG, INSERTIONS, MAX_ATTEMPTS or None,
                                      np.random.default_rng(SEED), work, os.path.join(RESULTS_DIR, "figures", "slab_montage.png"))
                record_stage(RESULTS_DIR, "insert", **st)
                write_gan_summary(RESULTS_DIR)
    except AssertionError as e:  # e.g. no tumour patches in the listed volumes: leave a summary the orchestrator can read, then fail
        logger.error(f"cache pipeline failed in stage {stage}: {e}")
        if os.path.exists(os.path.join(RESULTS_DIR, "stage_metrics.json")):
            write_gan_summary(RESULTS_DIR, {"error": str(e), "accepted": 0})
        else:
            write_json_atomic(os.path.join(RESULTS_DIR, "gan_summary.json"), {"error": str(e), "accepted": 0, "rejected_reasons": {}})
        raise SystemExit(4)


# -----------------------------
# Entry point
# -----------------------------
def main():
    stage = os.environ.get("LITS_GAN_STAGE", "all")
    assert stage in ("all", "prepare", "train", "generate", "insert"), stage
    check_patch(PATCH)
    setup_logging(RESULTS_DIR)
    set_seed(SEED)
    device = pick_device()
    setup_backends(device)
    if GAN_SOURCE == "cache":
        logger.info(f"stage={stage} source=cache cache={CACHE_DIR} results={RESULTS_DIR} window=[{WIN_LO},{WIN_HI}] patch={PATCH} "
                    f"work={WORK_MM} mm fusion={FUSION} device={device} num_aug={NUM_AUG} per_host={INSERTIONS}")
        main_cache(stage, device)
        logger.info("done")
        return
    patches = os.path.join(RESULTS_DIR, "patches.npz")
    samples = os.path.join(RESULTS_DIR, "synthetic", "samples.npz")
    logger.info(f"stage={stage} data={DATA_DIR} results={RESULTS_DIR} "
                f"window=[{WIN_LO},{WIN_HI}] patch={PATCH} fusion={FUSION} device={device}")
    n_train = len(train_part_ids(DATA_DIR))
    ids = split_cases(restrict=parse_volume_list(VOLUME_LIST) if VOLUME_LIST else None)
    logger.info(f"{len(ids)} of {n_train} train-part volumes used (original split: test_size=0.2, random_state=42): "
                f"{ids[:10]}{'...' if len(ids) > 10 else ''}")
    if stage in ("all", "prepare"):
        prepare_patches(ids, patches, source=NiftiSource(DATA_DIR, WORK_MM))
    if stage in ("all", "train"):
        train(patches, EPOCHS, device)
    if stage in ("all", "generate"):
        model = MultiGAN.load(os.path.join(RESULTS_DIR, "checkpoints", "gan_last.pt"), device)
        p, m = generate_samples(model, NUM_SAMPLES, FUSION, work_mm=WORK_MM)
        os.makedirs(os.path.dirname(samples), exist_ok=True)
        np.savez(samples, patches=p, masks=m, work_mm=np.float32(WORK_MM))
        if len(p):
            save_montage(os.path.join(RESULTS_DIR, "figures", "generated.png"), [(f"generated ({FUSION}) + mask head", p[:8], m[:8] > 0)])
    if stage in ("all", "insert"):
        run_insert(samples, ids, os.path.join(RESULTS_DIR, "aug"), os.path.join(RESULTS_DIR, "figures", "insertions.png"))
    logger.info("done")


if __name__ == "__main__":
    main()
