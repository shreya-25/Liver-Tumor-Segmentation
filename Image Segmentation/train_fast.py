"""Faster AIM-UNet training. Same model, loss, volume split and tumor oversampling as
train_orig_AIMUNetALF.py, but:
  * reads the uint8 slice cache built by prepare_data*.py (no NIfTI/PIL work at train time)
  * batches are indexed directly (no DataLoader workers): from RAM, or gathered from the memory-mapped cache
    (LITS_MMAP; needed on a 12 GB-RAM Colab machine) with a small prefetch thread
  * flip/rotate augmentation runs on the accelerator via affine_grid/grid_sample
  * mixed precision (LITS_AMP=auto|bf16|fp16|off), channels_last, cudnn.benchmark on cuda
  * confusion matrix / F1 computed exactly on-device; ROC AUC on a pixel subsample
  * validation patient-level 3D Dice (liver, tumor) from per-volume sums accumulated on-device

Config via env vars (all optional; unset = the behaviour of the earlier version of this script):
  paths/size : LITS_CACHE_DIR, LITS_RESULTS_DIR, LITS_EPOCHS (50), LITS_BATCH_SIZE (4), LITS_LR (1e-4),
               LITS_MAX_VOLUMES (smoke test: first N volumes before the split), LITS_SEED (unset = unseeded)
  split      : LITS_SUBSET_N (train volumes, 0 = all; stratified by thickness x tumor), LITS_SUBSET_VAL_N (validation
               part only), LITS_VAL_N (0 = old behaviour, val = the whole held-out part; >0: val = stratified subset of
               the held-out part, test = the rest of it, never used for any selection), LITS_PER_VOLUME
  data       : LITS_FLIP_AUG=0 (rotation only), LITS_EMPTY_KEEP (1.0), LITS_CONTEXT=1 (2.5D: slices k-1,k,k+1 of the same
               volume, clamped at borders), LITS_AUG_NPZ (+ LITS_AUG_MAX_SLABS): synthetic slabs added to the TRAINING pool
               only, LITS_EPOCH_LEN (samples per epoch, 0 = from the real training slices only)
  training   : LITS_LOSS=ce_dice|ce_dice_tversky, LITS_LR_SCHED=const|cosine, LITS_SELECT=loss|tumor3d (best checkpoint
               and early stopping), LITS_PATIENCE (7), LITS_RESUME=1 (continue from RESULTS_DIR/last.pt),
               LITS_STOP_AFTER=N (pause cleanly after epoch N; for time budgets and tests)
  device     : LITS_DEVICE=auto|cuda|mps|cpu, LITS_AMP=auto|bf16|fp16|off, LITS_CHANNELS_LAST=auto|0|1,
               LITS_MMAP=auto|0|1 (auto = on for cuda), LITS_SLICE_FLAGS (optional cache file for per-slice liver/tumor flags)
  utilities  : LITS_PROBE_ONLY=1 (print one 'PROBE {json}' throughput line and exit), LITS_SPLIT_ONLY=1 (write
               split_used.json and exit without loading images)
Exit codes: 0 ok / paused / probe, 3 non-finite training loss (stdout 'NONFINITE_LOSS'), 4 refused resume.

The split is the original one: sklearn train_test_split over the sorted volume names, test_size=0.2,
random_state=42 (104 train / 27 held-out volumes; each volume is one patient). Outputs in RESULTS_DIR:
train_AIMUNet.log, split_used.json (train, val[, test] names), history.json (per epoch), best.pt and last.pt
(dicts with 'model' and 'config'), loss_curves.png, done.json {epochs_run, best_epoch, best_val_tumor3d, stopped}.
Old runs stay loadable: eval_fast.py also accepts the raw state_dict files (best_epoch_NNN.pt) of earlier versions.
"""
import os
import csv
import sys
import json
import math
import time
import queue
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; data/ lives here
logger = logging.getLogger("train_fast")

AUC_PIXELS_PER_BATCH = 1000  # random pixels per val batch kept for ROC AUC
FLIP_AUG = os.environ.get("LITS_FLIP_AUG", "1") != "0"  # default for gpu_augment(); Cfg passes its own
EXIT_NONFINITE, EXIT_RESUME_REFUSED = 3, 4


# -----------------------------
# Config (read at call time so tests / the orchestrator can change the environment between runs)
# -----------------------------
class Cfg:
    CHOICES = {"loss": ("ce_dice", "ce_dice_tversky"), "lr_sched": ("const", "cosine"), "select": ("loss", "tumor3d"),
               "mmap": ("auto", "0", "1"), "amp": ("auto", "bf16", "fp16", "off"), "channels_last": ("auto", "0", "1"),
               "device": ("auto", "cuda", "mps", "cpu")}

    def __init__(self, env=None):
        e = os.environ if env is None else env
        g = lambda k, d: e.get(k, d)
        self.cache_dir = g("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache"))
        self.results_dir = g("LITS_RESULTS_DIR", os.path.join(ROOT, "results", "train_fast"))
        self.per_volume = g("LITS_PER_VOLUME", os.path.join(ROOT, "analysis", "output", "per_volume.csv"))
        self.epochs, self.batch_size = int(g("LITS_EPOCHS", 50)), int(g("LITS_BATCH_SIZE", 4))
        self.lr, self.patience = float(g("LITS_LR", 1e-4)), int(g("LITS_PATIENCE", 7))
        self.max_volumes = int(g("LITS_MAX_VOLUMES", 0))
        self.seed = int(e["LITS_SEED"]) if e.get("LITS_SEED") else None  # unset = unseeded, as before
        self.subset_n = int(g("LITS_SUBSET_N", 0))
        self.subset_val_n = int(g("LITS_SUBSET_VAL_N", self.subset_n))  # validation runs every epoch: allow fewer volumes
        self.val_n = int(g("LITS_VAL_N", 0))
        self.flip = g("LITS_FLIP_AUG", "1") != "0"
        self.empty_keep = float(g("LITS_EMPTY_KEEP", 1.0))
        self.context = g("LITS_CONTEXT", "0") == "1"
        self.in_channels = 3 if self.context else 1
        self.aug_npz = g("LITS_AUG_NPZ", "")
        self.aug_max_slabs = int(g("LITS_AUG_MAX_SLABS", 0))
        self.epoch_len = int(g("LITS_EPOCH_LEN", 0))
        self.loss = g("LITS_LOSS", "ce_dice").lower()
        self.lr_sched = g("LITS_LR_SCHED", "const").lower()
        self.select = g("LITS_SELECT", "loss").lower()
        self.resume = g("LITS_RESUME", "0") == "1"
        self.stop_after = int(g("LITS_STOP_AFTER", 0))
        self.device = g("LITS_DEVICE", "auto").lower()
        self.amp = g("LITS_AMP", "auto").lower()
        self.channels_last = g("LITS_CHANNELS_LAST", "auto").lower()
        self.mmap = g("LITS_MMAP", "auto").lower()
        self.slice_flags = g("LITS_SLICE_FLAGS", "")
        self.probe_only = g("LITS_PROBE_ONLY", "0") == "1"
        self.probe_steps = int(g("LITS_PROBE_STEPS", 40))
        self.split_only = g("LITS_SPLIT_ONLY", "0") == "1"
        for k, ok in self.CHOICES.items():
            if getattr(self, k) not in ok:
                raise ValueError(f"LITS_{k.upper()}={getattr(self, k)!r}, expected one of {ok}")


# -----------------------------
# Device logic (all queries defensive: a failing/absent CUDA query must not crash a long unattended run)
# -----------------------------
def pick_device(setting="auto"):
    if setting != "auto":
        return torch.device(setting)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def cuda_caps():
    """(compute capability tuple | None, bf16 supported | None); None = could not be queried."""
    try:
        cc = tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:
        cc = None
    try:
        ok = bool(torch.cuda.is_bf16_supported())  # True on a T4 too (emulated, slow): the capability check decides
    except Exception:
        ok = None
    return cc, ok


def resolve_amp(setting, dev_type, cc=None, bf16_ok=None):
    """-> (name, autocast dtype | None, use GradScaler). auto: bf16 on mps and on cuda with compute capability >= 8,
    fp16 + GradScaler on other cuda (T4 has no fast bf16), off on cpu. An explicit setting is honoured
    (cpu is always off, as before)."""
    s = setting.lower()
    if s not in ("auto", "bf16", "fp16", "off"):
        raise ValueError(f"LITS_AMP={setting!r}")
    if dev_type == "cpu" or s == "off":
        return "off", None, False
    if s == "auto":
        s = "bf16" if dev_type != "cuda" or (cc is not None and cc[0] >= 8 and bf16_ok is not False) else "fp16"
    return s, {"bf16": torch.bfloat16, "fp16": torch.float16}[s], s == "fp16" and dev_type == "cuda"


def resolve_flag(setting, dev_type, auto_on=("cuda",)):
    return dev_type in auto_on if setting == "auto" else setting == "1"


def device_plan(cfg, dev_type, cc=None, bf16_ok=None):
    """Every device-dependent switch in one pure function (unit-tested with mocked CUDA queries)."""
    amp, amp_dtype, use_scaler = resolve_amp(cfg.amp, dev_type, cc, bf16_ok)
    mmap = resolve_flag(cfg.mmap, dev_type)
    return dict(amp=amp, amp_dtype=amp_dtype, scaler=use_scaler, channels_last=resolve_flag(cfg.channels_last, dev_type),
                mmap=mmap, prefetch=mmap or dev_type == "cuda",  # gather thread: overlaps disk reads / pinning with compute
                pin=dev_type == "cuda",  # pinned host memory only helps cuda; non_blocking copies need it
                cudnn_benchmark=dev_type == "cuda")  # fixed 256x256 input: autotune once


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def np_state_to_py(s):  # numpy's MT state as tensors/ints so torch.load(weights_only=True) accepts it
    return {"alg": s[0], "keys": torch.from_numpy(s[1].astype(np.int64)), "pos": int(s[2]), "gauss": int(s[3]), "cached": float(s[4])}


def py_to_np_state(d):
    return (d["alg"], d["keys"].numpy().astype(np.uint32), d["pos"], d["gauss"], d["cached"])


def get_rng_state(dev):
    st = {"cpu": torch.get_rng_state(), "np": np_state_to_py(np.random.get_state())}
    try:
        if dev.type == "cuda":
            st["dev"] = torch.cuda.get_rng_state()
        elif dev.type == "mps":
            st["dev"] = torch.mps.get_rng_state()
    except Exception as e:
        logger.warning(f"device RNG state not saved: {e}")
    return st


def set_rng_state(st, dev):
    torch.set_rng_state(st["cpu"].cpu())
    np.random.set_state(py_to_np_state(st["np"]))
    try:
        if "dev" in st and dev.type == "cuda":
            torch.cuda.set_rng_state(st["dev"].cpu())
        elif "dev" in st and dev.type == "mps":
            torch.mps.set_rng_state(st["dev"].cpu())
    except Exception as e:
        logger.warning(f"device RNG state not restored: {e}")


# -----------------------------
# Data (uint8 slice cache; RAM or memory-mapped)
# -----------------------------
def stratified_subset(ids, names, n, per_volume):
    """n volumes from ids, deterministic, keeping the thickness x tumor/no-tumor mix. Largest-remainder
    allocation over those cells, then every tumor/no-tumor and thickness value present in ids gets >=1
    volume (taken from the largest cell that can spare it). Inside a cell: evenly spaced over tumor_ml."""
    ids = list(ids)
    if n >= len(ids):
        return ids
    info = {}
    for i in ids:
        r = per_volume[names[i]]
        sz, tml = float(r["sz_mm"]), float(r["tumor_ml"])
        info[i] = (("thin" if sz <= 1.0 else "mid" if sz <= 2.5 else "thick", tml > 0), tml)
    cells = {}
    for i in ids:
        cells.setdefault(info[i][0], []).append(i)
    cells = {c: sorted(m, key=lambda i: (info[i][1], i)) for c, m in sorted(cells.items())}
    quota = {c: n * len(m) / len(ids) for c, m in cells.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(cells, key=lambda c: (alloc[c] - quota[c], c))[:n - sum(alloc.values())]:
        alloc[c] += 1
    groups = [lambda c: c[1], lambda c: not c[1]] + [lambda c, t=t: c[0] == t for t in ("thin", "mid", "thick")]
    covered = lambda a: [any(a[c] > 0 for c in a if f(c)) for f in groups]
    if n >= 4:
        for gi, f in enumerate(groups):
            if covered(alloc)[gi] or not any(f(c) for c in cells):
                continue
            before = covered(alloc)
            give = max((c for c in cells if f(c)), key=lambda c: len(cells[c]))
            for take in sorted((c for c in alloc if alloc[c] > 1), key=lambda c: (-alloc[c], c)):
                alloc[take] -= 1
                alloc[give] += 1
                if all(after or not was for after, was in zip(covered(alloc), before)):
                    break
                alloc[take] += 1
                alloc[give] -= 1
    picked = []
    for c, m in cells.items():
        k = alloc[c]
        picked += [m[int((j + 0.5) * len(m) / k)] for j in range(k)]
    return sorted(picked)


def read_names(cache_dir):
    with open(os.path.join(cache_dir, "volumes.txt")) as fh:
        return fh.read().rstrip("\n").split("\n")


def split_volumes(names, cfg, write=True):
    """-> {'train': ids, 'val': ids[, 'test': ids]} over volumes.txt order. Writes results_dir/split_used.json.
    Original rule first (104 train / 27 held-out). LITS_VAL_N > 0 then splits the held-out part into a stratified
    validation subset (checkpoint selection, early stopping, recipe choice) and the rest = test (never selected on)."""
    ids = np.arange(len(names))
    if cfg.max_volumes:
        ids = ids[:cfg.max_volumes]
    # Same split as the original script (train_test_split over the sorted volume list, seed 42).
    train_ids, held_ids = train_test_split(ids, test_size=0.2, random_state=42)
    per_volume = None
    if cfg.subset_n or cfg.val_n:
        with open(cfg.per_volume) as fh:
            per_volume = {r["name"]: r for r in csv.DictReader(fh)}
    if cfg.val_n:
        if cfg.subset_val_n != cfg.subset_n:
            logger.warning("LITS_SUBSET_VAL_N is ignored when LITS_VAL_N > 0")
        val = stratified_subset(held_ids, names, cfg.val_n, per_volume)
        parts = {"train": np.array(stratified_subset(train_ids, names, cfg.subset_n, per_volume)) if cfg.subset_n else train_ids,
                 "val": np.array(val), "test": np.array(sorted(int(i) for i in held_ids if i not in set(val)), dtype=int)}
        if not len(parts["test"]):
            logger.warning(f"LITS_VAL_N={cfg.val_n} >= held-out part ({len(held_ids)}): the test part is empty")
    else:
        parts = {"train": train_ids, "val": held_ids}
        if cfg.subset_n:  # quick-run convenience inside the original split
            parts = {k: np.array(stratified_subset(v, names, cfg.subset_val_n if k == "val" else cfg.subset_n, per_volume))
                     for k, v in parts.items()}
    assert all(len(v) for k, v in parts.items() if k != "test"), f"empty split part: { {k: len(v) for k, v in parts.items()} }"
    everyone = [int(i) for v in parts.values() for i in v]
    assert len(everyone) == len(set(everyone)), "split parts overlap"
    used = {k: [names[i] for i in v] for k, v in parts.items()}
    for k, v in used.items():
        logger.info(f"Split part {k} ({len(v)} volumes): {', '.join(v)}")
    if write:
        with open(os.path.join(cfg.results_dir, "split_used.json"), "w") as fh:
            json.dump(used, fh, indent=1)
    return parts


def chunked_flags(get_chunk, n, chunk=2048):
    """(has_liver, has_tumor) bool arrays of n mask slices; get_chunk(a, b) -> uint8 (b-a, H, W). Chunked so a
    memory-mapped 4 GB mask file is never expanded to bool in one go."""
    liver, tumor = np.zeros(n, bool), np.zeros(n, bool)
    for a in range(0, n, chunk):
        m = get_chunk(a, min(a + chunk, n))
        liver[a:a + len(m)] = (m >= 1).reshape(len(m), -1).any(1)
        tumor[a:a + len(m)] = (m == 2).reshape(len(m), -1).any(1)
    return liver, tumor


def cache_flags(masks, cache_dir, flags_file):
    """Per-slice (liver, tumor) flags for the WHOLE cache, stored in flags_file (not in the cache dir: that one may be
    read-only / must stay untouched). Stored with the masks file size+mtime to detect a rebuilt cache."""
    st = os.stat(os.path.join(cache_dir, "masks.npy"))
    sig = np.array([masks.shape[0], st.st_size, st.st_mtime_ns], dtype=np.int64)
    if os.path.exists(flags_file):
        try:
            z = np.load(flags_file)
            if np.array_equal(z["sig"], sig):
                return z["liver"], z["tumor"]
        except Exception as e:
            logger.warning(f"slice flags cache unreadable ({e}), recomputing")
    t0 = time.time()
    liver, tumor = chunked_flags(lambda a, b: np.asarray(masks[a:b]), len(masks))
    logger.info(f"Per-slice flags for {len(masks)} cache slices in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(os.path.abspath(flags_file)), exist_ok=True)
    tmp = flags_file + ".tmp.npz"
    np.savez(tmp, sig=sig, liver=liver, tumor=tumor)
    os.replace(tmp, flags_file)
    return liver, tumor


def load_aug_npz(path, max_slabs, hw):
    """Synthetic slabs {images, masks, slab_id}: first max_slabs slabs (order of first appearance; 0 = all).
    Returns dict(images, masks, slab (0..K-1, contiguous runs), n_slabs)."""
    z = np.load(path)
    img, msk, sid = z["images"], z["masks"], np.asarray(z["slab_id"]).astype(np.int64)
    assert img.dtype == np.uint8 and msk.dtype == np.uint8, "slabs must be uint8 (cache window mapping, labels 0/1/2)"
    assert img.shape == msk.shape and img.shape[0] == len(sid) and img.shape[1:] == tuple(hw), \
        f"slab arrays {img.shape} / {msk.shape} / {sid.shape} do not match the cache slice size {tuple(hw)}"
    assert int(msk.max(initial=0)) <= 2, "slab masks must only contain labels 0, 1, 2"
    _, first = np.unique(sid, return_index=True)
    ids_in_order = sid[np.sort(first)]
    if max_slabs:
        ids_in_order = ids_in_order[:max_slabs]
        keep = np.isin(sid, ids_in_order)
        img, msk, sid = img[keep], msk[keep], sid[keep]
    remap = {int(s): k for k, s in enumerate(ids_in_order)}
    return dict(images=img, masks=msk, slab=np.array([remap[int(s)] for s in sid], dtype=np.int64), n_slabs=len(ids_in_order))


class SlicePool:
    """A list of 256x256 uint8 slices addressed by pool index: real cache rows first, then optional synthetic slabs.
    Neighbours (context mode) are the previous / next pool item of the SAME volume (each slab is its own
    pseudo-volume), clamped at its borders. mmap=False keeps everything in RAM (torch tensors), mmap=True gathers
    from the memory-mapped cache; both return identical batches."""

    def __init__(self, images, masks, rows, vol_of_row, syn=None, mmap=False, context=False):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n_real = len(self.rows)
        self.syn = syn
        self.n_syn = 0 if syn is None else len(syn["images"])
        self.N = self.n_real + self.n_syn
        self.context, self.mmap = context, mmap
        self.hw = tuple(images.shape[1:])
        vol = np.asarray(vol_of_row)[self.rows].astype(np.int64)
        if self.n_syn:  # slab ids live above every real volume id: a slab is never a neighbour of a real slice
            vol = np.concatenate([vol, int(np.max(vol_of_row)) + 1 + syn["slab"]])
        self.vol = vol
        same = vol[1:] == vol[:-1]
        assert 1 + int((~same).sum()) == len(np.unique(vol)), "slices of a volume / slab are not contiguous in the pool"
        ar = np.arange(self.N)
        self.prev = np.where(np.r_[False, same], ar - 1, ar)
        self.nxt = np.where(np.r_[same, False], ar + 1, ar)
        self.local_vol = np.unique(vol, return_inverse=True)[1].astype(np.int64)  # 0..n_volumes-1
        self.n_volumes = int(self.local_vol.max()) + 1 if self.N else 0
        self.images, self.masks = images, masks
        if not mmap:
            X, Y = np.empty((self.N,) + self.hw, np.uint8), np.empty((self.N,) + self.hw, np.uint8)
            for a in range(0, self.n_real, 2048):  # chunked: bounded temporaries while filling
                r = self.rows[a:a + 2048]
                X[a:a + len(r)], Y[a:a + len(r)] = images[r], masks[r]
            if self.n_syn:
                X[self.n_real:], Y[self.n_real:] = syn["images"], syn["masks"]
            self.X, self.Y = torch.from_numpy(X), torch.from_numpy(Y)

    def _take(self, which, ix):
        if not self.mmap:
            return (self.X if which == "x" else self.Y)[torch.from_numpy(ix)]
        out = np.empty((len(ix),) + self.hw, np.uint8)
        real = ix < self.n_real
        if real.any():
            out[real] = (self.images if which == "x" else self.masks)[self.rows[ix[real]]]
        if not real.all():
            out[~real] = self.syn["images" if which == "x" else "masks"][ix[~real] - self.n_real]
        return torch.from_numpy(out)

    def get(self, idx):
        """pool indices (np int64, B) -> (uint8 (B, C, H, W), uint8 (B, H, W)); C = 3 (k-1, k, k+1) in context mode."""
        idx = np.asarray(idx, dtype=np.int64)
        if self.context:
            x = self._take("x", np.concatenate([self.prev[idx], idx, self.nxt[idx]])).reshape(3, len(idx), *self.hw).transpose(0, 1)
        else:
            x = self._take("x", idx).unsqueeze(1)
        return x.contiguous(), self._take("y", idx)


def iter_batches(pool, order, batch_size, pin=False):
    for i in range(0, len(order), batch_size):
        x, y = pool.get(order[i:i + batch_size])
        yield (x.pin_memory(), y.pin_memory()) if pin else (x, y)


def prefetch_iter(it, depth=3):
    """Run the iterator in a thread (disk gather / pinning overlaps the accelerator). Order is unchanged."""
    q, END = queue.Queue(depth), object()

    def work():
        try:
            for item in it:
                q.put(item)
            q.put(END)
        except BaseException as e:  # hand the failure to the consumer
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is END:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def build_pools(cfg, parts, mmap):
    """-> train SlicePool (real + synthetic), val SlicePool (real only), per-slice liver/tumor flags of the train pool."""
    vol_index = np.load(os.path.join(cfg.cache_dir, "vol_index.npy"))
    images = np.load(os.path.join(cfg.cache_dir, "images.npy"), mmap_mode="r")
    masks = np.load(os.path.join(cfg.cache_dir, "masks.npy"), mmap_mode="r")
    rows = lambda ids: np.flatnonzero(np.isin(vol_index, ids))  # cache order, as before
    tr_rows, va_rows = rows(parts["train"]), rows(parts["val"])
    if cfg.slice_flags:
        liver_all, tumor_all = cache_flags(masks, cfg.cache_dir, cfg.slice_flags)
        liver, tumor = liver_all[tr_rows], tumor_all[tr_rows]
    else:
        liver, tumor = chunked_flags(lambda a, b: masks[tr_rows[a:b]], len(tr_rows))
    syn = None
    if cfg.aug_npz:
        syn = load_aug_npz(cfg.aug_npz, cfg.aug_max_slabs, images.shape[1:])
        sl, st_ = chunked_flags(lambda a, b: syn["masks"][a:b], len(syn["masks"]))
        liver, tumor = np.concatenate([liver, sl]), np.concatenate([tumor, st_])
        logger.info(f"Synthetic pool: {len(syn['images'])} slices in {syn['n_slabs']} slabs "
                    f"({int(st_.sum())} with tumor) from {cfg.aug_npz}; training pool only")
    train = SlicePool(images, masks, tr_rows, vol_index, syn, mmap, cfg.context)
    val = SlicePool(images, masks, va_rows, vol_index, None, mmap, cfg.context)  # never any synthetic data
    return train, val, liver, tumor


def gpu_augment(x, y, flip=None):
    """Random h/v flip + rotation in [-10, 10] deg; bilinear for images, nearest for masks. One affine per
    sample, applied to every channel (context slices stay aligned)."""
    flip = FLIP_AUG if flip is None else flip
    B = x.shape[0]
    ang = (torch.rand(B, device=x.device) * 20 - 10) * math.pi / 180
    fx = torch.where(torch.rand(B, device=x.device) > 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, device=x.device) > 0.5, -1.0, 1.0)
    if not flip:  # keep a canonicalised orientation (prepare_data_v2) intact
        fx, fy = torch.ones_like(fx), torch.ones_like(fy)
    cos, sin = ang.cos(), ang.sin()
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1] = cos * fx, -sin * fy
    theta[:, 1, 0], theta[:, 1, 1] = sin * fx, cos * fy
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x, y[:, 0].round().long()

# -----------------------------
# AIM-Unet Model (classes only use nn/torch: eval_fast.py and the showcase scripts exec them straight from source)
# -----------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_c, in_c, 1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                     nn.Conv2d(in_c, in_c, 5, padding=2),
                                     nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                     nn.Conv2d(in_c, in_c, 3, padding=1),
                                     nn.Conv2d(in_c, in_c, 3, padding=1),
                                     nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                     nn.Conv2d(in_c, in_c, 1),
                                     nn.BatchNorm2d(in_c),
                                     nn.ReLU())

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class AIM_UNet(nn.Module):
    def __init__(self, in_channels=1):  # 3 = 2.5D context (slices k-1, k, k+1); 1 = the original model
        super().__init__()
        self.enc1, self.pool1 = ConvBlock(in_channels,64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = ConvBlock(64,128),  nn.MaxPool2d(2)
        self.enc3, self.pool3 = ConvBlock(128,256), nn.MaxPool2d(2)
        self.enc4, self.pool4 = ConvBlock(256,512), nn.MaxPool2d(2)
        self.bottleneck      = ConvBlock(512,1024)
        self.inc4, self.up4, self.dec4 = InceptionModule(512), nn.ConvTranspose2d(1024,512,2,2), ConvBlock(512*5,512)
        self.inc3, self.up3, self.dec3 = InceptionModule(256), nn.ConvTranspose2d(512,256,2,2), ConvBlock(256*5,256)
        self.inc2, self.up2, self.dec2 = InceptionModule(128), nn.ConvTranspose2d(256,128,2,2), ConvBlock(128*5,128)
        self.inc1, self.up1, self.dec1 = InceptionModule(64),  nn.ConvTranspose2d(128,64,2,2),  ConvBlock(64*5,64)
        self.out = nn.Conv2d(64,3,1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.inc4(e4), self.up4(b)],1))
        d3 = self.dec3(torch.cat([self.inc3(e3), self.up3(d4)],1))
        d2 = self.dec2(torch.cat([self.inc2(e2), self.up2(d3)],1))
        d1 = self.dec1(torch.cat([self.inc1(e1), self.up1(d2)],1))
        return self.out(d1)

# -----------------------------
# Losses / metrics
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=3).permute(0,3,1,2).float()
        inter   = (probs * one_hot).sum(dim=(0,2,3))
        union   = probs.sum(dim=(0,2,3)) + one_hot.sum(dim=(0,2,3))
        dice    = (2*inter + self.smooth)/(union + self.smooth)
        return 1.0 - dice.mean()

class TverskyLoss(nn.Module):
    """1 - TP / (TP + alpha FP + beta FN) of one class over the batch; beta > alpha punishes missed tumor more
    than false alarms (tumor voxels are ~0.5% of the data)."""
    def __init__(self, cls=2, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.cls, self.alpha, self.beta, self.smooth = cls, alpha, beta, smooth
    def forward(self, logits, target):
        p = F.softmax(logits, dim=1)[:, self.cls]
        t = (target == self.cls).float()
        tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

def make_loss(name, device):
    """-> callable(logits, target) -> scalar. ce_dice = the original (CE weights [1,2,1] + Dice);
    ce_dice_tversky = CE weights [1,2,4] + Dice + 5 x Tversky(0.3, 0.7) on the tumor class."""
    dice_loss = DiceLoss()
    if name == "ce_dice":
        ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0], device=device))
        return lambda out, y: ce(out, y) + dice_loss(out, y)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0], device=device))
    tv = TverskyLoss()
    return lambda out, y: ce(out, y) + dice_loss(out, y) + 5.0 * tv(out, y)

def dice_per_class(pred, target, num_classes=3, smooth=1e-6):
    """Per-class Dice for one batch -> tensor (num_classes,), kept on device ('legacy' per-batch metric)."""
    pred = torch.argmax(pred, dim=1)
    scores = []
    for c in range(num_classes):
        p = (pred == c).float().view(-1)
        t = (target == c).float().view(-1)
        scores.append((2*(p*t).sum() + smooth)/(p.sum() + t.sum() + smooth))
    return torch.stack(scores)

def confusion_counts(pred, target, num_classes=3):
    """Exact (true x pred) pixel counts for one batch via one-hot matmul (works on cpu/cuda/mps)."""
    t = F.one_hot(target.reshape(-1), num_classes).float()
    p = F.one_hot(pred.reshape(-1), num_classes).float()
    return (t.T @ p).round().long()

class VolumeDice:
    """Patient-level 3D Dice from per-volume sums accumulated on the device (long counts: exact, float32 would
    lose integers above 2^24 which one volume can exceed). Columns: liver inter/pred/gt, tumor inter/pred/gt.
    Liver = label >= 1 (the LiTS convention), tumor = label 2."""
    def __init__(self, n_volumes, device):
        self.acc = torch.zeros(n_volumes, 6, dtype=torch.long, device=device)

    def update(self, pred, target, vid):
        """pred, target (B,H,W) integer labels; vid (B,) long = volume number of each slice."""
        pl, gl, pt, gt = pred >= 1, target >= 1, pred == 2, target == 2
        s = torch.stack([(pl & gl).sum((1, 2)), pl.sum((1, 2)), gl.sum((1, 2)),
                         (pt & gt).sum((1, 2)), pt.sum((1, 2)), gt.sum((1, 2))], 1)
        self.acc.index_add_(0, vid, s.long())

    def result(self):
        a = self.acc.cpu().numpy().astype(np.float64)  # one (n_volumes, 6) copy per epoch
        liver = np.where(a[:, 1] + a[:, 2] > 0, 2 * a[:, 0] / np.maximum(a[:, 1] + a[:, 2], 1), np.nan)
        tumor = np.where(a[:, 4] + a[:, 5] > 0, 2 * a[:, 3] / np.maximum(a[:, 4] + a[:, 5], 1), np.nan)
        has_t = a[:, 5] > 0  # tumor Dice is averaged over validation volumes that contain tumor
        return dict(tumor3d=float(np.mean(tumor[has_t])) if has_t.any() else float("nan"),
                    liver3d=float(np.nanmean(liver)) if np.isfinite(liver).any() else float("nan"),
                    n_tumor_volumes=int(has_t.sum()), tumor_per_volume=tumor, liver_per_volume=liver)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter, self.improved = 0, False

    @property
    def best_loss(self):  # name used by the earlier version
        return self.best

    def step(self, value):
        better = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        self.improved = bool(better)
        if better:
            self.best, self.counter = value, 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def state_dict(self):
        return dict(best=self.best, counter=self.counter, improved=self.improved)

    def load_state_dict(self, d):
        self.best, self.counter, self.improved = d["best"], d["counter"], d["improved"]

# -----------------------------
# Small helpers
# -----------------------------
def nan_to_none(o):
    if isinstance(o, float):
        return None if o != o or o in (float("inf"), float("-inf")) else o
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    return o

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(nan_to_none(obj), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_atomic(obj, path):
    """torch.save to a temp file, fsync, rename: a disconnect mid-write never leaves a truncated checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_ckpt(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:  # our own files only (resume); weights_only rejects some optimizer/scheduler extras
        return torch.load(path, map_location=map_location, weights_only=False)

def setup_logging(path, append):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True,
                        handlers=[logging.FileHandler(path, mode="a" if append else "w"), logging.StreamHandler()])

def plot_history(history, path):
    ep = [h["epoch"] for h in history]
    nan = lambda v: float("nan") if v is None else v
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="Train Loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="Val Loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Training & Validation Loss"); ax[0].legend()
    ax[1].plot(ep, [nan(h["val_tumor3d"]) for h in history], label="val tumor 3D Dice")
    ax[1].plot(ep, [nan(h["val_liver3d"]) for h in history], label="val liver 3D Dice")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Dice"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Validation patient-level 3D Dice"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def window_from_cache(cache_dir):
    try:
        with open(os.path.join(cache_dir, "config.json")) as fh:
            c = json.load(fh)
        return [float(c["win_lo"]), float(c["win_hi"])]
    except Exception:
        return None

# -----------------------------
# Training Loop
# -----------------------------
def train_model(cfg):
    dev = pick_device(cfg.device)
    cc, bf16_ok = cuda_caps() if dev.type == "cuda" else (None, None)
    plan = device_plan(cfg, dev.type, cc, bf16_ok)
    if plan["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    amp, amp_dtype, use_scaler = plan["amp"], plan["amp_dtype"], plan["scaler"]
    use_amp = amp_dtype is not None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except AttributeError:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast = lambda: torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp)
    channels_last, mmap, prefetch, pin = plan["channels_last"], plan["mmap"], plan["prefetch"], plan["pin"]
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    out_dir = cfg.results_dir
    p = lambda f: os.path.join(out_dir, f)
    if cfg.resume and os.path.exists(p("done.json")):
        logger.info(f"Run already complete ({p('done.json')}): nothing to do")
        return 0

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        logger.info(f"Seed {cfg.seed}")

    t0 = time.time()
    names = read_names(cfg.cache_dir)
    resume_ck = load_ckpt(p("last.pt")) if cfg.resume and os.path.exists(p("last.pt")) else None
    old_split = read_json(p("split_used.json")) if resume_ck is not None and os.path.exists(p("split_used.json")) else None
    parts = split_volumes(names, cfg)
    if old_split is not None and read_json(p("split_used.json")) != old_split:
        logger.error("Refusing to resume: the split differs from the one of the interrupted run")
        return EXIT_RESUME_REFUSED
    train_pool, val_pool, has_liver_np, has_tumor_np = build_pools(cfg, parts, mmap)
    nr = train_pool.n_real
    logger.info(f"Device {dev}, AMP {amp if use_amp else 'off'}{' + GradScaler' if use_scaler else ''}, channels_last {channels_last}, "
                f"{'mmap' if mmap else 'RAM'} batches{' (prefetch thread)' if prefetch else ''}, context {int(cfg.context)} "
                f"(in_channels {cfg.in_channels}), loss {cfg.loss}, lr {cfg.lr} {cfg.lr_sched}, select {cfg.select} | "
                f"train slices {nr}{f' + {train_pool.n_syn} synthetic' if train_pool.n_syn else ''}, "
                f"val slices {val_pool.N} in {val_pool.n_volumes} volumes (loaded in {time.time()-t0:.1f}s)")

    # Oversample tumor slices (same weighting as the original: weight = n_non / n_tumor over the REAL training slices;
    # synthetic tumor slices get that same weight, so the synthetic share of the draws follows the data, not the epoch)
    n_tumor = int(has_tumor_np[:nr].sum()); n_non = nr - n_tumor
    has_tumor, has_liver = torch.from_numpy(has_tumor_np), torch.from_numpy(has_liver_np)
    weights = torch.where(has_tumor, n_non / max(n_tumor, 1), 1.0).float()
    # Optional: ~2/3 of LiTS slices contain no liver. LITS_EMPTY_KEEP<1 down-weights them and
    # shortens the epoch accordingly (1.0 = original behaviour: every slice counted once per epoch).
    keep = cfg.empty_keep
    weights = torch.where(has_liver, weights, weights * keep)
    # Epoch length from the real slices only, so adding synthetic slices does not change the compute per epoch.
    epoch_len = cfg.epoch_len or int(has_liver[:nr].sum() + keep * (~has_liver[:nr]).sum())
    logger.info(f"Slices/epoch: {epoch_len} (tumor {n_tumor}, liver {int(has_liver[:nr].sum())}, empty kept x{keep})")

    model = AIM_UNet(cfg.in_channels).to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - 1), eta_min=cfg.lr * 0.01) \
        if cfg.lr_sched == "cosine" else None  # per-epoch, ends at 1% of lr on the last planned epoch
    loss_fn = make_loss(cfg.loss, dev)
    earlystop = EarlyStopping(patience=cfg.patience, mode="min" if cfg.select == "loss" else "max")
    config = dict(loss=cfg.loss, context=int(cfg.context), in_channels=cfg.in_channels, amp=amp if use_amp else "off",
                  window=window_from_cache(cfg.cache_dir), lr_sched=cfg.lr_sched, select=cfg.select)
    history, best = [], dict(score=earlystop.best, epoch=0)
    start_ep, stopped = 1, None

    if resume_ck is not None:
        old = resume_ck.get("config", {})
        bad = {k: (old.get(k), config[k]) for k in ("loss", "context", "in_channels", "lr_sched", "select") if old.get(k) != config[k]}
        if bad:
            logger.error(f"Refusing to resume: config differs from the interrupted run (old, new): {bad}")
            return EXIT_RESUME_REFUSED
        if resume_ck.get("planned_epochs") != cfg.epochs:
            logger.warning(f"LITS_EPOCHS is {cfg.epochs}, the interrupted run planned {resume_ck.get('planned_epochs')}"
                           f"{': the cosine schedule keeps its original length' if sched else ''}")
        model.load_state_dict(resume_ck["model"])
        opt.load_state_dict(resume_ck["opt"])
        if sched is not None:
            sched.load_state_dict(resume_ck["sched"])
        scaler.load_state_dict(resume_ck["scaler"])
        earlystop.load_state_dict(resume_ck["earlystop"])
        history, best, start_ep = resume_ck["history"], resume_ck["best"], resume_ck["epoch"] + 1
        set_rng_state(resume_ck["rng"], dev)
        stopped = resume_ck.get("stopped")
        logger.info(f"Resumed from {p('last.pt')}: {resume_ck['epoch']} epochs done, best epoch {best['epoch']}, "
                    f"early-stopping counter {earlystop.counter}, lr now {opt.param_groups[0]['lr']:.3g}")
        del resume_ck

    def to_dev(xb, yb):
        xb = xb.to(dev, non_blocking=True).float() / 255.
        return xb.contiguous(memory_format=mem_fmt), yb.to(dev, non_blocking=True).long()

    def check_finite(running):
        if not bool(torch.isfinite(running)):
            logger.error("NONFINITE_LOSS: training loss is NaN/inf, aborting")
            print("NONFINITE_LOSS", flush=True)
            sys.exit(EXIT_NONFINITE)

    def train_step(xb, yb):
        xb, yb = to_dev(xb, yb)
        xb, yb = gpu_augment(xb, yb, cfg.flip)
        xb = xb.contiguous(memory_format=mem_fmt)
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(xb)
        out = out.float()
        loss = loss_fn(out, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    val_vid = torch.from_numpy(val_pool.local_vol).to(dev)
    val_order = np.arange(val_pool.N)

    def val_batches():
        it = iter_batches(val_pool, val_order, cfg.batch_size, pin)
        return prefetch_iter(it) if prefetch else it

    if cfg.probe_only:
        return probe(cfg, dev, model, train_step, to_dev, autocast, val_batches, train_pool, val_pool.N, weights,
                     epoch_len, prefetch, pin, amp if use_amp else "off", mmap, channels_last)

    if stopped is not None or start_ep > cfg.epochs:
        logger.info("Nothing left to train: finalizing")
    n_val_batches = math.ceil(val_pool.N / cfg.batch_size)
    for ep in range(start_ep, cfg.epochs + 1):
        if stopped is not None:
            break
        ep_start = time.time()
        lr_now = opt.param_groups[0]["lr"]
        # — Training
        model.train()
        running_train = torch.zeros((), device=dev)
        order = torch.multinomial(weights, epoch_len, replacement=True)
        order_np = order.numpy()
        syn_share = float((order_np >= nr).mean()) if len(order_np) else 0.0
        n_batches = math.ceil(len(order) / cfg.batch_size)
        it = iter_batches(train_pool, order_np, cfg.batch_size, pin)
        for i, (xb, yb) in enumerate(prefetch_iter(it) if prefetch else it):
            running_train += train_step(xb, yb)
            if (i + 1) % 50 == 0:
                check_finite(running_train)  # one host sync per 50 batches
        check_finite(running_train)
        avg_train = running_train.item() / n_batches
        train_time = time.time() - ep_start
        logger.info(f"Epoch {ep}/{cfg.epochs} — Train Loss: {avg_train:.4f} "
                    f"({train_time:.0f}s, {len(order)/train_time:.1f} slices/s, lr {lr_now:.3g})")
        if train_pool.n_syn:
            logger.info(f"Epoch {ep}/{cfg.epochs} — synthetic share of the drawn training samples: {100*syn_share:.1f}% "
                        f"({int(round(syn_share*len(order_np)))} of {len(order_np)})")

        # — Validation (loss + metrics)
        model.eval()
        running_val = torch.zeros((), device=dev)
        dice_sum = torch.zeros(3, device=dev)
        cm = torch.zeros(3, 3, dtype=torch.long, device=dev)
        vd = VolumeDice(val_pool.n_volumes, dev)
        auc_true, auc_score = [], []
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_batches()):
                xb, yb = to_dev(xb, yb)
                with autocast():
                    out = model(xb)
                out = out.float()
                running_val += loss_fn(out, yb)
                dice_sum += dice_per_class(out, yb)
                pred = out.argmax(1)
                cm += confusion_counts(pred, yb)
                vd.update(pred, yb, val_vid[i * cfg.batch_size:(i + 1) * cfg.batch_size])
                probs = F.softmax(out, dim=1).permute(0,2,3,1).reshape(-1, 3)
                pick = torch.randint(0, probs.shape[0], (AUC_PIXELS_PER_BATCH,), device=dev)
                auc_score.append(probs[pick]); auc_true.append(yb.reshape(-1)[pick])

        avg_val = running_val.item() / n_val_batches
        d = (dice_sum / n_val_batches).tolist()
        v3 = vd.result()
        if cfg.select == "tumor3d" and v3["n_tumor_volumes"] == 0:
            raise RuntimeError("LITS_SELECT=tumor3d needs at least one validation volume with tumor")
        logger.info(f"Epoch {ep}/{cfg.epochs} — Val Loss:   {avg_val:.4f}")
        logger.info(f"Validation Dice — Background: {d[0]:.4f}, Liver: {d[1]:.4f}, Tumor: {d[2]:.4f} (legacy per-batch mean)")
        logger.info(f"Validation 3D Dice (patient level) — tumor: {v3['tumor3d']:.4f} "
                    f"(mean over {v3['n_tumor_volumes']} of {val_pool.n_volumes} volumes with tumor), liver: {v3['liver3d']:.4f}")

        # — Early stopping / best model
        score = avg_val if cfg.select == "loss" else v3["tumor3d"]
        stop = earlystop.step(score)
        if earlystop.improved:
            best = dict(score=score, epoch=ep, val_loss=avg_val, val_tumor3d=v3["tumor3d"], val_liver3d=v3["liver3d"])
            save_atomic(dict(model=model.state_dict(), config=config, epoch=ep, val_loss=avg_val,
                             val_tumor3d=v3["tumor3d"], val_liver3d=v3["liver3d"]), p("best.pt"))
            logger.info(f"✅ Saved new best model (epoch {ep}, {cfg.select} {score:.4f}) to {p('best.pt')}")

        # — Other metrics (confusion matrix exact over all pixels)
        cm = cm.cpu().numpy()
        logger.info(f"Confusion Matrix:\n{cm}")
        total = cm.sum()
        for c in range(3):
            TP = cm[c,c]; FN = cm[c,:].sum() - TP
            FP = cm[:,c].sum() - TP; TN = total - TP - FN - FP
            sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
            spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
            logger.info(f"Class {c} — Sens: {sens:.4f}, Spec: {spec:.4f}")
        f1s = [2*cm[c,c]/(cm[c,:].sum() + cm[:,c].sum()) if (cm[c,:].sum() + cm[:,c].sum())>0 else 0.0
               for c in range(3)]
        try:
            auc = roc_auc_score(torch.cat(auc_true).cpu().numpy(), torch.cat(auc_score).float().cpu().numpy(),
                                multi_class='ovr', labels=[0,1,2])
        except ValueError:
            auc = float('nan')
        epoch_s = time.time() - ep_start
        logger.info(f"ROC AUC (pixel subsample): {auc:.4f} — Macro F1: {np.mean(f1s):.4f} | epoch time {epoch_s:.0f}s")

        history.append(dict(epoch=ep, train_loss=avg_train, val_loss=avg_val, val_tumor3d=v3["tumor3d"], val_liver3d=v3["liver3d"],
                            legacy_batch_dice=dict(background=d[0], liver=d[1], tumor=d[2]), lr=lr_now,
                            samples_per_s=len(order) / train_time, seconds=epoch_s, synthetic_share=syn_share,
                            val_tumor3d_per_volume=[float(x) for x in v3["tumor_per_volume"]],
                            val_liver3d_per_volume=[float(x) for x in v3["liver_per_volume"]]))
        write_json_atomic(history, p("history.json"))
        if sched is not None:
            sched.step()
        if stop:
            stopped = "early"
            logger.info(f"▶ Early stopping at epoch {ep} (no improvement for {cfg.patience} epochs)")
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict() if sched else None,
                         scaler=scaler.state_dict(), epoch=ep, best=best, earlystop=earlystop.state_dict(), history=history,
                         rng=get_rng_state(dev), config=config, planned_epochs=cfg.epochs, stopped=stopped), p("last.pt"))
        if stop:
            break
        if cfg.stop_after and ep >= cfg.stop_after and ep < cfg.epochs:
            logger.info(f"⏸ Paused after epoch {ep} (LITS_STOP_AFTER); rerun with LITS_RESUME=1 to continue")
            print(f"PAUSED epoch={ep}", flush=True)
            return 0

    # — Final: done.json, plot
    epochs_run = history[-1]["epoch"] if history else 0
    write_json_atomic(dict(epochs_run=epochs_run, best_epoch=best["epoch"], best_val_tumor3d=best.get("val_tumor3d"),
                           best_val_liver3d=best.get("val_liver3d"), best_val_loss=best.get("val_loss"),
                           stopped="early" if stopped == "early" else "max", select=cfg.select, planned_epochs=cfg.epochs,
                           device=dev.type, amp=amp if use_amp else "off"), p("done.json"))
    logger.info(f"🏁 Training complete: {epochs_run} epochs, best epoch {best['epoch']} (val tumor 3D {best.get('val_tumor3d')}).")
    if history:  # plot (saved, not shown, so unattended runs don't block)
        plot_history(history, p("loss_curves.png"))
    return 0


def probe(cfg, dev, model, train_step, to_dev, autocast, val_batches, train_pool, val_slices, weights, epoch_len,
          prefetch, pin, amp_name, mmap, channels_last):
    """~40 timed train steps (first quarter = warm-up: cudnn autotune, allocator) and ~10 val batches -> one PROBE line."""
    steps = cfg.probe_steps
    warm = max(1, steps // 4)
    order = torch.multinomial(weights, steps * cfg.batch_size, replacement=True).numpy()
    model.train()
    t0, n_timed, loss = None, 0, None
    it = iter_batches(train_pool, order, cfg.batch_size, pin)
    for i, (xb, yb) in enumerate(prefetch_iter(it) if prefetch else it):
        if i == warm:
            sync(dev)
            t0 = time.time()
        loss = train_step(xb, yb)
        if i >= warm:
            n_timed += len(xb)
    sync(dev)
    if loss is None or not bool(torch.isfinite(loss)):
        print("NONFINITE_LOSS", flush=True)
        return EXIT_NONFINITE
    train_sps = n_timed / max(time.time() - t0, 1e-9)
    model.eval()
    n_val, vwarm, nv, t1 = 10, 2, 0, None
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_batches()):
            if i == vwarm:
                sync(dev)
                t1 = time.time()
            xb, yb = to_dev(xb, yb)
            with autocast():
                out = model(xb)
            out.float().argmax(1)
            if i >= vwarm:
                nv += len(xb)
            if i + 1 >= n_val + vwarm:
                break
    sync(dev)
    val_sps = nv / max(time.time() - t1, 1e-9) if t1 else train_sps * 3  # < 3 val batches: assume inference ~3x faster
    res = dict(train_slices_per_s=round(train_sps, 2), val_slices_per_s=round(val_sps, 2), samples_per_epoch=int(epoch_len),
               val_slices=int(val_slices), batch_size=cfg.batch_size, device=dev.type, amp=amp_name, mmap=bool(mmap),
               channels_last=bool(channels_last), context=int(cfg.context),
               est_epoch_min=round((epoch_len / train_sps + val_slices / val_sps) / 60, 3))
    if dev.type == "cuda":
        res["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print("PROBE " + json.dumps(res), flush=True)
    return 0


def main():
    cfg = Cfg()
    os.makedirs(cfg.results_dir, exist_ok=True)
    # utility modes get their own log so they never truncate the log of the real run in the same directory
    log = "probe_AIMUNet.log" if cfg.probe_only else "split_AIMUNet.log" if cfg.split_only else "train_AIMUNet.log"
    setup_logging(os.path.join(cfg.results_dir, log), append=cfg.resume or cfg.probe_only or cfg.split_only)
    if cfg.split_only:
        split_volumes(read_names(cfg.cache_dir), cfg)
        return 0
    return train_model(cfg)


if __name__ == "__main__":
    sys.exit(main())
