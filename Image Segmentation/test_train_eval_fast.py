"""Tests for train_fast.py / eval_fast.py (run: .venv/bin/python -I "Image Segmentation/test_train_eval_fast.py").

Most tests use a tiny synthetic cache (20 volumes x 12 slices of 32x32, built in a temp dir) and run on CPU in this
process, so they take seconds. The CUDA path cannot run here: device decisions are one pure function (device_plan /
resolve_amp / cuda_caps) that is tested with mocked torch.cuda queries. Regression tests compare against a snapshot of
the previous train_fast.py / eval_fast.py (LITS_SNAPSHOT_DIR, default /tmp/colab_dev/snapshot) and are skipped when
it is missing. One test trains on a real 3-volume subset of data/cache_v2 (LITS_MAX_VOLUMES=3, skipped without data).
Env: LITS_TEST_SKIP_REAL=1 skips the real-data tests; LITS_REPO_ROOT = repo holding data/ and results/ (default: the parent
of this folder); LITS_SNAPSHOT_DIR = snapshot of the previous scripts (regression tests).
"""
import os
import io
import sys
import csv
import json
import shutil
import hashlib
import tempfile
import unittest
import contextlib
import functools
import ast
import importlib.util
from unittest import mock

import numpy as np
import torch

sys.dont_write_bytecode = True  # importing the scripts under test must not drop __pycache__ into the repo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LITS_REPO_ROOT", os.path.dirname(HERE))  # repo with data/, results/baseline, analysis/output (dev copies: set it)
SNAP = os.environ.get("LITS_SNAPSHOT_DIR", "/tmp/colab_dev/snapshot")
SNAP_SEG = os.path.join(SNAP, "Image Segmentation")
REAL_CACHE = os.environ.get("LITS_REAL_CACHE", os.path.join(ROOT, "data", "cache_v2"))
REAL_PV = os.path.join(ROOT, "analysis", "output", "per_volume.csv")


def load_module(name, path, env=None):
    with env_ctx(env or {}):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def env_ctx(env):
    """Replace every LITS_* variable by env for the duration (tests must not see the caller's settings)."""
    saved = dict(os.environ)
    for k in [k for k in os.environ if k.startswith("LITS_")]:
        del os.environ[k]
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


tf = load_module("train_fast_under_test", os.path.join(HERE, "train_fast.py"))
ev = load_module("eval_fast_under_test", os.path.join(HERE, "eval_fast.py"))
TMP = CACHE = PV = None


def make_cache(d, n_vols=20, n_slices=12, hw=32, seed=0):
    """Tiny cache with the real layout: images/masks uint8 (N,hw,hw), vol_index, volumes.txt (lexicographic names),
    config.json, plus a per_volume.csv. Tumor in 2 of every 3 volumes, thickness cycling thin/mid/thick."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(seed)
    names = sorted(f"volume-{i}.nii" for i in range(n_vols))
    imgs, msks, vi, meta = [], [], [], []
    yy, xx = np.mgrid[:hw, :hw]
    for k, n in enumerate(names):
        has_t = k % 3 != 2
        for z in range(n_slices):
            m = np.zeros((hw, hw), np.uint8)
            if 3 <= z < 9:
                m[((yy - 14) / 9.) ** 2 + ((xx - 16) / 11.) ** 2 < 1] = 1
                if has_t and z in (5, 6, 7):
                    m[((yy - 13) / 3.) ** 2 + ((xx - 18) / 3.) ** 2 < 1] = 2
            imgs.append((60 + 100 * (m >= 1) - 40 * (m == 2) + rng.randint(0, 20, (hw, hw))).clip(0, 255).astype(np.uint8))
            msks.append(m)
            vi.append(k)
        meta.append((n, 0.7 + (k % 3) * 1.8, 5.0 if has_t else 0.0))
    np.save(f"{d}/images.npy", np.stack(imgs))
    np.save(f"{d}/masks.npy", np.stack(msks))
    np.save(f"{d}/vol_index.npy", np.array(vi, np.int32))
    with open(f"{d}/volumes.txt", "w") as f:
        f.write("\n".join(names))
    with open(f"{d}/config.json", "w") as f:
        json.dump({"win_lo": -100.0, "win_hi": 250.0}, f)
    with open(f"{d}/per_volume.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sx_mm", "sy_mm", "sz_mm", "tumor_ml"])
        for n, sz, t in meta:
            w.writerow([n, 1.4, 1.4, sz, t])
    return d, f"{d}/per_volume.csv"


def make_slabs(path, hw=32, n_slabs=3, per=6, seed=1):
    rng = np.random.RandomState(seed)
    n = n_slabs * per
    img = rng.randint(0, 256, (n, hw, hw)).astype(np.uint8)
    msk = np.zeros((n, hw, hw), np.uint8)
    msk[:, 8:24, 8:24] = 1
    msk[::2, 14:18, 14:18] = 2
    np.savez(path, images=img, masks=msk, slab_id=np.repeat(np.arange(n_slabs) + 100, per).astype(np.int32))
    return path


def setUpModule():
    global TMP, CACHE, PV
    TMP = tempfile.mkdtemp(prefix="lits_test_")
    CACHE, PV = make_cache(os.path.join(TMP, "cache"))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def run_train(name, expect_exit=0, **env):
    """tf.main() in this process on CPU with LITS_* set as given. Returns (results_dir, exit code, stdout, stderr)."""
    res = os.path.join(TMP, name)
    base = dict(LITS_CACHE_DIR=CACHE, LITS_PER_VOLUME=PV, LITS_RESULTS_DIR=res, LITS_DEVICE="cpu", LITS_BATCH_SIZE=4,
                LITS_EPOCH_LEN=48)
    out, err = io.StringIO(), io.StringIO()
    with env_ctx({**base, **env}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = tf.main()
        except SystemExit as e:
            code = e.code
    return res, code, out.getvalue(), err.getvalue()


def run_eval(ckpt, out_dir, cache=None, per_volume=None, **env):
    out, err = io.StringIO(), io.StringIO()
    mod = ev
    with env_ctx({"LITS_CKPT": ckpt, "LITS_DEVICE": "cpu", **env}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch.object(mod, "CACHE_DIR", cache or CACHE), mock.patch.object(mod, "PER_VOLUME", per_volume or PV), \
            mock.patch.object(mod, "OUT_DIR", out_dir):
        try:
            code = mod.main()
        except SystemExit as e:
            code = e.code
    return code, err.getvalue()


def jl(path):
    with open(path) as f:
        return json.load(f)


def wj(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rd(path):
    with open(path) as f:
        return f.read()


def md5(t):
    return hashlib.md5(np.ascontiguousarray(t.numpy() if torch.is_tensor(t) else t).tobytes()).hexdigest()


def sd_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


STABLE = ("epoch", "train_loss", "val_loss", "val_tumor3d", "val_liver3d", "lr", "legacy_batch_dice", "synthetic_share",
          "val_tumor3d_per_volume", "val_liver3d_per_volume")  # the rest of a history row is timing


# ----------------------------------------------------------------------------------------------------------------
class TestSplit(unittest.TestCase):
    def test_original_rule_and_val_test_partition_on_real_names(self):
        """Default = the original 104/27 split; LITS_VAL_N=8 splits the 27 held-out into 8 val (stratified picker) + 19 test."""
        if not os.path.exists(os.path.join(REAL_CACHE, "volumes.txt")) or not os.path.exists(REAL_PV):
            self.skipTest("no real cache names / per_volume.csv")
        from sklearn.model_selection import train_test_split
        names = rd(os.path.join(REAL_CACHE, "volumes.txt")).split("\n")
        tr, held = train_test_split(np.arange(len(names)), test_size=0.2, random_state=42)
        fake = os.path.join(TMP, "names_only")  # only volumes.txt: SPLIT_ONLY must not need (or load) images
        os.makedirs(fake, exist_ok=True)
        shutil.copy(os.path.join(REAL_CACHE, "volumes.txt"), fake)
        base = dict(LITS_CACHE_DIR=fake, LITS_PER_VOLUME=REAL_PV, LITS_SPLIT_ONLY=1)
        with env_ctx({**base, "LITS_RESULTS_DIR": os.path.join(TMP, "split0")}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tf.main(), 0)
        s0 = jl(os.path.join(TMP, "split0", "split_used.json"))
        self.assertEqual(list(s0), ["train", "val"])  # same two keys as before
        self.assertEqual(s0["train"], [names[i] for i in tr])
        self.assertEqual(s0["val"], [names[i] for i in held])
        self.assertEqual((len(s0["train"]), len(s0["val"])), (104, 27))
        with env_ctx({**base, "LITS_RESULTS_DIR": os.path.join(TMP, "split8"), "LITS_VAL_N": 8}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tf.main(), 0)
        s8 = jl(os.path.join(TMP, "split8", "split_used.json"))
        self.assertEqual(s8["train"], s0["train"])
        self.assertEqual((len(s8["val"]), len(s8["test"])), (8, 19))
        self.assertEqual(sorted(s8["val"] + s8["test"]), sorted(s0["val"]))
        self.assertFalse(set(s8["val"]) & set(s8["test"]))
        pv = {r["name"]: r for r in rows_of(REAL_PV)}
        self.assertEqual(s8["val"], [names[i] for i in tf.stratified_subset(held, names, 8, pv)])
        self.assertTrue(any(float(pv[n]["tumor_ml"]) > 0 for n in s8["val"]))
        self.assertEqual(sorted(os.listdir(os.path.join(TMP, "split8"))), ["split_AIMUNet.log", "split_used.json"])

    def test_val_n_on_tiny_cache(self):
        res, code, _, _ = run_train("split_tiny", LITS_SPLIT_ONLY=1, LITS_VAL_N=2, LITS_SUBSET_N=6)
        s = jl(os.path.join(res, "split_used.json"))
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (6, 2, 2))
        self.assertFalse(set(s["train"]) & (set(s["val"]) | set(s["test"])))


class TestDeviceLogic(unittest.TestCase):
    def cfg(self, **env):
        with env_ctx(env):
            return tf.Cfg()

    def test_amp_auto_rules(self):
        r = tf.resolve_amp
        self.assertEqual(r("auto", "mps")[0::2], ("bf16", False))
        self.assertEqual(r("auto", "cpu"), ("off", None, False))
        self.assertEqual(r("auto", "cuda", (8, 0), True), ("bf16", torch.bfloat16, False))   # A100
        self.assertEqual(r("auto", "cuda", (7, 5), True), ("fp16", torch.float16, True))     # T4: is_bf16_supported() lies, cc decides
        self.assertEqual(r("auto", "cuda", (8, 6), False)[0], "fp16")                        # capability ok but bf16 reported unsupported
        self.assertEqual(r("auto", "cuda", None, None)[0::2], ("fp16", True))                # queries failed: safe default
        self.assertEqual(r("bf16", "cuda", (7, 5), True)[0], "bf16")                         # explicit setting honoured
        self.assertEqual(r("fp16", "cuda", (8, 0), True)[0::2], ("fp16", True))
        self.assertEqual(r("fp16", "mps")[0::2], ("fp16", False))                            # GradScaler only on cuda
        self.assertEqual(r("bf16", "cpu"), ("off", None, False))
        self.assertEqual(r("off", "cuda", (8, 0), True), ("off", None, False))
        with self.assertRaises(ValueError):
            r("fp8", "cuda")

    def test_cuda_caps_with_mocked_queries(self):
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), \
                mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(tf.cuda_caps(), ((7, 5), True))
        with mock.patch.object(torch.cuda, "get_device_capability", side_effect=RuntimeError("no driver")), \
                mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=AttributeError("old torch")):
            self.assertEqual(tf.cuda_caps(), (None, None))

    def test_pick_device_with_mocked_availability(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(tf.pick_device("auto").type, "cuda")
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(tf.pick_device("auto").type, "cpu")
        self.assertEqual(tf.pick_device("cpu").type, "cpu")

    def test_device_plan_t4_vs_a100_vs_mps_vs_cpu(self):
        t4 = tf.device_plan(self.cfg(), "cuda", (7, 5), True)
        self.assertEqual((t4["amp"], t4["scaler"], t4["channels_last"], t4["mmap"], t4["prefetch"], t4["pin"], t4["cudnn_benchmark"]),
                         ("fp16", True, True, True, True, True, True))
        a100 = tf.device_plan(self.cfg(), "cuda", (8, 0), True)
        self.assertEqual((a100["amp"], a100["scaler"]), ("bf16", False))
        mps = tf.device_plan(self.cfg(), "mps")
        self.assertEqual((mps["amp"], mps["scaler"], mps["channels_last"], mps["mmap"], mps["prefetch"], mps["pin"], mps["cudnn_benchmark"]),
                         ("bf16", False, False, False, False, False, False))
        cpu = tf.device_plan(self.cfg(), "cpu")
        self.assertEqual((cpu["amp"], cpu["amp_dtype"], cpu["pin"]), ("off", None, False))
        # overrides
        o = tf.device_plan(self.cfg(LITS_MMAP=0, LITS_CHANNELS_LAST=0, LITS_AMP="off"), "cuda", (7, 5), True)
        self.assertEqual((o["mmap"], o["channels_last"], o["amp"], o["prefetch"]), (False, False, "off", True))
        o = tf.device_plan(self.cfg(LITS_MMAP=1, LITS_CHANNELS_LAST=1), "cpu")
        self.assertEqual((o["mmap"], o["channels_last"], o["prefetch"]), (True, True, True))

    def test_gradscaler_enabled_only_for_fp16_cuda_in_train_model(self):
        """train_model builds torch.amp.GradScaler('cuda', enabled=plan['scaler']): scaler.scale() is the identity when off."""
        s = torch.amp.GradScaler("cuda", enabled=False)
        x = torch.tensor(3.0)
        self.assertIs(s.scale(x), x)

    def test_cpu_run_never_pins_memory(self):
        seen = []
        orig = tf.iter_batches
        with mock.patch.object(tf, "iter_batches", side_effect=lambda *a, **k: (seen.append(a[3] if len(a) > 3 else k.get("pin")), orig(*a, **k))[1]):
            run_train("nopin", LITS_EPOCHS=1, LITS_EPOCH_LEN=8)
        self.assertTrue(seen and not any(seen), seen)

    def test_bad_choice_rejected(self):
        for k, v in (("LITS_LOSS", "dice"), ("LITS_SELECT", "dice"), ("LITS_AMP", "fp8"), ("LITS_LR_SCHED", "step")):
            with env_ctx({k: v}), self.assertRaises(ValueError):
                tf.Cfg()


class TestVolumeDice(unittest.TestCase):
    @staticmethod
    def numpy_ref(pred, gt, vid, n):
        """Independent per-volume implementation on host copies."""
        liver, tumor = [], []
        for v in range(n):
            p, g = pred[vid == v], gt[vid == v]
            for out, (a, b) in ((liver, (p >= 1, g >= 1)), (tumor, (p == 2, g == 2))):
                s = a.sum() + b.sum()
                out.append(np.nan if s == 0 else 2.0 * (a & b).sum() / s)
        liver, tumor = np.array(liver), np.array(tumor)
        has_t = np.array([(gt[vid == v] == 2).any() for v in range(n)])
        return float(np.mean(tumor[has_t])) if has_t.any() else float("nan"), float(np.nanmean(liver)), tumor, liver

    def check(self, device):
        rng = np.random.RandomState(3)
        n, per = 5, 7
        gt = np.zeros((n * per, 16, 16), np.int64)
        for v in range(n):
            gt[v * per:(v + 1) * per, 2:12, 3:13] = 1
            if v != 3:  # volume 3: no GT tumor
                gt[v * per + 2:v * per + 5, 5:9, 5:9] = 2
        pred = np.where(rng.rand(*gt.shape) < 0.85, gt, rng.randint(0, 3, gt.shape))
        pred[4 * per:] = 0  # volume 4: empty prediction -> tumor Dice 0 (GT has tumor), liver Dice 0
        vid = np.repeat(np.arange(n), per)
        vd = tf.VolumeDice(n, device)
        for a in range(0, len(gt), 4):  # batches that cut across volumes
            vd.update(torch.from_numpy(pred[a:a + 4]).to(device), torch.from_numpy(gt[a:a + 4]).to(device),
                      torch.from_numpy(vid[a:a + 4]).to(device))
        res = vd.result()
        t3, l3, tp, lv = self.numpy_ref(pred, gt, vid, n)
        self.assertAlmostEqual(res["tumor3d"], t3, places=12)
        self.assertAlmostEqual(res["liver3d"], l3, places=12)
        np.testing.assert_allclose(res["tumor_per_volume"], tp, rtol=0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(res["liver_per_volume"], lv, rtol=0, atol=1e-12, equal_nan=True)
        self.assertEqual(res["n_tumor_volumes"], 4)
        self.assertEqual(res["tumor_per_volume"][4], 0.0)
        self.assertTrue(np.isnan(res["tumor_per_volume"][3]) or res["tumor_per_volume"][3] == 0.0)

    def test_matches_numpy_cpu(self):
        self.check(torch.device("cpu"))

    def test_matches_numpy_mps(self):
        if not torch.backends.mps.is_available():
            self.skipTest("no mps")
        self.check(torch.device("mps"))

    def test_counts_beyond_float32_exact_range(self):
        """One volume > 2^24 voxels: float32 accumulation would be off; the long accumulator must be exact."""
        vd = tf.VolumeDice(1, torch.device("cpu"))
        vid = torch.zeros(10, dtype=torch.long)
        pred = torch.ones(10, 1000, 1700, dtype=torch.long)  # 17.0M voxels > 16,777,216
        gt = pred.clone()
        gt[0, :1, :3] = 0  # 3 voxels fewer in GT
        vd.update(pred, gt, vid)
        a = vd.acc[0].tolist()
        self.assertEqual(a[0], 17_000_000 - 3)
        self.assertEqual(a[1], 17_000_000)
        self.assertEqual(a[2], 17_000_000 - 3)
        self.assertAlmostEqual(vd.result()["liver3d"], 2 * (17_000_000 - 3) / (34_000_000 - 3), places=12)

    def test_no_tumor_in_validation_gives_nan_not_crash(self):
        vd = tf.VolumeDice(1, torch.device("cpu"))
        z = torch.zeros(2, 4, 4, dtype=torch.long)
        vd.update(z, z, torch.zeros(2, dtype=torch.long))
        r = vd.result()
        self.assertTrue(np.isnan(r["tumor3d"]) and r["n_tumor_volumes"] == 0)


class TestLosses(unittest.TestCase):
    def test_tversky_hand_computed(self):
        logits = torch.zeros(1, 3, 2, 2)  # uniform probs 1/3
        logits[0, 2, 0, 0] = 20.0         # pixel (0,0): tumor prob ~1
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        target[0, 0, 0] = 2
        target[0, 0, 1] = 2
        p = torch.softmax(logits, 1)[0, 2]
        t = (target[0] == 2).float()
        tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
        expect = 1 - (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
        self.assertAlmostEqual(float(tf.TverskyLoss()(logits, target)), float(expect), places=6)

    def test_tversky_perfect_and_empty(self):
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 1:3, 1:3] = 2
        perfect = torch.zeros(1, 3, 4, 4)
        perfect[0, 2] = (target[0] == 2).float() * 60 - 30
        perfect[0, 0] = -perfect[0, 2]
        self.assertLess(float(tf.TverskyLoss()(perfect, target)), 1e-4)
        no_t = torch.zeros(1, 4, 4, dtype=torch.long)
        confident_bg = torch.zeros(1, 3, 4, 4)
        confident_bg[0, 0] = 30
        self.assertLess(float(tf.TverskyLoss()(confident_bg, no_t)), 1e-4)  # no tumor, none predicted: no penalty
        fp = torch.zeros(1, 3, 4, 4)
        fp[0, 2] = 30
        self.assertGreater(float(tf.TverskyLoss()(fp, no_t)), 0.99)         # all false positive: full penalty

    def test_make_loss_variants(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 8, 8)
        y = torch.randint(0, 3, (2, 8, 8))
        base, tv = tf.make_loss("ce_dice", "cpu"), tf.make_loss("ce_dice_tversky", "cpu")
        ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]))
        self.assertAlmostEqual(float(base(logits, y)), float(ce(logits, y) + tf.DiceLoss()(logits, y)), places=6)
        ce4 = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(float(tv(logits, y)), float(ce4(logits, y) + tf.DiceLoss()(logits, y) + 5 * tf.TverskyLoss()(logits, y)), places=5)


class TestPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slabs = make_slabs(os.path.join(TMP, "slabs.npz"))

    def pools(self, mmap, context=False, aug=True, max_slabs=0):
        env = dict(LITS_CACHE_DIR=CACHE, LITS_PER_VOLUME=PV, LITS_RESULTS_DIR=os.path.join(TMP, "unused"), LITS_VAL_N=2,
                   LITS_CONTEXT=int(context))
        if aug:
            env.update(LITS_AUG_NPZ=self.slabs, LITS_AUG_MAX_SLABS=max_slabs)
        with env_ctx(env):
            cfg = tf.Cfg()
        os.makedirs(cfg.results_dir, exist_ok=True)
        with contextlib.redirect_stderr(io.StringIO()):
            parts = tf.split_volumes(tf.read_names(CACHE), cfg)
            return parts, tf.build_pools(cfg, parts, mmap)

    def test_ram_and_mmap_return_identical_batches(self):
        rng = np.random.RandomState(0)
        for context in (False, True):
            _, (tr_r, va_r, lv_r, tm_r) = self.pools(False, context)
            _, (tr_m, va_m, lv_m, tm_m) = self.pools(True, context)
            self.assertEqual((tr_r.N, tr_r.n_syn), (tr_m.N, tr_m.n_syn))
            self.assertTrue(tr_r.n_syn > 0)
            np.testing.assert_array_equal(lv_r, lv_m)
            np.testing.assert_array_equal(tm_r, tm_m)
            for _ in range(5):  # random draws incl. synthetic indices, duplicates, unsorted
                idx = rng.randint(0, tr_r.N, 7)
                for a, b in zip(tr_r.get(idx), tr_m.get(idx)):
                    self.assertTrue(torch.equal(a, b))
            idx = np.arange(tr_r.N - 5, tr_r.N)  # last (synthetic) items
            for a, b in zip(tr_r.get(idx), tr_m.get(idx)):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(all(torch.equal(a, b) for (a, _), (b, _) in zip(tf.iter_batches(va_r, np.arange(va_r.N), 5), tf.iter_batches(va_m, np.arange(va_m.N), 5))))
            shape = (7, 3 if context else 1, 32, 32)
            self.assertEqual(tuple(tr_r.get(idx := rng.randint(0, tr_r.N, 7))[0].shape), shape)

    def test_prefetch_thread_preserves_order_and_propagates_errors(self):
        _, (tr, _, _, _) = self.pools(True, True)
        order = np.random.RandomState(1).randint(0, tr.N, 37)
        plain = list(tf.iter_batches(tr, order, 4))
        thread = list(tf.prefetch_iter(tf.iter_batches(tr, order, 4)))
        self.assertEqual(len(plain), len(thread))
        self.assertTrue(all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(plain, thread)))

        def boom():
            yield 1
            raise KeyError("gather failed")
        with self.assertRaises(KeyError):
            list(tf.prefetch_iter(boom()))

    def test_context_gather_clamps_at_volume_borders_and_slab_is_pseudo_volume(self):
        parts, (tr, va, _, _) = self.pools(False, True)
        self.assertEqual(tr.n_volumes, len(parts["train"]) + 3)  # 3 slabs = 3 extra pseudo-volumes
        n_slices, nr = 12, tr.n_real
        for k in range(len(parts["train"])):  # real volumes: 12 slices each
            first, last = k * n_slices, (k + 1) * n_slices - 1
            self.assertEqual((tr.prev[first], tr.nxt[first]), (first, first + 1))
            self.assertEqual((tr.prev[last], tr.nxt[last]), (last - 1, last))
            self.assertEqual((tr.prev[first + 5], tr.nxt[first + 5]), (first + 4, first + 6))
        for s in range(3):  # slabs: 6 slices each; never a neighbour across a slab boundary or into the real part
            first, last = nr + s * 6, nr + (s + 1) * 6 - 1
            self.assertEqual((tr.prev[first], tr.nxt[first]), (first, first + 1))
            self.assertEqual((tr.prev[last], tr.nxt[last]), (last - 1, last))
        self.assertEqual(tr.prev[nr], nr)  # first slab slice does NOT see the last real slice
        # content: channel order k-1, k, k+1 and a clamped border
        x, y = tr.get(np.array([0, 5, nr, nr + 6]))
        X = tr.X
        self.assertTrue(torch.equal(x[0, 0], X[0]) and torch.equal(x[0, 1], X[0]) and torch.equal(x[0, 2], X[1]))
        self.assertTrue(torch.equal(x[1, 0], X[4]) and torch.equal(x[1, 1], X[5]) and torch.equal(x[1, 2], X[6]))
        self.assertTrue(torch.equal(x[2, 0], X[nr]) and torch.equal(x[3, 0], X[nr + 6]))  # slab borders clamp to themselves
        self.assertTrue(torch.equal(y[1], tr.Y[5]))

    def test_synthetic_only_in_train_pool(self):
        parts, (tr, va, liver, tumor) = self.pools(False, False)
        self.assertEqual(va.n_syn, 0)
        self.assertTrue(tr.n_syn == 18 and tr.N == tr.n_real + 18)
        vol_index = np.load(os.path.join(CACHE, "vol_index.npy"))
        self.assertTrue(set(np.unique(vol_index[va.rows])) == set(parts["val"].tolist()))
        self.assertFalse(set(np.unique(vol_index[tr.rows])) & set(parts["val"].tolist()) | set(np.unique(vol_index[tr.rows])) & set(parts["test"].tolist()))
        self.assertEqual(len(liver), tr.N)
        self.assertTrue(tumor[tr.n_real:].sum() == 9 and liver[tr.n_real:].all())  # every second slab slice has tumor
        _, (tr2, _, _, _) = self.pools(False, False, max_slabs=2)
        self.assertEqual(tr2.n_syn, 12)  # first 2 slabs
        _, (tr0, _, l0, _) = self.pools(False, False, aug=False)
        self.assertEqual((tr0.n_syn, len(l0)), (0, tr0.n_real))

    def test_bad_slab_files_rejected(self):
        bad = os.path.join(TMP, "bad_slabs.npz")
        np.savez(bad, images=np.zeros((4, 16, 16), np.uint8), masks=np.zeros((4, 16, 16), np.uint8), slab_id=np.arange(4))
        with self.assertRaises(AssertionError):
            tf.load_aug_npz(bad, 0, (32, 32))
        np.savez(bad, images=np.zeros((4, 32, 32), np.uint8), masks=np.full((4, 32, 32), 3, np.uint8), slab_id=np.arange(4))
        with self.assertRaises(AssertionError):
            tf.load_aug_npz(bad, 0, (32, 32))

    def test_slice_flags_cache_roundtrip(self):
        ff = os.path.join(TMP, "flags", "f.npz")
        masks = np.load(os.path.join(CACHE, "masks.npy"), mmap_mode="r")
        a = tf.cache_flags(masks, CACHE, ff)
        self.assertTrue(os.path.exists(ff))
        b = tf.cache_flags(masks, CACHE, ff)  # second call reads the file
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], (np.asarray(masks) == 2).reshape(len(masks), -1).any(1))


class TestAugment(unittest.TestCase):
    def test_context_channels_get_identical_affine(self):
        torch.manual_seed(0)
        base = torch.rand(2, 1, 32, 32)
        x3 = base.repeat(1, 3, 1, 1)
        y = torch.zeros(2, 32, 32, dtype=torch.long)
        torch.manual_seed(5)
        xa, _ = tf.gpu_augment(x3, y, True)
        self.assertTrue(torch.equal(xa[:, 0], xa[:, 1]) and torch.equal(xa[:, 1], xa[:, 2]))
        torch.manual_seed(5)
        x1, _ = tf.gpu_augment(base, y, True)
        self.assertTrue(torch.allclose(xa[:, :1], x1))

    def test_model_default_and_context_input(self):
        self.assertEqual(tf.AIM_UNet().enc1.conv[0].in_channels, 1)
        self.assertEqual(tf.AIM_UNet(3).enc1.conv[0].in_channels, 3)
        m = tf.AIM_UNet(3).eval()
        self.assertEqual(tuple(m(torch.zeros(1, 3, 32, 32)).shape), (1, 3, 32, 32))


class TestTrainingRuns(unittest.TestCase):
    def test_outputs_and_history_schema(self):
        res, code, _, err = run_train("outputs", LITS_EPOCHS=2, LITS_VAL_N=2, LITS_SELECT="tumor3d", LITS_SEED=3)
        self.assertEqual(code, 0, err[-2000:])
        for f in ("train_AIMUNet.log", "split_used.json", "history.json", "best.pt", "last.pt", "loss_curves.png", "done.json"):
            self.assertTrue(os.path.exists(os.path.join(res, f)), f)
        self.assertFalse([f for f in os.listdir(res) if f.endswith(".tmp")])
        h, d = jl(os.path.join(res, "history.json")), jl(os.path.join(res, "done.json"))
        self.assertEqual([r["epoch"] for r in h], [1, 2])
        for k in ("train_loss", "val_loss", "val_tumor3d", "val_liver3d", "legacy_batch_dice", "samples_per_s", "seconds", "lr"):
            self.assertIn(k, h[0])
        self.assertEqual(set(h[0]["legacy_batch_dice"]), {"background", "liver", "tumor"})
        self.assertEqual(set(d) >= {"epochs_run", "best_epoch", "best_val_tumor3d", "stopped"}, True)
        self.assertEqual((d["epochs_run"], d["stopped"]), (2, "max"))
        best = max(h, key=lambda r: r["val_tumor3d"])  # select=tumor3d: first maximum (ties keep the earlier epoch)
        self.assertEqual(d["best_epoch"], min(r["epoch"] for r in h if r["val_tumor3d"] == best["val_tumor3d"]))
        ck = torch.load(os.path.join(res, "best.pt"), weights_only=True)
        self.assertEqual(ck["config"]["in_channels"], 1)
        self.assertEqual(ck["config"]["loss"], "ce_dice")
        self.assertEqual(ck["config"]["window"], [-100.0, 250.0])
        self.assertIn(ck["config"]["amp"], ("off", "bf16", "fp16"))
        s = jl(os.path.join(res, "split_used.json"))
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (16, 2, 2))
        self.assertEqual(len(h[0]["val_tumor3d_per_volume"]), 2)

    def test_cosine_schedule_values(self):
        res, code, _, err = run_train("cosine", LITS_EPOCHS=4, LITS_LR_SCHED="cosine", LITS_EPOCH_LEN=8, LITS_LR=1e-3, LITS_PATIENCE=99)
        self.assertEqual(code, 0, err[-1500:])
        lrs = [r["lr"] for r in jl(os.path.join(res, "history.json"))]
        expect = [0.01e-3 + (1e-3 - 0.01e-3) * (1 + np.cos(np.pi * e / 3)) / 2 for e in range(4)]
        np.testing.assert_allclose(lrs, expect, rtol=1e-9)
        self.assertAlmostEqual(lrs[0], 1e-3)
        self.assertAlmostEqual(lrs[-1], 1e-5)  # 1% of lr on the last planned epoch

    def test_const_schedule_is_flat(self):
        res, code, _, _ = run_train("const", LITS_EPOCHS=2, LITS_EPOCH_LEN=8, LITS_LR=3e-4)
        self.assertEqual({r["lr"] for r in jl(os.path.join(res, "history.json"))}, {3e-4})

    def test_tversky_loss_context_and_selection_run(self):
        res, code, _, err = run_train("tv_ctx", LITS_EPOCHS=2, LITS_LOSS="ce_dice_tversky", LITS_CONTEXT=1, LITS_VAL_N=2,
                                      LITS_SELECT="tumor3d", LITS_EPOCH_LEN=16, LITS_AMP="off")
        self.assertEqual(code, 0, err[-1500:])
        ck = torch.load(os.path.join(res, "best.pt"), weights_only=True)
        self.assertEqual((ck["config"]["loss"], ck["config"]["context"], ck["config"]["in_channels"]), ("ce_dice_tversky", 1, 3))
        self.assertEqual(ck["model"]["enc1.conv.0.weight"].shape[1], 3)
        # eval reads in_channels from the checkpoint and runs the context path on the test part
        out = os.path.join(TMP, "eval_ctx")
        code, err = run_eval(os.path.join(res, "best.pt"), out)
        self.assertIsNone(code, err[-1500:])
        e = jl(os.path.join(out, "eval.json"))
        self.assertEqual((e["part"], e["n_evaluated"], e["model_config"]["in_channels"]), ("test", 2, 3))

    def test_mmap_and_ram_runs_are_identical(self):
        a, ca, _, ea = run_train("ram", LITS_EPOCHS=2, LITS_SEED=5, LITS_MMAP=0, LITS_EPOCH_LEN=24)
        b, cb, _, eb = run_train("mmap", LITS_EPOCHS=2, LITS_SEED=5, LITS_MMAP=1, LITS_EPOCH_LEN=24)
        self.assertEqual((ca, cb), (0, 0), ea[-800:] + eb[-800:])
        ha, hb = jl(os.path.join(a, "history.json")), jl(os.path.join(b, "history.json"))
        self.assertEqual([{k: r[k] for k in STABLE} for r in ha], [{k: r[k] for k in STABLE} for r in hb])
        self.assertTrue(sd_equal(torch.load(os.path.join(a, "last.pt"), weights_only=True)["model"],
                                 torch.load(os.path.join(b, "last.pt"), weights_only=True)["model"]))
        self.assertIn("mmap batches", rd(os.path.join(b, "train_AIMUNet.log")))

    def test_synthetic_run_logs_share_and_leaves_val_alone(self):
        slabs = make_slabs(os.path.join(TMP, "slabs_run.npz"))
        r0, c0, _, e0 = run_train("syn0", LITS_EPOCHS=1, LITS_SEED=2, LITS_VAL_N=2)
        r1, c1, _, e1 = run_train("syn1", LITS_EPOCHS=1, LITS_SEED=2, LITS_VAL_N=2, LITS_AUG_NPZ=slabs)
        r2, c2, _, e2 = run_train("syn2", LITS_EPOCHS=1, LITS_SEED=2, LITS_VAL_N=2, LITS_AUG_NPZ=slabs, LITS_AUG_MAX_SLABS=1, LITS_EPOCH_LEN=48)
        self.assertEqual((c0, c1, c2), (0, 0, 0), e0[-500:] + e1[-500:] + e2[-500:])
        h0, h1 = jl(os.path.join(r0, "history.json"))[0], jl(os.path.join(r1, "history.json"))[0]
        self.assertEqual(h0["synthetic_share"], 0.0)
        self.assertGreater(h1["synthetic_share"], 0.0)
        self.assertEqual(jl(os.path.join(r0, "split_used.json")), jl(os.path.join(r1, "split_used.json")))
        self.assertEqual(len(h0["val_tumor3d_per_volume"]), len(h1["val_tumor3d_per_volume"]))
        log = rd(os.path.join(r1, "train_AIMUNet.log"))
        self.assertIn("synthetic share of the drawn training samples", log)
        self.assertIn("18 slices in 3 slabs", log)
        self.assertIn("Synthetic pool: 6 slices in 1 slabs", rd(os.path.join(r2, "train_AIMUNet.log")))

    def test_nonfinite_loss_aborts_with_exit_3(self):
        nan_loss = lambda out, y: out.sum() * float("nan")
        with mock.patch.object(tf, "make_loss", return_value=nan_loss):
            res, code, out, err = run_train("nan", LITS_EPOCHS=2, LITS_EPOCH_LEN=8)
        self.assertEqual(code, 3)
        self.assertIn("NONFINITE_LOSS", out)
        self.assertIn("NONFINITE_LOSS", rd(os.path.join(res, "train_AIMUNet.log")))
        self.assertFalse(os.path.exists(os.path.join(res, "done.json")))

    def test_channels_last_and_prefetch_paths_run_on_cpu(self):
        res, code, _, err = run_train("cl", LITS_EPOCHS=1, LITS_CHANNELS_LAST=1, LITS_MMAP=1, LITS_EPOCH_LEN=16, LITS_CONTEXT=1)
        self.assertEqual(code, 0, err[-1500:])
        log = rd(os.path.join(res, "train_AIMUNet.log"))
        self.assertIn("channels_last True", log)
        self.assertIn("prefetch thread", log)
        sd = torch.load(os.path.join(res, "last.pt"), weights_only=True)["model"]
        m = tf.AIM_UNet(3)
        m.load_state_dict(sd)  # a channels_last-trained checkpoint loads into a normal model

    def test_probe_only_prints_one_line(self):
        res, code, out, err = run_train("probe", LITS_PROBE_ONLY=1, LITS_PROBE_STEPS=8, LITS_VAL_N=2)
        self.assertEqual(code, 0, err[-1500:])
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(lines[0].startswith("PROBE "))
        p = json.loads(lines[0][6:])
        for k in ("train_slices_per_s", "val_slices_per_s", "samples_per_epoch", "val_slices", "est_epoch_min"):
            self.assertIn(k, p)
        self.assertGreater(p["train_slices_per_s"], 0)
        self.assertEqual((p["samples_per_epoch"], p["val_slices"]), (48, 24))
        self.assertAlmostEqual(p["est_epoch_min"], (48 / p["train_slices_per_s"] + 24 / p["val_slices_per_s"]) / 60, delta=0.01)
        self.assertFalse(any(os.path.exists(os.path.join(res, f)) for f in ("best.pt", "last.pt", "done.json", "history.json")))

    def test_probe_in_subprocess_stdout_is_only_the_probe_line(self):
        import subprocess
        env = {**{k: v for k, v in os.environ.items() if not k.startswith("LITS_")}, "LITS_CACHE_DIR": CACHE, "LITS_PER_VOLUME": PV,
               "LITS_RESULTS_DIR": os.path.join(TMP, "probe_sp"), "LITS_DEVICE": "cpu", "LITS_PROBE_ONLY": "1", "LITS_PROBE_STEPS": "8",
               "LITS_VAL_N": "2", "LITS_BATCH_SIZE": "4"}
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "train_fast.py")], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertTrue(r.stdout.startswith("PROBE {"))


class TestResume(unittest.TestCase):
    ENV = dict(LITS_EPOCHS=4, LITS_SEED=11, LITS_LR_SCHED="cosine", LITS_SELECT="tumor3d", LITS_VAL_N=2, LITS_PATIENCE=2,
               LITS_EPOCH_LEN=48, LITS_LR=5e-4)

    def stable(self, res):
        return [{k: r[k] for k in STABLE} for r in jl(os.path.join(res, "history.json"))]

    def test_straight_equals_stop_and_resume(self):
        # min_delta=1e9: nothing after epoch 1 counts as an improvement, so patience 2 stops after epoch 3 ('early')
        # in a straight run; if the early-stopping counter were lost on resume, the stop would move to epoch 4.
        es = functools.partial(tf.EarlyStopping, min_delta=1e9)
        with mock.patch.object(tf, "EarlyStopping", es):
            a, ca, _, ea = run_train("straight", **self.ENV)
            b, cb, outb, eb = run_train("part1", LITS_STOP_AFTER=1, **self.ENV)
            self.assertEqual(cb, 0)
            self.assertIn("PAUSED epoch=1", outb)
            self.assertFalse(os.path.exists(os.path.join(b, "done.json")))
            self.assertEqual(len(jl(os.path.join(b, "history.json"))), 1)
            b, cb2, _, eb2 = run_train("part1", LITS_RESUME=1, **self.ENV)
        self.assertEqual((ca, cb2), (0, 0), ea[-800:] + eb2[-800:])
        ha, hb = self.stable(a), self.stable(b)
        self.assertEqual(len(ha), 3)  # early stop at 3 of 4
        self.assertEqual(ha, hb)  # losses, metrics, per-volume dice, lr: bit-identical
        da, db = jl(os.path.join(a, "done.json")), jl(os.path.join(b, "done.json"))
        self.assertEqual(da, db)
        self.assertEqual((da["epochs_run"], da["best_epoch"], da["stopped"]), (3, 1, "early"))
        la, lb = (torch.load(os.path.join(d, "last.pt"), weights_only=True) for d in (a, b))
        self.assertTrue(sd_equal(la["model"], lb["model"]))
        self.assertEqual(la["epoch"], lb["epoch"])
        self.assertEqual(la["earlystop"], lb["earlystop"])
        self.assertEqual(la["earlystop"]["counter"], 2)
        self.assertEqual(la["sched"], lb["sched"])  # cosine state (last_epoch, last_lr, ...) continued, not restarted
        self.assertEqual(la["best"], lb["best"])
        self.assertTrue(sd_equal(torch.load(os.path.join(a, "best.pt"), weights_only=True)["model"],
                                 torch.load(os.path.join(b, "best.pt"), weights_only=True)["model"]))
        log = rd(os.path.join(b, "train_AIMUNet.log"))
        self.assertIn("Resumed from", log)
        self.assertIn("Epoch 1/4", log)  # appended, not truncated
        # resuming a finished run is a no-op
        _, c3, _, e3 = run_train("part1", LITS_RESUME=1, **self.ENV)
        self.assertEqual(c3, 0)
        self.assertEqual(self.stable(b), hb)

    def test_resume_without_early_stop_matches_straight_run(self):
        env = {**self.ENV, "LITS_EPOCHS": 3, "LITS_PATIENCE": 99}
        a, ca, _, _ = run_train("s2", **env)
        b, cb, _, _ = run_train("r2", LITS_STOP_AFTER=2, **env)
        b, cb2, _, _ = run_train("r2", LITS_RESUME=1, **env)
        self.assertEqual((ca, cb, cb2), (0, 0, 0))
        self.assertEqual(self.stable(a), self.stable(b))
        np.testing.assert_allclose([r["lr"] for r in jl(os.path.join(b, "history.json"))], [5e-4, 5e-4 * 0.505, 5e-6], rtol=1e-9)

    def test_finalizes_a_finished_early_stop_without_training(self):
        es = functools.partial(tf.EarlyStopping, min_delta=1e9)
        with mock.patch.object(tf, "EarlyStopping", es):
            a, c, _, _ = run_train("fin", **self.ENV)
            h = self.stable(a)
            os.remove(os.path.join(a, "done.json"))  # crash between the last checkpoint and done.json
            _, c2, _, e2 = run_train("fin", LITS_RESUME=1, **self.ENV)
        self.assertEqual((c, c2), (0, 0))
        self.assertEqual(self.stable(a), h)
        self.assertEqual(jl(os.path.join(a, "done.json"))["stopped"], "early")

    def test_resume_refused_on_config_or_split_change(self):
        a, c, _, _ = run_train("refuse", LITS_STOP_AFTER=1, **self.ENV)
        _, c2, _, e2 = run_train("refuse", LITS_RESUME=1, **{**self.ENV, "LITS_LOSS": "ce_dice_tversky"})
        self.assertEqual(c2, 4, e2[-500:])
        _, c3, _, e3 = run_train("refuse", LITS_RESUME=1, **{**self.ENV, "LITS_VAL_N": 1})
        self.assertEqual(c3, 4, e3[-500:])
        self.assertFalse(os.path.exists(os.path.join(a, "done.json")))

    def test_resume_without_checkpoint_starts_fresh(self):
        res, c, _, _ = run_train("fresh", LITS_RESUME=1, **{**self.ENV, "LITS_EPOCHS": 1})
        self.assertEqual(c, 0)
        self.assertEqual(jl(os.path.join(res, "done.json"))["epochs_run"], 1)

    def test_earlystopping_state_roundtrip(self):
        for mode, seq in (("min", [1.0, 0.9, 0.95, 0.96]), ("max", [0.1, 0.2, 0.15, 0.14])):
            a = tf.EarlyStopping(patience=2, mode=mode)
            for v in seq[:3]:
                a.step(v)
            b = tf.EarlyStopping(patience=2, mode=mode)
            b.load_state_dict(a.state_dict())
            self.assertEqual((a.step(seq[3]), a.counter), (b.step(seq[3]), b.counter))
            self.assertTrue(a.step(seq[3]) and b.step(seq[3]))


@unittest.skipUnless(os.path.exists(os.path.join(SNAP_SEG, "train_fast.py")), "no snapshot of the previous scripts")
class TestRegressionAgainstSnapshot(unittest.TestCase):
    """Default behaviour must equal the previous scripts: same tensors, same init, same augmentation, same 1-epoch run."""

    @classmethod
    def setUpClass(cls):
        cls.res = os.path.join(TMP, "snap_res")
        with mock.patch.object(torch.backends.mps, "is_available", return_value=False), contextlib.redirect_stderr(io.StringIO()):
            cls.snap = load_module("snap_train_fast", os.path.join(SNAP_SEG, "train_fast.py"),
                                   dict(LITS_CACHE_DIR=CACHE, LITS_RESULTS_DIR=cls.res, LITS_SUBSET_N=0))

    def test_split_and_tensors_md5(self):
        with env_ctx(dict(LITS_CACHE_DIR=CACHE, LITS_RESULTS_DIR=os.path.join(TMP, "new_split"))):
            cfg = tf.Cfg()
        os.makedirs(cfg.results_dir, exist_ok=True)
        with contextlib.redirect_stderr(io.StringIO()):
            (Xtr, Ytr), (Xte, Yte) = self.snap.load_split(0)
            parts = tf.split_volumes(tf.read_names(CACHE), cfg)
            tr, va, _, _ = tf.build_pools(cfg, parts, False)
        self.assertEqual(rd(os.path.join(self.res, "split_used.json")), rd(os.path.join(cfg.results_dir, "split_used.json")))
        self.assertEqual((md5(Xtr), md5(Ytr), md5(Xte), md5(Yte)), (md5(tr.X), md5(tr.Y), md5(va.X), md5(va.Y)))
        # flags / weights
        has_t = (Ytr == 2).flatten(1).any(1)
        liver, tumor = tf.chunked_flags(lambda a, b: tr.Y[a:b].numpy(), tr.N)
        np.testing.assert_array_equal(has_t.numpy(), tumor)
        np.testing.assert_array_equal((Ytr >= 1).flatten(1).any(1).numpy(), liver)

    def test_model_init_state_and_old_checkpoint_loading(self):
        torch.manual_seed(7)
        a = self.snap.AIM_UNet()
        torch.manual_seed(7)
        b = tf.AIM_UNet()
        self.assertTrue(sd_equal(a.state_dict(), b.state_dict()))
        b.load_state_dict(a.state_dict())  # raw state_dict of the old format loads strict
        x = torch.rand(2, 1, 32, 32)
        a.eval(); b.eval()
        self.assertTrue(torch.equal(a(x), b(x)))

    def test_augmentation_same_seed_same_tensors(self):
        x, y = torch.rand(4, 1, 32, 32), torch.randint(0, 3, (4, 32, 32))
        for flip in (True, False):
            self.snap.FLIP_AUG = flip
            torch.manual_seed(1)
            xa, ya = self.snap.gpu_augment(x, y)
            torch.manual_seed(1)
            xb, yb = tf.gpu_augment(x, y, flip)
            self.assertTrue(torch.equal(xa, xb) and torch.equal(ya, yb))
        self.snap.FLIP_AUG = True

    def test_seeded_one_epoch_run_matches_snapshot(self):
        res = os.path.join(TMP, "snap_epoch")
        os.makedirs(res, exist_ok=True)
        env = dict(LITS_CACHE_DIR=CACHE, LITS_SEED=9, LITS_RESULTS_DIR=res, LITS_EMPTY_KEEP=0.5, LITS_EPOCHS=1)
        with mock.patch.object(torch.backends.mps, "is_available", return_value=False), contextlib.redirect_stderr(io.StringIO()):
            snap = load_module("snap_train_fast2", os.path.join(SNAP_SEG, "train_fast.py"), env)
            with env_ctx(env), self.assertLogs("snap_train_fast2", "INFO") as cm:  # the snapshot's own logger
                snap.train_model(epochs=1, batch_size=4, lr=1e-4, patience=7)
        old = os.path.join(res, "aim_unet_final.pt")
        old_log = "\n".join(cm.output)
        res2, code, _, err = run_train("snap_epoch_new", LITS_SEED=9, LITS_EMPTY_KEEP=0.5, LITS_EPOCHS=1, LITS_EPOCH_LEN=0)
        self.assertEqual(code, 0, err[-1500:])
        new = torch.load(os.path.join(res2, "last.pt"), weights_only=True)["model"]
        self.assertTrue(sd_equal(torch.load(old, weights_only=True), new), "weights after the seeded epoch differ from the snapshot run")
        import re
        grab = lambda log, pat: re.findall(pat, log)
        for pat in (r"Train Loss: ([\d.]+)", r"Val Loss:\s+([\d.]+)", r"Background: ([\d.]+), Liver: ([\d.]+), Tumor: ([\d.]+)", r"Slices/epoch: (\d+)"):
            self.assertEqual(grab(old_log, pat), grab(rd(os.path.join(res2, "train_AIMUNet.log")), pat), pat)
        self.assertTrue(os.path.exists(os.path.join(res2, "best.pt")))

    def test_old_eval_and_new_eval_agree_on_a_raw_state_dict(self):
        """An old-format checkpoint (raw state_dict) through the previous evaluator and the new one: same metrics."""
        torch.manual_seed(4)
        m = self.snap.AIM_UNet()
        # make the net predict something non-trivial on the tiny cache
        with torch.no_grad():
            m.out.bias.copy_(torch.tensor([0.0, 0.3, 0.1]))
        raw = os.path.join(TMP, "old_ckpt", "best_epoch_001.pt")
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        torch.save(m.state_dict(), raw)
        ev_old = load_module("snap_eval_fast", os.path.join(SNAP_SEG, "eval_fast.py"), dict(LITS_CACHE_DIR=CACHE, LITS_PER_VOLUME=PV))
        o_old = os.path.join(TMP, "ev_old")
        with env_ctx(dict(LITS_CKPT=raw, LITS_EVAL_OUT=o_old, LITS_CACHE_DIR=CACHE)), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(ev_old.torch.backends.mps, "is_available", return_value=False):
            ev_old.device = torch.device("cpu")
            ev_old.OUT_DIR = o_old
            ev_old.main()
        o_new = os.path.join(TMP, "ev_new")
        code, err = run_eval(raw, o_new, LITS_EVAL_PART="heldout", LITS_AMP="off")
        self.assertIsNone(code, err[-1500:])
        a, b = jl(os.path.join(o_old, "eval.json")), jl(os.path.join(o_new, "eval.json"))
        self.assertEqual(a["aggregate"], b["aggregate"])
        self.assertEqual(a["lesions"], b["lesions"])
        self.assertTrue(set(a) - {"eval_part"} <= set(b), set(a) - set(b))  # existing keys kept
        old_cols = list(rows_of(os.path.join(o_old, "per_patient.csv"))[0])
        new_cols = list(rows_of(os.path.join(o_new, "per_patient.csv"))[0])
        self.assertEqual(new_cols[:len(old_cols)], old_cols)
        self.assertEqual(new_cols[len(old_cols):], ["legacy_dice_background", "legacy_dice_liver_label1", "legacy_dice_tumor"])


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, code, _, err = run_train("eval_src", LITS_EPOCHS=1, LITS_VAL_N=2, LITS_EPOCH_LEN=24, LITS_SEED=1)
        assert code == 0, err[-1500:]
        cls.ckpt = os.path.join(cls.res, "best.pt")

    def ev(self, name, **env):
        out = os.path.join(TMP, name)
        code, err = run_eval(self.ckpt, out, **env)
        return out, code, err

    def test_default_part_is_test_and_outputs(self):
        out, code, err = self.ev("ev_default")
        self.assertIsNone(code, err[-1500:])
        e = jl(os.path.join(out, "eval.json"))
        split = jl(os.path.join(self.res, "split_used.json"))
        self.assertEqual((e["part"], e["volumes"]), ("test", sorted(split["test"])))
        self.assertEqual(set(e["volumes"]) & (set(split["train"]) | set(split["val"])), set())
        for f in ("eval.json", "eval.md", "per_patient.csv"):
            self.assertTrue(os.path.exists(os.path.join(out, f)))
        self.assertEqual(set(e["legacy_slice_mean_dice"]), {"background", "liver_label1", "tumor", "n_slices"})
        self.assertIn("TEST part", rd(os.path.join(out, "eval.md")))
        self.assertIn("Legacy per-slice Dice", rd(os.path.join(out, "eval.md")))
        rows = rows_of(os.path.join(out, "per_patient.csv"))
        self.assertEqual(len(rows), 2)
        for k in ("legacy_dice_background", "legacy_dice_liver_label1", "legacy_dice_tumor"):
            self.assertIn(k, rows[0])

    def test_val_part_and_max_volumes(self):
        out, code, err = self.ev("ev_val", LITS_EVAL_PART="val")
        self.assertIsNone(code, err[-800:])
        e = jl(os.path.join(out, "eval.json"))
        self.assertEqual((e["part"], sorted(e["volumes"])), ("val", sorted(jl(os.path.join(self.res, "split_used.json"))["val"])))
        out, code, err = self.ev("ev_one", LITS_EVAL_MAX_VOLUMES=1)
        self.assertEqual(jl(os.path.join(out, "eval.json"))["n_evaluated"], 1)

    def test_heldout_part_warns_about_validation_volumes(self):
        with self.assertLogs("eval_fast", "WARNING") as cm:
            out, code, err = self.ev("ev_heldout", LITS_EVAL_PART="heldout")
        self.assertIsNone(code, err[-800:])
        self.assertEqual(jl(os.path.join(out, "eval.json"))["n_evaluated"], 4)
        self.assertIn("validation list", "\n".join(cm.output))

    def test_leakage_guard(self):
        d = os.path.join(TMP, "leak_ckpt")
        os.makedirs(d, exist_ok=True)
        shutil.copy(self.ckpt, d)
        s = jl(os.path.join(self.res, "split_used.json"))
        # a train volume in the evaluated test list
        wj(os.path.join(d, "split_used.json"), {**s, "test": [s["train"][0]] + s["test"][:1]})
        code, err = run_eval(os.path.join(d, "best.pt"), os.path.join(TMP, "leak_out"))
        self.assertIn("Leakage guard", str(code))
        self.assertIn(s["train"][0], str(code))
        # a validation volume in the test list: selection leak
        wj(os.path.join(d, "split_used.json"), {**s, "test": s["val"][:1] + s["test"][:1]})
        code, err = run_eval(os.path.join(d, "best.pt"), os.path.join(TMP, "leak_out"))
        self.assertIn("VALIDATION list", str(code))
        self.assertFalse(os.path.exists(os.path.join(TMP, "leak_out", "eval.json")))
        # no test list: asking for test is an error
        wj(os.path.join(d, "split_used.json"), {"train": s["train"], "val": s["val"]})
        code, err = run_eval(os.path.join(d, "best.pt"), os.path.join(TMP, "leak_out"), LITS_EVAL_PART="test")
        self.assertIn("non-empty 'test' list", str(code))

    def test_legacy_dice_equals_original_test_script_function(self):
        """legacy_slice_dice == the original test script's dice_per_class evaluated slice by slice (batch size 1)."""
        src = rd(os.path.join(ROOT, "Image Segmentation", "test_aimunet_orig_65.py")) if os.path.exists(os.path.join(ROOT, "Image Segmentation", "test_aimunet_orig_65.py")) else None
        if src is None:
            self.skipTest("original test script not present")
        import ast
        fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "dice_per_class"]
        ns = {"torch": torch}
        exec(compile(ast.Module(fn, []), "orig", "exec"), ns)
        rng = np.random.RandomState(0)
        gt = rng.randint(0, 3, (9, 16, 16))
        gt[2:5] = 0                      # empty-vs-empty slices (tumor 1.0 in the legacy metric)
        pred = np.where(rng.rand(*gt.shape) < 0.7, gt, rng.randint(0, 3, gt.shape))
        pred[3:5] = 0
        ours = ev.legacy_slice_dice(pred, gt)
        for i in range(len(gt)):
            logits = torch.nn.functional.one_hot(torch.from_numpy(pred[i:i + 1]).long(), 3).permute(0, 3, 1, 2).float()
            ref = ns["dice_per_class"](logits, torch.from_numpy(gt[i:i + 1]).long())
            for c, name in enumerate(ev.LEGACY):
                self.assertAlmostEqual(float(ours[name][i]), ref[c], places=5)
        self.assertAlmostEqual(float(ours["tumor"][3]), 1.0, places=5)  # empty vs empty = 1.0 (why it is inflated)

    def test_on_device_3d_dice_of_training_equals_host_dice_of_eval(self):
        """The per-epoch validation 3D Dice (on-device sums) vs eval_fast's host numpy Dice of the same checkpoint
        (best.pt = the only epoch) on the same validation volumes."""
        out, code, err = self.ev("ev_val_cmp", LITS_EVAL_PART="val", LITS_AMP="off")
        self.assertIsNone(code, err[-800:])
        rows = jl(os.path.join(out, "eval.json"))["per_patient"]
        h = jl(os.path.join(self.res, "history.json"))[0]
        self.assertEqual(len(rows), len(h["val_tumor3d_per_volume"]))
        for r, t, l in zip(rows, h["val_tumor3d_per_volume"], h["val_liver3d_per_volume"]):
            self.assertAlmostEqual(r["liver_dice"], l, places=5)
            if r["tumor_dice"] is None:  # no GT tumor and (maybe) no prediction
                self.assertTrue(t is None or r["n_lesions"] == 0)
            else:
                self.assertAlmostEqual(r["tumor_dice"], t, places=5)
        t_eval = [r["tumor_dice"] for r in rows if r["n_lesions"]]
        self.assertAlmostEqual(h["val_tumor3d"], float(np.mean(t_eval)), places=5)

    def test_legacy_pooled_mean_in_eval_json(self):
        out, code, _ = self.ev("ev_legacy")
        e = jl(os.path.join(out, "eval.json"))
        n = sum(r["n_slices"] for r in e["per_patient"])
        for k in ev.LEGACY:
            self.assertAlmostEqual(e["legacy_slice_mean_dice"][k], sum(r[f"legacy_dice_{k}"] * r["n_slices"] for r in e["per_patient"]) / n, places=9)
        self.assertEqual(e["legacy_slice_mean_dice"]["n_slices"], n)

    def test_context_predict_matches_manual_stacking(self):
        torch.manual_seed(0)
        m = tf.AIM_UNet(3).eval()
        imgs = np.random.RandomState(0).randint(0, 255, (7, 32, 32)).astype(np.uint8)
        pred = ev.predict(m, imgs, lambda: torch.autocast("cpu", enabled=False), 3, torch.device("cpu"), True)
        pad = np.concatenate([imgs[:1], imgs, imgs[-1:]])  # clamped neighbours
        x = torch.from_numpy(np.stack([pad[:-2], pad[1:-1], pad[2:]], 1)).float() / 255.
        with torch.no_grad():
            ref = torch.cat([m(x[i:i + 3]).argmax(1) for i in range(0, 7, 3)]).to(torch.uint8).numpy()
        np.testing.assert_array_equal(pred, ref)


@unittest.skipUnless(os.path.exists(os.path.join(REAL_CACHE, "images.npy")) and os.environ.get("LITS_TEST_SKIP_REAL") != "1",
                     "no real cache_v2 (or LITS_TEST_SKIP_REAL=1)")
class TestRealData(unittest.TestCase):
    def test_three_real_volumes_train_one_epoch_on_default_device_with_mmap(self):
        """LITS_MAX_VOLUMES=3 on the real cache: 2 train volumes / 1 val volume, mmap batches, auto device and AMP."""
        res = os.path.join(TMP, "real3")
        env = dict(LITS_CACHE_DIR=REAL_CACHE, LITS_RESULTS_DIR=res, LITS_MAX_VOLUMES=3, LITS_EPOCHS=1, LITS_EPOCH_LEN=12,
                   LITS_BATCH_SIZE=4, LITS_MMAP=1, LITS_SEED=1, LITS_SELECT="tumor3d", LITS_LR_SCHED="cosine",
                   LITS_SLICE_FLAGS=os.path.join(TMP, "real_flags.npz"))
        out, err = io.StringIO(), io.StringIO()
        with env_ctx(env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tf.main()
        self.assertEqual(code, 0, err.getvalue()[-2000:])
        names = rd(os.path.join(REAL_CACHE, "volumes.txt")).split("\n")[:3]
        s = jl(os.path.join(res, "split_used.json"))
        self.assertEqual((len(s["train"]), len(s["val"])), (2, 1))
        self.assertEqual(sorted(s["train"] + s["val"]), sorted(names))
        h = jl(os.path.join(res, "history.json"))
        self.assertEqual(len(h), 1)
        self.assertTrue(np.isfinite(h[0]["train_loss"]) and np.isfinite(h[0]["val_loss"]))
        ck = torch.load(os.path.join(res, "best.pt"), weights_only=True)
        self.assertEqual(ck["config"]["window"], [-100.0, 250.0])
        self.assertTrue(os.path.exists(os.path.join(TMP, "real_flags.npz")))
        # the data dir must be untouched: no new file next to the cache
        self.assertEqual(sorted(os.listdir(REAL_CACHE)), sorted(["config.json", "images.npy", "masks.npy", "meta.csv", "vol_index.npy", "volumes.txt"]))

    def test_old_baseline_checkpoint_loads_strict_and_matches_snapshot_model(self):
        ck = os.path.join(ROOT, "results", "baseline", "C_v2_window_noflip", "best_epoch_005.pt")
        if not os.path.exists(ck):
            self.skipTest("no baseline checkpoint")
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        m = tf.AIM_UNet()
        m.load_state_dict(sd)  # strict
        x = torch.from_numpy(np.load(os.path.join(REAL_CACHE, "images.npy"), mmap_mode="r")[100:102].astype(np.float32)).unsqueeze(1) / 255.
        m.eval()
        with torch.no_grad():
            out = m(x)
        self.assertEqual(tuple(out.shape), (2, 3, 256, 256))
        self.assertTrue(torch.isfinite(out).all())
        if os.path.exists(os.path.join(SNAP_SEG, "train_fast.py")):
            ns = {"nn": torch.nn, "torch": torch}
            src = rd(os.path.join(SNAP_SEG, "train_fast.py"))
            keep = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name in ("InceptionModule", "ConvBlock", "AIM_UNet")]
            exec(compile(ast.Module(keep, []), "snap", "exec"), ns)
            old = ns["AIM_UNet"]()
            old.load_state_dict(sd)
            old.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(old(x), out))
        # showcase-style loading: exec only the three classes of the NEW source with nn/torch in the namespace
        src = rd(os.path.join(HERE, "train_fast.py"))
        keep = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name in ("InceptionModule", "ConvBlock", "AIM_UNet")]
        ns2 = {"nn": torch.nn, "torch": torch}
        exec(compile(ast.Module(keep, []), "new", "exec"), ns2)
        ns2["AIM_UNet"]().load_state_dict(sd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
