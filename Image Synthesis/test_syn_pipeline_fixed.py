"""Unit / smoke tests for Syn_Img_Pipeline_fixed.py (networks, patch sampling, split, mask gate, host insertion, CLI stages).

    python "Image Synthesis/test_syn_pipeline_fixed.py"

The cache source (Colab: no raw NIfTI) is tested on a synthetic toy cache built in a temp dir (always runs) and, when data/raw and
data/cache_v2 exist, against the real cache (HU reconstruction vs the raw NIfTI, patch geometry, split equality).
Env vars: LITS_TEST_DEVICE=cpu|mps|cuda picks the device of the training-step tests (default cpu);
LITS_TEST_SKIP_E2E=1 skips the (slower) end-to-end subprocess run (it writes results/gan_e2e).
Tests that need real LiTS volumes (data/raw) are skipped when it is absent.
"""
import os
import sys
import csv
import json
import shutil
import tempfile
import unittest
import subprocess
import importlib.util
from unittest import mock
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
spec = importlib.util.spec_from_file_location("syn", os.path.join(HERE, "Syn_Img_Pipeline_fixed.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)
DEV = torch.device(os.environ.get("LITS_TEST_DEVICE", "cpu"))
HAVE_RAW = all(os.path.exists(os.path.join(RAW, f)) for f in ("volume-0.nii", "segmentation-0.nii"))
CACHE = os.path.join(ROOT, "data", "cache_v2")
PERVOL = os.path.join(ROOT, "analysis", "output", "per_volume.csv")
HAVE_CACHE = HAVE_RAW and all(os.path.exists(os.path.join(CACHE, f)) for f in ("images.npy", "config.json", "meta.csv")) and os.path.exists(PERVOL)


def rand_real(b, p, rng):
    x = torch.from_numpy(rng.uniform(-1, 1, (b, 1, p, p, p)).astype(np.float32)).to(DEV)
    m = torch.from_numpy((rng.random((b, 1, p, p, p)) < 0.1).astype(np.float32)).to(DEV)
    return x, m


def unchanged(before, module):
    return [n for n, p in module.named_parameters() if torch.equal(before[n], p.detach().cpu())]


def snap(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters()}


def toy_generated(p=64, radius=(12, 10, 11), base_hu=100.0, tumour_hu=60.0, seed=0):
    """Procedural stand-in for a generated patch (the insertion code does not care where the patch came from)."""
    g = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:p, :p, :p].astype(np.float32) - p / 2
    mask = (zz / radius[0]) ** 2 + (yy / radius[1]) ** 2 + (xx / radius[2]) ** 2 <= 1
    hu = np.where(mask, tumour_hu, base_hu) + g.normal(0, 5, (p, p, p))
    return syn.to_unit(hu.astype(np.float32)), mask


def fake_volume_dir(d, zooms_by_id):
    """Tiny volume-<id>.nii files whose headers carry the given zooms (the split code reads names and headers only)."""
    for i, z in zooms_by_id.items():
        img = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.diag([z[0], z[1], z[2], 1.0]))
        nib.save(img, os.path.join(d, f"volume-{i}.nii"))


class TestNetworks(unittest.TestCase):
    def test_shapes_all_networks(self):
        for p in (64, 48, 32):  # 64 is the default; 32/48 are the documented smaller sizes (multiple of 16)
            b, ld = 2, 16
            z = torch.randn(b, ld, 1, 1, 1)
            real = torch.rand(b, 1, p, p, p) * 2 - 1
            for cls in (syn.DCGAN3DGenerator, syn.WGAN3DGenerator):
                out = cls(ld, p, 8)(z)
                self.assertEqual(tuple(out.shape), (b, 1, p, p, p))
                self.assertTrue(out.min() >= -1 and out.max() <= 1)
            for cls in (syn.DCGAN3DDiscriminator, syn.WGAN3DCritic, syn.Aggregator3DDiscriminator):
                net = cls(8)
                self.assertEqual(tuple(net(real).shape), (b,), cls.__name__)
                self.assertFalse(any(isinstance(m, torch.nn.Sigmoid) for m in net.modules()))
            self.assertEqual(tuple(syn.Aggregator3D()(real, real, real).shape), (b, 1, p, p, p))
            self.assertEqual(tuple(syn.MaskHead3D()(real).shape), (b, 1, p, p, p))
            print(f"  shapes ok at patch {p}")
        with self.assertRaises(AssertionError):
            syn.DCGAN3DGenerator(16, 64, 8)(torch.randn(2, 16, 2, 2, 2))
        with self.assertRaises(AssertionError):
            syn.DCGAN3DGenerator(16, 40, 8)

    def test_one_step_updates_every_parameter(self):
        rng = np.random.default_rng(0)
        net = syn.MultiGAN(DEV, ld=16, patch=32, width=8)
        real, mask = rand_real(4, 32, rng)
        gens = [net.dc1_g, net.dc2_g, net.w_g]
        groups = {"dc1": [net.dc1_g, net.dc1_d], "dc2": [net.dc2_g, net.dc2_d], "wgan": [net.w_g, net.w_c],
                  "aggregator": [net.aggr, net.ag_d], "mask": [net.mask]}
        steps = {"dc1": lambda: net.step_dcgan(1, real), "dc2": lambda: net.step_dcgan(2, real),
                 "wgan": lambda: net.step_wgan(real, True), "aggregator": lambda: net.step_aggregator(real),
                 "mask": lambda: net.step_mask(real, mask)}
        for name, step in steps.items():
            before = {id(m): snap(m) for m in net.modules().values()}
            losses = step()
            self.assertTrue(all(np.isfinite(v) for v in losses.values()), f"{name}: {losses}")
            for m in groups[name]:
                self.assertEqual(unchanged(before[id(m)], m), [], f"{name}: params not updated")
            if name == "aggregator":  # generators must be detached
                for g in gens:
                    self.assertEqual(len(unchanged(before[id(g)], g)), len(list(g.parameters())))
            print(f"  {name}: all params updated, losses {({k: round(v, 3) for k, v in losses.items()})}")
        out = net.train_batch(real, mask)  # whole-batch path
        self.assertTrue(all(np.isfinite(v) for v in out.values()))

    def test_sampling_fusions_and_checkpoint_roundtrip(self):
        net = syn.MultiGAN(DEV, ld=16, patch=32, width=8)
        for fusion in ("aggregator", "mixture"):
            x = net.sample(5, fusion)
            self.assertEqual(tuple(x.shape), (5, 1, 32, 32, 32))
            syn.assert_unit_range(x.cpu().numpy())
        self.assertTrue(all(m.training for m in net.modules().values()), "sampling must restore train mode")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck", "gan.pt")
            net.save(path)
            net2 = syn.MultiGAN.load(path, DEV)
            net.train(False)
            net2.train(False)
            z = net.z(2)
            torch.testing.assert_close(net.aggr(net.dc1_g(z), net.dc2_g(z), net.w_g(z)), net2.aggr(net2.dc1_g(z), net2.dc2_g(z), net2.w_g(z)))
        print("  aggregator and mixture sampling ok, checkpoint round-trip identical")

    def test_gradient_penalty(self):
        critic = syn.WGAN3DCritic(8).to(DEV)
        real, _ = rand_real(3, 32, np.random.default_rng(1))
        fake = torch.tanh(torch.randn_like(real))
        gp = syn.gradient_penalty(critic, real, fake)
        self.assertTrue(torch.isfinite(gp) and gp.dim() == 0)
        gp.backward()
        grads = [p.grad for p in critic.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        # zero cross-sample gradient with GroupNorm; the BatchNorm control must show the coupling
        x = real.clone().requires_grad_(True)
        g, = torch.autograd.grad(critic(x)[0], x)
        self.assertEqual(float(g[1:].abs().max()), 0.0)
        self.assertGreater(float(g[0].abs().max()), 0.0)
        bn = syn.DCGAN3DDiscriminator(8).to(DEV).train()
        x = real.clone().requires_grad_(True)
        g, = torch.autograd.grad(bn(x)[0], x)
        self.assertGreater(float(g[1:].abs().max()), 0.0)
        print(f"  GP={gp.item():.4f}; GroupNorm cross-sample grad 0.0, BatchNorm control {float(g[1:].abs().max()):.2e}")


class TestData(unittest.TestCase):
    def test_range_assertion_and_window(self):
        x = np.random.default_rng(0).uniform(-1, 1, (4, 4, 4)).astype(np.float32)
        syn.assert_unit_range(x)
        with self.assertRaises(AssertionError):
            syn.assert_unit_range(x * 2 - 1 - 0.5)  # the original real*2-1 bug
        hu = np.array([-3024.0, -100.0, 75.0, 250.0, 3071.0], np.float32)
        u = syn.to_unit(hu)
        np.testing.assert_allclose(u, [-1, -1, 0.0, 1, 1], atol=1e-6)
        self.assertEqual(float(u.min()), -1.0)
        self.assertEqual(float(u.max()), 1.0)
        np.testing.assert_allclose(syn.from_unit(syn.to_unit(np.array([0.0, 100.0], np.float32))), [0, 100], atol=1e-3)

    def test_extract_patch_near_edges(self):
        n, p = 20, 16
        vol = np.random.default_rng(0).uniform(-0.9, 1, (n, n, n)).astype(np.float32)
        pad = syn.pad_air(vol, p, -1.0)
        for c in [(0, 0, 0), (n - 1, n - 1, n - 1), (0, n - 1, 5), (10, 10, 10)]:
            patch = syn.extract_patch(pad, c, p)
            self.assertEqual(patch.shape, (p, p, p))
            self.assertEqual(patch[p // 2, p // 2, p // 2], vol[c])  # centre voxel is the requested one
            exp = np.full((p, p, p), -1.0, np.float32)
            src = [(max(0, ci - p // 2), min(n, ci + p // 2)) for ci in c]
            dst = [(lo - (ci - p // 2), hi - (ci - p // 2)) for (lo, hi), ci in zip(src, c)]
            exp[tuple(slice(*d) for d in dst)] = vol[tuple(slice(*s) for s in src)]
            np.testing.assert_array_equal(patch, exp)
        far = syn.extract_patch(pad, (n - 1,) * 3, p)  # far corner: content on the low half, air on the right/high half
        self.assertTrue((far[p // 2 + 1:] == -1).all() and (far[:, p // 2 + 1:] == -1).all() and (far[:, :, p // 2 + 1:] == -1).all())
        near = syn.extract_patch(pad, (0, 0, 0), p)
        self.assertTrue((near[:p // 2] == -1).all())
        print("  corner patches centred, air padded on the correct side")

    def test_component_wise_sampling(self):
        t = np.zeros((90, 90, 90), bool)
        t[10:50, 10:50, 10:50] = True  # one huge tumour (64000 voxels)
        small = [(60, 60, 60), (70, 15, 70), (20, 75, 75), (75, 75, 15)]
        for c in small:
            t[c[0]:c[0] + 3, c[1]:c[1] + 3, c[2]:c[2] + 3] = True  # 27-voxel tumours
        rng = np.random.default_rng(0)
        cs = syn.sample_centres(t, rng, cap_comp=4, cap_vol=24, jitter=2, min_vox=20)
        lab, n = ndi.label(t, structure=np.ones((3, 3, 3)))
        self.assertEqual(n, 5)
        counts = {}
        for c, comp in cs:
            counts[comp] = counts.get(comp, 0) + 1
        self.assertEqual(len(counts), 5, "every component must receive patches")
        self.assertTrue(all(v <= 4 for v in counts.values()))
        for c in small:  # small tumours get >= 1 centre within jitter of their voxels
            near = [cc for cc, _ in cs if np.all(np.abs(cc - (np.array(c) + 1)) <= 2 + 2)]
            self.assertGreaterEqual(len(near), 1)
        tiny_total = sum(v for k, v in counts.items() if k != lab[30, 30, 30])
        self.assertGreater(tiny_total, counts[lab[30, 30, 30]])  # huge tumour does not dominate (voxel-uniform would give ~100%)
        capped = syn.sample_centres(t, rng, cap_comp=4, cap_vol=6, jitter=2, min_vox=20)
        self.assertEqual(len(capped), 6)
        self.assertEqual(len({k for _, k in capped}), 5)  # round-robin: all components covered before any gets a 2nd patch
        print(f"  per-component counts {sorted(counts.values())}, voxel share of huge tumour {64000 / (64000 + 4 * 27):.2%}")

    def test_healthy_patches_exclude_tumour(self):
        seg = np.zeros((40, 40, 40), np.uint8)
        seg[5:35, 5:35, 5:35] = 1
        seg[18:22, 18:22, 18:22] = 2
        pad = syn.pad_air(seg, 16, 0)
        cs = syn.sample_healthy(seg, pad, np.random.default_rng(0), size=16, cap=6)
        self.assertGreater(len(cs), 0)
        for c in cs:
            self.assertEqual((syn.extract_patch(pad, c, 16) == 2).sum(), 0)

    def test_clean_mask(self):
        p = np.zeros((32, 32, 32), np.float32)
        p[10:20, 10:20, 10:20] = 0.9
        p[25:27, 25:27, 25:27] = 0.9  # small stray component must be dropped
        m, _ = syn.clean_mask(p, min_vox=100, edge=4)
        self.assertEqual(int(m.sum()), 1000)
        self.assertEqual(syn.clean_mask(p * 0, 100, 4)[1], "empty")
        self.assertEqual(syn.clean_mask(p, 5000, 4)[1], "too_small")
        q = p.copy()
        q[0:12, 10:20, 10:20] = 0.9  # merges with the main blob and reaches the border
        self.assertEqual(syn.clean_mask(q, 100, 4)[1], "touches_border")

    def test_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("volume-1.nii", "segmentation-2.nii"):
                open(os.path.join(d, f), "w").close()
            with self.assertRaises(AssertionError):
                syn.pair_cases(d)

    def test_split_equals_sklearn_on_sorted_names(self):
        with tempfile.TemporaryDirectory() as d:
            fake_volume_dir(d, {i: (0.8, 0.8, 2.0) for i in range(23)})  # 23 names: sorted order is volume-0, -1, -10, -11, ... (not numeric)
            names = syn.sorted_volume_names(d)
            self.assertEqual(names[:4], ["volume-0.nii", "volume-1.nii", "volume-10.nii", "volume-11.nii"])
            exp_train, exp_test = train_test_split(names, test_size=0.2, random_state=42)  # the original segmenter call
            self.assertEqual(len(exp_test), 5)
            self.assertEqual(syn.train_part_ids(d), sorted(syn.case_id(n) for n in exp_train))
            self.assertEqual(syn.split_cases(d, 0), sorted(syn.case_id(n) for n in exp_train))
            self.assertEqual(syn.split_cases(d, 4), sorted(syn.case_id(n) for n in exp_train)[:4])
        if HAVE_RAW and len(syn.sorted_volume_names(RAW)) == 131:
            names = syn.sorted_volume_names(RAW)
            tr, te = train_test_split(names, test_size=0.2, random_state=42)
            self.assertEqual((len(tr), len(te)), (104, 27))
            self.assertEqual(syn.train_part_ids(RAW), sorted(syn.case_id(n) for n in tr))
            print(f"  real data: {len(tr)} train / {len(te)} held-out, GAN train part equals sklearn's split")
            vt = os.path.join(ROOT, "data", "cache", "volumes.txt")  # the segmenter's own volume list (read-only)
            if os.path.exists(vt):
                self.assertEqual(names, open(vt).read().split(), "sorted names must equal the segmenter's volumes.txt order")
                print("  real data: sorted names equal the segmenter's data/cache/volumes.txt")

    def test_held_out_and_placeholder_volumes_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            placeholder = {3, 8, 12, 17}
            fake_volume_dir(d, {i: (1.0, 1.0, 1.0) if i in placeholder else (0.7 + 0.01 * i, 0.7 + 0.01 * i, 2.5) for i in range(25)})
            names = syn.sorted_volume_names(d)
            tr, te = train_test_split(names, test_size=0.2, random_state=42)
            train, held = {syn.case_id(n) for n in tr}, {syn.case_id(n) for n in te}
            ids = syn.split_cases(d, 0)
            self.assertFalse(set(ids) & held, "no GAN training volume / insertion host may come from the held-out part")
            self.assertTrue(set(ids) <= train)
            self.assertFalse(set(ids) & placeholder, "placeholder-spacing volumes must be dropped")
            self.assertEqual(set(ids), train - placeholder)
            self.assertTrue(train & placeholder, "fixture must put placeholder volumes in the train part")
            # a volume with only SOME zooms equal to 1.0 is real spacing and must be kept
            fake_volume_dir(d, {99: (0.8, 0.8, 1.0)})
            self.assertTrue(syn.placeholder_spacing(os.path.join(d, "volume-3.nii")))
            self.assertFalse(syn.placeholder_spacing(os.path.join(d, "volume-99.nii")))
        if HAVE_RAW and len(syn.sorted_volume_names(RAW)) == 131:
            names = syn.sorted_volume_names(RAW)
            held = {syn.case_id(n) for n in train_test_split(names, test_size=0.2, random_state=42)[1]}
            ids = syn.split_cases(RAW, 0)
            self.assertFalse(set(ids) & held)
            self.assertFalse(set(ids) & set(range(28, 48)), "LiTS volumes 28-47 (placeholder 1x1x1 spacing) must be excluded")
            self.assertEqual(len(ids), 89)
            print(f"  real data: {len(ids)} usable train-part volumes, none held-out, none of volumes 28-47")

    def test_prepare_patches_toy_volume(self):
        with tempfile.TemporaryDirectory() as d:
            sp = (1.2, 1.2, 2.0)
            g = np.random.default_rng(0)
            zz, yy, xx = np.mgrid[:90, :90, :50].astype(np.float32)
            liver = ((zz - 45) / 35) ** 2 + ((yy - 45) / 30) ** 2 + ((xx - 25) / 20) ** 2 <= 1
            seg = liver.astype(np.uint8)
            hu = np.where(liver, 100.0, -1000.0) + g.normal(0, 8, liver.shape)
            for c, r in [((45, 45, 25), 9), ((20, 45, 25), 2), ((70, 40, 25), 2), ((45, 70, 25), 2)]:  # 1 huge + 3 tiny
                t = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
                t &= liver
                seg[t] = 2
                hu[t] = 60
            aff = np.diag([sp[0], sp[1], sp[2], 1.0])
            nib.save(nib.Nifti1Image(hu.astype(np.int16), aff), os.path.join(d, "volume-1.nii"))
            nib.save(nib.Nifti1Image(seg, aff), os.path.join(d, "segmentation-1.nii"))
            out = os.path.join(d, "p.npz")
            syn.prepare_patches([1], out, data_dir=d, size=32, rng=np.random.default_rng(0))
            z = syn.load_npz(out)
            self.assertEqual(z["images"].shape[1:], (32, 32, 32))
            self.assertEqual(len(z["images"]), len(z["masks"]))
            syn.assert_unit_range(z["images"].astype(np.float32))
            self.assertTrue(all(m.any() for m in z["masks"]), "every tumour patch must contain tumour")
            comps = z["meta"][:, 1]
            self.assertEqual(len(set(comps.tolist())), 4)
            self.assertLessEqual(max(np.bincount(comps)), syn.CAP_COMP)
            self.assertEqual(z["healthy"].shape[1:], (32, 32, 32))
            print(f"  toy prepare: {len(z['images'])} tumour patches from {len(set(comps.tolist()))} components, {len(z['healthy'])} healthy")


def toy_liver_patch(rng, tumour, p=32):
    """Smooth liver background; a tumour is a darker ellipsoid. Returns (patch in [-1,1], mask)."""
    bg = ndi.gaussian_filter(rng.normal(0, 1, (p, p, p)), 1.0) * 0.15 + 0.3
    mask = np.zeros((p, p, p), bool)
    if tumour:
        c = rng.integers(3 * p // 8, 5 * p // 8 + 1, 3)
        r = rng.uniform(0.15 * p, 0.25 * p, 3)
        zz, yy, xx = np.mgrid[:p, :p, :p]
        mask = (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1
        bg = np.where(mask, bg - 0.5, bg)
    return np.clip(bg, -1, 1).astype(np.float32), mask


class TestMaskHeadAndGate(unittest.TestCase):
    def test_holdout_is_by_volume(self):
        rng = np.random.default_rng(0)
        cids = np.array([1, 1, 2, 2, 3, 3, 4, 5, 5, 6])
        held = syn.split_holdout(cids, rng, 0.34)
        self.assertEqual(len(held), 2)
        self.assertEqual(syn.split_holdout(np.array([7, 7, 7]), rng), set())  # one volume: nothing to hold out
        self.assertEqual(len(syn.split_holdout(np.array([7, 8]), rng, 0.99)), 1)  # always leaves a training volume

    def test_negatives_make_gate_reject_non_tumour(self):
        """Mask head trained on tumour + healthy patches: air, flat grey and healthy liver are rejected, a held-out tumour is accepted."""
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        net = syn.MultiGAN(DEV, ld=16, patch=32, width=8)
        for _ in range(200):
            xs, ms = zip(*[toy_liver_patch(rng, i % 2 == 0) for i in range(4)])
            net.step_mask(torch.from_numpy(np.stack(xs))[:, None].to(DEV), torch.from_numpy(np.stack(ms).astype(np.float32))[:, None].to(DEV))
        net.train(False)

        def gate(x):
            with torch.no_grad():
                pr = torch.sigmoid(net.mask(torch.from_numpy(x)[None, None].to(DEV)))[0, 0].cpu().numpy()
            return syn.clean_mask(pr, min_vox=100, edge=4)

        air, grey = np.full((32,) * 3, -1.0, np.float32), np.zeros((32,) * 3, np.float32)
        self.assertIsNone(gate(air)[0], "all-air patch must be rejected")
        self.assertIsNone(gate(grey)[0], "flat grey patch must be rejected")
        healthy = [gate(toy_liver_patch(rng, False)[0])[0] for _ in range(10)]
        self.assertLessEqual(sum(h is not None for h in healthy), 1)
        ious = []
        for _ in range(10):
            x, m = toy_liver_patch(rng, True)
            g, _ = gate(x)
            ious.append(0.0 if g is None else (g & m).sum() / (g | m).sum())
        print(f"  gate: air/grey rejected, healthy accepted {sum(h is not None for h in healthy)}/10, held-out tumour IoU {np.round(ious, 2)}")
        self.assertGreater(np.mean(ious), 0.6)
        self.assertGreaterEqual(sum(i > 0.5 for i in ious), 8)

    def test_train_logs_per_key_means_and_heldout_metrics(self):
        """w_g is produced every N_CRITIC-th batch only; the logged mean must be over those batches (not diluted)."""
        rng = np.random.default_rng(0)
        n = 12
        imgs, msks = zip(*[toy_liver_patch(rng, True, 16) for _ in range(n)])
        hl = [toy_liver_patch(rng, False, 16)[0] for _ in range(6)]
        with tempfile.TemporaryDirectory() as d:
            pp = os.path.join(d, "p.npz")
            np.savez(pp, images=np.stack(imgs).astype(np.float16), masks=np.stack(msks).astype(np.uint8), healthy=np.stack(hl).astype(np.float16),
                     meta=np.array([(i // 4, 0) for i in range(n)]), healthy_cid=np.array([0, 0, 1, 1, 2, 2]))
            calls = {"n": 0}

            def stub(self, real, mask, mask_pool=None):
                calls["n"] += 1
                _, m = mask_pool()  # the mask pool must deliver tumour + healthy (all-zero) samples
                calls["healthy"] = calls.get("healthy", 0) + int((m.flatten(1).sum(1) == 0).sum())
                out = {"a": 1.0}
                if calls["n"] % 5 == 0:
                    out["w_g"] = 10.0
                return out

            orig = syn.MultiGAN.train_batch
            syn.MultiGAN.train_batch = stub
            try:
                with self.assertLogs("gan_fixed", level="INFO") as cm:
                    syn.train(pp, 5, DEV, d)
            finally:
                syn.MultiGAN.train_batch = orig
            text = "\n".join(cm.output)
            self.assertIn("w_g=10.000", text)
            self.assertIn("a=1.000", text)
            self.assertIn("mask head heldout:", text)
            self.assertIn("mask head train:", text)
            self.assertGreater(calls["healthy"], 0)
            print("  per-key mean w_g=10.000 logged (not diluted to 2.000); heldout + train mask metrics logged; healthy negatives in the pool")


class TestResampling(unittest.TestCase):
    def test_volume_preserved_on_host_grid(self):
        zz, yy, xx = np.mgrid[:64, :64, :64].astype(float) - 32
        worst = 1.0
        for r in (4, 6, 10, 20):
            m = (zz ** 2 + yy ** 2 + xx ** 2) <= r * r
            for sp in [(0.703, 0.703, 5.0), (0.7, 0.7, 2.5), (0.8, 0.8, 3.0), (0.873, 0.873, 1.0), (1.0, 1.0, 1.0)]:
                img, mr = syn.resample_patch_to_host(np.zeros(m.shape, np.float32), m, sp)
                ratio = mr.sum() * np.prod(sp) / m.sum()
                worst = max(worst, ratio, 1 / ratio)
                self.assertLess(abs(ratio - 1), 0.05, (r, sp, ratio))
                self.assertEqual(img.shape, mr.shape)
        print(f"  mask volume preserved within {worst - 1:.1%} for r 4-20 mm at 1-5 mm host slices")
        _, mr = syn.resample_patch_to_host(np.zeros((64,) * 3, np.float32), np.zeros((64,) * 3, bool), (1, 1, 1))
        self.assertFalse(mr.any())


@unittest.skipUnless(HAVE_RAW, "data/raw/volume-0.nii not available")
class TestInsertion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol, cls.seg, cls.sp, cls.aff = syn.load_case(os.path.join(RAW, "volume-0.nii"), os.path.join(RAW, "segmentation-0.nii"))
        cls.hu0 = syn.to_int16(cls.vol)
        cls.patch, cls.mask = toy_generated()

    def insert(self, n=1, seed=0):
        return syn.insert_tumors(self.vol, self.seg, self.sp, ([self.patch], [self.mask.astype(np.uint8)]), np.random.default_rng(seed), n)

    def test_insertion_on_real_host(self):
        print(f"  host volume-0 spacing {self.sp}, shape {self.vol.shape}, liver vox {(self.seg == 1).sum()}, tumour vox {(self.seg == 2).sum()}")
        hu, seg, infos = self.insert(1)
        self.assertEqual(len(infos), 1)
        info = infos[0]
        win = info["window"]
        inwin = np.zeros(hu.shape, bool)
        inwin[win] = True
        np.testing.assert_array_equal(hu[~inwin], self.hu0[~inwin])  # bit-identical outside the blend window
        sup = np.zeros(hu.shape, bool)
        sup[win] = info["support"]
        np.testing.assert_array_equal(hu[~sup], self.hu0[~sup])  # and inside the window outside alpha>0
        changed = hu != self.hu0
        self.assertTrue(changed.any() and not (changed & ~sup).any())
        print(f"  changed voxels {int(changed.sum())} of {hu.size}; alpha support {int(sup.sum())}; window {tuple((s.start, s.stop) for s in win)}")
        # labels: only liver voxels become tumour, nothing existing erased
        lab_changed = seg != self.seg
        self.assertTrue((self.seg[lab_changed] == 1).all() and (seg[lab_changed] == 2).all())
        self.assertTrue((seg[self.seg == 2] == 2).all() and (seg[self.seg == 0] == 0).all())
        n_new = int((seg == 2).sum() - (self.seg == 2).sum())
        self.assertGreater(n_new, 0)
        self.assertEqual(n_new, info["n_label"])
        dist = ndi.distance_transform_edt(self.seg != 2, sampling=self.sp)[lab_changed]
        self.assertGreater(dist.min(), 0)
        print(f"  +{n_new} tumour voxels, all inside liver; min distance to existing tumour {dist.min():.1f} mm; tumour radius {info['radius_mm']:.1f} mm")
        # physical size: resampled mask extent (mm) == generated mask extent (mm at 1 mm voxels)
        _, m_rs = syn.resample_patch_to_host(self.patch, self.mask, self.sp)

        def ext(m, s):
            idx = np.argwhere(m)
            return (idx.max(0) - idx.min(0) + 1) * np.asarray(s)

        e_gen, e_host = ext(self.mask, (1, 1, 1)), ext(m_rs, self.sp)
        print(f"  extent mm: generated {e_gen} -> host grid {np.round(e_host, 1)} (voxel {self.sp})")
        self.assertTrue(np.all(np.abs(e_gen - e_host) <= np.maximum(np.asarray(self.sp) + 1.0, 0.12 * e_gen)))  # extent: quantised by the 5 mm slices
        v_ratio = m_rs.sum() * np.prod(self.sp) / self.mask.sum()
        print(f"  physical volume ratio host/generated {v_ratio:.3f}")
        self.assertLess(abs(v_ratio - 1), 0.05)
        self.assertTrue(np.all(ext(info["mask"], self.sp) <= e_host + 1e-6))
        # intensity transfer keeps generated contrast (-40 HU) relative to the host liver mean; noise matches host
        contrast = float(hu[win][info["mask"]].mean()) - info["host_ref_hu"]
        print(f"  tumour-minus-host-liver contrast {contrast:.1f} HU (generated -40), shift {info['shift_hu']:.1f} HU")
        self.assertLess(abs(contrast - (-40.0)), 15.0)
        x = hu[win].astype(np.float32)
        hp = (x - ndi.gaussian_filter(x, syn.NOISE_SIGMA_MM / np.asarray(self.sp), mode="nearest"))[info["mask"]]
        s_in = 1.4826 * np.median(np.abs(hp - np.median(hp)))
        print(f"  noise std host {info['noise_std']:.1f} HU vs inside inserted tumour {s_in:.1f} HU")
        self.assertTrue(0.4 < s_in / max(info["noise_std"], 1e-6) < 2.5)

    def test_thick_slice_feather_does_not_outrun_label(self):
        """On 5 mm slices the visible (alpha > 0.5) tumour must not extend a whole slice beyond the label."""
        flat = (toy_generated(radius=(12, 10, 4))[0], toy_generated(radius=(12, 10, 4))[1].astype(np.uint8))
        worst = 0.0
        for seed in range(4):
            for smp in (([self.patch], [self.mask.astype(np.uint8)]), ([flat[0]], [flat[1]])):
                _, _, infos = syn.insert_tumors(self.vol, self.seg, self.sp, smp, np.random.default_rng(seed), 1)
                for i in infos:
                    vis = i["alpha"] > 0.5
                    extra = int((vis & ~i["mask"]).sum())
                    worst = max(worst, extra / max(int(i["mask"].sum()), 1))
        print(f"  alpha>0.5 voxels outside the label / label voxels, worst over 8 insertions: {worst:.2f} (before the fix 1.3-1.7)")
        self.assertLess(worst, 0.5)

    def test_alpha_only_on_host_liver(self):
        _, _, infos = self.insert(3, seed=1)
        for i in infos:
            sw = self.seg[i["window"]]
            self.assertTrue((sw[i["support"]] == 1).all(), "blend support must be host liver only (no background, no existing tumour)")

    def test_hu_roundtrip_through_nifti(self):
        hu, seg, _ = self.insert(1)
        with tempfile.TemporaryDirectory() as d:
            pv, ps = os.path.join(d, "v.nii"), os.path.join(d, "s.nii")
            syn.save_nifti(hu, self.aff, pv, np.int16)
            syn.save_nifti(seg, self.aff, ps, np.uint8)
            v, s = nib.load(pv), nib.load(ps)
            a, b = np.asarray(v.dataobj), np.asarray(s.dataobj)
            self.assertEqual(a.dtype, np.int16)
            self.assertEqual(b.dtype, np.uint8)
            self.assertEqual((a.min(), a.max()), (hu.min(), hu.max()))
            np.testing.assert_array_equal(a, hu)
            np.testing.assert_array_equal(b, seg)
            np.testing.assert_allclose(v.affine, self.aff, atol=1e-4)
            np.testing.assert_allclose(v.header.get_zooms()[:3], self.sp, atol=1e-4)
            print(f"  reload dtype {a.dtype}, min/max {a.min()}/{a.max()} (saved {hu.min()}/{hu.max()}), exact equal, affine/zooms preserved")

    def test_multiple_nonoverlapping_insertions_and_no_site(self):
        _, seg, infos = self.insert(3, seed=3)
        self.assertGreaterEqual(len(infos), 2)
        n_new = int((seg == 2).sum() - (self.seg == 2).sum())
        self.assertEqual(n_new, sum(i["n_label"] for i in infos))  # disjoint: no insertion overwrote another
        print(f"  {len(infos)} insertions, +{n_new} tumour voxels")
        tiny_liver = np.zeros_like(self.seg)
        tiny_liver[100:104, 100:104, 30:32] = 1  # no site can fit a tumour: must skip, not crash
        h, s, infos = syn.insert_tumors(self.vol, tiny_liver, self.sp, ([self.patch], [self.mask.astype(np.uint8)]), np.random.default_rng(0), 1)
        self.assertEqual(infos, [])
        np.testing.assert_array_equal(h, self.hu0)
        self.assertEqual(int(s.sum()), int(tiny_liver.sum()))


@unittest.skipIf(os.environ.get("LITS_TEST_SKIP_E2E") == "1", "e2e skipped")
@unittest.skipUnless(HAVE_RAW, "data/raw not available")
class TestEndToEnd(unittest.TestCase):
    """CLI run: prepare -> train (2 epochs) -> generate -> insert on small LiTS volumes. LITS_DATA_DIR is a temporary
    directory of symlinks to 11 small raw volumes (2 of them placeholder-spacing), so the original split (test_size=0.2,
    random_state=42) runs unchanged over that pool: 8 train / 3 held-out; the code path is the one used on data/raw.
    This is an artefact/plumbing test of the CLI stages, not a quality test of generated samples: a 2-epoch GAN
    only emits noise and the mask gate rejects it. When fewer than 2 samples survive, the insert stage is fed REAL tumour
    patches/masks (transplant). Generated-sample -> insertion is asserted separately in TestGenerateInsertPlumbing.
    The real 131-volume split (104 train / 27 held-out) is covered by test_split_equals_sklearn_on_sorted_names and
    test_held_out_and_placeholder_volumes_excluded."""

    def run_cli(self, env, stage):
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "Syn_Img_Pipeline_fixed.py")],
                           env=dict(env, LITS_GAN_STAGE=stage), capture_output=True, text=True)
        print("\n".join(r.stderr.strip().splitlines()[-14:]))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_end_to_end(self):
        out = os.path.join(ROOT, "results", "gan_e2e")
        shutil.rmtree(out, ignore_errors=True)
        pool_ids = [45, 46, 48, 49, 50, 52, 55, 56, 58, 78, 31]  # 45, 46, 31 have placeholder 1x1x1 spacing
        pool = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, pool, True)
        for i in pool_ids:
            for kind in ("volume", "segmentation"):
                os.symlink(os.path.join(RAW, f"{kind}-{i}.nii"), os.path.join(pool, f"{kind}-{i}.nii"))
        names = syn.sorted_volume_names(pool)
        tr, te = train_test_split(names, test_size=0.2, random_state=42)
        held = {syn.case_id(n) for n in te}
        usable = sorted(syn.case_id(n) for n in tr if syn.case_id(n) not in (45, 46, 31))
        want = usable[:3]  # LITS_GAN_MAX_VOLUMES=3 takes the first 3 usable train-part ids
        print(f"  pool split: train {sorted(syn.case_id(n) for n in tr)}, held-out {sorted(held)}, expected GAN volumes {want}")
        self.assertTrue(held and {45, 46} <= {syn.case_id(n) for n in tr}, "fixture: placeholder volumes must sit in the train part")
        env = dict(os.environ, LITS_GAN_EPOCHS="2", LITS_GAN_MAX_VOLUMES="3", LITS_GAN_NUM_SAMPLES="2", LITS_GAN_RESULTS_DIR=out,
                   LITS_DATA_DIR=pool, LITS_GAN_PATCH="64", LITS_GAN_INSERTIONS="1", LITS_GAN_MASK_STEPS="10")
        for stage in ("prepare", "train", "generate"):
            self.run_cli(env, stage)
        for f in ("patches.npz", "checkpoints/gan_last.pt", "figures/train_epoch002.png", "synthetic/samples.npz"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        sp = os.path.join(out, "synthetic", "samples.npz")
        n_gen = len(syn.load_npz(sp)["patches"])
        print(f"  generated samples accepted by the mask gate after 2 epochs: {n_gen}/2")
        if n_gen < 2:
            d = syn.load_npz(os.path.join(out, "patches.npz"))
            keep, masks = [], []
            for i, m in enumerate(d["masks"]):  # same gate as generation (largest component, min voxels, off the border), moderate size
                cm, _ = syn.clean_mask(m.astype(np.float32), 100, 4)
                if cm is not None and cm.sum() < 4000 and len(keep) < 2:
                    keep.append(i)
                    masks.append(cm.astype(np.uint8))
            np.savez(sp, patches=d["images"][keep].astype(np.float32), masks=np.stack(masks))
            print(f"  substitute: insert stage fed with {len(keep)} real tumour patches, mask voxels {[int(m.sum()) for m in masks]}")
            self.assertEqual(len(keep), 2)
        self.run_cli(env, "insert")
        augs = sorted(os.listdir(os.path.join(out, "aug")))
        print("  aug files:", augs)
        self.assertGreaterEqual(len([a for a in augs if a.startswith("aug_volume")]), 2)
        self.assertTrue(os.path.exists(os.path.join(out, "figures", "insertions.png")))
        ids = {int(a.split("-")[1].split(".")[0]) for a in augs}
        self.assertTrue(ids <= set(want), "hosts must come from the usable train-part volumes only")
        self.assertFalse(ids & held, "no host may come from the held-out part")
        self.assertFalse(ids & {45, 46, 31}, "placeholder-spacing volumes must not be hosts")
        with open(os.path.join(out, "pipeline.log")) as f:
            log = f.read()
        self.assertIn("dropping 2 train-part volumes with placeholder 1x1x1 header spacing: [45, 46]", log)
        self.assertFalse([f for f in os.listdir(RAW) if f.startswith("aug")], "nothing may be written into data/raw")


@unittest.skipUnless(HAVE_RAW, "data/raw not available")
class TestGenerateInsertPlumbing(unittest.TestCase):
    """generate_samples -> samples.npz -> run_insert on two real hosts, with a stand-in sampler/mask head (the GAN
    itself is not trained here; this asserts that accepted GENERATED samples flow through the gate into insertion)."""

    def test_generated_samples_reach_two_hosts(self):
        net = syn.MultiGAN(DEV, ld=16, patch=64, width=8)
        seeds = iter(range(100))
        thr = float(syn.to_unit(np.float32(80.0)))  # between tumour (60 HU) and liver (100 HU) of the toy generator

        def fake_sample(n, fusion):
            return torch.from_numpy(np.stack([toy_generated(seed=next(seeds))[0] for _ in range(n)]))[:, None].to(DEV)

        net.sample = fake_sample

        class Stub(torch.nn.Module):  # logits: darker than the threshold = tumour
            def forward(self, x):
                return (thr - x) * 50.0

        net.mask = Stub()
        for fusion in ("aggregator", "mixture"):
            p, m = syn.generate_samples(net, 2, fusion)
            self.assertEqual((len(p), len(m)), (2, 2))
            self.assertTrue(all(mi.any() for mi in m))
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "samples.npz")
            np.savez(sp, patches=p, masks=m)
            done = syn.run_insert(sp, [0, 48], os.path.join(d, "aug"), os.path.join(d, "ins.png"), n=1)
            self.assertEqual(done, 2)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "aug"))),
                             ["aug_segmentation-0.nii", "aug_segmentation-48.nii", "aug_volume-0.nii", "aug_volume-48.nii"])
            self.assertTrue(os.path.exists(os.path.join(d, "ins.png")))
        print("  2 generated samples (aggregator + mixture gate path) inserted into 2 real hosts (volume-0, volume-48)")


# ---------------------------------------------------------------------------------------------------------------------
# Cache source
# ---------------------------------------------------------------------------------------------------------------------
def make_toy_cache(d, ids, big, placeholder=(), z=60, sz=2.0, seed=0):
    """Synthetic cache_v2-style directory: volume-<id>.nii names (sorted lexicographically), uint8 slices (Z,256,256) in the
    -100..250 HU window, an ellipsoid liver with two spherical tumours in the `big` volumes (other volumes: 3 empty slices),
    plus a per_volume.csv with raw spacing (0.7, 0.7, sz) or the 1x1x1 placeholder. Returns the sorted names."""
    names = sorted(f"volume-{i}.nii" for i in ids)
    g = np.random.default_rng(seed)
    imgs, msks, vi, rows = [], [], [], []
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    for k, n in enumerate(names):
        cid = syn.case_id(n)
        zn = z if cid in big else 3
        img, msk = np.zeros((zn, 256, 256), np.uint8), np.zeros((zn, 256, 256), np.uint8)
        if cid in big:
            zz = np.arange(zn, dtype=np.float32)[:, None, None]
            liver = ((zz - zn / 2) / (zn * 0.4)) ** 2 + ((yy - 128) / 70) ** 2 + ((xx - 128) / 55) ** 2 <= 1
            hu = np.where(liver, 100.0, -1000.0) + g.normal(0, 8, liver.shape)
            msk[liver] = 1
            for cy, cx in ((110, 100), (150, 160)):
                t = liver & ((((zz - zn / 2) * sz) ** 2 + ((yy - cy) * 1.4) ** 2 + ((xx - cx) * 1.4) ** 2) <= 13.0 ** 2)
                msk[t] = 2
                hu[t] = 60 + g.normal(0, 8, int(t.sum()))
            img = np.rint(np.clip((hu + 100) / 350, 0, 1) * 255).astype(np.uint8)
        imgs.append(img)
        msks.append(msk)
        vi.append(np.full(zn, k, np.int16))
        sp = (1.0, 1.0, 1.0) if cid in placeholder else (0.7, 0.7, sz)
        rows.append(dict(name=n, nx=512, ny=512, nz=zn, sx_mm=sp[0], sy_mm=sp[1], sz_mm=sp[2], tumor_ml=1.0 if cid in big else 0.0))
    np.save(os.path.join(d, "images.npy"), np.concatenate(imgs))
    np.save(os.path.join(d, "masks.npy"), np.concatenate(msks))
    np.save(os.path.join(d, "vol_index.npy"), np.concatenate(vi))
    with open(os.path.join(d, "volumes.txt"), "w") as f:
        f.write("\n".join(names))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"meta_fmt": 2, "win_lo": -100.0, "win_hi": 250.0, "canon": True, "volumes": names}, f)
    with open(os.path.join(d, "per_volume.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return names


def toy_cache_layout(n=25, n_big=4, n_placeholder=2):
    """ids 0..n-1; returns (train ids, held-out ids, big ids (all train, real spacing), placeholder ids (train))."""
    names = sorted(f"volume-{i}.nii" for i in range(n))
    tr, te = train_test_split(names, test_size=0.2, random_state=42)
    train, held = sorted(syn.case_id(x) for x in tr), sorted(syn.case_id(x) for x in te)
    return train, held, train[:n_big], train[n_big:n_big + n_placeholder]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


class Recorder:
    """Wraps a memmap/ndarray and records every slice range read (to prove which volumes were touched)."""

    def __init__(self, a):
        self.a, self.reads = a, []
        self.shape = a.shape

    def __len__(self):
        return len(self.a)

    def __getitem__(self, k):
        self.reads.append((k.start, k.stop) if isinstance(k, slice) else (int(k), int(k) + 1))
        return self.a[k]


class TestCacheSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.train, cls.held, cls.big, cls.ph = toy_cache_layout()
        cls.names = make_toy_cache(cls.tmp, range(25), set(cls.big), set(cls.ph))
        cls.pv = os.path.join(cls.tmp, "per_volume.csv")
        cls.patch_ids = [i for i in cls.train if i not in cls.ph]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def src(self, work=1.5):
        return syn.CacheSource(self.tmp, self.pv, work)

    def test_reconstruction_is_exact_inverse_of_window(self):
        s = self.src()
        u8 = np.arange(256, dtype=np.uint8)
        hu = s.to_hu(u8)
        np.testing.assert_array_equal(syn.quantise_window(hu), u8)  # every level survives HU round trip
        self.assertAlmostEqual(float(hu[0]), -100.0, places=4)
        self.assertAlmostEqual(float(hu[255]), 250.0, places=3)
        np.testing.assert_allclose(syn.to_unit(hu), u8 / 255.0 * 2 - 1, atol=2e-6)  # x = u8/255*2-1 is the GAN's [-1,1] window
        v, seg, sp, aff = s.load(self.big[0])
        self.assertEqual(v.shape[:2], (256, 256))
        self.assertEqual(v.shape, seg.shape)
        self.assertEqual(v.shape[2], 60)  # slice axis moved last
        self.assertIsNone(aff)
        np.testing.assert_allclose(sp, (1.4, 1.4, 2.0))  # 512/256 x raw in-plane spacing, raw z spacing
        img, _ = s.load_u8(self.big[0])
        np.testing.assert_array_equal(np.moveaxis(img, 0, 2), syn.quantise_window(v))

    def test_split_rule_and_volume_list_restriction(self):
        s = self.src()
        self.assertEqual(s.train_part_ids(), self.train)  # sklearn over volumes.txt order
        self.assertEqual(len(self.train), 20)
        self.assertEqual(s.split_cases(), [i for i in self.train if i not in self.ph])  # placeholder volumes dropped
        listed = {self.train[0], self.train[5], self.held[0], self.held[1], self.ph[0], 999}  # held-out, placeholder, unknown ids in the list
        ids = s.split_cases(listed)
        self.assertEqual(ids, sorted({self.train[0], self.train[5]} - set(self.ph)))
        self.assertFalse(set(ids) & set(self.held))
        self.assertEqual(s.split_cases(set(self.train), 3), [i for i in self.train if i not in self.ph][:3])
        # json formats
        with tempfile.TemporaryDirectory() as d:
            def parse(obj):
                f = os.path.join(d, "l.json")
                with open(f, "w") as fh:
                    json.dump(obj, fh)
                return syn.parse_volume_list(f)
            self.assertEqual(parse(["volume-3.nii", "volume-10", 7]), {3, 10, 7})
            self.assertEqual(parse({"volumes": ["volume-3.nii"]}), {3})
            self.assertEqual(parse({"volume-4.nii": {"x": 1}, "volume-5.nii": 2}), {4, 5})
            with self.assertRaises(ValueError):
                parse({"16": ["volume-3.nii"], "32": ["volume-4.nii"]})  # size -> list dicts are ambiguous: refuse
        with self.assertRaises(AssertionError):
            s.split_cases({self.held[0]})  # nothing usable
        with self.assertRaises(AssertionError):
            s.load_u8(self.held[0])  # the held-out part can not even be read

    def test_patches_only_from_listed_volumes_and_no_other_slice_is_read(self):
        s = self.src()
        listed = {self.big[0], self.big[1], self.held[0], self.ph[0]}
        ids = s.split_cases(listed)
        self.assertEqual(ids, sorted([self.big[0], self.big[1]]))
        s.images, s.masks = Recorder(s.images), Recorder(s.masks)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "p.npz")
            syn.prepare_patches(ids, out, size=32, rng=np.random.default_rng(0), source=s)
            z = syn.load_npz(out)
        self.assertEqual({int(c) for c in z["meta"][:, 0]} | {int(c) for c in z["healthy_cid"]}, set(ids))
        self.assertEqual(z["ids"].tolist(), ids)
        self.assertEqual(float(z["work_mm"]), 1.5)
        allowed = [(int(s.start[s.cid_to_idx[i]]), int(s.start[s.cid_to_idx[i] + 1])) for i in ids]
        for a, b in s.images.reads + s.masks.reads:
            self.assertTrue(any(lo <= a and b <= hi for lo, hi in allowed), f"slice range {(a, b)} lies outside the listed volumes {allowed}")
        self.assertGreater(len(s.images.reads), 0)
        print(f"  patches {len(z['images'])} from volumes {ids}; {len(s.images.reads)} reads, all inside the listed volumes' slice ranges")

    def toy_samples(self, k=3, p=32, radius=(7, 6, 6)):
        pm = [toy_generated(p, radius, seed=i) for i in range(k)]
        return np.stack([x for x, _ in pm]), np.stack([m for _, m in pm]).astype(np.uint8)

    def test_slab_schema_and_invariants(self):
        s = self.src()
        hosts = [i for i in self.big]
        out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out_dir, True)
        path = os.path.join(out_dir, "aug_slabs.npz")
        fig = os.path.join(out_dir, "m.png")
        st = syn.run_insert_slabs(s, hosts, self.toy_samples(), path, target=6, per_host=2, max_attempts=30, rng=np.random.default_rng(1),
                                  work_mm=1.5, fig_path=fig)
        z = syn.load_npz(path)
        self.assertEqual(sorted(z.files), sorted(["images", "masks", "slab_id", "host_vol", "z0", "new_tumor_vox", "n_slices"]))
        M, S = len(z["images"]), len(z["host_vol"])
        self.assertEqual(S, 6)
        self.assertEqual(st["accepted"], 6)
        self.assertEqual(z["images"].shape, (M, 256, 256))
        self.assertEqual(z["masks"].shape, (M, 256, 256))
        self.assertEqual((z["images"].dtype, z["masks"].dtype), (np.uint8, np.uint8))
        self.assertEqual((z["slab_id"].dtype, z["slab_id"].shape), (np.int32, (M,)))
        self.assertEqual(int(z["n_slices"].sum()), M)
        self.assertTrue((np.diff(z["slab_id"]) >= 0).all())
        np.testing.assert_array_equal(np.bincount(z["slab_id"], minlength=S), z["n_slices"])  # slab_id / n_slices consistent
        self.assertTrue(set(np.unique(z["masks"])) <= {0, 1, 2})
        self.assertTrue(os.path.exists(fig))
        pos, changed_total = 0, 0
        for k in range(S):
            host = syn.case_id(str(z["host_vol"][k]))
            self.assertIn(host, hosts)
            self.assertNotIn(host, self.held + self.ph)
            n, z0 = int(z["n_slices"][k]), int(z["z0"][k])
            img0, msk0 = s.load_u8(host)
            sl = slice(pos, pos + n)
            a_img, a_msk = z["images"][sl], z["masks"][sl]
            h_img, h_msk = img0[z0:z0 + n], msk0[z0:z0 + n]
            self.assertEqual(len(h_img), n)
            # labels: only liver (1) -> tumour (2); nothing erased, nothing outside the liver
            diff = a_msk != h_msk
            self.assertTrue((h_msk[diff] == 1).all() and (a_msk[diff] == 2).all())
            self.assertTrue((a_msk[h_msk == 2] == 2).all() and (a_msk[h_msk == 0] == 0).all())
            self.assertEqual(int(diff.sum()), int(z["new_tumor_vox"][k]))
            self.assertGreater(int(z["new_tumor_vox"][k]), 0)
            # image: changed voxels lie inside one insertion window; their count is small relative to the slab
            ch = a_img != h_img
            changed_total += int(ch.sum())
            self.assertTrue(ch.any())
            self.assertLess(ch.mean(), 0.05)  # only the insertion windows change: a small share of the slab's voxels
            self.assertTrue((a_img[~ch] == h_img[~ch]).all())
            pos += n
        print(f"  {S} slabs, {M} slices, {changed_total} changed uint8 voxels; labels only 1->2; unchanged voxels bit-identical; no slice twice per host")

    def test_slabs_of_one_host_visit_never_overlap_in_z(self):
        """Several insertions into one host: windows that overlap in z are merged into one slab, so no slice appears twice."""
        s = self.src()
        for seed in range(6):
            slabs = syn.host_slabs(s, self.big[0], self.toy_samples(), np.random.default_rng(seed), per_host=4, work_mm=1.5)
            spans = sorted((x["z0"], x["z0"] + x["n_slices"]) for x in slabs)
            self.assertTrue(all(b0 <= a1 for (_, b0), (a1, _) in zip(spans, spans[1:])), spans)
            for x in slabs:
                self.assertEqual(x["images"].shape[0], x["n_slices"])
        # the toy liver is roomy enough that some seed puts two insertions into one slab (more new voxels than a single tumour has)
        single = syn.host_slabs(s, self.big[0], self.toy_samples(), np.random.default_rng(0), per_host=1, work_mm=1.5)[0]["new_tumor_vox"]
        merged = [x["new_tumor_vox"] for seed in range(6) for x in syn.host_slabs(s, self.big[0], self.toy_samples(), np.random.default_rng(seed), per_host=4, work_mm=1.5)]
        self.assertTrue(any(v > 1.5 * single for v in merged), (single, merged))

    def test_float_insertion_outside_window_identical(self):
        """The cache path inserts into float HU: outside the window and outside alpha > 0 every value is bit-identical."""
        s = self.src()
        hu, seg, sp, _ = s.load(self.big[0])
        samples = self.toy_samples()
        hu2, seg2, infos = syn.insert_tumors(hu, seg, sp, samples, np.random.default_rng(3), 2, 1.5, keep_float=True)
        self.assertEqual(hu2.dtype, np.float32)
        self.assertGreaterEqual(len(infos), 1)
        sup = np.zeros(hu.shape, bool)
        for i in infos:
            sup[i["window"]] |= i["support"]
        np.testing.assert_array_equal(hu2[~sup], hu[~sup])
        self.assertTrue((hu2 != hu).any())
        d = seg2 != seg
        self.assertTrue((seg[d] == 1).all() and (seg2[d] == 2).all())
        # the segmentation of the window region never leaves the host liver
        for i in infos:
            self.assertTrue((seg[i["window"]][i["support"]] == 1).all())

    def test_accept_until_target_loop(self):
        calls = []

        def attempt(i):
            calls.append(i)
            return [f"s{i}a", f"s{i}b"] if i % 3 == 2 else []  # every third attempt yields two slabs

        out, tried = syn.accept_until_target(attempt, 5, 100)
        self.assertEqual(len(out), 5)  # 3 accepting visits (6 slabs) cut to the target
        self.assertEqual(tried, 9)
        self.assertEqual(calls, list(range(9)))
        out, tried = syn.accept_until_target(lambda i: [], 4, 7)  # attempts exhausted
        self.assertEqual((out, tried), ([], 7))
        out, tried = syn.accept_until_target(lambda i: ["x"], 3, 2)  # budget smaller than the target
        self.assertEqual((len(out), tried), (2, 2))
        self.assertEqual(syn.accept_until_target(lambda i: ["x"], 0, 5), ([], 0))

    def test_run_insert_slabs_exhausts_attempts_with_reasons(self):
        s = self.src()
        out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out_dir, True)
        path = os.path.join(out_dir, "a.npz")
        big = toy_generated(64, (30, 30, 30))  # radius 45 mm at 1.5 mm voxels: no site fits in the toy liver
        samples = (np.stack([big[0]]), np.stack([big[1]]).astype(np.uint8))
        st = syn.run_insert_slabs(s, self.big, samples, path, target=3, per_host=1, max_attempts=5, rng=np.random.default_rng(0), work_mm=1.5)
        self.assertEqual((st["accepted"], st["attempts"]), (0, 5))
        self.assertGreaterEqual(st["insert_rejected"].get("no_site", 0), 5)
        z = syn.load_npz(path)
        self.assertEqual(len(z["images"]), 0)  # a valid, empty slab file
        self.assertEqual(z["images"].shape, (0, 256, 256))
        st = syn.run_insert_slabs(s, self.big, (np.zeros((0, 32, 32, 32), np.float32), np.zeros((0, 32, 32, 32), np.uint8)), path, target=3,
                                  max_attempts=5, rng=np.random.default_rng(0), work_mm=1.5)
        self.assertEqual((st["accepted"], st["attempts"]), (0, 0))  # no samples: nothing is attempted
        with self.assertRaises(AssertionError):
            syn.run_insert_slabs(s, [self.held[0]], self.toy_samples(), path, target=1, rng=np.random.default_rng(0), work_mm=1.5)

    def test_target_reached_early_and_hosts_cycle(self):
        s = self.src()
        out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out_dir, True)
        path = os.path.join(out_dir, "a.npz")
        st = syn.run_insert_slabs(s, self.big, self.toy_samples(), path, target=7, per_host=1, max_attempts=50, rng=np.random.default_rng(0), work_mm=1.5)
        self.assertEqual(st["accepted"], 7)
        self.assertLess(st["attempts"], 50)
        self.assertEqual(st["attempts"], 7)  # one slab per visit with per_host=1 and a roomy liver
        z = syn.load_npz(path)
        names = [str(x) for x in z["host_vol"]]
        self.assertEqual(len(set(names)), len(self.big))  # 4 hosts, 7 visits: every host used, hosts revisited after a full pass
        self.assertTrue(set(range(len(z["images"]))) and len(z["images"]) == int(z["n_slices"].sum()))

    def test_cache_cli_end_to_end_resume_and_summary(self):
        """CLI stages on the toy cache; the 2-epoch GAN is noise so the gate rejects it (accepted 0, empty slab file, summary written);
        then the sample file is replaced by toy patches and insert runs again; LITS_GAN_RESUME=1 keeps finished stages."""
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, True)
        lst = os.path.join(out, "list.json")
        with open(lst, "w") as f:
            json.dump([f"volume-{i}.nii" for i in self.big[:3]] + [f"volume-{self.held[0]}.nii", f"volume-{self.ph[0]}.nii"], f)
        env = dict(os.environ, LITS_GAN_SOURCE="cache", LITS_CACHE_DIR=self.tmp, LITS_PER_VOLUME=self.pv, LITS_GAN_RESULTS_DIR=out,
                   LITS_GAN_VOLUME_LIST=lst, LITS_GAN_PATCH="32", LITS_GAN_WIDTH="8", LITS_GAN_EPOCHS="2", LITS_GAN_NUM_AUG="3",
                   LITS_GAN_NUM_SAMPLES="4", LITS_GAN_GEN_ATTEMPTS="16", LITS_GAN_SAMPLE_EVERY="1", LITS_DEVICE="cpu", LITS_GAN_MAX_ATTEMPTS="6",
                   LITS_GAN_RESUME="1")

        def cli(stage, **kw):
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "Syn_Img_Pipeline_fixed.py")], env=dict(env, LITS_GAN_STAGE=stage, **kw),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            return r
        for stage in ("prepare", "train", "generate", "insert"):
            cli(stage)
        sm = syn.read_json(os.path.join(out, "gan_summary.json"))
        for k in ("patches", "epochs", "accepted", "rejected_reasons", "mask_head_train_dice", "mask_head_heldout_dice", "volumes_used", "seconds"):
            self.assertIn(k, sm)
        self.assertEqual(sm["epochs"], 2)
        self.assertEqual(sm["volumes_listed"], 3)  # held-out and placeholder entries of the list are ignored
        self.assertEqual(sm["volumes_used"], 3)
        self.assertEqual(sm["accepted"], 0)
        self.assertGreater(sum(v for k, v in sm["rejected_reasons"].items() if k.startswith("gate_")), 0)
        self.assertIsNotNone(sm["mask_head_train_dice"])
        z = syn.load_npz(os.path.join(out, "patches.npz"))
        self.assertTrue({int(c) for c in z["meta"][:, 0]} <= set(self.big[:3]))
        self.assertEqual(len(syn.load_npz(os.path.join(out, "aug_slabs.npz"))["images"]), 0)
        # replace the (all rejected) samples by toy patches and rerun insert only
        sp = os.path.join(out, "synthetic", "samples.npz")
        p, m = self.toy_samples(4)
        np.savez(sp, patches=p, masks=m, work_mm=np.float32(1.5))
        os.remove(os.path.join(out, "aug_slabs.npz"))
        cli("insert")
        z = syn.load_npz(os.path.join(out, "aug_slabs.npz"))
        self.assertEqual(len(z["host_vol"]), 3)
        self.assertTrue({syn.case_id(str(h)) for h in z["host_vol"]} <= set(self.big[:3]))  # hosts only from the listed volumes
        sm = syn.read_json(os.path.join(out, "gan_summary.json"))
        self.assertEqual(sm["accepted"], 3)
        self.assertTrue(os.path.exists(os.path.join(out, "figures", "slab_montage.png")))
        # resume: finished stages are kept (patches, GAN epochs, slabs untouched)
        mt = {f: os.path.getmtime(os.path.join(out, f)) for f in ("patches.npz", "aug_slabs.npz")}
        r = cli("all")
        self.assertEqual(mt, {f: os.path.getmtime(os.path.join(out, f)) for f in mt})
        self.assertIn("exists", r.stderr)
        self.assertNotIn("epoch 1/2", r.stderr.split("resumed from")[-1])  # the finished GAN is not retrained
        print("  cache CLI: prepare/train/generate/insert, zero-accept summary, toy-sample insert (3 slabs), resume keeps outputs")

    def test_no_tumour_patches_fails_with_summary(self):
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, True)
        thick = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, thick, True)
        make_toy_cache(thick, range(25), set(self.big), set(self.ph), sz=5.0)  # every volume thicker than LITS_GAN_MAX_SLICE_MM
        env = dict(os.environ, LITS_GAN_SOURCE="cache", LITS_CACHE_DIR=thick, LITS_PER_VOLUME=os.path.join(thick, "per_volume.csv"),
                   LITS_GAN_RESULTS_DIR=out, LITS_GAN_PATCH="32", LITS_GAN_STAGE="prepare", LITS_DEVICE="cpu")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "Syn_Img_Pipeline_fixed.py")], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 4, r.stderr[-1500:])
        sm = syn.read_json(os.path.join(out, "gan_summary.json"))
        self.assertEqual(sm["accepted"], 0)
        self.assertIn("no tumour patches", sm["error"])


@unittest.skipUnless(HAVE_CACHE, "needs data/raw and data/cache_v2")
class TestCacheVsNifti(unittest.TestCase):
    """The cache rebuilt as HU must agree with the raw NIfTI processed exactly like prepare_data_v2 (float resize instead of uint8)."""

    @classmethod
    def setUpClass(cls):
        cls.src = syn.CacheSource(CACHE, PERVOL, 1.5)

    def test_hu_reconstruction_error_and_labels_on_volume_0(self):
        from PIL import Image
        vol, seg, sp, _ = syn.load_case(os.path.join(RAW, "volume-0.nii"), os.path.join(RAW, "segmentation-0.nii"))
        meta = read_csv_rows(os.path.join(CACHE, "meta.csv"))["volume-0.nii"]
        flips = [] if meta["flipped_axes"] == "none" else [int(a[2]) for a in meta["flipped_axes"].split("+")]
        lo, hi = self.src.lo, self.src.hi
        v = np.clip(np.moveaxis(vol, 2, 0), lo, hi).astype(np.float32)
        ref = np.stack([np.asarray(Image.fromarray(x, mode="F").resize((256, 256), Image.BILINEAR)) for x in v])
        lab = np.stack([np.asarray(Image.fromarray(x).resize((256, 256), Image.NEAREST)) for x in np.moveaxis(seg, 2, 0)])
        for a in flips:
            ref, lab = np.flip(ref, a + 1), np.flip(lab, a + 1)
        ref, lab = np.moveaxis(ref, 0, 2), np.moveaxis(lab, 0, 2)
        hu, sg, spc, _ = self.src.load(0)
        err = np.abs(hu - ref)
        inside = (ref > lo + 1) & (ref < hi - 1)
        self.assertTrue(inside.any())
        level = (hi - lo) / 255
        q = {p_: float(np.quantile(err[inside], p_)) for p_ in (0.99, 0.999, 1.0)}
        print(f"  volume-0 HU reconstruction vs float-resized raw, voxels inside the window ({inside.mean():.0%}): "
              f"p99 {q[0.99]:.3f}, p99.9 {q[0.999]:.3f}, max {q[1.0]:.3f} HU, mean {err[inside].mean():.4f} HU (1 uint8 level = {level:.3f} HU)")
        # uint8 quantisation of the source (+-0.5 level) plus PIL's uint8 resize, which rounds after the horizontal AND the vertical pass
        # (+-0.5 level each): the hard bound is 1.5 levels = 2.06 HU; 99.9 % of voxels stay within 1 level (1.37 HU)
        self.assertLessEqual(q[0.999], 1.4)
        self.assertLessEqual(q[1.0], 1.5 * level)
        self.assertLess(float(err[inside].mean()), 0.5)
        np.testing.assert_array_equal(sg, lab)  # labels: same nearest resize + flips -> identical
        np.testing.assert_allclose(spc, (2 * sp[0], 2 * sp[1], sp[2]), atol=1e-3)  # per_volume.csv rounds to 4 decimals
        print(f"  spacing cache {tuple(round(x, 4) for x in spc)} vs 2x raw header {(2 * sp[0], 2 * sp[1], sp[2])}")

    def test_patch_geometry_matches_nifti_path_at_equal_working_spacing(self):
        cid = 27  # thin volume (0.7 mm) in the train part, 12 tumours
        self.assertIn(cid, self.src.train_ids)
        nsrc = syn.NiftiSource(RAW, 1.5)
        nsrc.prepare([cid])
        st = {}
        for name, src in (("cache", self.src), ("nifti", nsrc)):
            vol, seg, sp, _ = src.load(cid)
            sl = syn.liver_crop(seg, sp, (16 + syn.EDGE_MARGIN) * 1.5)
            seg_rs = syn.resample_iso(seg[sl], sp, 0, 1.5)
            st[name] = dict(extent_mm=np.array(seg_rs.shape) * 1.5, liver_ml=(seg_rs > 0).sum() * 1.5 ** 3 / 1000, tumor_ml=(seg_rs == 2).sum() * 1.5 ** 3 / 1000)
        print("  " + "; ".join(f"{k}: extent {v['extent_mm']} mm, liver {v['liver_ml']:.0f} ml, tumour {v['tumor_ml']:.1f} ml" for k, v in st.items()))
        self.assertTrue(np.all(np.abs(st["cache"]["extent_mm"] - st["nifti"]["extent_mm"]) <= 6.0))
        self.assertLess(abs(st["cache"]["liver_ml"] / st["nifti"]["liver_ml"] - 1), 0.05)
        self.assertLess(abs(st["cache"]["tumor_ml"] / st["nifti"]["tumor_ml"] - 1), 0.10)
        with tempfile.TemporaryDirectory() as d:
            outs = {}
            for name, src in (("cache", self.src), ("nifti", nsrc)):
                outs[name] = os.path.join(d, name + ".npz")
                syn.prepare_patches([cid], outs[name], size=32, rng=np.random.default_rng(0), source=src)
            a, b = syn.load_npz(outs["cache"]), syn.load_npz(outs["nifti"])
            self.assertEqual(a["images"].shape[1:], b["images"].shape[1:])
            self.assertTrue(all(m.any() for m in a["masks"]))
            self.assertLessEqual(abs(len(a["images"]) - len(b["images"])), 6)
            ra, rb = a["masks"].sum((1, 2, 3)).mean(), b["masks"].sum((1, 2, 3)).mean()
            print(f"  patches cache {len(a['images'])} / nifti {len(b['images'])}; mean tumour voxels per patch {ra:.0f} / {rb:.0f}")
            self.assertLess(abs(ra / rb - 1), 0.3)
            syn.assert_unit_range(a["images"].astype(np.float32))

    def test_split_and_exclusions_equal_nifti_rule(self):
        self.assertEqual(self.src.train_part_ids(), syn.train_part_ids(RAW))
        ids = self.src.split_cases()
        self.assertEqual(ids, syn.split_cases(RAW, 0))
        self.assertEqual(len(ids), 89)
        self.assertFalse(set(ids) & set(range(28, 48)))
        held = {syn.case_id(n) for n in train_test_split(syn.sorted_volume_names(RAW), test_size=0.2, random_state=42)[1]}
        self.assertFalse(set(ids) & held)
        meta = read_csv_rows(os.path.join(CACHE, "meta.csv"))  # spacing_reliable flag agrees with the per_volume rule
        for n in self.src.names:
            self.assertEqual(self.src.placeholder(syn.case_id(n)), meta[n]["spacing_reliable"] == "False", n)
        print(f"  real cache: {len(ids)} usable train-part volumes, equal to the NIfTI path; placeholder rule equals meta.csv spacing_reliable")

    def test_real_host_insertion_yields_valid_slab(self):
        hosts = [27]
        p, m = toy_generated(32, (7, 6, 6))
        with tempfile.TemporaryDirectory() as t:
            st = syn.run_insert_slabs(self.src, hosts, (p[None], m[None].astype(np.uint8)), os.path.join(t, "s.npz"), target=2, per_host=1,
                                      max_attempts=6, rng=np.random.default_rng(0), work_mm=1.5)
            z = syn.load_npz(os.path.join(t, "s.npz"))
            self.assertEqual(st["accepted"], 2)
            img0, msk0 = self.src.load_u8(27)
            pos = 0
            for k in range(2):
                n, z0 = int(z["n_slices"][k]), int(z["z0"][k])
                a, h = z["images"][pos:pos + n], img0[z0:z0 + n]
                diff = z["masks"][pos:pos + n] != msk0[z0:z0 + n]
                self.assertTrue((msk0[z0:z0 + n][diff] == 1).all() and (z["masks"][pos:pos + n][diff] == 2).all())
                self.assertGreater(int((a != h).sum()), 0)
                pos += n
            print(f"  real host volume-27 (0.7 mm slices): slabs {z['n_slices'].tolist()} slices, +{z['new_tumor_vox'].tolist()} tumour voxels at z0 {z['z0'].tolist()}")


class TestDeviceAndTrainLoop(unittest.TestCase):
    def test_pick_device_with_mocks(self):
        mps_on = mock.patch.object(torch.backends.mps, "is_available", return_value=True)
        with mock.patch.dict(os.environ, {"LITS_DEVICE": "auto"}):
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                self.assertEqual(syn.pick_device().type, "cuda")  # auto prefers cuda
                with mps_on:
                    self.assertEqual(syn.pick_device().type, "cuda")
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                self.assertEqual(syn.pick_device().type, "cpu")
                with mps_on:
                    self.assertEqual(syn.pick_device().type, "cpu")  # mps is opt-in (slower here)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with mock.patch.dict(os.environ, {"LITS_DEVICE": "cuda"}):
                with self.assertRaises(RuntimeError):
                    syn.pick_device()  # explicit cuda on a machine without it must not silently become cpu
            with mock.patch.dict(os.environ, {"LITS_DEVICE": "cpu"}):
                self.assertEqual(syn.pick_device().type, "cpu")
        with mock.patch.dict(os.environ, {"LITS_DEVICE": "mps"}), mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                syn.pick_device()
        with mock.patch.dict(os.environ, {"LITS_DEVICE": "mps"}), mps_on:
            self.assertEqual(syn.pick_device().type, "mps")

    def test_setup_backends_fp32_cudnn_autotune_on_cuda_only(self):
        keep = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        try:
            torch.backends.cudnn.benchmark = False
            syn.setup_backends(torch.device("cpu"))
            self.assertFalse(torch.backends.cudnn.benchmark)
            syn.setup_backends(torch.device("cuda"))  # flags only; no cuda needed
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = keep

    def test_step_losses_stay_on_device_when_sync_off(self):
        net = syn.MultiGAN(DEV, ld=16, patch=32, width=8)
        real, mask = rand_real(4, 32, np.random.default_rng(0))
        net.check_range = False
        net.sync = False
        out = net.train_batch(real, mask)
        self.assertTrue(all(isinstance(v, torch.Tensor) and v.dim() == 0 and not v.requires_grad for v in out.values()))
        net.sync = True
        out = net.train_batch(real, mask)
        self.assertTrue(all(isinstance(v, float) for v in out.values()))
        bad = real * 3
        net.check_range = True
        with self.assertRaises(AssertionError):
            net.train_batch(bad, mask)  # range check still on by default
        net.check_range = False
        net.train_batch(bad, mask)  # off: no per-batch host sync (the whole set was validated once by train())

    def test_patch_store_and_mask_pool_transfer_pattern_with_mocked_cuda(self):
        """Mock .to(cuda) as a recorded no-op to check the cuda branches: device-resident set when it fits, pinned + non_blocking otherwise."""
        arrays = {"images": np.random.default_rng(0).uniform(-1, 1, (6, 8, 8, 8)).astype(np.float16), "masks": np.ones((6, 8, 8, 8), np.uint8),
                  "healthy": np.zeros((3, 8, 8, 8), np.float16)}
        calls = []
        real_to, real_pin = torch.Tensor.to, torch.Tensor.pin_memory

        def fake_to(self, *a, **k):
            if a and isinstance(a[0], torch.device) and a[0].type == "cuda":
                calls.append(("to_cuda", k.get("non_blocking", False)))
                return self
            return real_to(self, *a, **k)

        def fake_pin(self):
            calls.append(("pin", None))
            return self
        cuda = torch.device("cuda")
        with mock.patch.object(torch.Tensor, "to", fake_to), mock.patch.object(torch.Tensor, "pin_memory", fake_pin):
            st = syn.PatchStore(arrays, cuda, limit_mb=64)  # fits: lives on the device, gathers need no copy
            self.assertTrue(st.on_device)
            self.assertEqual(sum(c[0] == "to_cuda" for c in calls), 3)
            calls.clear()
            x = st.gather("images", torch.tensor([0, 2]))
            self.assertEqual(tuple(x.shape), (2, 1, 8, 8, 8))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(calls, [])
            st = syn.PatchStore(arrays, cuda, limit_mb=0)  # too big: pinned host memory, per-batch non_blocking copy
            self.assertFalse(st.on_device)
            self.assertEqual(sum(c[0] == "pin" for c in calls), 3)
            calls.clear()
            st.gather("images", torch.tensor([1, 3]))
            self.assertEqual(calls, [("pin", None), ("to_cuda", True)])
            calls.clear()
            pool = syn.MaskPool(st, np.arange(6), np.arange(3), 4, np.random.default_rng(0), cuda)
            pool.refill(2)
            x, m = pool()
            self.assertEqual(tuple(x.shape), (4, 1, 8, 8, 8))
            self.assertEqual(int((m.flatten(1).sum(1) == 0).sum()), 2 + 0)  # 2 tumour entries (all-ones masks) + 2 healthy entries (all-zero masks)
            for _ in range(5):  # more calls than refilled rows: refills itself
                pool()

    def test_train_resume_and_time_budget(self):
        rng = np.random.default_rng(0)
        n = 12
        imgs, msks = zip(*[toy_liver_patch(rng, True, 16) for _ in range(n)])
        hl = [toy_liver_patch(rng, False, 16)[0] for _ in range(6)]
        with tempfile.TemporaryDirectory() as d:
            pp = os.path.join(d, "p.npz")
            np.savez(pp, images=np.stack(imgs).astype(np.float16), masks=np.stack(msks).astype(np.uint8), healthy=np.stack(hl).astype(np.float16),
                     meta=np.array([(i // 4, 0) for i in range(n)]), healthy_cid=np.array([0, 0, 1, 1, 2, 2]), work_mm=np.float32(1.5))
            syn.train(pp, 2, DEV, d, resume=False, max_seconds=0)
            m = syn.read_json(os.path.join(d, "train_metrics.json"))
            self.assertEqual(m["epochs_done"], 2)
            self.assertIn("mask_head_train", m)
            ck = torch.load(os.path.join(d, "checkpoints", "gan_last.pt"), weights_only=True)
            self.assertEqual(ck["epoch"], 2)
            self.assertIn("opt", ck)
            self.assertFalse(os.path.exists(os.path.join(d, "checkpoints", "gan_last.pt.tmp")))
            before = syn.MultiGAN.load(os.path.join(d, "checkpoints", "gan_last.pt"), DEV)
            with self.assertLogs("gan_fixed", level="INFO") as cm:
                syn.train(pp, 4, DEV, d, resume=True, max_seconds=0)  # continues: epochs 3 and 4 only
            text = "\n".join(cm.output)
            self.assertIn("resumed", text)
            self.assertIn("epoch 3/4", text)
            self.assertNotIn("epoch 1/4", text)
            m = syn.read_json(os.path.join(d, "train_metrics.json"))
            self.assertEqual((m["epochs_done"], m["stopped"]), (4, "max"))
            after = syn.MultiGAN.load(os.path.join(d, "checkpoints", "gan_last.pt"), DEV)
            self.assertFalse(torch.equal(before.dc1_g.net[0].weight, after.dc1_g.net[0].weight))  # it really trained on
            with self.assertLogs("gan_fixed", level="INFO") as cm:  # finished run: nothing to do
                syn.train(pp, 4, DEV, d, resume=True, max_seconds=0)
            self.assertNotIn("epoch 5/4", "\n".join(cm.output))
            self.assertFalse(any("epoch " in o and "/4 (" in o for o in cm.output))
            # time budget: stops after the epoch that crosses it
            d2 = os.path.join(d, "b")
            with self.assertLogs("gan_fixed", level="WARNING") as cm:
                syn.train(pp, 6, DEV, d2, resume=False, max_seconds=1e-6)
            m = syn.read_json(os.path.join(d2, "train_metrics.json"))
            self.assertEqual((m["epochs_done"], m["stopped"]), (1, "time"))
            print("  train: resume continues at epoch 3, finished run is a no-op, time budget stops after epoch 1 (checkpoint saved atomically)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
