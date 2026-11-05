"""100 tumour patches (50 small + 50 medium) from checkpoints/gan_best.pt -> synthetic/samples_100.npz, synthetic/patches_100/*.nii, figures."""
import os, sys, time
REPO = r"C:\Users\shrey\Downloads\Liver-Tumor-Segmentation-GitHub\Liver-Tumor-Segmentation-GitHub"
RES = os.path.join(REPO, "results", "gan_lits130")
os.environ.update(LITS_DATA_DIR=os.path.join(REPO, "data", "raw"), LITS_GAN_RESULTS_DIR=RES, LITS_DEVICE="cpu")
sys.path.insert(0, os.path.join(REPO, "Image Synthesis"))
import numpy as np
import torch
import nibabel as nib
import syn_training_improved as T
S = T.S

TARGET, PER_CLASS = 100, {"small": 50, "medium": 50}
S.setup_logging(RES)
torch.manual_seed(2026)
np.random.seed(2026)
model = T.ImprovedMultiGAN.load(os.path.join(RES, "checkpoints", "gan_best.pt"), torch.device("cpu"))
t0, st = time.time(), {}
by_size = T.generate_by_size(model, max(PER_CLASS.values()), tuple(PER_CLASS), S.FUSION, 1.0, max_attempts=30000, stats=st)
by_size = {c: tuple(a[:PER_CLASS[c]] for a in v) for c, v in by_size.items()}
got = {c: len(v[0]) for c, v in by_size.items()}
short = TARGET - sum(got.values())
if short > 0:  # a class ran out of attempts: top up from the other one
    extra = T.generate_by_size(model, short, ("medium" if got["small"] < PER_CLASS["small"] else "small",), S.FUSION, 1.0, max_attempts=30000)
    for c, v in extra.items():
        by_size[c] = tuple(np.concatenate([a, b]) for a, b in zip(by_size[c], v))
out = T.save_samples(os.path.join(RES, "synthetic", "samples_100.npz"), by_size, 1.0)
d = np.load(out)
n = len(d["patches"])
S.write_json_atomic(os.path.join(RES, "synthetic", "samples_100_stats.json"), dict(st, total=n, seconds=round(time.time() - t0), checkpoint="gan_best.pt (epoch 36)"))

nd = os.path.join(RES, "synthetic", "patches_100")
os.makedirs(nd, exist_ok=True)
for i, (p, m, r, c) in enumerate(zip(d["patches"], d["masks"], d["radius_mm"], d["size_class"])):
    tag = f"tumor_{i:03d}_{c}_r{r:.1f}mm"
    nib.save(nib.Nifti1Image(np.rint(S.from_unit(p)).astype(np.int16), np.eye(4)), os.path.join(nd, tag + "_image.nii.gz"))
    nib.save(nib.Nifti1Image((m > 0).astype(np.uint8), np.eye(4)), os.path.join(nd, tag + "_mask.nii.gz"))
rows = [(f"{c} #{j}-{j + len(idx) - 1}", d["patches"][idx], d["masks"][idx] > 0)
        for c in ("small", "medium") for idx0 in [np.flatnonzero(d["size_class"] == c)] for j in range(0, len(idx0), 10) for idx in [idx0[j:j + 10]]]
S.save_montage(os.path.join(RES, "figures", "generated_100.png"), rows)
print(f"DONE: {n} patches {dict(zip(*np.unique(d['size_class'], return_counts=True)))} radius {d['radius_mm'].min():.1f}-{d['radius_mm'].max():.1f} mm "
      f"in {time.time() - t0:.0f}s, tried {st.get('tried')}", flush=True)
