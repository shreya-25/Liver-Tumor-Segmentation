"""50-epoch improved multi-GAN run on the LiTS train part, then size-controlled generation, insertion and evaluation.
Re-running it resumes training from checkpoints/gan_last.pt (saved after every epoch)."""
import os, sys, time, subprocess
REPO = r"C:\Users\shrey\Downloads\Liver-Tumor-Segmentation-GitHub\Liver-Tumor-Segmentation-GitHub"
RES = os.path.join(REPO, "results", "gan_lits130")
os.environ.update(LITS_DATA_DIR=os.path.join(REPO, "data", "raw"), LITS_GAN_RESULTS_DIR=RES, LITS_GAN_EPOCHS="50",
                  LITS_GAN_STEPS_PER_EPOCH="110", LITS_GAN_CKPT_EVERY="1", LITS_GAN_RESUME="1", LITS_DEVICE="cpu", LITS_EVAL_N="128")
sys.path.insert(0, os.path.join(REPO, "Image Synthesis"))
import numpy as np
import torch
import syn_training_improved as T
import insertion_improved as I
S = T.S

S.setup_logging(RES)
S.set_seed(S.SEED)
dev = torch.device("cpu")
t0 = time.time()
T.train_improved(os.path.join(RES, "patches.npz"), 50, dev, RES, resume=True, patience=0)
S.logger.info(f"STAGE train done in {(time.time() - t0) / 3600:.2f} h")

ck = os.path.join(RES, "checkpoints", "gan_best.pt")
ck = ck if os.path.exists(ck) else os.path.join(RES, "checkpoints", "gan_last.pt")
model = T.ImprovedMultiGAN.load(ck, dev)
st = {}
by_size = T.generate_by_size(model, 16, ("small", "medium", "large"), S.FUSION, 1.0, max_attempts=3000, stats=st)
samples = T.save_samples(os.path.join(RES, "synthetic", "samples.npz"), by_size, 1.0)
S.write_json_atomic(os.path.join(RES, "synthetic", "generation_stats.json"), dict(st, checkpoint=os.path.basename(ck)))
rows = [(f"{c} (r {r.min():.1f}-{r.max():.1f} mm)", p[:8], m[:8] > 0) for c, (p, m, r) in by_size.items() if len(p)]
if rows:
    S.save_montage(os.path.join(RES, "figures", "generated_by_size.png"), rows)
S.logger.info(f"STAGE generate done: {st.get('accepted')} from {ck}")

if sum(len(v[0]) for v in by_size.values()):
    hosts = S.split_cases(S.DATA_DIR)[:10]  # train part, real spacing only
    I.run_insert_improved(samples, hosts, os.path.join(RES, "aug"), data_dir=S.DATA_DIR, n=1,
                          params=I.InsertParams(target_radius_mm=10.0), rng=np.random.default_rng(S.SEED))
    S.logger.info("STAGE insert done")
r = subprocess.run([sys.executable, os.path.join(REPO, "Image Synthesis", "evaluate_synthesis.py")], env=dict(os.environ))
S.logger.info(f"STAGE evaluate exit {r.returncode}; ALL DONE in {(time.time() - t0) / 3600:.2f} h")
