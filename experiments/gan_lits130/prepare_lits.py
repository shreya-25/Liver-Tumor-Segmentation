"""Extract tumour patches from all usable LiTS volumes (both folders, linked into data/raw)."""
import os, sys, glob, time
REPO = r"C:\Users\shrey\Downloads\Liver-Tumor-Segmentation-GitHub\Liver-Tumor-Segmentation-GitHub"
os.environ.update(LITS_DATA_DIR=os.path.join(REPO, "data", "raw"), LITS_GAN_RESULTS_DIR=os.path.join(REPO, "results", "gan_lits130"),
                  LITS_GAN_CAP_COMP="8", LITS_GAN_CAP_VOL="48", LITS_GAN_MAX_SLICE_MM="0")
sys.path.insert(0, os.path.join(REPO, "Image Synthesis"))
import numpy as np
import Syn_Img_Pipeline_fixed as S

S.setup_logging(S.RESULTS_DIR)
S.set_seed(S.SEED)
paths = {S.case_id(p): p for p in glob.glob(os.path.join(S.DATA_DIR, "volume-*.nii*"))}
ids = sorted(i for i, p in paths.items() if not S.placeholder_spacing(p))
held = sorted(set(paths) - set(S.train_part_ids(S.DATA_DIR)))
S.logger.info(f"{len(paths)} volumes, {len(ids)} with real spacing (placeholder dropped: {sorted(set(paths) - set(ids))}); "
              f"of those, in the segmenter's 80/20 held-out part: {[i for i in ids if i in held]}")
t0 = time.time()
S.prepare_patches(ids, os.path.join(S.RESULTS_DIR, "patches.npz"), source=S.NiftiSource(S.DATA_DIR, 1.0))
z = S.load_npz(os.path.join(S.RESULTS_DIR, "patches.npz"))
S.logger.info(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(z['images'])} tumour patches, {len(z['healthy'])} healthy, "
              f"{len(np.unique(z['meta'][:, 0]))} volumes, {len({tuple(r) for r in z['meta']})} tumour components")
