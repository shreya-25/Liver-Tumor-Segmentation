"""v1 slice cache: raw LiTS NIfTI volumes -> compact uint8 axial slices (run once).

Same preprocessing as train_orig_AIMUNetALF.load_volume (per-volume min-max, uint8 quantisation, PIL bilinear
resize for images / nearest for masks), but stored as two uint8 arrays so training never touches NIfTI/PIL again.
Kept as the original-preprocessing reference; prepare_data_v2.py is the fixed-window / canonical-orientation cache.

    python prepare_data.py            # data/raw -> data/cache

Env vars: LITS_DATA_DIR (raw volume-*/segmentation-* NIfTI, default data/raw), LITS_CACHE_DIR (default data/cache).
Output: images.npy, masks.npy (N, 256, 256 uint8), vol_index.npy (volume of each slice), volumes.txt (sorted names).
"""
import os
import sys
import time
import numpy as np
import nibabel as nib
from PIL import Image
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; data/ lives here
RAW_DIR = os.environ.get("LITS_DATA_DIR", os.path.join(ROOT, "data", "raw"))
CACHE_DIR = os.environ.get("LITS_CACHE_DIR", os.path.join(ROOT, "data", "cache"))
SIZE = (256, 256)


def volume_pairs(folder):
    # Same pairing rule as the training script: sorted 'vol*' and 'seg*' file names.
    names = os.listdir(folder)
    vols = sorted(os.path.join(folder, f) for f in names if 'vol' in f.lower())
    segs = sorted(os.path.join(folder, f) for f in names if 'seg' in f.lower())
    return vols, segs


def process(args):
    vp, mp = args
    vol = np.asarray(nib.load(vp).dataobj).astype(np.float32)
    seg = np.asarray(nib.load(mp).dataobj).astype(np.uint8)
    vol, seg = np.moveaxis(vol, 2, 0), np.moveaxis(seg, 2, 0)  # raw LiTS is (H, W, Z) -> slice along Z
    vol = (vol + 200) / 400.0  # affine pre-step as in the original script (cancels in the min-max below)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    u8 = (vol * 255).astype(np.uint8)
    imgs = np.stack([np.asarray(Image.fromarray(s).resize(SIZE, Image.BILINEAR)) for s in u8])
    msks = np.stack([np.asarray(Image.fromarray(s).resize(SIZE, Image.NEAREST)) for s in seg])
    return imgs, msks


def main():
    vols, segs = volume_pairs(RAW_DIR)
    assert vols and len(vols) == len(segs), f"found {len(vols)} volumes / {len(segs)} masks in {RAW_DIR}"
    counts = [nib.load(v).shape[2] for v in vols]
    total = sum(counts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    images = np.lib.format.open_memmap(os.path.join(CACHE_DIR, "images.npy"), "w+", np.uint8, (total, *SIZE))
    masks = np.lib.format.open_memmap(os.path.join(CACHE_DIR, "masks.npy"), "w+", np.uint8, (total, *SIZE))
    vol_index = np.repeat(np.arange(len(vols), dtype=np.int16), counts)

    t0, pos = time.time(), 0
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        for i, (imgs, msks) in enumerate(pool.imap(process, zip(vols, segs))):
            images[pos:pos + len(imgs)], masks[pos:pos + len(msks)] = imgs, msks
            pos += len(imgs)
            print(f"[{i + 1}/{len(vols)}] {os.path.basename(vols[i])}: {len(imgs)} slices ({time.time() - t0:.0f}s)", flush=True)
    images.flush(), masks.flush()
    np.save(os.path.join(CACHE_DIR, "vol_index.npy"), vol_index)
    with open(os.path.join(CACHE_DIR, "volumes.txt"), "w") as f:
        f.write("\n".join(os.path.basename(v) for v in vols))
    print(f"Done: {total} slices -> {CACHE_DIR} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
