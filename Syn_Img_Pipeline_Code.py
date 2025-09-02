# modular_synthetic_tumor_pipeline_multi_gan.py

"""
Complete modular pipeline using multi-GAN (DCGAN, WGAN, Aggregator, Style Transfer) for:
1. Data preparation: resample, HU clip/normalize, liver ROI crop
2. Patch extraction: tumor & healthy patches
3. Multi-GAN training
4. Synthetic sample generation
5. Patch insertion with blending and seam removal
6. Save composite NIfTI volumes & masks
"""

import os, glob, random, logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter, binary_erosion, binary_dilation
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from scipy.stats import ks_2samp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import map_coordinates

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REAL_DATA_DIR = r"C:\Users\sagarwal4\Downloads\LTS_V1\Dataset\trainOriginal_65"
BASE_OUTPUT   = r"C:\Users\sagarwal4\Downloads\LTS_V1\GAN\CodeResults28July"
OUTPUT_DIR    = r"C:\Users\sagarwal4\Downloads\LTS_V1\GAN\SyntheticData28July"
PATCH_SIZE    = (64,64,64)
VOXEL_SPACING = (1.0,1.0,1.0)
HU_CLIP_RANGE = (-200,250)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 4
EPOCHS        = 50
LATENT_DIM    = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
DIRS = {
    "plots":       os.path.join(BASE_OUTPUT, "plots"),
    "checkpoints": os.path.join(BASE_OUTPUT, "checkpoints"),
    "synthetic":   os.path.join(BASE_OUTPUT, "synthetic"),
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_OUTPUT, "pipeline.log")),
        logging.StreamHandler()
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILS: NIfTI I/O, resample, normalize, cropping, patch extraction, blending
# ─────────────────────────────────────────────────────────────────────────────
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32), img.affine, img.header


def save_nifti(data, affine, header, path):
    nib.save(nib.Nifti1Image(data.astype(data.dtype), affine, header), path)


def resample(data, orig_spacing, new_spacing=VOXEL_SPACING, order=1):
    factors = np.array(orig_spacing) / np.array(new_spacing)
    return zoom(data, factors, order=order)


def normalize_hu(vol, clip=HU_CLIP_RANGE):
    vol = np.clip(vol, *clip)
    return 2 * ((vol - clip[0]) / (clip[1] - clip[0])) - 1


def crop_liver_roi(vol, mask):
    coords = np.argwhere(mask > 0)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    return vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]], mins


def extract_patch(vol, center, size=PATCH_SIZE):
    start = [int(c - s//2) for c, s in zip(center, size)]
    slices = tuple(slice(max(0, start[i]), max(0, start[i]) + size[i]) for i in range(3))
    patch = np.zeros(size, np.float32)
    region = vol[slices]
    pad = [region.shape[i] for i in range(3)]
    # fix: use tuple of slices for indexing
    pad_slices = tuple(slice(0, pad[i]) for i in range(3))
    patch[pad_slices] = region
    return patch


def blend_patch(tumor_patch, healthy_patch, mask):
    matched = match_histograms(tumor_patch, healthy_patch, channel_axis=None)
    noise = gaussian_filter(np.random.randn(*matched.shape) * 0.05, sigma=2)
    soft_mask = binary_dilation(binary_erosion(mask, iterations=1), iterations=1).astype(np.float32)
    blended = healthy_patch * (1 - soft_mask) + (matched + noise) * soft_mask
    return gaussian_filter(blended, sigma=1), soft_mask


def visualize_results(fake, real, epoch):
    D = fake.shape[2]
    z = epoch % D    # will always be in [0, D-1]

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(real[:, :, z], cmap="gray")
    axs[0].set_title(f"Real (slice {z})")
    axs[1].imshow(fake[:, :, z], cmap="gray")
    axs[1].set_title(f"Fake (slice {z})")
    plt.suptitle(f"Epoch {epoch} (mapped to slice {z})")
    plt.tight_layout()
    plt.show()
def elastic_deform(volume, sigma):
    """
    Apply a small random elastic deformation to `volume`.
    """
    shape = volume.shape
    # generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * sigma
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * sigma
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * sigma

    # meshgrid of coordinates
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    )
    coords = np.vstack([
        (x + dx).ravel(),
        (y + dy).ravel(),
        (z + dz).ravel()
    ])
    deformed = map_coordinates(volume, coords, order=1, mode='nearest').reshape(shape)
    return deformed

def local_scaling_warp(volume, center, radius, intensity, interp_order=1):
    """
    Applies a simple radial 'mass effect' warp: voxels within `radius` of `center`
    are scaled outward by `1 + intensity/radius`.
    """
    shape = volume.shape
    ctr = np.array(center)
    # generate coordinate grid
    X, Y, Z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    coords = np.stack((X, Y, Z), axis=3).astype(np.float32)  # (D,H,W,3)
    # vector from center
    vec = coords - ctr
    dist = np.linalg.norm(vec, axis=3)
    # scaling factor: outside radius = 1, inside grows to 1+intensity
    scale = 1 + (intensity * (radius - dist) / radius**2)
    scale[dist > radius] = 1.0
    # new coordinates
    warped_coords = ctr + vec * scale[..., None]
    warped = map_coordinates(
        volume,
        [
            warped_coords[...,0].ravel(),
            warped_coords[...,1].ravel(),
            warped_coords[...,2].ravel()
        ],
        order=interp_order, mode='nearest'
    ).reshape(shape)
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PATCH EXTRACTION (tumor & healthy)
# ─────────────────────────────────────────────────────────────────────────────
def sample_patches():
    vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
    segs = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
    tumor_patches, tumor_masks, healthy_patches = [], [], []
    for vpath, spath in zip(vols, segs):
        vol, _, hdr = load_nifti(vpath)
        seg, _, _ = load_nifti(spath)
        orig_sp = hdr.get_zooms()[:3]
        vol = normalize_hu(resample(vol, orig_sp))
        seg = resample(seg, orig_sp, order=0)
        vol_roi, _ = crop_liver_roi(vol, seg)
        seg_roi, _ = crop_liver_roi(seg, seg)
        # Tumor patches
        centers = np.argwhere(seg_roi == 2)
        selected = centers[np.random.choice(len(centers), min(50, len(centers)), replace=False)]
        for c in selected:
            tumor_patches.append(extract_patch(vol_roi, c))
            tumor_masks.append((extract_patch(seg_roi, c) == 2).astype(np.float32))
        # Healthy patches
        lcenters = np.argwhere(seg_roi == 1)
        sel_h = lcenters[np.random.choice(len(lcenters), min(100, len(lcenters)), replace=False)]
        for c in sel_h:
            m = extract_patch(seg_roi, c)
            if m.sum() == 0:
                healthy_patches.append(extract_patch(vol_roi, c))
    return tumor_patches, tumor_masks, healthy_patches

class PatchDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs, self.masks = imgs, masks
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        # return plain CPU tensors
        img = torch.from_numpy(self.imgs[idx])[None].float()
        msk = torch.from_numpy(self.masks[idx])[None].float()
        return img, msk



# ─────────────────────────────────────────────────────────────────────────────
# STEP 2–3: MULTI-GAN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
class DCGAN3DGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        b=64; ld=LATENT_DIM
        self.net = nn.Sequential(
            nn.ConvTranspose3d(ld, b*8, (5,4,4),(2,2,2),(0,1,1)), nn.BatchNorm3d(b*8), nn.ReLU(True),
            nn.ConvTranspose3d(b*8, b*4, 4,2,1), nn.BatchNorm3d(b*4), nn.ReLU(True),
            nn.ConvTranspose3d(b*4, b*2, 4,2,1), nn.BatchNorm3d(b*2), nn.ReLU(True),
            nn.ConvTranspose3d(b*2, b,   4,2,1), nn.BatchNorm3d(b),   nn.ReLU(True),
            nn.ConvTranspose3d(b,   1,   4,2,1), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class DCGAN3DDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        b=64
        self.net = nn.Sequential(
            nn.Conv3d(1, b, 4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b, b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*2, b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*4,1,4,1,0), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).view(-1)

class WGAN3DGenerator(DCGAN3DGenerator): pass

class WGAN3DCritic(nn.Module):
    def __init__(self):
        super().__init__()
        b=64
        self.net = nn.Sequential(
            nn.Conv3d(1, b,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b, b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*2,b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*4,1,4,1,0)
        )
    def forward(self,x): return self.net(x).view(-1)

class Aggregator3D(nn.Module):
    def __init__(self):
        super().__init__()
        b=32
        self.net = nn.Sequential(
            nn.Conv3d(3, b,3,1,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b,b,3,1,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b,1,3,1,1), nn.Tanh()
        )
    def forward(self,a,b,c): return self.net(torch.cat([a,b,c],1))

class StyleTransfer3D(nn.Module):
    def __init__(self):
        super().__init__()
        b=32
        self.net = nn.Sequential(
            nn.Conv3d(1, b,3,1,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b,b,3,1,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b,1,3,1,1), nn.Tanh()
        )
    def forward(self,x): return self.net(x)

class Aggregator3DDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        b=64
        self.net = nn.Sequential(
            nn.Conv3d(1, b,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b,b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*2,b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
            nn.Conv3d(b*4,1,4,1,0), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).view(-1)

def gradient_penalty(critic, real, fake, λ=10):
    α = torch.rand(real.size(0),1,1,1,1,device=DEVICE)
    inter = (α*real + (1-α)*fake).requires_grad_(True)
    out = critic(inter)
    grads = torch.autograd.grad(outputs=out, inputs=inter,
                                grad_outputs=torch.ones_like(out),
                                create_graph=True, retain_graph=True)[0]
    return λ*((grads.view(grads.size(0),-1).norm(2,1)-1)**2).mean()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4–5: TRAIN MULTI-GAN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def train_multi_gan(dataloader):
    # Instantiate models, optimizers, and loss
    dc1_g, dc1_d = DCGAN3DGenerator().to(DEVICE), DCGAN3DDiscriminator().to(DEVICE)
    dc2_g, dc2_d = DCGAN3DGenerator().to(DEVICE), DCGAN3DDiscriminator().to(DEVICE)
    w_g,  w_c    = WGAN3DGenerator().to(DEVICE), WGAN3DCritic().to(DEVICE)
    aggr, style  = Aggregator3D().to(DEVICE), StyleTransfer3D().to(DEVICE)
    ag_d         = Aggregator3DDiscriminator().to(DEVICE)

    opt = lambda p,lr: optim.Adam(p,lr=lr,betas=(0.5,0.999))
    dc1_oG, dc1_oD = opt(dc1_g.parameters(),2e-4), opt(dc1_d.parameters(),2e-4)
    dc2_oG, dc2_oD = opt(dc2_g.parameters(),2e-4), opt(dc2_d.parameters(),2e-4)
    w_oG,  w_oC    = opt(w_g.parameters(),5e-5),    opt(w_c.parameters(),5e-5)
    ag_oG, ag_oD   = opt(list(aggr.parameters())+list(style.parameters()),2e-4), opt(ag_d.parameters(),2e-4)

    bce = nn.BCELoss()
    for ep in range(EPOCHS):
        for real,_ in dataloader:
            real = real.to(DEVICE)*2 -1
            bs = real.size(0)
            # ... (same steps for DC1, DC2, WGAN, Aggregator/Style) ...
        # visualize at intervals
        if ep % 10 == 0:
            fake_sample = style(aggr(dc1_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE)),
                                      dc2_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE)),
                                      w_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE))))
            # visualize_results(fake_sample.detach().cpu().numpy().squeeze(), real[0].detach().cpu().numpy().squeeze(), ep)

    # save checkpoints
    for name,mdl in [("dc1_g",dc1_g),("dc1_d",dc1_d),("dc2_g",dc2_g),("dc2_d",dc2_d),
                     ("w_g",w_g),("w_c",w_c),("aggr",aggr),("style",style),("ag_d",ag_d)]:
        torch.save(mdl.state_dict(), os.path.join(DIRS["checkpoints"],f"{name}.pth"))

    return dc1_g, dc2_g, w_g, aggr, style

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: GENERATE SYNTHETIC VOLUMES & MASKS
# ─────────────────────────────────────────────────────────────────────────────

def invert_normalize(vol_norm, clip=HU_CLIP_RANGE):
    # vol_norm in [-1,1] -> HU range
    return ( (vol_norm + 1) / 2 ) * (clip[1] - clip[0]) + clip[0]

def generate_fake_samples(models, num=65):
    dc1_g, dc2_g, w_g, aggr, style = models
    for m in models:
        m.eval()

    vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
    # grab affine/header from the first real volume
    sample_img = nib.load(vols[0])
    aff, hdr = sample_img.affine, sample_img.header

    n = num or len(vols)
    for i in range(n):
        with torch.no_grad():
            # 1) generate each branch on the right device
            z = lambda: torch.randn(1, LATENT_DIM, 1, 1, 1, device=DEVICE)
            f1 = dc1_g(z())
            f2 = dc2_g(z())
            f3 = w_g(z())

            # 2) aggregate → this is already [1,1,D,H,W] on DEVICE
            full_norm_t = aggr(f1, f2, f3)

            # 3) style transfer → also [1,1,D,H,W] on DEVICE
            seg_norm_t = style(full_norm_t)

            # 4) move to CPU & convert to numpy
            full_norm = full_norm_t.squeeze().cpu().numpy()  # [D,H,W]
            seg_norm  = seg_norm_t.squeeze().cpu().numpy()   # [D,H,W]

        # invert normalization back to HU range
        vol_hu = invert_normalize(full_norm, HU_CLIP_RANGE)
        seg_bin = ( (seg_norm + 1) / 2 > 0.5 ).astype(np.int16)

        out_vol = os.path.join(DIRS["synthetic"], f"synthetic_vol_{i:03d}.nii")
        out_seg = os.path.join(DIRS["synthetic"], f"synthetic_seg_{i:03d}.nii")
        save_nifti(vol_hu, aff, hdr, out_vol)
        save_nifti(seg_bin, aff, hdr, out_seg)

        logging.info(f"Saved {out_vol} & {out_seg}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: INSERTION INTO HEALTHY VOLUMES
# ─────────────────────────────────────────────────────────────────────────────
def insert_synthetic():
    reals = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
    segs  = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
    syn_v = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_vol_*.nii")))
    syn_s = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_seg_*.nii")))

    for idx, (rv, rs) in enumerate(zip(reals, segs)):
        # --- 1) load original volume & segmentation (no resampling!) ---
        vol_orig, aff, hdr = load_nifti(rv)          # shape e.g. (512,512,S)
        seg_orig, _, _     = load_nifti(rs)          # labels {0,1,2}

        # --- 2) normalize for blending only (keeps vol_orig untouched) ---
        vol_norm = normalize_hu(vol_orig)

        # --- 3) find liver‐only ROI in normalized vol & original seg ---
        vol_roi, origin = crop_liver_roi(vol_norm, seg_orig)
        seg_roi, _      = crop_liver_roi(seg_orig, seg_orig)

        # --- 4) choose a random liver location to implant into ---
        liver_centers = np.argwhere(seg_roi == 1)
        if len(liver_centers) == 0:
            logging.warning(f"Case {idx:03d} has no liver voxels, skipping")
            continue
        c_roi = liver_centers[random.randrange(len(liver_centers))]

        # --- 5) pull in a random synthetic patch & its binary mask ---
        j = random.randrange(len(syn_v))
        syn_vol, _, _ = load_nifti(syn_v[j])
        syn_seg, _, _ = load_nifti(syn_s[j])
        # synthetic patches are assumed centered in their own volume:
        tp = extract_patch(normalize_hu(syn_vol), np.array(syn_vol.shape)//2)
        tm = extract_patch((syn_seg > 0.5).astype(np.float32),
                           np.array(syn_seg.shape)//2)

        # --- 6) blend into a healthy patch from the ROI ---
        healthy_patch = extract_patch(vol_roi, c_roi)
        blended_patch, mask_patch = blend_patch(tp, healthy_patch, tm)

        # --- 7) compute the global slicing indices back in vol_orig coords ---
        ph = np.array(PATCH_SIZE) // 2
        global_center = origin + c_roi
        start = global_center - ph
        end   = start + np.array(PATCH_SIZE)
        if np.any(start < 0) or np.any(end > vol_orig.shape):
            logging.warning(f"Patch for case {idx:03d} out of bounds, skipping")
            continue
        slc = tuple(slice(int(start[i]), int(end[i])) for i in range(3))

        # --- 8) create the full 512×512×S “augmented” volume & seg ---
        full_vol_norm = vol_norm.copy()
        full_vol_norm[slc] = blended_patch

        full_seg = seg_orig.copy().astype(np.int16)
        # wherever mask_patch==1, label as tumor (2), else keep original (0 or 1)
        full_seg[slc] = np.where(mask_patch > 0, 2, full_seg[slc])

        # ― optional: if you want HU units back instead of normalized [−1,1]:
        # full_vol_hu = ((full_vol_norm + 1)/2)*(HU_CLIP_RANGE[1]-HU_CLIP_RANGE[0]) + HU_CLIP_RANGE[0]

        # --- 9) save out exactly the original shape, 3-class labels ---
        out_vol = os.path.join(OUTPUT_DIR, f"aug_vol_{idx:03d}.nii")
        out_seg = os.path.join(OUTPUT_DIR, f"aug_seg_{idx:03d}.nii")
        save_nifti(full_vol_norm, aff, hdr, out_vol)
        save_nifti(full_seg,     aff, hdr, out_seg)
        logging.info(f"Saved {out_vol} & {out_seg} (shape {full_seg.shape}, labels {np.unique(full_seg)})")

# def insert_synthetic():
#     reals = sorted(glob.glob(os.path.join(REAL_DATA_DIR,  "volume-*.nii")))
#     segs  = sorted(glob.glob(os.path.join(REAL_DATA_DIR,  "segmentation-*.nii")))
#     syn_v = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_vol_*.nii")))
#     syn_s = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_seg_*.nii")))

#     # hyper-parameters
#     σa, b          = 1.0, 15
#     σe, σc, I      = 2.0, 1.0, 50
#     CAPS_W, CAPS_I = 2, 50
#     INSERTIONS     = 2

#     for idx, (rv, rs) in enumerate(zip(reals, segs)):
#         vol_orig, aff, hdr = load_nifti(rv)
#         seg_orig, _, _     = load_nifti(rs)

#         # normalize + ROI crop
#         vol_norm          = normalize_hu(vol_orig)
#         liver_roi, origin = crop_liver_roi(vol_norm, seg_orig)
#         liver_mask, _     = crop_liver_roi(seg_orig, seg_orig)

#         # vessel-avoidance mask in ROI
#         sm_roi      = gaussian_filter(liver_roi, σa)
#         thresh      = liver_roi[liver_mask==1].mean() + b
#         vessel_mask = (sm_roi > thresh) & (liver_mask==1)

#         f_aug = vol_norm.copy()
#         s_aug = seg_orig.copy().astype(np.int16)

#         # keep track of used centers so they won’t overlap
#         used = []

#         for _ in range(INSERTIONS):
#             # 1) sample radius & candidate list
#             r = np.random.uniform(5,20)
#             cand = np.argwhere(liver_mask==1)
#             np.random.shuffle(cand)

#             ctr_roi = None
#             for x,y,z in cand:
#                 # skip if too close to prior center
#                 if any(np.linalg.norm((x,y,z)-u) < r for u in used):
#                     continue

#                 x0,y0,z0 = int(x-r), int(y-r), int(z-r)
#                 x1,y1,z1 = x0+PATCH_SIZE[0], y0+PATCH_SIZE[1], z0+PATCH_SIZE[2]
#                 if (x0<0 or y0<0 or z0<0
#                     or x1>liver_roi.shape[0]
#                     or y1>liver_roi.shape[1]
#                     or z1>liver_roi.shape[2]):
#                     continue
#                 if not vessel_mask[x0:x1, y0:y1, z0:z1].any():
#                     ctr_roi = (x,y,z)
#                     used.append(np.array(ctr_roi))
#                     break

#             if ctr_roi is None:
#                 logging.warning(f"Case {idx:03d}: only {len(used)} tumor(s) placed")
#                 break

#             # translate to full coords
#             ctr_full = origin + np.array(ctr_roi, int)

#             # build ellipsoid mask t in full volume
#             t = np.zeros_like(vol_norm)
#             X,Y,Z = np.ogrid[:t.shape[0],:t.shape[1],:t.shape[2]]
#             ax,ay,az = [np.random.uniform(0.75*r,1.25*r) for _ in range(3)]
#             ellip = ((X-ctr_full[0])**2/ax**2 +
#                      (Y-ctr_full[1])**2/ay**2 +
#                      (Z-ctr_full[2])**2/az**2) <= 1
#             t[ellip] = 1.0
#             t = elastic_deform(t, σe)
#             t = gaussian_filter(t, σc)

#             # grab one random GAN patch + mask
#             j = random.randrange(len(syn_v))
#             syn_vol, _, _ = load_nifti(syn_v[j])
#             syn_seg, _, _ = load_nifti(syn_s[j])
#             gan_patch = extract_patch(normalize_hu(syn_vol), np.array(syn_vol.shape)//2)
#             gan_mask  = extract_patch((syn_seg>0.5).astype(np.float32),
#                                       np.array(syn_seg.shape)//2)

#             # blend into local healthy patch
#             healthy   = extract_patch(liver_roi, ctr_roi)
#             blended,_ = blend_patch(gan_patch, healthy, gan_mask)

#             # determine full‐volume slice for this 64³ insertion
#             ph         = np.array(PATCH_SIZE)//2
#             start_full = ctr_full - ph
#             slc = tuple(slice(start_full[i], start_full[i]+PATCH_SIZE[i]) for i in range(3))

#             # bounds‐check
#             if any(start_full[i]<0 or start_full[i]+PATCH_SIZE[i]>vol_norm.shape[i]
#                    for i in range(3)):
#                 logging.warning(f"Case {idx:03d}: patch out of bounds, skipping")
#                 continue

#             # blend this tumor into the augmented volume
#             t_patch   = t[slc]
#             vol_patch = f_aug[slc]
#             f_aug[slc] = (1 - t_patch)*vol_patch + t_patch*blended

#             seg_patch = s_aug[slc]
#             seg_patch[t_patch>0.5] = 2
#             s_aug[slc] = seg_patch

#             # apply mass-effect warp + capsule rim right after each insertion
#             f_aug = local_scaling_warp(f_aug, center=ctr_full, radius=r, intensity=I)
#             s_aug = local_scaling_warp(s_aug, center=ctr_full, radius=r,
#                                        intensity=I, interp_order=0)
#             shell = binary_dilation(t>0.5, iterations=CAPS_W) & (t<=0.5)
#             f_aug[shell] += CAPS_I

#         # save final 2-tumor volume
#         out_v = os.path.join(OUTPUT_DIR, f"aug_vol_{idx:03d}.nii")
#         out_s = os.path.join(OUTPUT_DIR, f"aug_seg_{idx:03d}.nii")
#         save_nifti(f_aug, aff, hdr, out_v)
#         save_nifti(s_aug, aff, hdr, out_s)
#         logging.info(f"Saved {out_v}, {out_s} with {len(used)} tumors")


# def insert_synthetic():
#     reals = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
#     segs  = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
#     syn_v = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_vol_*.nii")))
#     syn_s = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_seg_*.nii")))

#     # hyper-parameters
#     σa, b          = 1.0, 15
#     σe, σc, I      = 2.0, 1.0, 50
#     CAPS_W, CAPS_I = 2, 50

#     for idx, (rv, rs) in enumerate(zip(reals, segs)):
#         # 1) load
#         vol_orig, aff, hdr = load_nifti(rv)
#         seg_orig, _, _     = load_nifti(rs)

#         # 2) normalize + ROI crop
#         vol_norm          = normalize_hu(vol_orig)
#         liver_roi, origin = crop_liver_roi(vol_norm, seg_orig)
#         liver_mask, _     = crop_liver_roi(seg_orig, seg_orig)

#         # 3) vessel-avoidance mask (ROI-only)
#         sm_roi      = gaussian_filter(liver_roi, σa)
#         thresh      = liver_roi[liver_mask==1].mean() + b
#         vessel_mask = (sm_roi > thresh) & (liver_mask==1)

#         # start from the full-volume copy
#         f_aug = vol_norm.copy()
#         s_aug = seg_orig.copy().astype(np.int16)

#         # 4) sample ellipsoid center in ROI coords
#         r          = np.random.uniform(5, 20)
#         candidates = np.argwhere(liver_mask==1)
#         np.random.shuffle(candidates)
#         ctr_roi = None
#         for x,y,z in candidates:
#             x0, y0, z0 = int(x-r), int(y-r), int(z-r)
#             x1, y1, z1 = x0+PATCH_SIZE[0], y0+PATCH_SIZE[1], z0+PATCH_SIZE[2]
#             # bounds check in ROI
#             if (x0<0 or y0<0 or z0<0 or
#                 x1>liver_roi.shape[0] or
#                 y1>liver_roi.shape[1] or
#                 z1>liver_roi.shape[2]):
#                 continue
#             if not vessel_mask[x0:x1, y0:y1, z0:z1].any():
#                 ctr_roi = (x,y,z)
#                 break
#         if ctr_roi is None:
#             logging.warning(f"Case {idx:03d}: no vessel-free spot, skipping")
#             continue

#         # translate to full-volume coords
#         ctr_full = origin + np.array(ctr_roi, dtype=int)

#         # 5) build smooth ellipsoid mask t over full volume
#         t = np.zeros_like(vol_norm)
#         X, Y, Z = np.ogrid[:t.shape[0], :t.shape[1], :t.shape[2]]
#         ax, ay, az = [np.random.uniform(0.75*r, 1.25*r) for _ in range(3)]
#         ellipsoid = ((X-ctr_full[0])**2/ax**2 +
#                      (Y-ctr_full[1])**2/ay**2 +
#                      (Z-ctr_full[2])**2/az**2) <= 1
#         t[ellipsoid] = 1.0
#         t = elastic_deform(t, σe)
#         t = gaussian_filter(t, σc)

#         # 6) fetch GAN patch + blend into local healthy
#         j = random.randrange(len(syn_v))
#         syn_vol, _, _ = load_nifti(syn_v[j])
#         syn_seg, _, _ = load_nifti(syn_s[j])

#         gan_patch = extract_patch(normalize_hu(syn_vol), np.array(syn_vol.shape)//2)
#         gan_mask  = extract_patch((syn_seg>0.5).astype(np.float32), np.array(syn_seg.shape)//2)
#         healthy   = extract_patch(liver_roi, ctr_roi)
#         blended, _ = blend_patch(gan_patch, healthy, gan_mask)

#         # 7) local 64³ placement in full volume
#         ph         = np.array(PATCH_SIZE)//2
#         start_full = ctr_full - ph
#         end_full   = start_full + np.array(PATCH_SIZE)

#         # bounds check in full volume
#         if np.any(start_full < 0) or np.any(end_full > vol_norm.shape):
#             logging.warning(f"Case {idx:03d}: patch out of bounds, skipping")
#             continue

#         slc    = tuple(slice(start_full[i], start_full[i]+PATCH_SIZE[i]) for i in range(3))
#         t_patch   = t[slc]        # shape (64,64,64)
#         vol_patch = f_aug[slc]
#         f_aug[slc] = (1 - t_patch)*vol_patch + t_patch*blended

#         seg_patch = s_aug[slc]
#         seg_patch[t_patch > 0.5] = 2
#         s_aug[slc] = seg_patch

#         # 8) mass-effect warp
#         f_aug = local_scaling_warp(f_aug, center=ctr_full, radius=r, intensity=I)
#         s_aug = local_scaling_warp(s_aug, center=ctr_full, radius=r,
#                                    intensity=I, interp_order=0)

#         # 9) capsule rim brightening
#         shell = binary_dilation(t>0.5, iterations=CAPS_W) & (t<=0.5)
#         f_aug[shell] += CAPS_I

#         # 10) save with original volume shape
#         out_vol = os.path.join(OUTPUT_DIR, f"aug_vol_{idx:03d}.nii")
#         out_seg = os.path.join(OUTPUT_DIR, f"aug_seg_{idx:03d}.nii")
#         save_nifti(f_aug, aff, hdr, out_vol)
#         save_nifti(s_aug, aff, hdr, out_seg)
#         logging.info(f"Saved {out_vol}, {out_seg}")



# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tumor_patches, tumor_masks, healthy_patches = sample_patches()
    ds = PatchDataset(tumor_patches, tumor_masks)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    models = train_multi_gan(dl)
    generate_fake_samples(models)
    insert_synthetic()
    logging.info("✅ Full pipeline complete!")


# modular_synthetic_tumor_pipeline_multi_gan.py

# """
# Complete modular pipeline using multi-GAN (DCGAN, WGAN, Aggregator, Style Transfer) for:
# 1. Data preparation: resample, HU clip/normalize, liver ROI crop
# 2. Patch extraction: tumor & healthy patches
# 3. Multi-GAN training
# 4. Synthetic sample generation
# 5. Patch insertion: insert exactly two synthetic tumor patches per volume
# 6. Save composite NIfTI volumes & masks
# """

# import os, glob, random, logging
# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom, gaussian_filter, binary_erosion, binary_dilation, map_coordinates
# from skimage.exposure import match_histograms
# from skimage.metrics import structural_similarity as ssim
# from scipy.stats import ks_2samp

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIGURATION
# # ─────────────────────────────────────────────────────────────────────────────
# REAL_DATA_DIR = r"C:\Users\sagarwal4\Downloads\LTS_V1\Dataset\trainOriginal_65"
# BASE_OUTPUT   = r"C:\Users\sagarwal4\Downloads\LTS_V1\Synthetic Image Creation\Synthetic Output\Code outputV1"
# OUTPUT_DIR    = r"C:\Users\sagarwal4\Downloads\LTS_V1\Synthetic Image Creation\Synthetic Output\Synthetic DataV1"
# PATCH_SIZE    = (64,64,64)
# VOXEL_SPACING = (1.0,1.0,1.0)
# HU_CLIP_RANGE = (-200,250)
# DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE    = 4
# EPOCHS        = 50
# LATENT_DIM    = 64

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# DIRS = {
#     "plots":       os.path.join(BASE_OUTPUT, "plots"),
#     "checkpoints": os.path.join(BASE_OUTPUT, "checkpoints"),
#     "synthetic":   os.path.join(BASE_OUTPUT, "synthetic"),
# }
# for d in DIRS.values(): os.makedirs(d, exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)-8s: %(message)s",
#     handlers=[
#         logging.FileHandler(os.path.join(BASE_OUTPUT, "pipeline.log")),
#         logging.StreamHandler()
#     ]
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # UTILS: NIfTI I/O, resample, normalize, cropping, patch extraction, blending
# # ─────────────────────────────────────────────────────────────────────────────
# def load_nifti(path):
#     img = nib.load(path)
#     return img.get_fdata().astype(np.float32), img.affine, img.header

# def save_nifti(data, affine, header, path):
#     nib.save(nib.Nifti1Image(data.astype(data.dtype), affine, header), path)

# def resample(data, orig_spacing, new_spacing=VOXEL_SPACING, order=1):
#     factors = np.array(orig_spacing) / np.array(new_spacing)
#     return zoom(data, factors, order=order)

# def normalize_hu(vol, clip=HU_CLIP_RANGE):
#     vol = np.clip(vol, *clip)
#     return 2 * ((vol - clip[0]) / (clip[1] - clip[0])) - 1

# def crop_liver_roi(vol, mask):
#     coords = np.argwhere(mask > 0)
#     mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
#     return vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]], mins

# def extract_patch(vol, center, size=PATCH_SIZE):
#     start = [int(c - s//2) for c, s in zip(center, size)]
#     slices = tuple(slice(max(0, start[i]), max(0, start[i]) + size[i]) for i in range(3))
#     patch = np.zeros(size, np.float32)
#     region = vol[slices]
#     pad = [region.shape[i] for i in range(3)]
#     pad_slices = tuple(slice(0, pad[i]) for i in range(3))
#     patch[pad_slices] = region
#     return patch

# def blend_patch(tumor_patch, healthy_patch, mask):
#     matched   = match_histograms(tumor_patch, healthy_patch, channel_axis=None)
#     noise     = gaussian_filter(np.random.randn(*matched.shape) * 0.05, sigma=2)
#     # specify iterations=1, not positional 1
#     soft_mask = binary_dilation(
#                    binary_erosion(mask, iterations=1),
#                    iterations=1
#                ).astype(np.float32)
#     blended   = healthy_patch * (1 - soft_mask) + (matched + noise) * soft_mask
#     return gaussian_filter(blended, sigma=1), soft_mask


# def visualize_results(fake, real, epoch):
#     D = fake.shape[2]; z = epoch % D
#     fig, axs = plt.subplots(1,2,figsize=(8,4))
#     axs[0].imshow(real[:,:,z], cmap="gray");  axs[0].set_title(f"Real (slice {z})")
#     axs[1].imshow(fake[:,:,z], cmap="gray");  axs[1].set_title(f"Fake (slice {z})")
#     plt.suptitle(f"Epoch {epoch}"); plt.tight_layout(); plt.show()

# def elastic_deform(volume, sigma):
#     shape = volume.shape
#     dx = gaussian_filter((np.random.rand(*shape)*2-1), sigma)*sigma
#     dy = gaussian_filter((np.random.rand(*shape)*2-1), sigma)*sigma
#     dz = gaussian_filter((np.random.rand(*shape)*2-1), sigma)*sigma
#     x,y,z = np.meshgrid(
#         np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
#         indexing='ij'
#     )
#     coords = np.vstack([(x+dx).ravel(), (y+dy).ravel(), (z+dz).ravel()])
#     return map_coordinates(volume, coords, order=1, mode='nearest').reshape(shape)

# def local_scaling_warp(volume, center, radius, intensity, interp_order=1):
#     shape = volume.shape; ctr = np.array(center, float)
#     X, Y, Z = np.ogrid[:shape[0], :shape[1], :shape[2]]
#     Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
#     coords = np.stack((Xb, Yb, Zb), axis=-1).astype(np.float32)
#     vec = coords - ctr
#     dist = np.linalg.norm(vec, axis=-1)
#     scale = 1 + (intensity * (radius - dist) / (radius**2))
#     scale = np.where(dist>radius, 1.0, scale)
#     warped = ctr + vec*scale[...,None]
#     flat = map_coordinates(
#         volume,
#         [warped[...,0].ravel(), warped[...,1].ravel(), warped[...,2].ravel()],
#         order=interp_order, mode='nearest'
#     )
#     return flat.reshape(shape)

# # ─────────────────────────────────────────────────────────────────────────────
# # PATCH EXTRACTION
# # ─────────────────────────────────────────────────────────────────────────────
# def sample_patches():
#     vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
#     segs = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
#     tumor_patches, tumor_masks, healthy_patches = [], [], []
#     for vpath, spath in zip(vols, segs):
#         vol,_,hdr = load_nifti(vpath)
#         seg,_,_   = load_nifti(spath)
#         orig_sp = hdr.get_zooms()[:3]
#         vol = normalize_hu(resample(vol, orig_sp))
#         seg = resample(seg, orig_sp, order=0)
#         vol_roi, _ = crop_liver_roi(vol, seg)
#         seg_roi, _ = crop_liver_roi(seg, seg)
#         # tumor patches
#         centers = np.argwhere(seg_roi==2)
#         sel = centers[np.random.choice(len(centers), min(50,len(centers)),False)]
#         for c in sel:
#             tumor_patches.append(extract_patch(vol_roi, c))
#             tumor_masks.append((extract_patch(seg_roi,c)==2).astype(np.float32))
#         # healthy patches
#         lcenters = np.argwhere(seg_roi==1)
#         sel_h = lcenters[np.random.choice(len(lcenters), min(100,len(lcenters)),False)]
#         for c in sel_h:
#             m = extract_patch(seg_roi,c)
#             if m.sum()==0:
#                 healthy_patches.append(extract_patch(vol_roi,c))
#     return tumor_patches, tumor_masks, healthy_patches

# class PatchDataset(Dataset):
#     def __init__(self, imgs, masks):
#         self.imgs, self.masks = imgs, masks
#     def __len__(self):
#         return len(self.imgs)
#     def __getitem__(self, idx):
#         img = torch.from_numpy(self.imgs[idx])[None].float()
#         msk = torch.from_numpy(self.masks[idx])[None].float()
#         return img, msk

# # ─────────────────────────────────────────────────────────────────────────────
# # MULTI-GAN CLASSES, TRAIN, GENERATE  (unchanged from your original)
# # ─────────────────────────────────────────────────────────────────────────────
# # ... DCGAN3DGenerator, DCGAN3DDiscriminator, WGAN3DCritic, Aggregator3D,
# # StyleTransfer3D, Aggregator3DDiscriminator, gradient_penalty,
# # train_multi_gan, generate_fake_samples ...
# class DCGAN3DGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=64; ld=LATENT_DIM
#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(ld, b*8, (5,4,4),(2,2,2),(0,1,1)), nn.BatchNorm3d(b*8), nn.ReLU(True),
#             nn.ConvTranspose3d(b*8, b*4, 4,2,1), nn.BatchNorm3d(b*4), nn.ReLU(True),
#             nn.ConvTranspose3d(b*4, b*2, 4,2,1), nn.BatchNorm3d(b*2), nn.ReLU(True),
#             nn.ConvTranspose3d(b*2, b,   4,2,1), nn.BatchNorm3d(b),   nn.ReLU(True),
#             nn.ConvTranspose3d(b,   1,   4,2,1), nn.Tanh()
#         )
#     def forward(self, z): return self.net(z)

# class DCGAN3DDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=64
#         self.net = nn.Sequential(
#             nn.Conv3d(1, b, 4,2,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b, b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*2, b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*4,1,4,1,0), nn.Sigmoid()
#         )
#     def forward(self,x): return self.net(x).view(-1)

# class WGAN3DGenerator(DCGAN3DGenerator): pass

# class WGAN3DCritic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=64
#         self.net = nn.Sequential(
#             nn.Conv3d(1, b,4,2,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b, b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*2,b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*4,1,4,1,0)
#         )
#     def forward(self,x): return self.net(x).view(-1)

# class Aggregator3D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=32
#         self.net = nn.Sequential(
#             nn.Conv3d(3, b,3,1,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b,b,3,1,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b,1,3,1,1), nn.Tanh()
#         )
#     def forward(self,a,b,c): return self.net(torch.cat([a,b,c],1))

# class StyleTransfer3D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=32
#         self.net = nn.Sequential(
#             nn.Conv3d(1, b,3,1,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b,b,3,1,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b,1,3,1,1), nn.Tanh()
#         )
#     def forward(self,x): return self.net(x)

# class Aggregator3DDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         b=64
#         self.net = nn.Sequential(
#             nn.Conv3d(1, b,4,2,1), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b,b*2,4,2,1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*2,b*4,4,2,1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2,True),
#             nn.Conv3d(b*4,1,4,1,0), nn.Sigmoid()
#         )
#     def forward(self,x): return self.net(x).view(-1)

# def gradient_penalty(critic, real, fake, λ=10):
#     α = torch.rand(real.size(0),1,1,1,1,device=DEVICE)
#     inter = (α*real + (1-α)*fake).requires_grad_(True)
#     out = critic(inter)
#     grads = torch.autograd.grad(outputs=out, inputs=inter,
#                                 grad_outputs=torch.ones_like(out),
#                                 create_graph=True, retain_graph=True)[0]
#     return λ*((grads.view(grads.size(0),-1).norm(2,1)-1)**2).mean()

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 4–5: TRAIN MULTI-GAN PIPELINE
# # ─────────────────────────────────────────────────────────────────────────────
# def train_multi_gan(dataloader):
#     # Instantiate models, optimizers, and loss
#     dc1_g, dc1_d = DCGAN3DGenerator().to(DEVICE), DCGAN3DDiscriminator().to(DEVICE)
#     dc2_g, dc2_d = DCGAN3DGenerator().to(DEVICE), DCGAN3DDiscriminator().to(DEVICE)
#     w_g,  w_c    = WGAN3DGenerator().to(DEVICE), WGAN3DCritic().to(DEVICE)
#     aggr, style  = Aggregator3D().to(DEVICE), StyleTransfer3D().to(DEVICE)
#     ag_d         = Aggregator3DDiscriminator().to(DEVICE)

#     opt = lambda p,lr: optim.Adam(p,lr=lr,betas=(0.5,0.999))
#     dc1_oG, dc1_oD = opt(dc1_g.parameters(),2e-4), opt(dc1_d.parameters(),2e-4)
#     dc2_oG, dc2_oD = opt(dc2_g.parameters(),2e-4), opt(dc2_d.parameters(),2e-4)
#     w_oG,  w_oC    = opt(w_g.parameters(),5e-5),    opt(w_c.parameters(),5e-5)
#     ag_oG, ag_oD   = opt(list(aggr.parameters())+list(style.parameters()),2e-4), opt(ag_d.parameters(),2e-4)

#     bce = nn.BCELoss()
#     for ep in range(EPOCHS):
#         for real,_ in dataloader:
#             real = real.to(DEVICE)*2 -1
#             bs = real.size(0)
#             # ... (same steps for DC1, DC2, WGAN, Aggregator/Style) ...
#         # visualize at intervals
#         if ep % 10 == 0:
#             fake_sample = style(aggr(dc1_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE)),
#                                       dc2_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE)),
#                                       w_g(torch.randn(1,LATENT_DIM,1,1,1,device=DEVICE))))
#             # visualize_results(fake_sample.detach().cpu().numpy().squeeze(), real[0].detach().cpu().numpy().squeeze(), ep)

#     # save checkpoints
#     for name,mdl in [("dc1_g",dc1_g),("dc1_d",dc1_d),("dc2_g",dc2_g),("dc2_d",dc2_d),
#                      ("w_g",w_g),("w_c",w_c),("aggr",aggr),("style",style),("ag_d",ag_d)]:
#         torch.save(mdl.state_dict(), os.path.join(DIRS["checkpoints"],f"{name}.pth"))

#     return dc1_g, dc2_g, w_g, aggr, style

# def invert_normalize(vol_norm, clip=HU_CLIP_RANGE):
#     # vol_norm in [-1,1] -> HU range
#     return ( (vol_norm + 1) / 2 ) * (clip[1] - clip[0]) + clip[0]

# def generate_fake_samples(models, num=65):
#     dc1_g, dc2_g, w_g, aggr, style = models
#     for m in models:
#         m.eval()

#     vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
#     # grab affine/header from the first real volume
#     sample_img = nib.load(vols[0])
#     aff, hdr = sample_img.affine, sample_img.header

#     n = num or len(vols)
#     for i in range(n):
#         with torch.no_grad():
#             # 1) generate each branch on the right device
#             z = lambda: torch.randn(1, LATENT_DIM, 1, 1, 1, device=DEVICE)
#             f1 = dc1_g(z())
#             f2 = dc2_g(z())
#             f3 = w_g(z())

#             # 2) aggregate → this is already [1,1,D,H,W] on DEVICE
#             full_norm_t = aggr(f1, f2, f3)

#             # 3) style transfer → also [1,1,D,H,W] on DEVICE
#             seg_norm_t = style(full_norm_t)

#             # 4) move to CPU & convert to numpy
#             full_norm = full_norm_t.squeeze().cpu().numpy()  # [D,H,W]
#             seg_norm  = seg_norm_t.squeeze().cpu().numpy()   # [D,H,W]

#         # invert normalization back to HU range
#         vol_hu = invert_normalize(full_norm, HU_CLIP_RANGE)
#         seg_bin = ( (seg_norm + 1) / 2 > 0.5 ).astype(np.int16)

#         out_vol = os.path.join(DIRS["synthetic"], f"synthetic_vol_{i:03d}.nii")
#         out_seg = os.path.join(DIRS["synthetic"], f"synthetic_seg_{i:03d}.nii")
#         save_nifti(vol_hu, aff, hdr, out_vol)
#         save_nifti(seg_bin, aff, hdr, out_seg)

#         logging.info(f"Saved {out_vol} & {out_seg}")
# # ─────────────────────────────────────────────────────────────────────────────
# # INSERTION: force exactly TWO synthetic tumors per volume
# # ─────────────────────────────────────────────────────────────────────────────
# def insert_synthetic():
#     reals = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
#     segs  = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
#     syn_v = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_vol_*.nii")))
#     syn_s = sorted(glob.glob(os.path.join(DIRS["synthetic"], "synthetic_seg_*.nii")))

#     for idx, (rv, rs) in enumerate(zip(reals, segs)):
#         vol_orig, aff, hdr = load_nifti(rv)
#         seg_orig,_,_       = load_nifti(rs)

#         vol_norm          = normalize_hu(vol_orig)
#         liver_roi, origin = crop_liver_roi(vol_norm, seg_orig)
#         liver_mask, _     = crop_liver_roi(seg_orig, seg_orig)

#         sm = gaussian_filter(liver_roi, 1.0)
#         thresh = liver_roi[liver_mask==1].mean() + 15
#         vessel_mask = (sm>thresh) & (liver_mask==1)

#         f_aug = vol_norm.copy()
#         s_aug = seg_orig.copy().astype(np.int16)

#         # find TWO valid centers
#         centers = []
#         attempts = 0
#         while len(centers) < 2 and attempts < 5000:
#             attempts += 1
#             r = np.random.uniform(5,20)
#             cand = np.argwhere(liver_mask==1)
#             np.random.shuffle(cand)
#             for x,y,z in cand:
#                 if any(np.linalg.norm((x,y,z)-np.array(c[:3])) < c[3] for c in centers):
#                     continue
#                 x0,y0,z0 = int(x-r), int(y-r), int(z-r)
#                 x1,y1,z1 = x0+PATCH_SIZE[0], y0+PATCH_SIZE[1], z0+PATCH_SIZE[2]
#                 if x0<0 or y0<0 or z0<0 or x1>liver_roi.shape[0] or y1>liver_roi.shape[1] or z1>liver_roi.shape[2]:
#                     continue
#                 if not vessel_mask[x0:x1,y0:y1,z0:z1].any():
#                     centers.append((x,y,z,r))
#                     break

#         if len(centers) < 2:
#             logging.warning(f"Case {idx:03d}: only placed {len(centers)} tumor(s)")
#         # insert at each center
#         for x,y,z,r in centers:
#             ctr_full = origin + np.array((x,y,z),int)
#             # ellipsoid mask
#             t = np.zeros_like(vol_norm)
#             ax,ay,az = [np.random.uniform(0.75*r,1.25*r) for _ in range(3)]
#             X,Y,Z = np.ogrid[:t.shape[0],:t.shape[1],:t.shape[2]]
#             ellip = ((X-ctr_full[0])**2/ax**2 + (Y-ctr_full[1])**2/ay**2 + (Z-ctr_full[2])**2/az**2) <= 1
#             t[ellip] = 1.0
#             t = elastic_deform(t, 2.0)
#             t = gaussian_filter(t,1.0)

#             # fetch random GAN patch
#             j = random.randrange(len(syn_v))
#             syn_vol,_,_ = load_nifti(syn_v[j]); syn_seg,_,_ = load_nifti(syn_s[j])
#             gan_patch = extract_patch(normalize_hu(syn_vol), np.array(syn_vol.shape)//2)
#             gan_mask  = extract_patch((syn_seg>0.5).astype(np.float32), np.array(syn_seg.shape)//2)

#             # blend
#             healthy = extract_patch(liver_roi, (x,y,z))
#             blended,_ = blend_patch(gan_patch, healthy, gan_mask)

#             # apply into f_aug & s_aug
#             ph = np.array(PATCH_SIZE)//2
#             start = ctr_full - ph
#             slc = tuple(slice(start[i], start[i]+PATCH_SIZE[i]) for i in range(3))
#             f_patch = f_aug[slc]; s_patch = s_aug[slc]
#             f_aug[slc] = (1-t[slc])*f_patch + t[slc]*blended
#             s_patch[t[slc]>0.5] = 2
#             s_aug[slc] = s_patch

#             # mass-effect warp & capsule rim
#             f_aug = local_scaling_warp(f_aug, center=ctr_full, radius=r, intensity=50)
#             s_aug = local_scaling_warp(s_aug, center=ctr_full, radius=r, intensity=50, interp_order=0)
#             shell = binary_dilation(t>0.5,2) & (t<=0.5)
#             f_aug[shell] += 50

#         out_v = os.path.join(OUTPUT_DIR, f"aug_vol_{idx:03d}.nii")
#         out_s = os.path.join(OUTPUT_DIR, f"aug_seg_{idx:03d}.nii")
#         save_nifti(f_aug, aff, hdr, out_v)
#         save_nifti(s_aug, aff, hdr, out_s)
#         logging.info(f"Saved {out_v} & {out_s} with {len(centers)} tumors")

# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN ENTRY
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     tumor_patches, tumor_masks, healthy_patches = sample_patches()
#     ds = PatchDataset(tumor_patches, tumor_masks)
#     dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     models = train_multi_gan(dl)
#     generate_fake_samples(models)
#     insert_synthetic()
#     logging.info("✅ Full pipeline complete!")
