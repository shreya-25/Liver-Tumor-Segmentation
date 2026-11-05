"""Builds the v2 (small-tumour) AIM-UNet Colab notebook (2-fold CV on LiTS 0-79, val 80-100, test on synthetic and LiTS 101-130) into ../"""
import json, os

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # experiments/aimunet_colab

COMMON = r'''
# ===== 3. Shared code: files, data, legacy AIM-UNet, loss, metrics, plots =====
import os, re, time, json, csv, random, logging
import numpy as np, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import scipy.ndimage as ndi
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ('background', 'liver', 'tumor')
NUM_WORKERS = 2

def setup_log(path):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(path, mode='a'), logging.StreamHandler()])
    return logging.getLogger('aimunet')

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---- find files by name anywhere under a folder ----
def index_files(root):
    """{file stem: [paths]} for every .nii / .nii.gz under root and its subfolders."""
    assert os.path.isdir(root), f'folder not found: {root}'
    idx = {}
    for d, sub, files in os.walk(root, followlinks=True):
        sub.sort()
        for f in sorted(files):
            stem = f[:-7] if f.endswith('.nii.gz') else f[:-4] if f.endswith('.nii') else None
            if stem:
                idx.setdefault(stem, []).append(os.path.join(d, f))
    return idx

def stems(v):
    """(image stem, mask stem): int = LiTS case id, str = synthetic image name."""
    return (f'volume-{v}', f'segmentation-{v}') if isinstance(v, int) else (v, v + '_label')

def resolve(idx, cases):
    """-> {case: (image path, mask path)}; raises with the list of every missing file."""
    files, missing = {}, []
    for v in cases:
        got = []
        for s in stems(v):
            p = idx.get(s)
            if not p:
                missing.append(s + '.nii[.gz]')
                continue
            if len(p) > 1:
                print(f'{s}: {len(p)} copies, using {p[0]}')
            got.append(p[0])
        if len(got) == 2:
            files[v] = tuple(got)
    if missing:
        msg = f'{len(missing)} files not found: ' + ', '.join(missing)
        if not SKIP_MISSING:
            raise FileNotFoundError(msg + '  (set SKIP_MISSING = True in the settings to continue without these cases)')
        print(f'WARNING: {msg} -> continuing without those cases')
    return files

def find_batch_dirs(root, n):
    """Outermost folders under root named like 'Training Batch <n>' / 'Training_Batch<n>' / 'batch-<n>...'."""
    pat = re.compile(rf'batch[\s_-]*{n}(?!\d)', re.I)
    hits = []
    for d, sub, _ in os.walk(root, followlinks=True):
        sub.sort()
        hits += [os.path.join(d, s) for s in sub if pat.search(s)]
    return [h for h in hits if not any(h != o and h.startswith(o + os.sep) for o in hits)]

def lits_files(ids):
    """LiTS case ids -> {id: (volume path, segmentation path)}: ids in BATCH1_IDS from the 'Training Batch 1' folder,
    all others from 'Training Batch 2', both found under LITS_ROOT (or BATCH1_DIR / BATCH2_DIR if set)."""
    files = {}
    for n, want, fixed in ((1, [i for i in ids if i in BATCH1_IDS], BATCH1_DIR), (2, [i for i in ids if i not in BATCH1_IDS], BATCH2_DIR)):
        if not want:
            continue
        dirs = [fixed] if fixed else find_batch_dirs(LITS_ROOT, n)
        assert dirs, f'no "Training Batch {n}" folder found under {LITS_ROOT}: set BATCH{n}_DIR in cell 1'
        idx = {}
        for d in dirs:
            for s, p in index_files(d).items():
                idx.setdefault(s, []).extend(p)
        print(f'Training Batch {n}: {", ".join(dirs)} -> volumes {want[0]}-{want[-1]} ({len(want)})')
        files.update(resolve(idx, want))
    return files

def name(v):
    return f'lits-{v}' if isinstance(v, int) else v

def host(v):
    """Scan a volume comes from: 'lits-<i>', or the synthetic image's host ('chaos-8', 'lits-105')."""
    if isinstance(v, int):
        return f'lits-{v}'
    m = re.match(r'synthetic_\d+_(.+)_r\d+mm$', v)
    return m.group(1) if m else v

def radius(v):
    m = re.search(r'_r(\d+)mm$', v) if isinstance(v, str) else None
    return int(m.group(1)) if m else None

# ---- data ----
def load_volume(path_pair):
    """-> (CT (Z,SIZE,SIZE) uint8, mask (Z,SIZE,SIZE) uint8, (pixel mm after resizing, slice mm)):
    axial liver slices +- MARGIN. Only that slab of the CT is read."""
    ip, mp = path_pair
    seg_img = nib.load(mp)
    seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    zoom = [float(z) for z in seg_img.header.get_zooms()[:3]]
    zl = np.flatnonzero((seg > 0).any((0, 1)))
    if MARGIN is None or len(zl) == 0:
        z0, z1 = 0, seg.shape[2]
    else:
        z0, z1 = max(zl[0] - MARGIN, 0), min(zl[-1] + MARGIN + 1, seg.shape[2])
    vol = np.asarray(nib.load(ip).dataobj[:, :, z0:z1], dtype=np.float32)
    x = torch.from_numpy(np.clip((vol - WIN_LO) / (WIN_HI - WIN_LO), 0, 1)).permute(2, 0, 1)[:, None]
    y = torch.from_numpy(seg[:, :, z0:z1]).permute(2, 0, 1)[:, None].float()
    x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', antialias=True, align_corners=False)
    y = F.interpolate(y, size=(SIZE, SIZE), mode='nearest')
    x = (x * 255).round().to(torch.uint8)[:, 0].numpy()   # the legacy dataset works on uint8 anyway
    y = y.to(torch.uint8)[:, 0].numpy()
    return x, y, (zoom[0] * seg.shape[0] / SIZE, zoom[2])

def load_all(files, log):
    data = {}
    for i, (v, pp) in enumerate(files.items(), 1):
        data[v] = load_volume(pp)
        x, m, sp = data[v]
        d, _ = lesions(m == 2, sp)
        log.info(f'loaded {i}/{len(files)} {name(v)}: {len(x)} slices, tumour slices {int((m == 2).any((1, 2)).sum())}, '
                 f'{len(d)} tumours (< {SMALL_MM} mm: {int((d < SMALL_MM).sum())}), pixel {sp[0]:.2f} mm')
    return data

def lesions(t, sp):
    """3D connected tumour regions of the boolean volume t -> (equivalent diameters in mm, label volume)."""
    lab, n = ndi.label(t)
    vox = sp[0] * sp[0] * sp[1]
    return (6 * np.bincount(lab.ravel(), minlength=n + 1)[1:] * vox / np.pi) ** (1 / 3), lab

def small_tumour_slices(m, sp):
    """Slices that contain part of a tumour smaller than SMALL_MM (3D equivalent diameter)."""
    d, lab = lesions(m == 2, sp)
    return np.isin(lab, np.flatnonzero(d < SMALL_MM) + 1).any((1, 2))

class LiverDataset(Dataset):
    """Slices of (CT, mask) volumes; input = slices z-CONTEXT..z+CONTEXT as channels (2.5D, edge slices repeated).
    Legacy augmentation: flips + rotation in [-10, 10] deg."""
    def __init__(self, vols, augment=False):
        self.items = [(x, m, z) for x, m in vols for z in range(len(x))]
        self.augment = augment
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        x, m, z = self.items[idx]
        img = torch.from_numpy(x[np.clip(np.arange(z - CONTEXT, z + CONTEXT + 1), 0, len(x) - 1)]).float() / 255
        msk = torch.from_numpy(m[z][None])
        if self.augment:
            if random.random() > 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() > 0.5:
                img, msk = TF.vflip(img), TF.vflip(msk)
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, angle, interpolation=InterpolationMode.NEAREST)
        return img, msk[0].long()

# ---- legacy AIM-UNet ----
class InceptionModule(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_c, in_c, 1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_c, in_c, 1), nn.Conv2d(in_c, in_c, 5, padding=2), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_c, in_c, 1), nn.Conv2d(in_c, in_c, 3, padding=1),
                                     nn.Conv2d(in_c, in_c, 3, padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_c, in_c, 1), nn.BatchNorm2d(in_c), nn.ReLU())
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(), nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU())
    def forward(self, x):
        return self.conv(x)

class AIM_UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc1, self.pool1 = ConvBlock(in_ch, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = ConvBlock(64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = ConvBlock(128, 256), nn.MaxPool2d(2)
        self.enc4, self.pool4 = ConvBlock(256, 512), nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)
        self.inc4, self.up4, self.dec4 = InceptionModule(512), nn.ConvTranspose2d(1024, 512, 2, 2), ConvBlock(512 * 5, 512)
        self.inc3, self.up3, self.dec3 = InceptionModule(256), nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(256 * 5, 256)
        self.inc2, self.up2, self.dec2 = InceptionModule(128), nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(128 * 5, 128)
        self.inc1, self.up1, self.dec1 = InceptionModule(64), nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(64 * 5, 64)
        self.out = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.inc4(e4), self.up4(b)], 1))
        d3 = self.dec3(torch.cat([self.inc3(e3), self.up3(d4)], 1))
        d2 = self.dec2(torch.cat([self.inc2(e2), self.up2(d3)], 1))
        d1 = self.dec1(torch.cat([self.inc1(e1), self.up1(d2)], 1))
        return self.out(d1)

new_model = lambda: AIM_UNet(2 * CONTEXT + 1)

# ---- loss: weighted CE + soft Dice, Tversky for the tumour class (legacy = CE weights 1,2,1 + soft Dice) ----
ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(CE_W, device=device))
TV_B = torch.tensor([0.5, 0.5, TVERSKY_B], device=device)   # weight on missed pixels per class (0.5 = Dice)
def overlap_loss(logits, target, smooth=1e-6):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    tp, sp, so = (probs * one_hot).sum((0, 2, 3)), probs.sum((0, 2, 3)), one_hot.sum((0, 2, 3))
    fp, fn = sp - tp, so - tp
    return 1.0 - ((tp + smooth) / (tp + (1 - TV_B) * fp + TV_B * fn + smooth)).mean()
loss_fn = lambda out, y: ce_loss(out, y) + overlap_loss(out, y)

# ---- metrics ----
def confusion_gpu(pred, target):
    return torch.bincount((target.reshape(-1) * 3 + pred.reshape(-1)), minlength=9).reshape(3, 3).cpu().numpy()

def dice_from(cm):
    """Pixel Dice per class; NaN for a class that is in neither ground truth nor prediction."""
    tp = np.diag(cm)
    return {f'dice_{c}': (float(2 * tp[i] / (cm[i].sum() + cm[:, i].sum())) if cm[i].sum() + cm[:, i].sum() else float('nan'))
            for i, c in enumerate(CLASSES)}

def metrics_from(cm, scores, truth):
    tp = np.diag(cm).astype(float)
    m = dice_from(cm)
    m['accuracy'] = float(tp.sum() / cm.sum())
    m['balanced_accuracy'] = float(np.mean(tp / np.maximum(cm.sum(1), 1)))
    for i, c in enumerate(CLASSES):
        fn, fp = cm[i].sum() - tp[i], cm[:, i].sum() - tp[i]
        m[f'sens_{c}'] = float(tp[i] / max(tp[i] + fn, 1))
        m[f'spec_{c}'] = float((cm.sum() - tp[i] - fn - fp) / max(cm.sum() - tp[i] - fn, 1))
        pos = truth == i
        m[f'auc_{c}'] = float(roc_auc_score(pos, scores[:, i])) if 0 < pos.sum() < len(pos) else float('nan')
    m['auc_macro'] = float(np.nanmean([m[f'auc_{c}'] for c in CLASSES]))
    return m

@torch.no_grad()
def predict_volume(model, x, m, amp, tta=False):
    """-> softmax probabilities (Z,3,H,W) fp16 on the GPU, summed loss. tta = also average the flipped predictions."""
    model.eval()
    P, run = [], 0.0
    for xb, yb in DataLoader(LiverDataset([(x, m)]), batch_size=8, shuffle=False, num_workers=NUM_WORKERS):
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(xb)
        out = out.float()
        run += loss_fn(out, yb).item() * len(xb)
        p = F.softmax(out, 1)
        if tta:
            for dims in ((3,), (2,)):
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    o = model(torch.flip(xb, dims))
                p += torch.flip(F.softmax(o.float(), 1), dims)
            p /= 3
        P.append(p.half())
    return torch.cat(P), run

def postprocess(p, thr=None, fill=False):
    """Probabilities (Z,3,H,W) -> labels (Z,H,W). thr: tumour wherever p(tumour) >= thr (None = argmax, legacy);
    fill: background holes enclosed by liver/tumour in a slice become tumour (dark lesions the model left out)."""
    if thr is None:
        pred = p.argmax(1)
    else:
        pred = p[:, :2].argmax(1)
        pred[p[:, 2] >= thr] = 2
    if fill:
        fg = (pred > 0).cpu().numpy()
        holes = np.stack([ndi.binary_fill_holes(s) for s in fg]) & ~fg
        pred[torch.from_numpy(holes).to(pred.device)] = 2
    return pred

LESION_BINS = ((0, 10), (10, 20), (20, 40), (40, 1e9))   # tumour equivalent diameter, mm
bin_name = lambda a, b: f'lt{b}mm' if a == 0 else f'gt{a}mm' if b > 1e8 else f'{a}_{b}mm'

def lesion_hits(pred_t, true_t, sp):
    """Per ground-truth tumour (3D connected): diameter mm and found (any overlap); + false-positive predicted tumours."""
    d, lab = lesions(true_t, sp)
    hit = np.bincount(lab[pred_t], minlength=len(d) + 1)[1:] > 0
    plab, pn = ndi.label(pred_t)
    return d, hit, pn - len(np.unique(plab[true_t & (plab > 0)]))

@torch.no_grad()
def evaluate(model, vols, data, amp=False, post=None, tta=False):
    """Loss + metrics pooled over every slice of vols, the confusion matrix, ROC scores, and per-volume Dice and tumours found.
    post = {'thr', 'fill'} (None = legacy argmax). AUC / ROC use all tumour pixels plus a random sample of the others."""
    post = post or {}
    frac = min(1.0, AUC_PIXELS / max(sum(len(data[v][0]) for v in vols) * SIZE * SIZE, 1))
    cm, run, n, S, T, per, D, H, FP = np.zeros((3, 3), np.int64), 0.0, 0, [], [], {}, [], [], 0
    for v in vols:
        x, m, sp = data[v]
        y = torch.from_numpy(m).to(device).long()
        p, l = predict_volume(model, x, m, amp, tta)
        run += l; n += len(x)
        pred = postprocess(p, post.get('thr'), post.get('fill', False))
        cv = confusion_gpu(pred, y)
        p, t = p.permute(0, 2, 3, 1).reshape(-1, 3), y.reshape(-1)
        keep = (t == 2) | (torch.rand(t.shape, device=device) < frac)
        S.append(p[keep].float().cpu().numpy()); T.append(t[keep].cpu().numpy())
        d, hit, fp = lesion_hits((pred == 2).cpu().numpy(), m == 2, sp)
        D.append(d); H.append(hit); FP += fp
        cm += cv
        per[name(v)] = {'host': host(v), 'radius_mm': radius(v), 'tumor_pixels_true': int(cv[2].sum()), **dice_from(cv),
                        'tumours': len(d), 'tumours_found': int(hit.sum()), 'false_pos_tumours': fp}
    m = metrics_from(cm, np.concatenate(S), np.concatenate(T))
    m['loss'] = run / n
    D, H = np.concatenate(D), np.concatenate(H)
    m['tumours'], m['fp_tumours_per_volume'] = int(len(D)), FP / len(vols)
    m['tumour_recall'] = float(H.mean()) if len(D) else float('nan')
    for a, b in LESION_BINS:
        s = (D >= a) & (D < b)
        m[f'tumours_{bin_name(a, b)}'] = int(s.sum())
        m[f'tumour_recall_{bin_name(a, b)}'] = float(H[s].mean()) if s.any() else float('nan')
    return m, cm, (np.concatenate(S), np.concatenate(T)), per

@torch.no_grad()
def tune_post(model, vols, data, amp, log):
    """Pick the tumour threshold and/or hole filling (the settings that are 'auto') with the best pooled tumour Dice on vols."""
    thrs = THR_GRID if TUMOR_THR == 'auto' else [TUMOR_THR]
    fills = (False, True) if FILL_HOLES == 'auto' else (FILL_HOLES,)
    combos = [(t, f) for t in thrs for f in fills]
    cms = {c: np.zeros((3, 3), np.int64) for c in combos}
    for v in vols:
        x, m, _ = data[v]
        y = torch.from_numpy(m).to(device).long()
        p, _ = predict_volume(model, x, m, amp, TTA)
        for c in combos:
            cms[c] += confusion_gpu(postprocess(p, *c), y)
    dice = {c: dice_from(cms[c])['dice_tumor'] for c in combos}
    best = max(combos, key=lambda c: -1 if np.isnan(dice[c]) else dice[c])
    log.info('  tumour Dice by (threshold, fill holes): ' + ', '.join(f'({t}, {f}) {dice[(t, f)]:.4f}' for t, f in combos))
    log.info(f'  -> threshold {best[0]}, fill holes {best[1]} (TTA {TTA})')
    return {'thr': best[0], 'fill': best[1], 'tta': TTA, 'tuned_dice_tumor': dice[best]}

def get_post(model, vols, data, amp, log):
    """Prediction settings: TUMOR_THR / FILL_HOLES that are 'auto' are tuned on vols, the others used as set."""
    if 'auto' in (TUMOR_THR, FILL_HOLES):
        return tune_post(model, vols, data, amp, log)
    return {'thr': TUMOR_THR, 'fill': FILL_HOLES, 'tta': TTA}

post_name = lambda p: f"TTA {p.get('tta')}, tumour threshold {'argmax' if p['thr'] is None else p['thr']}, fill holes {p['fill']}"

def fmt(m):
    return (f"loss {m['loss']:.4f} | Dice bg {m['dice_background']:.4f} liver {m['dice_liver']:.4f} tumor {m['dice_tumor']:.4f} | "
            f"acc {m['accuracy']:.4f} bal-acc {m['balanced_accuracy']:.4f} | AUC bg {m['auc_background']:.4f} "
            f"liver {m['auc_liver']:.4f} tumor {m['auc_tumor']:.4f} macro {m['auc_macro']:.4f} | sens tumor {m['sens_tumor']:.4f} | "
            f"tumours found {m['tumour_recall']:.3f} of {m['tumours']} (" +
            ', '.join(f"{bin_name(a, b)} {m[f'tumour_recall_{bin_name(a, b)}']:.3f}/{m[f'tumours_{bin_name(a, b)}']}" for a, b in LESION_BINS) +
            f") | false-pos tumours/vol {m['fp_tumours_per_volume']:.1f}")

def plot_roc(path, sc, title, show=False):
    scores, truth = sc
    plt.figure(figsize=(5, 5))
    for i, c in enumerate(CLASSES):
        pos = truth == i
        if 0 < pos.sum() < len(pos):
            fpr, tpr, _ = roc_curve(pos, scores[:, i])
            plt.plot(fpr, tpr, label=f'{c} (AUC {roc_auc_score(pos, scores[:, i]):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title(title); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=110)
    plt.show() if show else plt.close()

def plot_cm(path, cm, title, show=True):
    """Pixel confusion matrix: counts, and row-normalised (= per-class recall)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    for a, mat, t in ((ax[0], cm, 'pixel counts'), (ax[1], norm, 'row-normalised (recall)')):
        im = a.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f'{mat[i, j]:,}' if mat is cm else f'{mat[i, j]:.3f}', ha='center', va='center',
                       color='white' if norm[i, j] > 0.5 else 'black', fontsize=9)
        a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(CLASSES); a.set_yticklabels(CLASSES)
        a.set_xlabel('predicted'); a.set_ylabel('true'); a.set_title(t)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.suptitle(title)
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.show() if show else plt.close()

def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)

ms = lambda xs: [float(np.nanmean(xs)), float(np.nanstd(xs))] if np.isfinite(xs).any() else [float('nan'), float('nan')]
METRIC_KEYS = ['loss', 'dice_background', 'dice_liver', 'dice_tumor', 'accuracy', 'balanced_accuracy',
               'sens_liver', 'sens_tumor', 'spec_liver', 'spec_tumor', 'auc_background', 'auc_liver', 'auc_tumor', 'auc_macro',
               'tumour_recall'] + [f'tumour_recall_{bin_name(a, b)}' for a, b in LESION_BINS] + ['fp_tumours_per_volume']
'''

DRIVE = r'''
# ===== 1. Connect Google Drive and set paths (EDIT THESE) =====
from google.colab import drive
drive.mount('/content/drive')
'''

PIP = r'''
!pip -q install nibabel scikit-learn
!nvidia-smi -L
'''

# ---------------------------------------------------------------- notebook 1: train
TRAIN_MD = r'''
# AIM-UNet v2 (small-tumour detection): train on LiTS 0-79 (2-fold CV, validation 80-100, 20 epochs), then test on synthetic images and on LiTS 101-130

**Runtime -> Change runtime type -> GPU (T4 or better)** before running.

**What v2 changes for tumour detection, especially small tumours** (all in cell 2, *Small-tumour settings*;
setting each to its "legacy" value gives back the original notebook):

| | legacy | v2 | why |
|---|---|---|---|
| CE class weights (bg, liver, tumour) | 1, 2, 1 | 1, 1, 4 | tumour pixels were weighted lowest -> model too conservative (high precision, low recall) |
| tumour overlap loss | soft Dice | Tversky, beta 0.7 | missed tumour pixels cost more than false ones |
| input | 1 slice | 3 slices (z-1, z, z+1) | a small tumour is round in 3D, a vessel is a tube: neighbour slices tell them apart |
| sampling | tumour slices oversampled | + slices with tumours < 20 mm 3x more | small tumours are rare in the pixels |
| LR | constant 1e-4 | cosine 1e-4 -> 0 | steadier last epochs |
| best epoch | lowest val loss | highest val tumour Dice | the goal is tumour |
| prediction | argmax | flip TTA + tumour threshold and hole filling tuned on validation 80-100 | recall up at little precision cost |
| metrics | pixel Dice | + **tumours found by size** (< 10, 10-20, 20-40, > 40 mm) and false-positive tumours per volume | pixel Dice hides small tumours |

`SIZE = 512` (native CT resolution) is the biggest single gain for small tumours (a 1 cm tumour is ~13 px instead of ~6 px)
but is ~4x slower and needs ~8 GB RAM for the slices: use an A100/L4 GPU with High-RAM. Default stays 256.

**Testing models already trained with the original notebook**: set `OUT_DIR` to that folder and run cells 1-3b, then
Part B / C only. The tumour threshold and hole filling are tuned for those models on validation 80-100 (which is loaded for it),
and results go to `test_*_v2/` beside the old ones. The raw (argmax) scores are logged next to the improved ones.

The notebook has three parts. Run the setup cells (1-3) first, then:
* **Part A - Training**: trains and saves the 2 fold models.
* **Part B - Test on synthetic images**: loads the saved models and tests them.
* **Part C - Test on LiTS 101-130**: loads the saved models and tests them.

Parts B and C only read the saved models from `OUT_DIR`, so they can be run later in a new session
(run cells 1-3, then jump to Part B or C) without training again.

* **Data**: `LITS_ROOT` = the LiTS challenge folder. Volumes 0-27 (`volume-<i>.nii` + `segmentation-<i>.nii`) are read from its
  **Training Batch 1** folder and volumes 28-130 from its **Training Batch 2** folder; both folders are found automatically
  (any depth, names like `Training Batch 1` / `Training_Batch1`), or can be given as `BATCH1_DIR` / `BATCH2_DIR`. Synthetic images
  (`synthetic_XX_<host>_r<R>mm.nii.gz` + `..._label.nii.gz`) under `SYN_ROOT`.
* **Split** (by patient):
  * volumes **0-79** are split into **2 folds** of 40; each fold model trains on 40 and is scored on the other 40 (cross-validation)
  * volumes **80-100** = fixed **validation** set: checked after every epoch; the epoch with the best validation tumour Dice is kept
    (`best.pt`); the tumour threshold / hole filling are also tuned on it
  * volumes **101-130** = test set of Part C (never used in training or validation)
  * synthetic images = test set of Part B (any whose host scan is in 0-100 is left out automatically)
* **Training** = legacy `train_orig_AIMUNetALF.py` AIM-UNet (Adam 1e-4, batch 4, flip/rotate augmentation, 256 px, axial slices,
  fixed HU window -100..250, 20 epochs per fold) with the v2 changes above.
* **Saved** in `OUT_DIR/fold<k>/`: `best.pt` (highest validation tumour Dice), `final.pt` (epoch 20), `last.pt` (resume state),
  confusion matrices + ROC curves; `OUT_DIR/manifest.json` lists the models for Parts B and C.
* **Tests** (Parts B, C): each fold model is scored on the test set; per fold a **confusion matrix plot**, ROC curve, metrics and
  per-volume Dice; then mean +- std over the folds (and, for synthetic images, tumour Dice by tumour size).
  Results go to `OUT_DIR/test_synthetic_v2/` and `OUT_DIR/test_lits101_130_v2/`.
* **Logs**: `train_AIMUNet.log` (per epoch train loss / Dice / accuracy and validation loss, Dice for background, liver,
  tumour, accuracy, balanced accuracy, sensitivity, specificity, ROC AUC, confusion matrix; per fold the best model on 80-100
  and on the held-out 40 volumes; mean +- std over the 2 folds), `test_synthetic.log`, `test_lits101_130.log`.
* **Disconnected?** Run cells 1-3 and Part A again with the same `OUT_DIR`: training continues from the last finished epoch.
'''

TRAIN_PATHS = DRIVE + r'''
LITS_ROOT = '/content/drive/MyDrive/<PATH_TO_LITS_CHALLENGE_FOLDER>'   # LiTS challenge folder holding 'Training Batch 1' and 'Training Batch 2'
# Volumes 0-27 are taken from the 'Training Batch 1' folder and 28-130 from 'Training Batch 2'; both are found
# automatically anywhere under LITS_ROOT. If your folders are named differently, give them here instead of None:
BATCH1_DIR = None   # e.g. '/content/drive/MyDrive/LiTS/Training Batch 1'
BATCH2_DIR = None   # e.g. '/content/drive/MyDrive/LiTS/Training Batch 2'
SYN_ROOT = '/content/drive/MyDrive/<PATH_TO_SYNTHETIC_FOLDER>'   # synthetic images + _label masks (any subfolders)
OUT_DIR   = '/content/drive/MyDrive/<PATH_TO>/aimunet_lits0_100_v2'  # models, logs, plots and test results are written here
# (a NEW folder for v2 training; point it at the old aimunet_lits0_100 folder only to re-test those models in Parts B / C)
'''

TRAIN_SET = r'''
# ===== 2. Settings =====
BATCH1_IDS = set(range(0, 28))       # LiTS ids stored in Training Batch 1 (the rest are in Training Batch 2)
CV_IDS     = list(range(0, 80))      # cross-validation volumes (2 folds of 40)
VAL_IDS    = list(range(80, 101))    # fixed validation volumes, checked every epoch
FOLDS      = 2
EPOCHS     = 20                      # per fold
SIZE       = 256                     # slice size (legacy = 256); 512 = native, best for small tumours (A100/L4 + High-RAM)
BATCH      = 4                       # legacy = 4
LR         = 1e-4
STEPS      = None                    # batches per epoch; None = one pass over the training slices (legacy)
MARGIN     = 3                       # liver slices +- MARGIN are used; None = every slice (legacy, much slower)
USE_AMP    = True                    # mixed precision on GPU (~2x faster); False = fp32 as in the legacy code
SELECT     = 'dice_tumor'            # best epoch by validation 'dice_tumor' or 'loss' (legacy)
SAVE_EVERY_EPOCH = False             # True = also keep epoch_XX.pt for every epoch (227 MB each)
RESUME     = True                    # continue an interrupted run in OUT_DIR
SKIP_MISSING = False                 # True = continue without volumes whose files are not found (a warning lists them)
WIN_LO, WIN_HI = -100.0, 250.0
AUC_PIXELS = 400_000                 # pixels sampled for AUC / ROC (all tumour pixels are always kept)
SEED       = 42
# ---- small-tumour settings (legacy values in brackets) ----
CONTEXT    = 1                       # 2.5D: slices z-CONTEXT..z+CONTEXT as input channels [0 = single slice]
CE_W       = [1.0, 1.0, 4.0]         # cross-entropy weights background, liver, tumour [1, 2, 1]
TVERSKY_B  = 0.7                     # tumour overlap loss: weight on missed tumour pixels, 0.5 = soft Dice [0.5]
SMALL_MM   = 20                      # 'small tumour' = 3D equivalent diameter below this (mm)
SMALL_BOOST = 2.0                    # slices with a small tumour are sampled (1 + SMALL_BOOST)x more than other tumour slices [0]
SCHED      = 'cosine'                # learning rate: 'cosine' (LR -> 0 over EPOCHS) or None (constant) [None]
TTA        = True                    # average with horizontally / vertically flipped predictions [False]
TUMOR_THR  = 'auto'                  # tumour wherever p(tumour) >= threshold; 'auto' = tuned on validation 80-100, a number, or None = argmax [None]
THR_GRID   = [None, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]   # thresholds tried by 'auto'
FILL_HOLES = 'auto'                  # holes inside the predicted liver -> tumour: True / False / 'auto' (= keep if it helps on validation) [False]
# testing (Parts B and C)
TEST_WHICH = 'best'                  # 'best' (best validation epoch) or 'final' (last epoch) model of each fold
TEST_SYN   = 'all'                   # Part B: 'all' synthetic images found in SYN_ROOT, or a list of names
TEST_LITS  = list(range(101, 131))   # Part C: LiTS volumes 101-130
'''

TRAIN_SETUP = r'''
# ===== A1. Find the training / validation files and load the slices =====
from sklearn.model_selection import KFold
os.makedirs(OUT_DIR, exist_ok=True)
AMP = USE_AMP and device.type == 'cuda'
_m = os.path.join(OUT_DIR, 'manifest.json')
if RESUME and os.path.exists(_m):
    _old = json.load(open(_m))
    assert (_old['size'], _old.get('context', 0)) == (SIZE, CONTEXT), \
        f'{OUT_DIR} holds models trained with SIZE {_old["size"]}, CONTEXT {_old.get("context", 0)}: use a new OUT_DIR'
log = setup_log(os.path.join(OUT_DIR, 'train_AIMUNet.log'))
seed_all(SEED)
log.info(f'v2 settings: CONTEXT {CONTEXT} | CE_W {CE_W} | TVERSKY_B {TVERSKY_B} | small < {SMALL_MM} mm boost {SMALL_BOOST} | '
         f'SCHED {SCHED} | SELECT {SELECT} | TTA {TTA} | TUMOR_THR {TUMOR_THR} | FILL_HOLES {FILL_HOLES}')
FILES = lits_files(CV_IDS + VAL_IDS)
CV_IDS, VAL_IDS = [i for i in CV_IDS if i in FILES], [i for i in VAL_IDS if i in FILES]   # minus missing ones (SKIP_MISSING)
for v, (ip, mp) in FILES.items():
    print(f'  {name(v):9s} {ip}')
log.info(f'AIM-UNet (legacy) | CV volumes {CV_IDS[0]}-{CV_IDS[-1]} ({FOLDS} folds) | validation {VAL_IDS[0]}-{VAL_IDS[-1]} | '
         f'{EPOCHS} epochs | {SIZE}px | HU [{WIN_LO}, {WIN_HI}] | {device} amp={AMP}')
data = load_all(FILES, log)
'''

TRAIN_FOLD = r'''
# ===== A2. Train one fold =====
def save_atomic(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)

def meta(k, tr, cvh, ep):
    return {'fold': k, 'epoch': ep, 'train': tr, 'cv_holdout': cvh, 'val': VAL_IDS, 'size': SIZE,
            'window': [WIN_LO, WIN_HI], 'margin': MARGIN, 'context': CONTEXT}

def train_fold(k, tr, cvh):
    fd = os.path.join(OUT_DIR, f'fold{k}')
    done = os.path.join(fd, 'result.json')
    if RESUME and os.path.exists(done):
        log.info(f'[fold {k}] already finished, loaded {done}')
        return json.load(open(done))
    os.makedirs(fd, exist_ok=True)
    seed_all(SEED + k)
    train_ds = LiverDataset([data[v][:2] for v in tr], augment=True)
    has_t = np.concatenate([(data[v][1] == 2).any((1, 2)) for v in tr])
    small = np.concatenate([small_tumour_slices(data[v][1], data[v][2]) for v in tr])
    w = np.where(has_t, (~has_t).sum() / max(has_t.sum(), 1), 1.0)   # legacy tumour-slice oversampling
    w = w * np.where(small, 1 + SMALL_BOOST, 1.0)                     # + small-tumour slices
    n_samples = STEPS * BATCH if STEPS else len(w)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=WeightedRandomSampler(w.tolist(), n_samples, replacement=True),
                          num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    log.info(f'FOLD {k}: train {len(tr)} volumes {tr} ({len(train_ds)} slices, {has_t.sum()} with tumour, {small.sum()} with a '
             f'tumour < {SMALL_MM} mm) | CV hold-out {cvh} | validation {VAL_IDS[0]}-{VAL_IDS[-1]}')
    model = new_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS) if SCHED == 'cosine' else None
    scaler = torch.amp.GradScaler(device.type, enabled=AMP)
    sign = 1.0 if SELECT == 'loss' else -1.0          # lower is better for loss, higher for Dice
    hist, best, best_ep, start = [], float('inf'), 0, 1
    last = os.path.join(fd, 'last.pt')
    if RESUME and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        if sched and ck.get('sched'):
            sched.load_state_dict(ck['sched'])
        hist, best, best_ep, start = ck['hist'], ck['best'], ck['best_epoch'], ck['epoch'] + 1
        log.info(f'[fold {k}] resuming after epoch {ck["epoch"]} (best so far: epoch {best_ep})')
    for ep in range(start, EPOCHS + 1):
        t0 = time.time()
        model.train()
        run, cm = 0.0, np.zeros((3, 3), np.int64)
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device.type, dtype=torch.float16, enabled=AMP):
                out = model(xb)
            out = out.float()
            loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            cm += confusion_gpu(out.detach().argmax(1), yb)
        lr = opt.param_groups[0]['lr']
        if sched:
            sched.step()
        trm = dict(loss=run / len(train_dl), accuracy=float(np.diag(cm).sum() / cm.sum()), lr=lr, **dice_from(cm))
        vm, vcm, vsc, _ = evaluate(model, VAL_IDS, data, AMP)
        score = sign * np.nan_to_num(vm[SELECT], nan=sign * 1e9)   # NaN = worst
        if score < best:
            best, best_ep = score, ep
            save_atomic({'model': model.state_dict(), **meta(k, tr, cvh, ep), 'val_metrics': vm}, os.path.join(fd, 'best.pt'))
        if SAVE_EVERY_EPOCH:
            save_atomic({'model': model.state_dict(), **meta(k, tr, cvh, ep), 'val_metrics': vm}, os.path.join(fd, f'epoch_{ep:02d}.pt'))
        hist.append({'fold': k, 'epoch': ep, **{f'train_{a}': b for a, b in trm.items()}, **{f'val_{a}': b for a, b in vm.items()}})
        save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': ep,
                     'sched': sched.state_dict() if sched else None, 'hist': hist, 'best': best, 'best_epoch': best_ep}, last)
        log.info(f"[fold {k}] Epoch {ep}/{EPOCHS} TRAIN loss {trm['loss']:.4f} | Dice bg {trm['dice_background']:.4f} "
                 f"liver {trm['dice_liver']:.4f} tumor {trm['dice_tumor']:.4f} | acc {trm['accuracy']:.4f} | lr {lr:.2e}")
        log.info(f'[fold {k}] Epoch {ep}/{EPOCHS} VAL   {fmt(vm)}')
        log.info(f'[fold {k}] Epoch {ep}/{EPOCHS} VAL confusion (rows true bg/liver/tumor):\n{vcm}')
        log.info(f'[fold {k}] Epoch {ep}/{EPOCHS} {"* new best * " if best_ep == ep else ""}({time.time() - t0:.0f}s)')
    save_atomic({'model': model.state_dict(), **meta(k, tr, cvh, EPOCHS)}, os.path.join(fd, 'final.pt'))

    # end of fold: tune threshold / hole filling on validation, then best model on validation (80-100) and on the CV hold-out half
    model.load_state_dict(torch.load(os.path.join(fd, 'best.pt'), map_location=device, weights_only=False)['model'])
    log.info(f'[fold {k}] tuning the tumour threshold / hole filling on validation:')
    post = get_post(model, VAL_IDS, data, AMP, log)
    res = {'fold': k, 'best_epoch': best_ep, 'history': hist, 'train': tr, 'cv_holdout': cvh, 'post': post}
    for tag, vols, label in (('val', VAL_IDS, f'validation LiTS {VAL_IDS[0]}-{VAL_IDS[-1]}'), ('cv', cvh, f'CV hold-out ({len(cvh)} LiTS)')):
        m0 = evaluate(model, vols, data, AMP)[0]
        m, c, sc, per = evaluate(model, vols, data, AMP, post, TTA)
        log.info(f'[fold {k}] BEST (epoch {best_ep}) on {label}, raw argmax: {fmt(m0)}')
        log.info(f'[fold {k}] BEST (epoch {best_ep}) on {label}, {post_name(post)}: {fmt(m)}')
        log.info(f'[fold {k}] confusion on {label} ({post_name(post)}):\n{c}')
        plot_cm(os.path.join(fd, f'confusion_{tag}.png'), c, f'Fold {k}, best model (epoch {best_ep}) - {label}\n{post_name(post)}')
        plot_roc(os.path.join(fd, f'roc_{tag}.png'), sc, f'Fold {k} - {label}')
        res[tag] = m; res[f'{tag}_raw'] = m0; res[f'{tag}_confusion'] = c.tolist(); res[f'{tag}_per_volume'] = per
    json.dump(res, open(done, 'w'), indent=1)
    return res
'''

TRAIN_RUN = r'''
# ===== A3. Run the 2 folds =====
t_all = time.time()
results = []
for k, (i_tr, i_ho) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(CV_IDS), 1):
    tr, cvh = [CV_IDS[i] for i in i_tr], [CV_IDS[i] for i in i_ho]
    results.append(train_fold(k, tr, cvh))
    manifest = {'folds': [{'fold': r['fold'], 'best': os.path.join(OUT_DIR, f"fold{r['fold']}", 'best.pt'),
                           'final': os.path.join(OUT_DIR, f"fold{r['fold']}", 'final.pt'), 'best_epoch': r['best_epoch'],
                           'train': r['train'], 'cv_holdout': r['cv_holdout'], 'post': r['post']} for r in results],
                'val': VAL_IDS, 'size': SIZE, 'window': [WIN_LO, WIN_HI], 'margin': MARGIN, 'context': CONTEXT, 'epochs': EPOCHS}
    json.dump(manifest, open(os.path.join(OUT_DIR, 'manifest.json'), 'w'), indent=1)
'''

TRAIN_SUM = r'''
# ===== A4. Training summary over the folds =====
hist = [h for r in results for h in r['history']]
write_csv(os.path.join(OUT_DIR, 'history.csv'), hist)
write_csv(os.path.join(OUT_DIR, 'fold_results.csv'),
          [{'fold': r['fold'], 'best_epoch': r['best_epoch'], **{f'val_{a}': r['val'][a] for a in METRIC_KEYS},
            **{f'cv_{a}': r['cv'][a] for a in METRIC_KEYS}} for r in results])
summary = {'val_80_100_best_model': {a: ms([r['val'][a] for r in results]) for a in METRIC_KEYS},
           'cv_holdout_best_model': {a: ms([r['cv'][a] for r in results]) for a in METRIC_KEYS},
           'cv_holdout_best_model_raw_argmax': {a: ms([r['cv_raw'][a] for r in results]) for a in METRIC_KEYS},
           'post': [r['post'] for r in results],
           'best_epochs': [r['best_epoch'] for r in results], 'minutes_this_session': round((time.time() - t_all) / 60, 1)}
json.dump(summary, open(os.path.join(OUT_DIR, 'summary.json'), 'w'), indent=1)
log.info(f'SUMMARY (best model per fold, mean +- std over {FOLDS} folds; val 80-100 is optimistic: threshold tuned on it)')
for a in METRIC_KEYS:
    v, c, c0 = (summary[s][a] for s in ('val_80_100_best_model', 'cv_holdout_best_model', 'cv_holdout_best_model_raw_argmax'))
    log.info(f'  {a:24s} val 80-100 {v[0]:.4f} +- {v[1]:.4f} | CV hold-out {c[0]:.4f} +- {c[1]:.4f} (raw argmax {c0[0]:.4f})')
plot_cm(os.path.join(OUT_DIR, 'confusion_val_all_folds.png'), sum(np.array(r['val_confusion']) for r in results),
        f'Validation LiTS {VAL_IDS[0]}-{VAL_IDS[-1]}, best models of {FOLDS} folds summed')

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for r in results:
    h, k = r['history'], r['fold']; e = [x['epoch'] for x in h]
    ax[0].plot(e, [x['train_loss'] for x in h], '--', c=f'C{k}'); ax[0].plot(e, [x['val_loss'] for x in h], c=f'C{k}', label=f'fold {k}')
    for j, c in enumerate(('liver', 'tumor')):
        ax[1 + j].plot(e, [x[f'val_dice_{c}'] for x in h], c=f'C{k}', label=f'fold {k}')
    ax[3].plot(e, [x['val_auc_macro'] for x in h], c=f'C{k}', label=f'fold {k}')
    for a in ax:
        a.axvline(r['best_epoch'], c=f'C{k}', ls=':', lw=0.8)
for a, t in zip(ax, ('Loss (dashed train, solid val)', 'Val Dice liver', 'Val Dice tumor', 'Val macro AUC')):
    a.set_title(t); a.set_xlabel('epoch'); a.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'curves.png'), dpi=110); plt.show()
log.info(f'ALL DONE -> {OUT_DIR}  (models: fold<k>/best.pt, listed in manifest.json)')
'''


# ---------------------------------------------------------------- testing (Parts B and C)
TEST_FUNCS = r'''
# ===== 3b. Shared test code (used by Parts B and C) =====
import gc

def load_manifest():
    """Fold models saved by Part A; also sets the preprocessing (SIZE, HU window, MARGIN, CONTEXT) used in training.
    Works for models of the original notebook too (CONTEXT 0)."""
    global SIZE, WIN_LO, WIN_HI, MARGIN, CONTEXT
    p = os.path.join(OUT_DIR, 'manifest.json')
    assert os.path.exists(p), f'{p} not found: run Part A (training) first'
    man = json.load(open(p))
    SIZE, (WIN_LO, WIN_HI), MARGIN, CONTEXT = man['size'], man['window'], man['margin'], man.get('context', 0)
    man['used_hosts'] = {host(v) for f in man['folds'] for v in f['train'] + f['cv_holdout']} | {host(v) for v in man['val']}
    return man

VAL_DATA = None
def fold_post(man, f, model, amp, log):
    """Prediction settings of a fold model: those tuned at the end of training, or (models of the original notebook,
    or TTA changed) tuned now on the validation volumes, which are loaded once for it."""
    global VAL_DATA
    if 'auto' not in (TUMOR_THR, FILL_HOLES):
        return {'thr': TUMOR_THR, 'fill': FILL_HOLES, 'tta': TTA}
    if f.get('post') and f['post'].get('tta') == TTA:
        return f['post']
    if VAL_DATA is None:
        log.info(f'loading validation volumes {man["val"][0]}-{man["val"][-1]} to tune the tumour threshold / hole filling')
        VAL_DATA = load_all(lits_files(man['val']), log)
    log.info(f'[fold {f["fold"]}] tuning on validation:')
    return tune_post(model, [v for v in man['val'] if v in VAL_DATA], VAL_DATA, amp, log)

def model_path(f):
    p = f[TEST_WHICH]
    return p if os.path.exists(p) else os.path.join(OUT_DIR, f'fold{f["fold"]}', os.path.basename(p))  # OUT_DIR moved

def test_models(man, test, files, results_dir, log_name, test_name, show=6):
    """Score every fold model on the test volumes: per fold metrics + confusion matrix + ROC + per-volume Dice,
    then mean +- std over the folds. Writes into results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    amp = USE_AMP and device.type == 'cuda'
    log = setup_log(os.path.join(results_dir, log_name))
    seed_all(SEED)
    log.info(f'{len(man["folds"])} fold models ({TEST_WHICH}) from {OUT_DIR} | test: {test_name}, {len(test)} volumes | {SIZE}px | {device}')
    data = load_all(files, log)
    rows, raw_rows, per_rows, cms, scs, posts = [], [], [], [], [], []
    for f in man['folds']:
        k = f['fold']
        model = new_model().to(device)
        ck = torch.load(model_path(f), map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        post = fold_post(man, f, model, amp, log); posts.append(post)
        m0 = evaluate(model, test, data, amp)[0]
        m, c, sc, per = evaluate(model, test, data, amp, post, post['tta'])
        log.info(f'[fold {k}] {TEST_WHICH} model (epoch {ck["epoch"]}) on {test_name}, raw argmax: {fmt(m0)}')
        log.info(f'[fold {k}] {TEST_WHICH} model (epoch {ck["epoch"]}) on {test_name}, {post_name(post)}: {fmt(m)}')
        log.info(f'[fold {k}] confusion (rows true bg/liver/tumor):\n{c}')
        for v, d in per.items():
            log.info(f"[fold {k}]   {v:30s} Dice liver {d['dice_liver']:.4f} tumor {d['dice_tumor']:.4f} | "
                     f"tumours found {d['tumours_found']}/{d['tumours']}, false-pos {d['false_pos_tumours']}")
        plot_cm(os.path.join(results_dir, f'confusion_fold{k}.png'), c,
                f'Fold {k} {TEST_WHICH} model (epoch {ck["epoch"]}) - {test_name}\n{post_name(post)}')
        plot_roc(os.path.join(results_dir, f'roc_fold{k}.png'), sc, f'Fold {k} - {test_name}', show=True)
        rows.append({'fold': k, 'epoch': ck['epoch'], **m}); raw_rows.append(m0); cms.append(c); scs.append(sc)
        per_rows += [{'fold': k, 'volume': v, **d} for v, d in per.items()]

    write_csv(os.path.join(results_dir, 'test_per_fold.csv'), rows)
    write_csv(os.path.join(results_dir, 'test_per_volume.csv'), per_rows)
    summary = {'test_set': test_name, 'volumes': [name(v) for v in test], 'model': TEST_WHICH, 'post': posts,
               'metrics_mean_std_over_folds': {a: ms([r[a] for r in rows]) for a in METRIC_KEYS},
               'raw_argmax_metrics_mean_std_over_folds': {a: ms([r[a] for r in raw_rows]) for a in METRIC_KEYS},
               'per_volume': {name(v): {c: ms([r[f'dice_{c}'] for r in per_rows if r['volume'] == name(v)]) for c in ('liver', 'tumor')}
                              for v in test}}
    sizes = sorted({r['radius_mm'] for r in per_rows if r['radius_mm'] is not None})
    if sizes:
        summary['tumor_dice_by_radius_mm'] = {r: ms([x['dice_tumor'] for x in per_rows if x['radius_mm'] == r]) for r in sizes}
    json.dump(summary, open(os.path.join(results_dir, 'summary.json'), 'w'), indent=1)
    log.info(f'SUMMARY {test_name}, {TEST_WHICH} model, mean +- std over {len(rows)} folds')
    for a in METRIC_KEYS:
        v, v0 = summary['metrics_mean_std_over_folds'][a], summary['raw_argmax_metrics_mean_std_over_folds'][a]
        log.info(f'  {a:24s} {v[0]:.4f} +- {v[1]:.4f}   (raw argmax {v0[0]:.4f} +- {v0[1]:.4f})')
    for r, d in summary.get('tumor_dice_by_radius_mm', {}).items():
        log.info(f'  tumour Dice, {r:2d} mm tumours: {d[0]:.4f} +- {d[1]:.4f}')
    plot_cm(os.path.join(results_dir, 'confusion_all_folds.png'), sum(cms), f'{test_name}: all fold models summed')
    plot_roc(os.path.join(results_dir, 'roc_all_folds.png'),
             (np.concatenate([s for s, _ in scs]), np.concatenate([t for _, t in scs])), f'{test_name}: fold models pooled', show=True)

    # slice with the most tumour: CT / ground truth / fold-1 prediction
    vols = [v for v in test if (data[v][1] == 2).any()][:show]
    if vols:
        model = new_model().to(device)
        model.load_state_dict(torch.load(model_path(man['folds'][0]), map_location=device, weights_only=False)['model'])
        fig, ax = plt.subplots(len(vols), 3, figsize=(12, 4 * len(vols)), squeeze=False)
        for r, v in enumerate(vols):
            x, m, _ = data[v]
            z = int((m == 2).sum((1, 2)).argmax())
            lo, hi = max(z - CONTEXT, 0), z + CONTEXT + 1   # the slice and its context neighbours
            p = postprocess(predict_volume(model, x[lo:hi], m[lo:hi], amp, posts[0]['tta'])[0], posts[0]['thr'], posts[0]['fill'])
            for a, im, t in zip(ax[r], (x[z] / 255, m[z], p[z - lo].cpu().numpy()), ('CT (HU -100..250)', 'ground truth', 'fold-1 prediction')):
                a.imshow(im, cmap='gray' if t.startswith('CT') else 'viridis', vmin=0, vmax=1 if t.startswith('CT') else 2)
                a.set_title(f'{name(v)}\n{t}' if t.startswith('CT') else t); a.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, 'examples.png'), dpi=90); plt.show()
    log.info(f'ALL DONE -> {results_dir}')
    del data; gc.collect()
    return summary
'''

PART_A_MD = r'''
---
## Part A — Training (2-fold CV on LiTS 0-79, validation on LiTS 80-100, 20 epochs per fold)
'''

PART_B_MD = r'''
---
## Part B — Test the trained models on the synthetic images
Needs only cells 1-3b and the models saved by Part A (it can run in a new session).
'''

PART_B = r'''
# ===== B. Test on synthetic images =====
data = None; gc.collect()                       # free the training slices if Part A ran in this session
man = load_manifest()
sidx = index_files(SYN_ROOT)
cand = sorted(s for s in sidx if s.startswith('synthetic_') and s + '_label' in sidx) if TEST_SYN == 'all' else list(TEST_SYN)
TEST_B = [s for s in cand if host(s) not in man['used_hosts']]
if len(TEST_B) < len(cand):
    print('left out (host scan used in training / validation):', [s for s in cand if host(s) in man['used_hosts']])
assert TEST_B, f'no synthetic test images found under {SYN_ROOT}'
summary_B = test_models(man, TEST_B, resolve(sidx, TEST_B), os.path.join(OUT_DIR, 'test_synthetic_v2'),
                        'test_synthetic.log', 'synthetic images')
'''

PART_C_MD = r'''
---
## Part C — Test the trained models on original LiTS volumes 101-130
Needs only cells 1-3b and the models saved by Part A (it can run in a new session).
'''

PART_C = r'''
# ===== C. Test on LiTS 101-130 =====
data = None; gc.collect()
man = load_manifest()
TEST_C = list(TEST_LITS)
assert not [v for v in TEST_C if host(v) in man['used_hosts']], 'some test volumes were used in training / validation'
FILES_C = lits_files(TEST_C)
TEST_C = [i for i in TEST_C if i in FILES_C]
summary_C = test_models(man, TEST_C, FILES_C, os.path.join(OUT_DIR, 'test_lits101_130_v2'),
                        'test_lits101_130.log', f'LiTS {TEST_C[0]}-{TEST_C[-1]}')
'''


def notebook(path, cells):
    nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for kind, src in cells:
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(nb, open(path, "w"), indent=1)
    print(path)


notebook(os.path.join(OUT, "AIMUNet_v2_SmallTumour_Train_LiTS0-100_Test_Synthetic_LiTS101-130.ipynb"),
         [("markdown", TRAIN_MD), ("code", TRAIN_PATHS), ("code", PIP), ("code", TRAIN_SET), ("code", COMMON), ("code", TEST_FUNCS),
          ("markdown", PART_A_MD), ("code", TRAIN_SETUP), ("code", TRAIN_FOLD), ("code", TRAIN_RUN), ("code", TRAIN_SUM),
          ("markdown", PART_B_MD), ("code", PART_B),
          ("markdown", PART_C_MD), ("code", PART_C)])
