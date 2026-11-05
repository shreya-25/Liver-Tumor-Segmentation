"""Legacy AIM-UNet (legacy/train_orig_AIMUNetALF.py): train on 10 original LiTS volumes, test on 4 synthetic volumes.

Split (by volume):
  train / validation = 10 real LiTS volumes in 5-fold cross-validation (each fold: 8 train, 2 val)
  test               = 4 synthetic volumes from results/synthetic_healthy40 (never seen in training; their hosts
                       are not among the 10 LiTS training volumes)
Each fold trains 4 epochs, saves the weights after every epoch, and its final model is scored on the test volumes
(pooled over the 4 and per volume).

Taken unchanged from the legacy file: AIM_UNet, the loss (CE with weights [1,2,1] + soft Dice), Adam lr 1e-4,
batch 4, the PIL flip/rotate augmentation (LiverDataset.__getitem__) and the tumour-slice oversampling.
Changed (see README.md): axial slices, fixed HU window -100..250, 128 px, liver slices +- MARGIN only and STEPS
batches per epoch (CPU only: 256 px costs ~4 s per slice).
Env: AIM_OUT, AIM_EPOCHS, AIM_STEPS, AIM_SIZE, AIM_FOLDS.
"""
import os, time, json, csv, logging, random, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import nibabel as nib
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"C:\Users\shrey\Downloads\Liver-Tumor-Segmentation-GitHub\Liver-Tumor-Segmentation-GitHub"
LITS_DIR = os.path.join(REPO, "data", "raw")
SYN_DIR = os.path.join(REPO, "results", "synthetic_healthy40", "images")
OUT = os.environ.get("AIM_OUT", os.path.join(REPO, "results", "aimunet_lits10_syn4"))
EPOCHS, STEPS = int(os.environ.get("AIM_EPOCHS", 4)), int(os.environ.get("AIM_STEPS", 75))
SIZE, FOLDS, BATCH, LR = int(os.environ.get("AIM_SIZE", 128)), int(os.environ.get("AIM_FOLDS", 5)), 4, 1e-4
WIN_LO, WIN_HI, MARGIN, AUC_PIXELS, SEED = -100.0, 250.0, 3, 400_000, 42
CLASSES = ("background", "liver", "tumor")
# 10 LiTS Training Batch 1 volumes (ids 0-22), all with tumour, mixed tumour size and slice thickness; none is a test host
TRAIN_LITS = [0, 2, 6, 8, 10, 13, 14, 16, 19, 21]
# 4 synthetic test images: 3 CHAOS hosts + 1 LiTS host, tumour radius 5 / 10 / 15 / 15 mm
TEST_SYN = ["synthetic_02_chaos-2_r10mm", "synthetic_03_chaos-8_r15mm", "synthetic_10_chaos-24_r5mm",
            "synthetic_18_lits-91_r15mm"]

# the legacy module configures logging at import: point it at OUT so its log file is ours
os.makedirs(OUT, exist_ok=True)
os.environ["LITS_RESULTS_DIR"] = OUT
_spec = importlib.util.spec_from_file_location("legacy_aimunet", os.path.join(REPO, "legacy", "train_orig_AIMUNetALF.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
log = logging.getLogger("aimunet_cv")
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def paths(v):
    if isinstance(v, int):
        return os.path.join(LITS_DIR, f"volume-{v}.nii"), os.path.join(LITS_DIR, f"segmentation-{v}.nii")
    return os.path.join(SYN_DIR, v + ".nii.gz"), os.path.join(SYN_DIR, v + "_label.nii.gz")


def name(v):
    return f"lits-{v}" if isinstance(v, int) else v


def load_volume(v):
    """-> list of (image (SIZE,SIZE,1) float [0,1], mask (SIZE,SIZE,1) uint8); axial liver slices +- MARGIN."""
    ip, mp = paths(v)
    vol = np.asarray(nib.load(ip).dataobj, dtype=np.float32)
    seg = np.asarray(nib.load(mp).dataobj).astype(np.uint8)
    z = np.flatnonzero((seg > 0).any((0, 1)))
    z = np.arange(max(z[0] - MARGIN, 0), min(z[-1] + MARGIN + 1, seg.shape[2]))
    x = torch.from_numpy(np.clip((vol[:, :, z] - WIN_LO) / (WIN_HI - WIN_LO), 0, 1)).permute(2, 0, 1)[:, None]
    y = torch.from_numpy(seg[:, :, z]).permute(2, 0, 1)[:, None].float()
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    y = F.interpolate(y, size=(SIZE, SIZE), mode="nearest")
    return list(zip(x.permute(0, 2, 3, 1).numpy(), y.permute(0, 2, 3, 1).numpy().astype(np.uint8)))


class SliceDataset(L.LiverDataset):
    """The legacy dataset (same __getitem__ / augmentation) over slices that are already loaded."""
    def __init__(self, data, augment=False):
        self.data, self.augment = data, augment


DICE = None
def loss_fn(out, y):
    global DICE
    if DICE is None:  # the legacy loss: CE weights [1, 2, 1] + soft Dice (DiceLoss is local to legacy train_model)
        ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0], device=device))
        def dice(logits, target, smooth=1e-6):
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
            inter, union = (probs * one_hot).sum((0, 2, 3)), probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
            return 1.0 - ((2 * inter + smooth) / (union + smooth)).mean()
        DICE = lambda o, t: ce(o, t) + dice(o, t)
    return DICE(out, y)


def dice_from(cm):
    tp = np.diag(cm)
    return {f"dice_{c}": float(2 * tp[i] / max(cm[i].sum() + cm[:, i].sum(), 1)) for i, c in enumerate(CLASSES)}


def metrics_from(cm, scores, truth):
    """cm (3x3 true x pred), scores (n, 3) softmax / truth (n,) on a pixel subsample -> dict."""
    tp = np.diag(cm).astype(float)
    m = dice_from(cm)
    m["accuracy"] = float(tp.sum() / cm.sum())
    m["balanced_accuracy"] = float(np.mean(tp / np.maximum(cm.sum(1), 1)))
    for i, c in enumerate(CLASSES):
        fn, fp = cm[i].sum() - tp[i], cm[:, i].sum() - tp[i]
        m[f"sens_{c}"], m[f"spec_{c}"] = float(tp[i] / max(tp[i] + fn, 1)), float((cm.sum() - tp[i] - fn - fp) / max(cm.sum() - tp[i] - fn, 1))
        pos = truth == i
        m[f"auc_{c}"] = float(roc_auc_score(pos, scores[:, i])) if 0 < pos.sum() < len(pos) else float("nan")
    m["auc_macro"] = float(np.nanmean([m[f"auc_{c}"] for c in CLASSES]))
    return m


@torch.no_grad()
def evaluate(model, vols, data, rng):
    """Loss + metrics pooled over every slice of vols, plus per-volume Dice. AUC / ROC from all tumour pixels plus
    a random sample of the rest (AUC_PIXELS in total)."""
    model.eval()
    total = sum(len(data[v]) for v in vols) * SIZE * SIZE
    frac = min(1.0, AUC_PIXELS / total)
    cm, run, n, S, T, per = np.zeros((3, 3), np.int64), 0.0, 0, [], [], {}
    for v in vols:
        cv = np.zeros((3, 3), np.int64)
        for xb, yb in DataLoader(SliceDataset(data[v]), batch_size=8, shuffle=False, num_workers=0):
            out = model(xb)
            run += loss_fn(out, yb).item() * len(xb); n += len(xb)
            p = F.softmax(out, 1).permute(0, 2, 3, 1).reshape(-1, 3).numpy()
            t = yb.reshape(-1).numpy()
            cv += confusion_matrix(t, p.argmax(1), labels=[0, 1, 2])
            keep = (t == 2) | (rng.random(len(t)) < frac)
            S.append(p[keep]); T.append(t[keep])
        cm += cv
        per[name(v)] = dice_from(cv)
    S, T = np.concatenate(S), np.concatenate(T)
    m = metrics_from(cm, S, T)
    m["loss"] = run / n
    return m, cm, (S, T), per


def plot_roc(path, sc, title):
    scores, truth = sc
    plt.figure(figsize=(5, 5))
    for i, c in enumerate(CLASSES):
        pos = truth == i
        if 0 < pos.sum() < len(pos):
            fpr, tpr, _ = roc_curve(pos, scores[:, i])
            plt.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(pos, scores[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def fmt(m):
    return (f"loss {m['loss']:.4f} | Dice bg {m['dice_background']:.4f} liver {m['dice_liver']:.4f} tumor {m['dice_tumor']:.4f} | "
            f"acc {m['accuracy']:.4f} bal-acc {m['balanced_accuracy']:.4f} | AUC bg {m['auc_background']:.4f} "
            f"liver {m['auc_liver']:.4f} tumor {m['auc_tumor']:.4f} macro {m['auc_macro']:.4f}")


def train_fold(k, tr, va, data):
    fd = os.path.join(OUT, f"fold{k}")
    os.makedirs(fd, exist_ok=True)
    seed_all(SEED + k)
    train_ds = SliceDataset([s for v in tr for s in data[v]], augment=True)
    has_t = np.array([(m[..., 0] == 2).any() for _, m in train_ds.data])
    w = np.where(has_t, (~has_t).sum() / max(has_t.sum(), 1), 1.0)  # legacy oversampling of tumour slices
    sampler = WeightedRandomSampler(w.tolist(), num_samples=STEPS * BATCH, replacement=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    log.info(f"FOLD {k}: train {[name(v) for v in tr]} ({len(train_ds)} slices, {has_t.sum()} with tumour) | "
             f"val {[name(v) for v in va]} ({sum(len(data[v]) for v in va)} slices)")
    model = L.AIM_UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng, hist = np.random.default_rng(SEED + k), []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        run, cm = 0.0, np.zeros((3, 3), np.int64)
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            run += loss.item()
            cm += confusion_matrix(yb.reshape(-1).numpy(), out.argmax(1).reshape(-1).numpy(), labels=[0, 1, 2])
        trm = dict(loss=run / len(train_dl), accuracy=float(np.diag(cm).sum() / cm.sum()), **dice_from(cm))
        vm, vcm, vsc, _ = evaluate(model, va, data, rng)
        ck = os.path.join(fd, f"aimunet_fold{k}_epoch{ep:02d}.pt")
        torch.save({"model": model.state_dict(), "epoch": ep, "fold": k, "train": [name(v) for v in tr],
                    "val": [name(v) for v in va], "val_metrics": vm, "size": SIZE, "window": [WIN_LO, WIN_HI]}, ck)
        plot_roc(os.path.join(fd, f"roc_val_epoch{ep:02d}.png"), vsc, f"Fold {k} validation (LiTS), epoch {ep}")
        log.info(f"[fold {k}] Epoch {ep}/{EPOCHS} TRAIN loss {trm['loss']:.4f} | Dice bg {trm['dice_background']:.4f} "
                 f"liver {trm['dice_liver']:.4f} tumor {trm['dice_tumor']:.4f} | acc {trm['accuracy']:.4f}")
        log.info(f"[fold {k}] Epoch {ep}/{EPOCHS} VAL   {fmt(vm)}")
        log.info(f"[fold {k}] Epoch {ep}/{EPOCHS} VAL confusion (rows true bg/liver/tumor):\n{vcm}")
        log.info(f"[fold {k}] Epoch {ep}/{EPOCHS} saved {ck} ({time.time() - t0:.0f}s)")
        hist.append({"fold": k, "epoch": ep, **{f"train_{a}": b for a, b in trm.items()}, **{f"val_{a}": b for a, b in vm.items()}})
    tm, tcm, tsc, per = evaluate(model, TEST_SYN, data, rng)
    plot_roc(os.path.join(fd, "roc_test_final.png"), tsc, f"Fold {k} model on synthetic test set")
    log.info(f"[fold {k}] TEST synthetic (epoch {EPOCHS} model) {fmt(tm)}")
    log.info(f"[fold {k}] TEST confusion:\n{tcm}")
    for v, d in per.items():
        log.info(f"[fold {k}] TEST {v}: Dice liver {d['dice_liver']:.4f} tumor {d['dice_tumor']:.4f}")
    return hist, tm, tsc, per


def main():
    t_all = time.time()
    seed_all(SEED)
    log.info(f"AIM-UNet (legacy): train/val on {len(TRAIN_LITS)} LiTS volumes ({FOLDS}-fold CV), test on {len(TEST_SYN)} "
             f"synthetic | {EPOCHS} epochs x {STEPS} batches of {BATCH} | {SIZE}px | HU [{WIN_LO}, {WIN_HI}] | cpu x{torch.get_num_threads()}")
    data = {}
    for v in TRAIN_LITS + TEST_SYN:
        data[v] = load_volume(v)
        log.info(f"loaded {name(v)}: {len(data[v])} slices, tumour slices {sum(int((m == 2).any()) for _, m in data[v])}")
    log.info(f"TEST (synthetic, held out from every fold): {TEST_SYN}")
    split = {"test": TEST_SYN, "folds": []}
    hist, tests, tsc_all, per_rows = [], [], [], []
    for k, (i_tr, i_va) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(TRAIN_LITS), 1):
        tr, va = [TRAIN_LITS[i] for i in i_tr], [TRAIN_LITS[i] for i in i_va]
        split["folds"].append({"fold": k, "train": [name(v) for v in tr], "val": [name(v) for v in va]})
        h, tm, tsc, per = train_fold(k, tr, va, data)
        hist += h; tests.append({"fold": k, **tm}); tsc_all.append(tsc)
        per_rows += [{"fold": k, "volume": v, **d} for v, d in per.items()]
        with open(os.path.join(OUT, "history.json"), "w") as f:
            json.dump({"split": split, "history": hist, "test": tests, "test_per_volume": per_rows}, f, indent=1)

    for fn, rows in (("history.csv", hist), ("test_per_fold.csv", tests), ("test_per_volume.csv", per_rows)):
        with open(os.path.join(OUT, fn), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    keys = ["loss", "dice_background", "dice_liver", "dice_tumor", "accuracy", "balanced_accuracy",
            "auc_background", "auc_liver", "auc_tumor", "auc_macro"]
    ms = lambda xs: [float(np.nanmean(xs)), float(np.nanstd(xs))]
    last = [h for h in hist if h["epoch"] == EPOCHS]
    summary = {"split": split, "epochs": EPOCHS, "steps_per_epoch": STEPS, "size": SIZE,
               "cv_val_lits_final_epoch": {k: ms([h[f"val_{k}"] for h in last]) for k in keys},
               "test_synthetic": {k: ms([t[k] for t in tests]) for k in keys},
               "test_synthetic_per_volume": {v: {c: ms([r[f"dice_{c}"] for r in per_rows if r["volume"] == v]) for c in ("liver", "tumor")}
                                             for v in TEST_SYN},
               "minutes": round((time.time() - t_all) / 60, 1)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log.info("SUMMARY (mean +- std over %d folds, epoch %d): validation = real LiTS, test = synthetic", FOLDS, EPOCHS)
    for k in keys:
        a, b = summary["cv_val_lits_final_epoch"][k], summary["test_synthetic"][k]
        log.info(f"  {k:18s} val(LiTS) {a[0]:.4f} +- {a[1]:.4f} | test(synthetic) {b[0]:.4f} +- {b[1]:.4f}")
    for v, d in summary["test_synthetic_per_volume"].items():
        log.info(f"  {v:28s} Dice liver {d['liver'][0]:.4f} +- {d['liver'][1]:.4f} | tumor {d['tumor'][0]:.4f} +- {d['tumor'][1]:.4f}")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for k in range(1, FOLDS + 1):
        h = [r for r in hist if r["fold"] == k]
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["train_loss"] for r in h], "--", c=f"C{k}"); ax[0].plot(e, [r["val_loss"] for r in h], c=f"C{k}", label=f"fold {k}")
        for j, c in enumerate(("liver", "tumor")):
            ax[1 + j].plot(e, [r[f"val_dice_{c}"] for r in h], c=f"C{k}", label=f"fold {k}")
        ax[3].plot(e, [r["val_auc_macro"] for r in h], c=f"C{k}", label=f"fold {k}")
    for a, t in zip(ax, ("Loss (dashed train, solid val)", "Val Dice liver", "Val Dice tumor", "Val macro AUC")):
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "curves.png"), dpi=110); plt.close()
    plot_roc(os.path.join(OUT, "roc_test_all_folds.png"),
             (np.concatenate([s for s, _ in tsc_all]), np.concatenate([t for _, t in tsc_all])), "Synthetic test set, fold models pooled")
    log.info(f"ALL DONE in {summary['minutes']} min")


if __name__ == "__main__":
    main()
