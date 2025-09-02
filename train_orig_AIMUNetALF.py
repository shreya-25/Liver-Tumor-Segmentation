import os
import time
import logging
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt

# -----------------------------
# Logging Configuration
# -----------------------------
results_dir = r"C:\Users\sagarwal4\Downloads\LTS_V1\AimUnet\Orig\After liver fixing\trainresult"
os.makedirs(results_dir, exist_ok=True)

LOG_PATH = os.path.join(results_dir, "train_AIMUNet.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data Utilities
# -----------------------------
def get_volume_mask_paths(folder):
    volumes, masks = [], []
    for fname in os.listdir(folder):
        if 'vol' in fname.lower():    volumes.append(os.path.join(folder, fname))
        elif 'seg' in fname.lower():  masks.append(os.path.join(folder, fname))
    return sorted(volumes), sorted(masks)

def load_volume(image_path, mask_path, size=(256, 256)):
    vol = nib.load(image_path).get_fdata().astype(np.float32)
    vol = (vol + 200) / 400.0
    seg = nib.load(mask_path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    images, masks = [], []
    for i in range(vol.shape[0]):
        img = Image.fromarray((vol[i] * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        msk = Image.fromarray(seg[i]).resize(size, Image.NEAREST)
        images.append(np.array(img)[..., None] / 255.)
        masks.append(np.array(msk).astype(np.uint8)[..., None])
    return np.stack(images), np.stack(masks)

class LiverDataset(Dataset):
    def __init__(self, vol_paths, mask_paths, augment=False):
        self.data = []
        for vp, mp in zip(vol_paths, mask_paths):
            imgs, msks = load_volume(vp, mp)
            self.data += list(zip(imgs, msks))
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, msk = self.data[idx]
        img_pil = Image.fromarray((img[...,0]*255).astype(np.uint8))
        msk_pil = Image.fromarray(msk[...,0].astype(np.uint8), mode='L')
        if self.augment:
            if random.random() > 0.5:
                img_pil, msk_pil = TF.hflip(img_pil), TF.hflip(msk_pil)
            if random.random() > 0.5:
                img_pil, msk_pil = TF.vflip(img_pil), TF.vflip(msk_pil)
            angle = random.uniform(-10, 10)
            img_pil = TF.rotate(img_pil, angle, resample=Image.BILINEAR)
            msk_pil = TF.rotate(msk_pil, angle, resample=Image.NEAREST)
        img_tensor = TF.to_tensor(img_pil)
        msk_tensor = torch.as_tensor(np.array(msk_pil), dtype=torch.long)
        return img_tensor, msk_tensor

# -----------------------------
# AIM-Unet Model
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
    def __init__(self):
        super().__init__()
        self.enc1, self.pool1 = ConvBlock(1,64),    nn.MaxPool2d(2)
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
# Dice Score
# -----------------------------
def dice_per_class(pred, target, num_classes=3, smooth=1e-6):
    scores = {}
    pred = torch.argmax(pred, dim=1)
    for c in range(num_classes):
        p = (pred == c).float().view(-1)
        t = (target == c).float().view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        scores[c] = (2*inter + smooth)/(union + smooth)
    return scores

# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# -----------------------------
# Training Loop
# -----------------------------
def train_model(data_path, epochs=50, batch_size=4, lr=1e-4, patience=7):
    # Prepare data
    vols, masks = get_volume_mask_paths(data_path)
    vt, vtest, mt, mtest = train_test_split(vols, masks, test_size=0.2, random_state=42)
    train_ds = LiverDataset(vt, mt, augment=True)
    test_ds  = LiverDataset(vtest, mtest, augment=False)

    # Oversample tumor slices
    has_tumor = [(msk[...,0]==2).sum()>0 for _, msk in train_ds.data]
    n_tumor, n_non = sum(has_tumor), len(has_tumor)-sum(has_tumor)
    weights = [(n_non/n_tumor) if ht else 1.0 for ht in has_tumor]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)

    # Model, optimizer, losses
    model = AIM_UNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor([1.0,2.0,1.0], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
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
    dice_loss = DiceLoss()

    earlystop   = EarlyStopping(patience=patience)
    train_losses, val_losses = [], []

    for ep in range(1, epochs+1):
        # — Training
        model.train()
        running_train = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = ce_loss(out, yb) + dice_loss(out, yb)
            loss.backward()
            opt.step()
            running_train += loss.item()
        avg_train = running_train / len(train_dl)
        train_losses.append(avg_train)
        logger.info(f"Epoch {ep}/{epochs} — Train Loss: {avg_train:.4f}")

        # — Validation (loss + metrics)
        model.eval()
        running_val = 0.0
        all_dice    = {0: [], 1: [], 2: []}
        y_true_all, y_pred_all, y_score_all = [], [], []

        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = ce_loss(out, yb) + dice_loss(out, yb)
                running_val += loss.item()

                # per-class dice
                dsc = dice_per_class(out, yb)
                for c in dsc:
                    all_dice[c].append(dsc[c].item())

                probs = F.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                truths= yb.cpu().numpy()

                y_score_all.append(probs.reshape(-1, 3))
                y_pred_all.append(preds.flatten())
                y_true_all.append(truths.flatten())

        avg_val = running_val / len(test_dl)
        val_losses.append(avg_val)
        logger.info(f"Epoch {ep}/{epochs} — Val Loss:   {avg_val:.4f}")

        # — Log per-class dice
        dice_bg    = np.mean(all_dice[0])
        dice_liver = np.mean(all_dice[1])
        dice_tumor = np.mean(all_dice[2])
        logger.info(
            f"Validation Dice — Background: {dice_bg:.4f}, "
            f"Liver: {dice_liver:.4f}, Tumor: {dice_tumor:.4f}"
        )

        # — Early stopping
        if earlystop.step(avg_val):
            logger.info(f"▶ Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

        # — Save best model
        if avg_val == earlystop.best_loss:
            best_ckpt = os.path.join(results_dir, f"best_epoch_{ep:03d}.pt")
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"✅ Saved new best model to {best_ckpt}")

        # — Other metrics
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_score= np.concatenate(y_score_all)

        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        logger.info(f"Confusion Matrix:\n{cm}")

        total = cm.sum()
        for i, cls in enumerate([0,1,2]):
            TP = cm[i,i]; FN = cm[i,:].sum() - TP
            FP = cm[:,i].sum() - TP; TN = total - TP - FN - FP
            sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
            spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
            logger.info(f"Class {cls} — Sens: {sens:.4f}, Spec: {spec:.4f}")

        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except ValueError:
            auc = float('nan')
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        logger.info(f"ROC AUC: {auc:.4f} — Macro F1: {f1:.4f}")

    # — Final save
    final_ckpt = os.path.join(results_dir, "aim_unet_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    logger.info("🏁 Training complete. Final model saved.")

    # — Plot Loss Curves
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model(
        r"C:\Users\sagarwal4\Downloads\LTS_V1\Dataset\trainOriginal_65",
        epochs=50,
        batch_size=4,
        lr=1e-4,
        patience=7
    )
