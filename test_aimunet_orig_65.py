import os
import numpy as np
from PIL import Image
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import logging

# -----------------------------
# Paths (adjust as needed)
# -----------------------------
TEST_FOLDER = r"C:\Users\sagarwal4\Downloads\LTS_V1\Dataset\testOriginal_65"
SAVE_DIR    = r"C:\Users\sagarwal4\Downloads\LTS_V1\AimUnet\Orig\Train_result"
CHECKPOINT  = os.path.join(SAVE_DIR, "aim_unet_epoch_035.pt")

LOG_PATH = os.path.join(SAVE_DIR, "test_AIMUNet.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Config to limit memory for ROC sampling
# -----------------------------
MAX_SAMPLES_PER_SLICE = 1000  # max points per slice to include in ROC

# -----------------------------
# Data Utilities
# -----------------------------
def get_volume_mask_paths(folder):
    vols, masks = [], []
    for fname in os.listdir(folder):
        if 'vol' in fname.lower(): vols.append(os.path.join(folder, fname))
        elif 'seg' in fname.lower(): masks.append(os.path.join(folder, fname))
    return sorted(vols), sorted(masks)


def load_volume(image_path, mask_path, size=(256, 256)):
    vol = nib.load(image_path).get_fdata()
    seg = nib.load(mask_path).get_fdata()
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)
    images, masks = [], []
    for i in range(vol.shape[0]):
        img = Image.fromarray((vol[i] * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        msk = Image.fromarray(seg[i]).resize(size, Image.NEAREST)
        images.append(np.array(img)[..., None] / 255.)
        masks.append(np.array(msk).astype(np.uint8)[..., None])
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.uint8)

class LiverDataset(Dataset):
    def __init__(self, vols, masks, augment=False):
        self.data = []
        for v_path, m_path in zip(vols, masks):
            imgs, msks = load_volume(v_path, m_path)
            self.data += list(zip(imgs, msks))
        self.augment = augment
        self.transforms = transforms.Compose([
            transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.RandomRotation(10), transforms.ToTensor()
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, msk = self.data[idx]
        if self.augment:
            img = self.transforms((img * 255).astype(np.uint8).squeeze())
        else:
            img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        return img, torch.tensor(msk.squeeze(), dtype=torch.long)

# -----------------------------
# Model Definition (AIM_UNet)
# -----------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_c,in_c,1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_c,in_c,1), nn.Conv2d(in_c,in_c,5,padding=2), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_c,in_c,1), nn.Conv2d(in_c,in_c,3,padding=1), nn.Conv2d(in_c,in_c,3,padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(3,stride=1,padding=1), nn.Conv2d(in_c,in_c,1), nn.BatchNorm2d(in_c), nn.ReLU())
    def forward(self, x): return torch.cat([b(x) for b in (self.branch1,self.branch2,self.branch3,self.branch4)], dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1), nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1), nn.ReLU()
        )
    def forward(self, x): return self.conv(x)

class AIM_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1,64);    self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64,128);  self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128,256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256,512); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512,1024)
        self.inc4 = InceptionModule(512); self.up4 = nn.ConvTranspose2d(1024,512,2,stride=2); self.dec4 = ConvBlock(512*5,512)
        self.inc3 = InceptionModule(256); self.up3 = nn.ConvTranspose2d(512,256,2,stride=2); self.dec3 = ConvBlock(256*5,256)
        self.inc2 = InceptionModule(128); self.up2 = nn.ConvTranspose2d(256,128,2,stride=2); self.dec2 = ConvBlock(128*5,128)
        self.inc1 = InceptionModule(64);  self.up1 = nn.ConvTranspose2d(128,64,2,stride=2);  self.dec1 = ConvBlock(64*5,64)
        self.out  = nn.Conv2d(64,3,1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1)); e3=self.enc3(self.pool2(e2)); e4=self.enc4(self.pool3(e3)); b=self.bottleneck(self.pool4(e4))
        d4=self.dec4(torch.cat([self.inc4(e4), self.up4(b)], dim=1))
        d3=self.dec3(torch.cat([self.inc3(e3), self.up3(d4)], dim=1))
        d2=self.dec2(torch.cat([self.inc2(e2), self.up2(d3)], dim=1))
        d1=self.dec1(torch.cat([self.inc1(e1), self.up1(d2)], dim=1))
        return self.out(d1)

# Dice per class
def dice_per_class(pred, target, num_classes=3, smooth=1e-6):
    scores={}
    labels = torch.argmax(pred, dim=1)
    for c in range(num_classes):
        p=(labels==c).float().view(-1)
        t=(target==c).float().view(-1)
        inter=(p*t).sum(); union=p.sum()+t.sum()
        scores[c]=((2*inter+smooth)/(union+smooth)).item()
    return scores

# -----------------------------
# Testing Function
# -----------------------------
def test_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vols, masks = get_volume_mask_paths(TEST_FOLDER)
    ds = LiverDataset(vols, masks, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model=AIM_UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    total_TP=total_FP=total_FN=total_TN=0
    dice_scores={c:[] for c in range(3)}
    roc_t_true=[]; roc_t_score=[]
    cm=np.zeros((3,3),dtype=int)

    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            out=model(x); probs=F.softmax(out,dim=1); pred=torch.argmax(probs,dim=1)

            # Dice
            d=dice_per_class(out,y)
            for c,val in d.items(): dice_scores[c].append(val)

            # Flatten arrays
            p_flat=pred.view(-1).cpu().numpy()
            t_flat=y.view(-1).cpu().numpy()

            # Confusion matrix
            cm+=confusion_matrix(t_flat, p_flat, labels=[0,1,2])

            # Micro metrics counts
            for c in [0,1,2]:
                pc=(pred==c).view(-1); tc=(y==c).view(-1)
                TP=(pc&tc).sum().item(); FP=(pc&~tc).sum().item()
                FN=(~pc&tc).sum().item(); TN=(~pc&~tc).sum().item()
                total_TP+=TP; total_FP+=FP; total_FN+=FN; total_TN+=TN

            # ROC for tumor class
            t_bin=(t_flat==2)
            s_bin=probs[:,2,:,:].view(-1).cpu().numpy()
            if len(t_bin)>MAX_SAMPLES_PER_SLICE:
                idx=np.random.choice(len(t_bin),MAX_SAMPLES_PER_SLICE,replace=False)
                t_bin=t_bin[idx]; s_bin=s_bin[idx]
            roc_t_true.extend(t_bin.tolist()); roc_t_score.extend(s_bin.tolist())

    # Results
    class_names={0:'Background',1:'Liver',2:'Tumor'}
    print("\nTest Dice per Class:")
    logger.info("Test Dice per Class:")
    for c in class_names:
        logger.info(f"  {class_names[c]}: {np.mean(dice_scores[c]):.4f}")
        print(f" {class_names[c]}: {np.mean(dice_scores[c]):.4f}")

    sens=total_TP/(total_TP+total_FN+1e-8)
    spec=total_TN/(total_TN+total_FP+1e-8)
    logger.info(f"Overall Sensitivity: {sens:.4f}")
    logger.info(f"Overall Specificity: {spec:.4f}")
    logger.info("Confusion Matrix (rows=true, cols=pred):")
    logger.info(f"\n{cm}")
    print(f"\nOverall Sensitivity: {sens:.4f}")
    print(f"Overall Specificity: {spec:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # ROC AUC
    fpr,tpr,_=roc_curve(roc_t_true, roc_t_score)
    auc_val=auc(fpr,tpr)
    logger.info(f"Tumor AUC: {auc_val:.4f}")
    print(f"\nTumor AUC: {auc_val:.4f}")

    # Plot ROC
    plt.figure(); plt.plot(fpr,tpr,label=f"Tumor (AUC {auc_val:.3f})");
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR');
    plt.title('ROC Curve: Tumor Detection'); plt.legend(); plt.show()

if __name__=="__main__":
    test_model()
