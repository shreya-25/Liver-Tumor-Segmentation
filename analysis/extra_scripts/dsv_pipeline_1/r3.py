import numpy as np, nibabel as nib
from sklearn.model_selection import train_test_split
R="data/raw/"
C="data/cache/"
names=open(C+"volumes.txt").read().split("\n")
tr,te=train_test_split(np.arange(131),test_size=0.2,random_state=42)
tot=0; tum=0; per=[]
for i in tr:
    sn=names[i].replace("volume","segmentation")
    s=np.asarray(nib.load(R+sn).dataobj)
    H,W,Z=s.shape
    wi=np.floor((np.arange(256)+0.5)*W/256).astype(int)
    zi=np.floor((np.arange(256)+0.5)*Z/256).astype(int)
    sub=s[:,:,zi][:,wi,:]  # (H,256,256)
    t=(sub==2).reshape(H,-1).any(1)
    tot+=H; tum+=t.sum(); per.append(t.sum())
print(tot,tum,tum/tot)
per=np.array(per); q=per/per.sum(); print("patients",(per>0).sum(),"ess",1/(q**2).sum())
nn=tot-tum; print("w",nn/tum,"draws",0.5*tot/tum)
