import json, numpy as np
R=""
S={d['name']:d for d in json.load(open(R+"analysis/output/stats.json"))["per_volume"]}
names=open(R+"data/cache/volumes.txt").read().split("\n")
vi=np.load(R+"data/cache/vol_index.npy")
im=np.load(R+"data/cache/images.npy",mmap_mode='r'); mk=np.load(R+"data/cache/masks.npy",mmap_mode='r')
print(im.shape,len(names),names[:3])
rows=[]
for i,n in enumerate(names):
    idx=np.where(vi==i)[0]; a,b=idx[0],idx[-1]+1
    I=np.asarray(im[a:b]).astype(np.float64); M=np.asarray(mk[a:b])
    d=S[n]
    liv=I[M==1]; tum=I[M==2]
    pred=255*(d['liver_hu_mean']-d['hu_min'])/(d['hu_max']-d['hu_min'])
    rows.append((liv.mean(),pred,(tum.mean()-liv.mean()) if len(tum) else np.nan,255*d.get('contrast_hu',np.nan)/(d['hu_max']-d['hu_min']) if d.get('contrast_hu') is not None else np.nan, liv.std(), len(tum)))
r=np.array(rows,dtype=float)
m,p,cm,cp,sd,nt=r.T
print("liver r",np.corrcoef(m,p)[0,1],"mad",np.abs(m-p).mean(), "bias",(m-p).mean())
ok=~np.isnan(cm)&~np.isnan(cp)
print("n tumor",ok.sum(),"contrast r",np.corrcoef(cm[ok],cp[ok])[0,1])
print("measured contrast med/IQR",np.median(cm[ok]),np.percentile(cm[ok],[25,75]),"<3:",np.mean(np.abs(cm[ok])<3),"<=1:",np.mean(np.abs(cm[ok])<=1))
print("liver grey med,p5,p95,min,max",np.median(m),*np.percentile(m,[5,95]),m.min(),m.max())
print("liver SD med",np.median(sd))
np.save("rows.npy",r)
