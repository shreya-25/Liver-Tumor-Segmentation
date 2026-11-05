import numpy as np, json
from sklearn.model_selection import train_test_split
O="analysis/output/stats.json"
S=json.load(open(O))["per_volume"]
C="data/cache/"
names=open(C+"volumes.txt").read().split("\n"); vi=np.load(C+"vol_index.npy")
f=np.load("/tmp/dsv_pipeline_1/flags.npz"); t=f["t"]
tr,te=train_test_split(np.arange(131),test_size=0.2,random_state=42)
print([S[i]["name"] for i in tr[:3]],names[:3])
# check names map
bn={s["name"]:s for s in S}
print(list(bn)[:3])
import collections
cnt=np.bincount(vi[t],minlength=131)
rows=[]
for i in tr:
    s=bn.get(names[i]) or bn.get(names[i].replace('.nii',''))
    if s is None: print("missing",names[i]);break
    rows.append((i,s["spacing"][2],cnt[i],s["slices_tumor"],s["n_tumors"],s["tumor_ml"],s["n_slices"]))
rows=np.array(rows)
print("slices_tumor match cache count:",(rows[:,2]==rows[:,3]).mean())
th=rows[:,1]
g=np.digitize(th,[1.0000001,2.5000001])
for k in range(3):
    m=(g==k)&(rows[:,2]>0)
    print(k,"tumor patients",m.sum(),m.sum()/(rows[:,2]>0).sum(),"tumor slices",rows[m,2].sum()/rows[:,2].sum(),"lesions",rows[m,4].sum()/rows[rows[:,2]>0,4].sum(),"median",np.median(rows[m,2]), "all pts",(g==k).sum(), "total slices share",rows[g==k,6].sum()/rows[:,6].sum())
print("lesions total",rows[rows[:,2]>0,4].sum())
print("thickness unique", np.unique(th))
