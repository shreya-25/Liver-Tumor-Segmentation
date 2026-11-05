import numpy as np, json, nibabel as nib
d=np.load("/tmp/dsv_pipeline_0/flags.npz"); tum,vi,tr=d['tum'],d['vi'],d['tr']
S=json.load(open("analysis/output/stats.json"))
pv=S['per_volume']; names=open("data/cache/volumes.txt").read().split("\n")
byname={p['name']:p for p in pv}
print(list(byname)[:3], names[:3])
# align
def key(n): return n
P=[byname.get(n) or byname.get(n.replace('.nii','')) for n in names]
print(sum(p is None for p in P))
th=np.array([p['spacing'][2] for p in P])
cnt=np.bincount(vi[tum],minlength=131)
nl=np.array([p['n_tumors'] for p in P])
def grp(t): return 0 if t<=1 else (1 if t<=2.5 else 2)
g=np.array([grp(t) for t in th])
tp=[i for i in tr if cnt[i]>0]
G=np.array([g[i] for i in tp])
print("tumor patients",len(tp),[ (G==k).mean() for k in range(3)])
ts=np.array([cnt[i] for i in tp]); print("slice share",[ts[G==k].sum()/ts.sum() for k in range(3)])
print("median slices",[np.median(ts[G==k]) for k in range(3)])
nlt=np.array([nl[i] for i in tp]); print("lesions stats.json",nlt.sum(),[nlt[G==k].sum()/nlt.sum() for k in range(3)])
# n_slices vs cache counts
print(all(P[i]['n_slices']==(vi==i).sum() for i in range(131)))
# compare slices_tumor from stats vs cache
diff=[(i,P[i]['slices_tumor'],cnt[i]) for i in range(131)]
print("sum stats train", sum(P[i]['slices_tumor'] for i in tr), "cache",cnt[tr].sum())
print("thick14?", sum(1 for i in tp if g[i]==2))
ext=[max(c['max_extent_mm'] for c in P[i]['components']) for i in tp if P[i]['components']]
# axis-0 tumor slices
tot=0; T=0
for i in tr:
    seg=np.asarray(nib.load("data/raw/segmentation-%s.nii"%names[i].split('-')[1].replace('.nii','')).dataobj)
    H,W,Z=seg.shape
    zi=np.minimum(((np.arange(256)+0.5)*Z/256).astype(int),Z-1)
    wi=(np.arange(256)*2+1) if W==512 else np.minimum(((np.arange(256)+0.5)*W/256).astype(int),W-1)
    t=(seg==2)
    t=t[:,wi][:,:,zi]
    T+=t.any(axis=(1,2)).sum(); tot+=H
print("axis0 train slices",tot,"tumor",T)
print("thick train vols",sum(1 for i in tr if g[i]==2),"all",sum(g==2))
print("tumor extent median per patient (z extent from slices*thickness?)")
ex=[cnt[i]*th[i] for i in tp]; print(np.median(ex))
