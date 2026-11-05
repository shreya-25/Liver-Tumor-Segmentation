import json, numpy as np, os
R=""
S=json.load(open(R+"analysis/output/stats.json"))
pv=sorted(S["per_volume"], key=lambda p:p["name"])   # lexicographic == cache order
N=len(pv)
names=[p["name"] for p in pv]
sz=np.array([p["spacing"][2] for p in pv]); sx=np.array([p["spacing"][0] for p in pv])
tg=np.where(sz<=1.0,0,np.where(sz<=2.5,1,2))   # thickness group
tml=np.array([p["tumor_ml"] for p in pv]); nt=np.array([p["n_tumors"] for p in pv])
lml=np.array([p["liver_ml"] for p in pv]); lhu=np.array([p["liver_hu_mean"] for p in pv])
nsl=np.array([p["n_slices"] for p in pv]); sltum=np.array([p["slices_tumor"] for p in pv])
hmin=np.array([p["hu_min"] for p in pv]); hmax=np.array([p["hu_max"] for p in pv])
has=nt>0
