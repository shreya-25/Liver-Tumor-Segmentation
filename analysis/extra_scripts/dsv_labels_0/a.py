import json,numpy as np
from sklearn.model_selection import train_test_split
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
print(len(pv))
tv=sum(p['tumor_vox'] for p in pv);lv=sum(p['liver_vox'] for p in pv)
tot=sum(int(np.prod(p['shape'])) for p in pv)
print(tv/tot,lv/tot,tv/lv,(tot-lv)/tv,(lv-tv)/tv)
ns=sum(p['n_slices'] for p in pv);st=sum(p['slices_tumor'] for p in pv);print(ns,st,st/ns)
print(sum(p['tumor_vox']==0 for p in pv),sum(p['tumor_ml']<1 for p in pv),sum(p['tumor_ml']<5 for p in pv))
t=sorted([p['tumor_ml'] for p in pv],reverse=True);print(sum(t[:13])/sum(t),t[0]/sum(t))
names=sorted(p['name'] for p in pv)
tr,te=train_test_split(range(131),test_size=0.2,random_state=42)
byn={p['name']:p for p in pv}
def s(idx):
    P=[byn[names[i]] for i in idx]
    n=sum(p['n_slices'] for p in P);k=sum(p['slices_tumor'] for p in P)
    return len(P),n,k,k/n,(n-k)/k,sum(p['tumor_vox']==0 for p in P),sum(p['n_tumors'] for p in P)
print(s(tr),s(te))
print([names[i] for i in te if byn[names[i]]['tumor_vox']==0])
print(sum(1 for i in te if byn[names[i]]['spacing'][2]>2.5), sum(1 for p in pv if p['spacing'][2]>2.5))
tt=sorted([byn[names[i]]['slices_tumor'] for i in tr],reverse=True);print(sum(tt[:10])/sum(tt), sum(1 for i in tr if byn[names[i]]['slices_tumor']>0))
print(sorted(names)[:5])
