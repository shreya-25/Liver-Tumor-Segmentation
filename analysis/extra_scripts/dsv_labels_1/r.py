import json, numpy as np
from sklearn.model_selection import train_test_split
R='.'
pv={p['name']:p for p in json.load(open(R+'/analysis/output/stats.json'))['per_volume']}
names=open(R+'/data/cache/volumes.txt').read().split('\n')
print(len(names), names[:3], len(pv))
P=[pv[n] for n in names]
nvox=sum(np.prod(p['shape']) for p in P); tv=sum(p['tumor_vox'] for p in P); lv=sum(p['liver_vox'] for p in P)
print('tumor frac',tv/nvox,'liver',lv/nvox,'t/l',tv/lv,(nvox-lv)/tv,(lv-tv)/tv)
ns=sum(p['n_slices'] for p in P); st=sum(p['slices_tumor'] for p in P); print(ns,st,st/ns)
tm=np.array([p['tumor_ml'] for p in P]); print((tm==0).sum(),(tm<1).sum(),(tm<5).sum())
s=np.sort(tm)[::-1]; print(s[:13].sum()/tm.sum(), s[0]/tm.sum())
# tumor vox share
tvx=np.array([p['tumor_vox'] for p in P]);s=np.sort(tvx)[::-1];print('vox',s[:13].sum()/tvx.sum(),s[0]/tvx.sum())
tr,te=train_test_split(list(range(131)),test_size=0.2,random_state=42)
print(len(tr),len(te))
for nm,ids in(('tr',tr),('te',te)):
    print(nm,'nslices',sum(P[i]['n_slices'] for i in ids),'tum slices raw',sum(P[i]['slices_tumor'] for i in ids),'notumor',[names[i] for i in ids if P[i]['tumor_vox']==0],'lesions',sum(P[i]['n_tumors'] for i in ids),'thick>2.5',sum(P[i]['spacing'][2]>2.5 for i in ids))
print('all lesions',sum(p['n_tumors'] for p in P))
print('tumor vols train', sum(P[i]['tumor_vox']>0 for i in tr))
# thickness
th=np.array([p['spacing'][2] for p in P]); print(np.percentile(th,[0,25,50,75,100]), (th>2.5).sum())
print('tumor ml by thickness: n thick>2.5 tumor share', tm[th>2.5].sum()/tm.sum())
# slices counts raw by thickness
ts=np.array([p['slices_tumor'] for p in P]); print('thin share of tumor slices', ts[th<=1].sum()/ts.sum(), 'of slices', np.array([p['n_slices'] for p in P])[th<=1].sum()/ns)
