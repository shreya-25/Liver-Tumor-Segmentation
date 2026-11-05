import json, numpy as np
from sklearn.model_selection import train_test_split
B=''
S=json.load(open(B+'analysis/output/stats.json')); P={d['name']:d for d in S['per_volume']}
names=open(B+'data/cache/volumes.txt').read().split('\n')
vi=np.load(B+'data/cache/vol_index.npy')
ids=np.arange(len(names)); tr,te=train_test_split(ids,test_size=0.2,random_state=42)
print(len(tr),len(te))
def grp(t): return 'thin' if t<=1 else ('mid' if t<=2.5 else 'thick')
for nm,s in [('train',tr),('test',te)]:
    ds=[P[names[i]] for i in s]
    g=[grp(d['spacing'][2]) for d in ds]
    ns=sum(d['n_slices'] for d in ds); st=sum(d['slices_tumor'] for d in ds)
    print(nm,'vols',len(ds),{k:g.count(k) for k in ['thin','mid','thick']},'slices',ns,'tumor slices',st,'no-tumor vols',sum(d['n_tumors']==0 for d in ds),
          'slice share by grp',{k:round(sum(d['n_slices'] for d,gg in zip(ds,g) if gg==k)/ns,3) for k in ['thin','mid','thick']}, 'tumor ml',sum(d['tumor_ml'] for d in ds))
    print('  tumor slice share by grp',{k:round(sum(d['slices_tumor'] for d,gg in zip(ds,g) if gg==k)/st,3) for k in ['thin','mid','thick']})
# sanity vol_index counts equal n_slices
cnt=np.bincount(vi); print(all(cnt[i]==P[names[i]]['n_slices'] for i in range(131)))
# per-volume min-max effect: contrast in uint8 levels
rows=[]
for n,d in P.items():
    rng=d['hu_max']-d['hu_min']
    rows.append((rng, d['contrast_hu'], d['liver_hu_std'], d['hu_min'], d['hu_max'],n))
rng=np.array([r[0] for r in rows]); con=np.array([abs(r[1]) for r in rows if r[1]==r[1]]); 
lv=np.array([r[2] for r in rows])
print('range HU median',np.median(rng),'min',rng.min(),'max',rng.max(), 'pct>5000',(rng>5000).mean())
gray_per_HU=255/rng
print('uint8 levels per HU: median',np.median(gray_per_HU),'min',gray_per_HU.min(),'max',gray_per_HU.max())
print('liver SD in uint8 levels: median',np.median(lv*gray_per_HU),'min',(lv*gray_per_HU).min(),'max',(lv*gray_per_HU).max())
c=np.array([r[1] for r in rows]); ok=~np.isnan(c)
cl=np.abs(c[ok])*gray_per_HU[ok]; print('|tumor contrast| in uint8 levels median',np.median(cl),'min',cl.min(),'max',cl.max(), 'share<2 levels',(cl<2).mean(),'share<1',(cl<1).mean())
# by min-hu group
mins=np.array([r[3] for r in rows]); 
for lo in [-1024,-2048,-3024]:
    m=mins==lo; print('min',lo,m.sum(),'range med',np.median(rng[m]),'levels/HU med',np.median(gray_per_HU[m]))
print('outliers: sorted ranges top5',sorted(rows,key=lambda r:-r[0])[:6])
print('hu_min<-3024',[(r[5],r[3]) for r in rows if r[3]<-3024 or r[3]>-900 and r[3]!=-1024 and r[3]>-1024][:20])
