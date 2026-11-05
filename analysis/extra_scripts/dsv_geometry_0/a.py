import json, collections, numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
print(len(pv))
print(collections.Counter(p['vol_dtype'] for p in pv))
print(collections.Counter(p['seg_dtype'] for p in pv))
print(collections.Counter(tuple(p['labels']) if not isinstance(p['labels'],list) else tuple(p['labels']) for p in pv))
print(sum(p['seg_shape_match'] for p in pv))
rng=np.array([p['hu_max']-p['hu_min'] for p in pv])
print(rng.min(),rng.max(),np.median(rng),(rng>4095).sum(),(rng>8000).sum())
print(collections.Counter(p['hu_min'] for p in pv).most_common(6))
idx=np.argsort(rng)[::-1][:6]
for i in idx: print(pv[i]['name'],pv[i]['hu_min'],pv[i]['hu_max'])
g=255/rng; print(np.median(g),g.min(),g.max())
tv=[(abs(p['contrast_hu'])*255/(p['hu_max']-p['hu_min']),p['liver_hu_std']*255/(p['hu_max']-p['hu_min'])) for p in pv if p['tumor_vox']>0]
t=np.array(tv);print(len(t),np.median(t[:,0]),(t[:,0]<2).mean(),(t[:,0]<1).mean(),np.median(t[:,1]))
print([p['name'] for p in pv if p['tumor_vox']==0])
print(np.median([abs(p['contrast_hu']) for p in pv if p['tumor_vox']>0]))
print(collections.Counter((p['vol_dtype'],p['seg_dtype']) for p in pv))
