import json,numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
n0=[v for v in pv if not v['n_tumors']];print(len(n0),[ (v['name'],v['tumor_vox']) for v in n0][:3])
l=np.array([v['liver_hu_mean'] for v in pv])
for lo,hi in [(0,60),(60,80),(80,100),(100,120),(120,999)]:print(lo,hi,((l>=lo)&(l<hi)).sum())
t=[v for v in pv if v['n_tumors']]
# thickness subsets
for k in [1.0,]:
    pass
import collections
print(collections.Counter(round(v['spacing'][2],1) for v in t))
c=np.array([v['contrast_hu'] for v in t]);sz=np.array([v['spacing'][2] for v in t])
for m in [sz<=1,sz>1]:print(m.sum(),np.median(c[m]),np.median(np.array([v['cnr'] for v in t])[m]))
