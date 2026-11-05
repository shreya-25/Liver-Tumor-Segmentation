import json,numpy as np
from scipy.stats import spearmanr
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
bins=[0,5,10,20,40,1e9]
rows=[]
for v in pv:
    for c in v['components']:
        rows.append((c['eq_diam_mm'],c['hu_mean']-v['liver_hu_mean'],c['vox'],v['spacing'][2],c['hu_mean'],v['contrast_hu'],v['name']))
r=np.array([x[:6] for x in rows]);print(len(r))
for i in range(5):
    m=(r[:,0]>=bins[i])&(r[:,0]<bins[i+1])
    print(bins[i],m.sum(),np.median(r[m,1]),np.median(r[m,2]),np.median(r[m,1]/r[m,5]))
print(spearmanr(abs(r[:,1]),r[:,0]))
print((r[:,2]==1).sum(),(r[:,2]<=2).sum(), ((r[:,0]<5)).sum())
print('voxw',np.sum(r[:,1]*r[:,2])/r[:,2].sum(),np.median(r[:,1]))
sz=r[:,3]
for lo,hi in [(0,1.0001),(1.0001,2.5),(2.5,99)]:
    for dl,dh in [(0,10),(10,1e9)]:
        m=(sz>=lo)&(sz<hi)&(r[:,0]>=dl)&(r[:,0]<dh)
        print(lo,hi,dl,m.sum(),np.median(r[m,1]))
print(pv[0]['spacing'], [v['spacing'][2] for v in pv][:10])
print('---')
for lo,hi in [(0,1.0001),(1.0001,2.5001),(2.5001,99)]:
    for dl,dh in [(0,10),(10,1e9)]:
        m=(sz>=lo)&(sz<hi)&(r[:,0]>=dl)&(r[:,0]<dh)
        print(lo,hi,dl,m.sum(),np.median(r[m,1]))
import collections;print(collections.Counter(round(v['spacing'][2],2) for v in pv))
