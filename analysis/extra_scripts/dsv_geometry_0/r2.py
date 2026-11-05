import json, numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
for thr in (1.25,1.5,2.0,2.5,3.0):
    t=[c['eq_diam_mm'] for p in pv if p['spacing'][2]>=thr for c in p['components']]
    n=[c['eq_diam_mm'] for p in pv if p['spacing'][2]<thr for c in p['components']]
    print(thr,len(t),len(n),np.median(t),np.median(n),np.mean(np.array(t)<10),np.mean(np.array(n)<10), sum(p['spacing'][2]>=thr for p in pv))
# small vs slice thickness
allc=[(c['eq_diam_mm'],p['spacing'][2]) for p in pv for c in p['components']]
a=np.array(allc); print(len(a), np.mean(a[:,0]<a[:,1]))
th=a[a[:,1]>=3]; print(len(th), np.mean(th[:,0]<5))
th=a[a[:,1]>=2]; print(len(th), np.mean(th[:,0]<5), np.mean(th[:,0]<th[:,1]))
# n between 2 and 130
print(sum(1 for p in pv[2:] if p['spacing'][2]!=5.0))
