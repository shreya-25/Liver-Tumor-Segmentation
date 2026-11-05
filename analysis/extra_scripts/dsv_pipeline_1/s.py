import json,numpy as np
from sklearn.model_selection import train_test_split
S=json.load(open('analysis/output/stats.json'))
pv=S['per_volume']; names=sorted(v['name'] for v in pv); d={v['name']:v for v in pv}
def nz(x): return 0 if x is None else x
cov={'thick':lambda v:v['spacing'][2],'nsl':lambda v:v['n_slices'],'liver':lambda v:v['liver_ml'],'tumor':lambda v:v['tumor_ml'],
'ntum':lambda v:v['n_tumors'],'hu':lambda v:v['liver_hu_mean'],'hustd':lambda v:v['liver_hu_std'],'cnr':lambda v:nz(v.get('cnr')),'contr':lambda v:nz(v.get('contrast_hu')),'hasT':lambda v:float(v['n_tumors']>0),'sx':lambda v:v['spacing'][0]}
X=np.array([[nz(f(d[n])) for f in cov.values()] for n in names],float)
idx=np.arange(131)
tr,va=train_test_split(idx,test_size=0.2,random_state=42)
def smd(X,va):
    m=np.zeros(131,bool);m[va]=True
    a,b=X[m],X[~m]
    return np.abs(a.mean(0)-b.mean(0))/np.sqrt((a.var(0,ddof=1)+b.var(0,ddof=1))/2+1e-12)
s=smd(X,va); print(dict(zip(cov,s.round(3))))
rng=np.random.default_rng(0)
for cols in (list(range(9)),[0,1,2,3,4,5,6,7,8],[0,2,3,4,5,7,9]):
    ref=smd(X,va)[cols].mean()
    r=np.array([smd(X,rng.permutation(131)[:27])[cols].mean() for _ in range(20000)])
    print(cols,ref,np.median(r),(r<ref).mean())
# lesion-level
big=lambda n:[c for c in d[n]['components'] if c['eq_diam_mm']>=20]
print(sum(len(big(n)) for n in [names[i] for i in va]), sum(1 for i in va if big(names[i])))
print(sum(1 for i in va for c in d[names[i]]['components'] if c['max_extent_mm']>64), d['volume-108.nii']['tumor_ml'], d['volume-108.nii']['n_tumors'])
print(sum(1 for i in va if d[names[i]]['tumor_ml']>100))
