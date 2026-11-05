import json,numpy as np
from sklearn.model_selection import train_test_split
S=json.load(open('analysis/output/stats.json'))
pv=S['per_volume']; names=sorted(v['name'] for v in pv); d={v['name']:v for v in pv}
tr,va=train_test_split(np.arange(131),test_size=0.2,random_state=42)
m=np.zeros(131,bool);m[va]=True
tb=np.array([d[n]['n_tumors']>0 for n in names]); 
rng=np.random.default_rng(1)
for k in ['tumor_hu_mean','contrast_hu','cnr','liver_ml','liver_hu_std']:
    x=np.array([np.nan if d[n].get(k) is None else d[n][k] for n in names],float)
    ok=~np.isnan(x)
    xv=x[ok];mv=m[ok]
    obs=abs(xv[mv].mean()-xv[~mv].mean())
    c=0;N=20000
    for _ in range(N):
        p=rng.permutation(len(xv));mm=np.zeros(len(xv),bool);mm[p[:mv.sum()]]=True
        c+= abs(xv[mm].mean()-xv[~mm].mean())>=obs
    print(k,mv.sum(),obs,c/N, xv[mv].mean(),xv[~mv].mean())
