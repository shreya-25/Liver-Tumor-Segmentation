import json,numpy as np,itertools
from sklearn.model_selection import train_test_split
S=json.load(open('analysis/output/stats.json'))
pv=S['per_volume']; names=sorted(v['name'] for v in pv); d={v['name']:v for v in pv}
print([k for k in pv[0]])
keys=['n_slices','liver_ml','tumor_ml','tumor_frac_of_liver','slices_liver','slices_tumor','n_tumors','largest_tumor_share','liver_hu_mean','liver_hu_std','tumor_hu_mean','contrast_hu','cnr','hu_min','hu_p001','hu_p999']
def g(n,k):
    v=d[n].get(k); return np.nan if v is None else v
X=np.array([[g(n,k) for k in keys]+[d[n]['spacing'][2],d[n]['spacing'][0],float(d[n]['n_tumors']>0)] for n in names],float)
keys+=['thick','sx','hasT']
tr,va=train_test_split(np.arange(131),test_size=0.2,random_state=42)
m=np.zeros(131,bool);m[va]=True
def smd(a,b):
    return np.abs(np.nanmean(a,0)-np.nanmean(b,0))/np.sqrt((np.nanvar(a,0,ddof=1)+np.nanvar(b,0,ddof=1))/2+1e-12)
s=smd(X[m],X[~m]);print({k:round(float(x),3) for k,x in zip(keys,s)})
tb=np.array([d[n]['n_tumors']>0 for n in names])
s2=smd(X[m&tb],X[~m&tb]);print('tumor-bearing only',{k:round(float(x),3) for k,x in zip(keys,s2)})
