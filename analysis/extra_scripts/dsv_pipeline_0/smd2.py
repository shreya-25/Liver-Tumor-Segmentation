import json,numpy as np,os
from sklearn.model_selection import train_test_split
R=''
names=sorted(f for f in os.listdir(R+'data/raw') if 'vol' in f.lower())
segs=sorted(f for f in os.listdir(R+'data/raw') if 'seg' in f.lower())
_,vv,_,_=train_test_split(names,segs,test_size=0.2,random_state=42)
vids=set(int(n.split('-')[1].split('.')[0]) for n in vv)
S=json.load(open(R+'analysis/output/stats.json'))['per_volume']
d={int(v['name'].split('-')[1].split('.')[0]):v for v in S}
ids=sorted(d)
print([k for k in d[0] if True][:40])
def g(v,k):
    x=v.get(k); return np.nan if x is None else x
keys=['hu_min','hu_p999','liver_hu_mean','liver_hu_std','tumor_ml','n_tumors','tumor_frac_of_liver','cnr','contrast_hu','tumor_hu_mean','largest_tumor_share','liver_ml','n_slices','slices_tumor']
X=np.array([[g(d[i],k) for k in keys]+[d[i]['spacing'][2],d[i]['spacing'][0]] for i in ids],float)
lab=keys+['sz','sx']
m=np.array([i in vids for i in ids])
def smd(m):
    A=X[m];B=X[~m]
    return np.abs(np.nanmean(A,0)-np.nanmean(B,0))/np.sqrt((np.nanvar(A,0,ddof=1)+np.nanvar(B,0,ddof=1))/2)
s=smd(m);print(dict(zip(lab,np.round(s,3))))
rng=np.random.default_rng(2);R_=[]
for _ in range(20000):
    mm=np.zeros(len(ids),bool);mm[rng.permutation(len(ids))[:27]]=True;R_.append(smd(mm))
R_=np.array(R_)
print('mean over all',s.mean(),(R_.mean(1)<s.mean()).mean())
# per-covariate percentile
print(dict(zip(lab,np.round((R_<s).mean(0),3))))
