import json,numpy as np,os
from sklearn.model_selection import train_test_split
R=''
names=sorted(f for f in os.listdir(R+'data/raw') if 'vol' in f.lower())
segs=sorted(f for f in os.listdir(R+'data/raw') if 'seg' in f.lower())
_,vv,_,_=train_test_split(names,segs,test_size=0.2,random_state=42)
vids=[int(n.split('-')[1].split('.')[0]) for n in vv]
S=json.load(open(R+'analysis/output/stats.json'))['per_volume']
d={int(v['name'].split('-')[1].split('.')[0]):v for v in S}
ids=sorted(d)
def f(v):
    return [v['spacing'][2],v['spacing'][0],v['n_slices'],v['liver_hu_mean'],v['liver_hu_std'],v['liver_ml'],v['tumor_ml'],v['n_tumors'],float(v['n_tumors']>0)]
lab=['sz','sx','nsl','hu','hustd','liver_ml','tumor_ml','n_tum','has_tum']
X=np.array([f(d[i]) for i in ids],float)
m=np.array([i in set(vids) for i in ids])
def smd(m):
    A=X[m];B=X[~m]
    return np.abs(A.mean(0)-B.mean(0))/np.sqrt((A.var(0,ddof=1)+B.var(0,ddof=1))/2)
s=smd(m);print(dict(zip(lab,np.round(s,3))),s.mean())
rng=np.random.default_rng(1);r=[]
for _ in range(100000):
    mm=np.zeros(len(ids),bool);mm[rng.permutation(len(ids))[:27]]=True;r.append(smd(mm).mean())
r=np.array(r);print(np.median(r),np.percentile(r,90),(r<s.mean()).sum(),(r<s.mean()).mean())
