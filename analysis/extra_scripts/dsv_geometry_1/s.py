import json,numpy as np
from sklearn.model_selection import train_test_split
R=''
d=json.load(open(R+'analysis/output/stats.json'))['per_volume']
names=[l.strip() for l in open(R+'data/cache/volumes.txt') if l.strip()]
print(names[:3],[v['name'] for v in d[:3]])
by={v['name']:v for v in d}
def key(x):
    import os;return x if x in by else os.path.basename(x)
for nm,ids in zip(['train','test'],train_test_split(names,test_size=0.2,random_state=42)):
    vs=[by.get(key(i)) or by.get(key(i).replace('.nii','')) for i in ids]
    if None in vs: print('missing');continue
    sz=np.array([v['spacing'][2] for v in vs]);n=np.array([v['n_slices'] for v in vs]);t=sum(v['slices_tumor'] for v in vs)
    g=[(sz<=1),(sz>1)&(sz<=2.5),sz>2.5]
    print(nm,len(vs),[int(m.sum()) for m in g],[round(n[m].sum()/n.sum()*100,1) for m in g],t,n[g[2]].sum())
