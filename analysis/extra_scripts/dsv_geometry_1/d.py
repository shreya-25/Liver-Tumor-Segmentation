import numpy as np,json
B=''
names=open(B+'data/cache/volumes.txt').read().split('\n');vi=np.load(B+'data/cache/vol_index.npy')
mk=np.load(B+'data/cache/masks.npy',mmap_mode='r')
cl=json.load(open('/tmp/ds_geom/cls.json'))
starts=np.r_[0,np.cumsum(np.bincount(vi))]
from collections import defaultdict
res=defaultdict(list)
for k,nm in enumerate(names):
    if not nm: continue
    n=int(nm.split('-')[1].split('.')[0])
    m=np.asarray(mk[starts[k]:starts[k+1]])>=1
    idx=np.argwhere(m)
    c=idx.mean(0)
    res[cl[str(n)]].append((c[1]>127.5,c[2]>127.5))
for k,v in res.items():
    v=np.array(v);print(k,len(v),v.mean(0))
