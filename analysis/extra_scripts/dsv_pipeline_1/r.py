import json,numpy as np
from sklearn.model_selection import train_test_split
S=json.load(open('analysis/output/stats.json'))
pv=S['per_volume']; names=sorted(v['name'] for v in pv)
print(len(names), names[:3])
d={v['name']:v for v in pv}
tr,va=train_test_split(names,test_size=0.2,random_state=42)
print(len(tr),len(va),[n.split('-')[-1].split('.')[0] for n in va])
def grp(v):
    t=v['spacing'][2]; return 0 if t<=1 else (1 if t<=2.5 else 2)
for nm,s in (('tr',tr),('va',va)):
    g=np.bincount([grp(d[n]) for n in s],minlength=3)
    print(nm,g,sum(d[n]['n_tumors']>0 for n in s),sum(d[n]['n_tumors'] for n in s),sum(d[n]['tumor_ml'] for n in s), np.mean([d[n]['liver_hu_mean'] for n in s]),
      sum(d[n]['slices_tumor'] for n in s)/sum(d[n]['n_slices'] for n in s))
print(open('data/cache/volumes.txt').read().split('\n')[-3:])
# batch-level empty fraction
import numpy as np
vi=np.load('data/cache/vol_index.npy'); print(vi.shape,vi[:5])
