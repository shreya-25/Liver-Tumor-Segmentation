import json,numpy as np,os
from scipy import stats
from sklearn.model_selection import train_test_split
R=''
names=sorted(f for f in os.listdir(R+'data/raw') if 'vol' in f.lower())
segs=sorted(f for f in os.listdir(R+'data/raw') if 'seg' in f.lower())
_,vv,_,_=train_test_split(names,segs,test_size=0.2,random_state=42)
vids=set(int(n.split('-')[1].split('.')[0]) for n in vv)
S=json.load(open(R+'analysis/output/stats.json'))['per_volume']
d={int(v['name'].split('-')[1].split('.')[0]):v for v in S}
tb=[i for i in sorted(d) if d[i]['n_tumors']>0]
print(len(tb), sum(i in vids for i in tb))
def lesion(ids,k='contrast'):
    return np.array([c[k] for i in ids for c in d[i]['components']])
cnr=lambda ids:np.array([c['contrast']/d[i]['liver_hu_std'] for i in ids for c in d[i]['components']])
v=[i for i in tb if i in vids];t=[i for i in tb if i not in vids]
print('lesion KS cnr',stats.ks_2samp(cnr(v),cnr(t)))
obs=stats.ks_2samp(cnr(v),cnr(t)).statistic
rng=np.random.default_rng(0);c=0;N=3000
for _ in range(N):
    p=rng.permutation(tb);vv_=p[:len(v)];tt=p[len(v):]
    c+= stats.ks_2samp(cnr(vv_),cnr(tt)).statistic>=obs
print('patient perm p',(c+1)/(N+1))
# per-patient
pc=np.array([d[i]['cnr'] for i in tb]);m=np.array([i in vids for i in tb])
print('patient cnr mean val/train',pc[m].mean(),pc[~m].mean(),stats.mannwhitneyu(pc[m],pc[~m]),stats.ttest_ind(pc[m],pc[~m]))
pc=np.array([d[i]['contrast_hu'] for i in tb])
print('contrast',pc[m].mean(),pc[~m].mean(),stats.mannwhitneyu(pc[m],pc[~m]))
