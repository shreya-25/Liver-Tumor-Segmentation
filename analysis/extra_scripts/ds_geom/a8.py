import json, numpy as np, collections
S=json.load(open('analysis/output/stats.json'))['per_volume']
P={int(d['name'].split('-')[1].split('.')[0]):d for d in S}
cl=json.load(open('/tmp/ds_geom/cls.json'))
# thickness series by index
runs=[];prev=None
for i in range(131):
    t=round(P[i]['spacing'][2],2)
    key=t
    if key!=prev: runs.append([t,i,i]);prev=key
    else: runs[-1][2]=i
print(runs)
# by thickness value
for t in sorted(set(round(P[i]['spacing'][2],2) for i in range(131))):
    ii=[i for i in range(131) if round(P[i]['spacing'][2],2)==t]
    ip=[P[i]['spacing'][0] for i in ii]; n=[P[i]['n_slices'] for i in ii]
    print(t,len(ii),'idx',ii[:4],'..',ii[-1],'ip',round(min(ip),3),round(float(np.median(ip)),3),round(max(ip),3),'slices med',int(np.median(n)),'range',min(n),max(n),'class',dict(collections.Counter(cl[str(i)] for i in ii)))
# z-extent covered
ext=np.array([P[i]['shape'][2]*P[i]['spacing'][2] for i in range(131)]); print('z extent mm',ext.min(),np.median(ext),ext.max())
th=np.array([P[i]['spacing'][2] for i in range(131)]); ns=np.array([P[i]['n_slices'] for i in range(131)])
# 2.5mm-5mm region: identity of 'thick'
for i in range(131):
    if ns[i]>=800 or ns[i]<=100: print('outlier slices',i,ns[i],th[i])
# in-plane
ip=np.array([P[i]['spacing'][0] for i in range(131)]); print('ip pct',np.percentile(ip,[0,10,50,90,100]))
# test/train class composition
from sklearn.model_selection import train_test_split
names=open('data/cache/volumes.txt').read().split('\n')
tr,te=train_test_split(np.arange(131),test_size=0.2,random_state=42)
num=lambda k:int(names[k].split('-')[1].split('.')[0])
for nm,s in [('train',tr),('test',te)]:
    print(nm,collections.Counter(cl[str(num(k))] for k in s), 'slices',collections.Counter())
    c=collections.Counter()
    for k in s: c[cl[str(num(k))]]+=P[num(k)]['n_slices']
    print(c)
print('test idx',sorted(num(k) for k in te))
