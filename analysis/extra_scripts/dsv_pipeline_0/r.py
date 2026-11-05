import json,numpy as np
s=json.load(open('analysis/output/stats.json'))
pv=s['per_volume']
print(len(pv))
tb=[v for v in pv if v['n_tumors']>0]
print('tumor vols',len(tb))
comps=[(v['name'],c) for v in pv for c in v['components']]
print('comps',len(comps))
d=np.array([c['eq_diam_mm'] for _,c in comps]);ex=np.array([c['max_extent_mm'] for _,c in comps])
for t in (10,20,40,64):
    m=d>=t; print('eqdiam>=',t,m.sum(),len({n for (n,_),k in zip(comps,m) if k}))
    m=ex>=t; print('extent>=',t,m.sum(),len({n for (n,_),k in zip(comps,m) if k}))
# patches
tot=0
for v in pv:
    tot+=min(50,1000*v['tumor_ml']) if v['tumor_ml']>0 else 0
print('patches',tot)
# contrast
con=np.array([c['contrast'] for _,c in comps if c.get('contrast') is not None])
print(len(con),np.median(con),np.percentile(con,[25,75,5,95]),con.std(ddof=1),(con>0).mean())
# ess
ml=np.array([c['ml'] for _,c in comps])
q=[]
for v in pv:
    if v['tumor_ml']>0:
        P=min(50,1000*v['tumor_ml'])
        for c in v['components']: q.append(P*c['ml']/sum(x['ml'] for x in v['components']))
q=np.array(q);q/=q.sum();print('ess full',1/(q**2).sum())
# thickness
th=[(v['spacing'][2],v['n_tumors']) for v in pv]
print(sum(n for z,n in th if z>2.5), sum(1 for v in pv if v['spacing'][2]>2.5))
nt=np.array([v['n_tumors'] for v in pv]);print((nt>=20).sum(),nt.max())
print(sorted(nt)[-5:], sum(sorted(nt)[-3:])/nt.sum())
# subset
rng=np.random.default_rng(0)
R=[]
for _ in range(10000):
    idx=rng.choice(131,65,replace=False)
    sub=[pv[i] for i in idx]
    q=[];P=0;nl=0;n20=0;p20=set()
    for v in sub:
        if v['tumor_ml']>0:
            Pv=min(50,1000*v['tumor_ml']);P+=Pv
            T=sum(x['ml'] for x in v['components'])
            for c in v['components']:
                q.append(Pv*c['ml']/T)
                if c['eq_diam_mm']>=20:n20+=1
    q=np.array(q);nl=len(q);qq=q/q.sum()
    R.append((nl,n20,P,1/(qq**2).sum()))
R=np.array(R)
for i,n in enumerate(['lesions','>=20','patches','ess']):
    print(n,R[:,i].mean(),np.percentile(R[:,i],[2.5,97.5]))
