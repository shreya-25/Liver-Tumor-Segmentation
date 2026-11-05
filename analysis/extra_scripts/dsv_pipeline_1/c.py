import json,numpy as np
d=json.load(open('analysis/output/stats.json'))['per_volume']
n=len(d);print(n)
tb=np.array([v['tumor_ml']>0 for v in d]);print(tb.sum())
les=np.array([len(v['components']) for v in d]);nt=np.array([v['n_tumors'] for v in d]);print(les.sum(),nt.sum())
ml=np.array([v['tumor_ml'] for v in d])
big=np.array([sum(c['eq_diam_mm']>=20 for c in v['components']) for v in d]);print(big.sum())
rng=np.random.default_rng(0)
R=[];D=[];M=[];V=[]
for _ in range(10000):
    p=rng.permutation(n);a=p[:65];b=p[65:]
    R.append((tb[a].sum(),les[a].sum(),les[b].sum()));D.append(abs(les[a].sum()-les[b].sum()));M.append(abs(ml[a].sum()-ml[b].sum())/ml.sum())
    v=a[:13];V.append((tb[v].sum(),les[v].sum(),big[v].sum()))
R=np.array(R);V=np.array(V)
print(R.mean(0),np.percentile(R[:,1],[5,95]));print(np.mean(D),np.percentile(D,95),np.mean(M),np.percentile(M,95))
print(V.mean(0),np.percentile(V[:,1],[5,95]),np.percentile(V[:,2],[5,95]))
sp=np.array([v['spacing'][2] for v in d]);print(np.unique(sp,return_counts=True))
print(sorted(les)[-5:], les.std())
