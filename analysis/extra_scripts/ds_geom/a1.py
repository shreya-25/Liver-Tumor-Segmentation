import json, numpy as np, collections
S=json.load(open('analysis/output/stats.json'))
P=S['per_volume']
def idx(n): return int(n.split('-')[1].split('.')[0])
P.sort(key=lambda d: idx(d['name']))
print(P[0].keys())
th=np.array([d['spacing'][2] for d in P]); ip=np.array([d['spacing'][0] for d in P]); ip2=np.array([d['spacing'][1] for d in P])
ns=np.array([d['n_slices'] for d in P]); ix=np.array([idx(d['name']) for d in P])
print('sx!=sy',(np.abs(ip-ip2)>1e-3).sum())
c=collections.Counter(np.round(th,3)); print(sorted(c.items()))
# blocks
runs=[];prev=None
for i,d in zip(ix,np.round(th,2)):
    g='thick>2.5' if d>2.5 else ('mid' if d>1 else 'thin')
    if g!=prev: runs.append([g,i,i]);prev=g
    else: runs[-1][2]=i
print(runs)
for g,f in [('thin',lambda t:t<=1),('mid',lambda t:(t>1)&(t<=2.5)),('thick',lambda t:t>2.5)]:
    m=f(th); print(g,m.sum(),'slices',ns[m].sum(),ns[m].sum()/ns.sum(),'median n',np.median(ns[m]), 'ip med',np.median(ip[m]), ip[m].min(), ip[m].max())
    print(' idx range', ix[m].min(), ix[m].max(), 'median idx', np.median(ix[m]))
from scipy.stats import spearmanr
print('th vs idx',spearmanr(th,ix)); print('th vs ns',spearmanr(th,ns)); 
tm=np.array([d['tumor_ml'] for d in P]); lm=np.array([d['liver_ml'] for d in P]); nt=np.array([d['n_tumors'] for d in P]); st=np.array([d['slices_tumor'] for d in P])
print('th vs tumor_ml',spearmanr(th,tm)); print('th vs liver ml',spearmanr(th,lm)); print('th vs ntumors',spearmanr(th,nt))
print('ip vs idx',spearmanr(ip,ix))
for g,f in [('thin',lambda t:t<=1),('mid',lambda t:(t>1)&(t<=2.5)),('thick',lambda t:t>2.5)]:
    m=f(th); print(g,'tumor_ml med',np.median(tm[m]),'mean',tm[m].mean(),'ntum med',np.median(nt[m]),'sum',nt[m].sum(),'no-tumor',(nt[m]==0).sum(),'liver med',np.median(lm[m]))
    print('  tumor slices total',st[m].sum(), st[m].sum()/ns[m].sum())
print('total slices',ns.sum(),'tumor slices',st.sum(),'liver slices',sum(d['slices_liver'] for d in P))
# per-patient weight under slice sampling
for g,f in [('thin',lambda t:t<=1),('mid',lambda t:(t>1)&(t<=2.5)),('thick',lambda t:t>2.5)]:
    m=f(th); print(g,'share of patients',m.mean(),'share of slices',ns[m].sum()/ns.sum(),'share of tumor slices',st[m].sum()/st.sum(),'share tumor ml',tm[m].sum()/tm.sum())
# per patient weight ratio
print('ratio max/min slices',ns.max()/ns.min(), 'top5 volumes slices',sorted(ns)[-5:],'share top 5',sum(sorted(ns)[-5:])/ns.sum(), 'top10', sum(sorted(ns)[-10:])/ns.sum())
# axcodes vs thickness, idx
ax=[d['axcodes'] for d in P]
print(collections.Counter(zip(ax,[ 'thick' if t>2.5 else ('mid' if t>1 else 'thin') for t in th])))
runs=[];prev=None
for i,a in zip(ix,ax):
    if a!=prev: runs.append([a,i,i]);prev=a
    else: runs[-1][2]=i
print(runs[:60], len(runs))
# dtype
print(collections.Counter((d['vol_dtype'],d['seg_dtype']) for d in P))
print(collections.Counter((d['vol_dtype'],d['seg_dtype'],d['axcodes']) for d in P))
print('labels',collections.Counter(str(d['labels']) for d in P))
print('seg shape match',collections.Counter(d['seg_shape_match'] for d in P))
print('shape',collections.Counter(tuple(d['shape'][:2]) for d in P))
print([ (idx(d['name']),d['vol_dtype'],d['hu_min'],d['hu_max']) for d in P if d['vol_dtype']=='int32'])
print('thickness unique', sorted(set(np.round(th,3))))
print('ip unique count',len(set(np.round(ip,4))))
# thin spacing (non-integral)
# upsample factors
for d in P[:0]: pass
