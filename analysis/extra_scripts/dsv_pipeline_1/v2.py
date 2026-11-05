import json,numpy as np
from scipy import stats
d=json.load(open('analysis/output/stats.json'))['per_volume']
C=[];G=[];sz=[];th=[]
for i,v in enumerate(d):
    for c in v['components']:
        if c['contrast'] is None or c['hu_mean'] is None: continue
        C.append(c['contrast']);G.append(i);sz.append(c['eq_diam_mm']);th.append(v['spacing'][2])
C=np.array(C);G=np.array(G);sz=np.array(sz);th=np.array(th)
print(len(C),np.median(C),np.percentile(C,[25,75,5,95]),C.std(),(C>0).mean())
print('spearman',stats.spearmanr(sz,C))
print('thick>2.5 lesions',(th>2.5).sum(), (th>2.5).mean())
# ICC1
ids=np.unique(G);k=len(ids);N=len(C)
gm=C.mean();ni=np.array([(G==g).sum() for g in ids]);mi=np.array([C[G==g].mean() for g in ids])
ssb=(ni*(mi-gm)**2).sum();ssw=sum(((C[G==g]-C[G==g].mean())**2).sum() for g in ids)
msb=ssb/(k-1);msw=ssw/(N-k);n0=(N-(ni**2).sum()/N)/(k-1)
print(k,N,'ICC',(msb-msw)/(msb+(n0-1)*msw), 'n0',n0)
# excluding tiny lesions
for m in (0,5,10):
    s=sz>=m;Cs=C[s];Gs=G[s];ids=np.unique(Gs);k=len(ids);N=len(Cs)
    ni=np.array([(Gs==g).sum() for g in ids]);mi=np.array([Cs[Gs==g].mean() for g in ids]);gm=Cs.mean()
    ssb=(ni*(mi-gm)**2).sum();ssw=sum(((Cs[Gs==g]-Cs[Gs==g].mean())**2).sum() for g in ids)
    msb=ssb/(k-1);msw=ssw/(N-k);n0=(N-(ni**2).sum()/N)/(k-1)
    print(m,k,N,(msb-msw)/(msb+(n0-1)*msw))
ls=np.array([v['liver_hu_std'] for v in d if v['tumor_ml']>0]);print('liver sd median',np.median(ls), np.median(C)*2/450, np.median(ls)*2/450)
# per-volume contrast thick
pv=[(v["tumor_hu_mean"]-v["liver_hu_mean"],v["spacing"][2]) for v in d if v["tumor_ml"]>0]
a=np.array([x for x,t in pv if t<=2.5]);b=np.array([x for x,t in pv if t>2.5]);print(len(a),len(b),stats.mannwhitneyu(a,b),stats.ttest_ind(a,b))
# small-lesion contrast bias: contrast by size bins
for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,999)]:
    s=(sz>=lo)&(sz<hi);print(lo,hi,s.sum(),np.median(C[s]))
# design effect
