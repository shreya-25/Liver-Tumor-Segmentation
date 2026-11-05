import json,numpy as np
d=json.load(open('analysis/output/stats.json'))['per_volume']
n=len(d)
tb=[v for v in d if v['tumor_ml']>0]
print(n,len(tb),sum(len(v['components']) for v in d))
for t in (10,20,40,64): 
    print(t,sum(c['eq_diam_mm']>=t for v in d for c in v['components']),sum(any(c['eq_diam_mm']>=t for c in v['components']) for v in d))
print('maxext>=64',sum(c['max_extent_mm']>64 for v in d for c in v['components']))
# expected patches
def q(v,real=True):
    tv=v['tumor_ml']*1000
    P=min(50,tv)
    return [P*c['ml']/v['tumor_ml'] for c in v['components']]
tot=sum(sum(q(v)) for v in tb);print('exp patches',tot)
# sanity: sum of component ml vs tumor_ml
print(max(abs(sum(c['ml'] for c in v['components'])-v['tumor_ml']) for v in tb))
print(sorted([(v['tumor_vox'],v['tumor_ml'],v['spacing']) for v in tb])[:5])
rng=np.random.default_rng(1)
ess=[];pat=[];n20=[];lt=[];top3=[]
for _ in range(5000):
    a=rng.permutation(n)[:65]
    qs=np.concatenate([q(d[i]) for i in a if d[i]['tumor_ml']>0]);pat.append(qs.sum())
    ess.append(qs.sum()**2/(qs**2).sum())  # ess on normalized
    ls=[len(d[i]['components']) for i in a];top=sorted(ls)[-3:];top3.append(sum(top)/sum(ls))
print(np.mean(ess),np.percentile(ess,[5,95]),np.mean(pat),np.mean(top3))
# ESS excluding tiny lesions: how many lesions have expected patches>=1
allq=np.concatenate([q(v) for v in tb]);print((allq>=1).sum(),(allq>=5).sum(),len(allq))
print('patch share by lesion size', [ (t, sum(x for v in tb for x,c in zip(q(v),v['components']) if c['eq_diam_mm']>=t)/allq.sum()) for t in (5,10,20,40,64)])
# patient-level ESS
pq=np.array([sum(q(v)) for v in tb]);print('volume-level ESS',(pq.sum()**2/(pq**2).sum()))
