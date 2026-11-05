import json,numpy as np
st=json.load(open('analysis/output/stats.json'))
pv=st['per_volume']
c=[x for d in pv for x in d['components']]
print(len(c),sum(d['n_tumors'] for d in pv))
v=np.array([x['vox'] for x in c]);d=np.array([x['eq_diam_mm'] for x in c]);ml=np.array([x['ml'] for x in c])
print((v<5).sum(),(v==1).sum(),v[v<5].sum(),v.sum(),np.median(d),(d<10).mean())
s=np.sort(ml)[::-1];t=s.sum();n=len(s)
print(s[:n//10].sum()/t,s[:n//100].sum()/t,s[n//2:].sum()/t)
