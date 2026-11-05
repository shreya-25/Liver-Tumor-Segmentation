import json, numpy as np, collections
o=json.load(open('out.json'))
print(len(o), sum(x['diag'] for x in o))
for ax in ['LPS','LAS','RAS']:
    g=[x for x in o if x['ax']==ax]
    for k in (0,1):
        r=np.array([x['rel'][k] for x in g])
        print(ax,k,len(g),'neg',(r<0).sum(),'pos',(r>0).sum(),r.mean().round(3),r.std().round(3),r.min().round(3),r.max().round(3))
ras=[x for x in o if x['ax']=='RAS']
print('RAS ax0 pos n:',[x['n'] for x in ras if x['rel'][0]>0])
print('RAS ax0 neg n:',[x['n'] for x in ras if x['rel'][0]<0])
def cls(x):
    return ('A' if x['rel'][1]<0 else 'B') if x['rel'][0]<0 else 'C'
c=collections.Counter(cls(x) for x in o); print(c)
print({k:(c[k], sum(x['shape'][2] for x in o if cls(x)==k)) for k in c}, sum(x['shape'][2] for x in o))
print('cross',collections.Counter((cls(x),x['ax']) for x in o))
print('C ids',sorted(x['n'] for x in o if cls(x)=='C'))
z=np.array([x['zfrac'] for x in o]); zb=np.array([x['zb'] for x in o]);print(z.mean(), (z<0.5).sum(), z.min())
# liver pos relative to body z-extent: need body extent; approx
print('liver low ax0',sum(x['rel'][0]<0 for x in o))
print('smallest |rel|',sorted(abs(x['rel'][0]) for x in o)[:5], sorted(abs(x['rel'][1]) for x in o)[:5])
