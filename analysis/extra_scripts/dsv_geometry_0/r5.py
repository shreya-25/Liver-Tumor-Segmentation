import json, numpy as np
P='analysis/output/stats.json'
st=json.load(open(P))['per_volume']
a=st[28:48]
print(sum(s['n_slices'] for s in a), sum(s['n_slices'] for s in st), sum(s['n_slices'] for s in a)/sum(s['n_slices'] for s in st))
print(sum(s['n_tumors'] for s in a), len(sum([s['components'] for s in a],[])))
print('thin<=1', sum(1 for s in st if s['spacing'][2]<=1.0), 'excl 28-47', sum(1 for i,s in enumerate(st) if s['spacing'][2]<=1.0 and not 28<=i<=47))
print([round(s['liver_hu_mean']) for s in st[48:53]])
ec=[c['eq_diam_mm'] for s in st for c in s['components']]; print(np.median(ec))
ec2=[c['eq_diam_mm'] for s in st[28:48] for c in s['components']]; print(np.median(ec2), np.median([c['ml'] for s in st for c in s['components'] ]))
# correction effect on eq diam for a: scale by (0.76^2*1.6)^(1/3)
print((0.76**2*1.6)**(1/3), 0.76**2*1.65)
sp=np.array([s['spacing'] for s in st]); o=np.r_[0:28,48:131]; print(np.median(sp[o],0), np.mean(sp[o,0]))
