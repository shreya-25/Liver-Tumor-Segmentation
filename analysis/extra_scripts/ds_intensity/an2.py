import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
import collections
print('min counts',collections.Counter(vmin.astype(int)).most_common(8))
print('max: ==3071',(vmax==3071).sum(),'>3071',(vmax>3071).sum(),'<3071',(vmax<3071).sum(),'>4000',(vmax>4000).sum(),'>10000',(vmax>10000).sum(),'max q',q(vmax))
print('max <2000',(vmax<2000).sum(), 'max values <3071 sample',sorted(vmax[vmax<3071])[:15])
fmin=A('frac_at_min'); print('frac_at_min',q(fmin),'by min value',{int(v):round(float(np.median(fmin[vmin==v])),3) for v in set(vmin)  if (vmin==v).sum()>2})
print('inter -1024 vols:',[(r['name'],S[r['k']]['vol_dtype'],int(vmin[r['k']])) for r in raw if r['slope_inter'][1]!=0][:20], sum(r['slope_inter'][1]!=0 for r in raw))
rng=vmax-vmin; print('range HU',q(rng,(5,25,50,75,95)))
print('range>=3000',(rng>=3000).sum(), '>=4000',(rng>=4000).sum(),'>=10000',(rng>=10000).sum(), '<=2000',(rng<=2000).sum())
# outliers: vmin extremes
print('vmin sorted lowest', sorted(vmin)[:6], 'vmax top', sorted(vmax)[-6:], [S[i]['name'] for i in np.argsort(vmax)[-4:]],[S[i]['name'] for i in np.argsort(vmin)[:2]])
# p001/p999 
p001=A('hu_p001');p999=A('hu_p999'); print('p001 q',q(p001),'p999',q(p999))
# effective scheme A vs cache
LA=np.array([cn[s['name']]['lm'] for s in S])/255; SA=np.array([cn[s['name']]['ls'] for s in S])/255
LA_an=(lm-vmin)/rng; SA_an=ls/rng
print('cache vs analytic liver mean abs diff max',np.nanmax(abs(LA-LA_an)),'med',np.median(abs(LA-LA_an)),' SD ratio med',np.median(SA/SA_an))
# per-volume cache min/max sanity
print('cache xmin/xmax',collections.Counter((cn[s['name']]['xmin'],cn[s['name']]['xmax']) for s in S).most_common(3))
