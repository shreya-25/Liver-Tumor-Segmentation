import json,numpy as np
d=json.load(open('analysis/output/stats.json'))
e=np.array(d['hu_bins']);print(len(e),e[:3],e[-3:])
for k in ('liver','tumor'):
    h=np.array(d['hu_hist'][k],float);n=h.sum()
    cum=np.concatenate([[0],np.cumsum(h)])/n
    f=lambda x:np.interp(x,e[:len(cum)],cum)
    for lo,hi in [(-100,250),(-200,250),(-100,200),(0,200)]:
        print(k,lo,hi,'clipped %',100*(f(lo)+1-f(hi)))
pv=d['per_volume']
lm=np.array([v.get('liver_hu_mean') if v.get('liver_hu_mean') is not None else np.nan for v in pv],float);tm=np.array([v.get('tumor_hu_mean') if v.get('tumor_hu_mean') is not None else np.nan for v in pv],float)
ok=~np.isnan(tm);print('r liver mean vs tumor mean',np.corrcoef(lm[ok],tm[ok])[0,1], 'slope',np.polyfit(lm[ok],tm[ok],1))
c=tm[ok]-lm[ok];print('contrast sd',c.std(), 'liver mean sd',lm[ok].std(),'tumor mean sd',tm[ok].std())
