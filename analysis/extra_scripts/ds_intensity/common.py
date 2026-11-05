import json, pickle, numpy as np
R=''
S=json.load(open(R+'analysis/output/stats.json'))['per_volume']
raw=pickle.load(open('/tmp/ds_intensity/raw_scan.pkl','rb')); ca=pickle.load(open('/tmp/ds_intensity/cache_scan.pkl','rb'))
cn={c['name']:c for c in ca}
A=lambda k:np.array([s.get(k) if s.get(k) is not None else np.nan for s in S],float)
sz=np.array([s['spacing'][2] for s in S]); sx=np.array([s['spacing'][0] for s in S])
lm=A('liver_hu_mean'); ls=A('liver_hu_std'); tm=A('tumor_hu_mean'); ts=A('tumor_hu_std'); vmin=A('hu_min'); vmax=A('hu_max')
contrast=A('contrast_hu'); cnr=A('cnr'); ntum=A('n_tumors')
def q(x,p=(5,25,50,75,95)):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; return dict(n=len(x),min=round(x.min(),2),**{f'p{i}':round(float(np.percentile(x,i)),2) for i in p},max=round(x.max(),2),mean=round(x.mean(),2))
