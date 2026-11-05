import json, numpy as np
R=""
S=json.load(open(R+"analysis/output/stats.json"))["per_volume"]
print(len(S), S[0]['name'])
rng=np.array([d['hu_max']-d['hu_min'] for d in S])
print("range median/min/max",np.median(rng),rng.min(),rng.max())
print("hu min/max extremes",min(d['hu_min'] for d in S),max(d['hu_max'] for d in S))
gl=rng/255
print("HU per grey", np.median(gl),gl.min(),gl.max())
lm=np.array([255*(d['liver_hu_mean']-d['hu_min'])/(d['hu_max']-d['hu_min']) for d in S])
print("liver grey median,p5,p95,min,max",np.median(lm),*np.percentile(lm,[5,95]),lm.min(),lm.max())
c=np.array([255*d['contrast_hu']/(d['hu_max']-d['hu_min']) for d in S])
print("contrast grey median, IQR",np.median(c),np.percentile(c,[25,75]))
print("abs<3",np.mean(np.abs(c)<3),"abs<=1",np.mean(np.abs(c)<=1))
sd=np.array([255*d['liver_hu_std']/(d['hu_max']-d['hu_min']) for d in S])
print("liver sd grey median",np.median(sd))
print("liver HU SD across vols",np.std([d['liver_hu_mean'] for d in S]),np.std([d['liver_hu_mean'] for d in S],ddof=1))
print("liver hu mean range",min(d['liver_hu_mean'] for d in S),max(d['liver_hu_mean'] for d in S))
print("contrast HU median",np.median([d['contrast_hu'] for d in S if d['contrast_hu'] is not None]))
nt=[d for d in S if d['tumor_vox']==0]; print("no tumor vols",len(nt))
