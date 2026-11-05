import json,numpy as np
d=json.load(open('analysis/output/stats.json'))
bins=[0,5,10,20,40,64,1e9]
share=np.zeros(6);cnt=np.zeros(6);tot=0
big=0;big_ext=0
for v in d['per_volume']:
    cs=v['components']
    if not cs: continue
    sp=v['spacing']; vox1=np.array([c['ml']*1000 for c in cs])  # 1mm^3 voxels
    n=min(50,vox1.sum()); p=n*vox1/vox1.sum()
    for c,pp in zip(cs,p):
        i=np.digitize(c['eq_diam_mm'],bins)-1
        share[i]+=pp; tot+=pp
        if c['max_extent_mm']>64: big_ext+=pp
print(np.round(100*share/tot,1), tot, 'ext>64',big_ext/tot, 'ext<10', )
