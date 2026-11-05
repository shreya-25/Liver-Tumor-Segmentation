import json, collections
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
print(collections.Counter((p['vol_dtype'],p['seg_dtype'],p['axcodes'] if isinstance(p['axcodes'],str) else ''.join(p['axcodes'])) for p in pv))
f=[int(p['name'].split('-')[1].split('.')[0]) for p in pv if p['seg_dtype']=='float64'];print(sorted(f))
i=[int(p['name'].split('-')[1].split('.')[0]) for p in pv if p['vol_dtype']=='int32'];print(sorted(i))
print(sorted((p['hu_min'],int(p['name'].split('-')[1].split('.')[0])) for p in pv if p['hu_min'] not in(-1024,-2048,-3024)))
# volumes where range>4095 and tumor contrast
import numpy as np
for k,rg in [(0,2000),(2000,4096),(4096,8000),(8000,1e9)]:
    s=[p for p in pv if k<=p['hu_max']-p['hu_min']<rg and p['n_tumors']]
    print(k,len(s),np.median([abs(p['contrast_hu'])*255/(p['hu_max']-p['hu_min']) for p in s]) if s else None)
