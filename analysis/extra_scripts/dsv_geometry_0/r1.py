import json, numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
print(len(pv))
for i,p in enumerate(pv):
    print(i,p['name'],p['spacing'],p['axcodes'],p['shape'],round(p['liver_ml']),p['liver_bbox_mm'],p['n_slices'],p['n_tumors'])
