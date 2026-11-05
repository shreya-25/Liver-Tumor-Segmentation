import json
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
def idx(n):
    import re
    return int(re.findall(r'\d+',n)[-1])
print(pv[0]['name'])
for v in pv:
    i=idx(v['name'])
    if i in (4,17,18,22,23,27,28,31,45,46,47,48,52,105,127): print(i,v['shape'],v['spacing'],v['axcodes'],v['n_slices'],[round(x) for x in v['liver_bbox_mm']],v['slices_liver'])
import numpy as np
bb=np.array([v['liver_bbox_mm'] for v in pv]);print(np.sort(bb[:,2])[:6],np.median(bb[:,2]),bb[:,2].max())
ns=np.array([v['n_slices'] for v in pv]);print(ns.min(),ns.max(),sorted(ns)[-12:])
print(sorted([(round(v['tumor_ml']),idx(v['name'])) for v in pv])[-8:])
big=[(idx(v['name']),c['max_extent_mm']) for v in pv for c in v['components'] if c['max_extent_mm']>64]
print(len(big),big)
big2=[(idx(v['name']),c['eq_diam_mm']) for v in pv for c in v['components'] if c['eq_diam_mm']>64]
print(len(big2))
print(sum(1 for v in pv if v['hu_min']==-3024), [(idx(v['name']),v['hu_max']) for v in pv if v['hu_max']>5000])
print([(idx(v['name']),v['contrast_hu']) for v in pv if v['contrast_hu'] and v['contrast_hu']>0], sum(1 for v in pv if v['n_tumors']>0))
