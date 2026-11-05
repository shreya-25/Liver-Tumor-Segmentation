import json
pv=json.load(open('analysis/output/stats.json'))['per_volume']
print(sum(v['spacing'][2]<=1.0 for v in pv), sum(v['spacing'][2]<=1.0 for v in pv[:28]+pv[48:]))
print(sum(v['n_slices'] for v in pv[28:48]), sum(v['n_slices'] for v in pv), sum(v['n_tumors'] for v in pv[28:48]))
import numpy as np
print(np.median([np.prod(v['spacing']) for v in pv[28:48]]))
