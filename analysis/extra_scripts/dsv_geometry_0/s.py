import json,re,statistics as st
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
for v in pv: v['i']=int(re.findall(r'\d+',v['name'])[0])
pv.sort(key=lambda v:v['i'])
z=[v['liver_bbox_mm'][2] for v in pv]
print('z min/med/max',min(z),st.median(z),max(z),'slices',min(v['n_slices'] for v in pv),max(v['n_slices'] for v in pv))
for i in (22,23,31,45,46):
    v=pv[i];print(i,v['spacing'],v['n_slices'],v['liver_bbox_mm'],v['axcodes'],v['liver_bbox_vox'])
print('spacing 28-52')
for i in range(26,54):
    v=pv[i];print(i,v['spacing'],v['axcodes'],v['n_slices'])
print('>=800 slices',[(v['i'],v['n_slices']) for v in pv if v['n_slices']>=800])
print('top slices',sorted([(v['n_slices'],v['i'],v['spacing'][2]) for v in pv])[-5:])
print('lowest z',sorted([(v['liver_bbox_mm'][2],v['i'],v['n_slices']) for v in pv])[:6])
print('tumor',sorted([(round(v['tumor_ml']),v['i']) for v in pv])[-8:])
print('hu',sorted([(v['hu_max'],v['i']) for v in pv])[-8:])
print('min -3024',[v['i'] for v in pv if v['hu_min']==-3024],len([v for v in pv if v['hu_min']==-3024]))
print('hu range>10000',[(v['i'],v['hu_min'],v['hu_max']) for v in pv if v['hu_max']-v['hu_min']>6000])
c=[v['contrast_hu'] for v in pv if v['n_tumors'] and v['contrast_hu'] is not None]
print(len(c),'hyper',[(v['i'],v['contrast_hu']) for v in pv if v['n_tumors'] and v['contrast_hu'] is not None and v['contrast_hu']>0])
big=[(v['i'],round(c_['max_extent_mm'])) for v in pv for c_ in v['components'] if c_['max_extent_mm']>64]
print(len(big),big)
print('eq>64',sum(1 for v in pv for c_ in v['components'] if c_['eq_diam_mm']>64))
print('tumor volumes',sum(1 for v in pv if v['n_tumors']>0))
