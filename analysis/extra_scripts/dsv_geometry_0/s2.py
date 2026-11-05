import json,re,collections
d=json.load(open('analysis/output/stats.json'))
pv=sorted(d['per_volume'],key=lambda v:int(re.findall(r'\d+',v['name'])[0]))
for i,v in enumerate(pv): v['i']=i
print([ (v['i'],round(c['eq_diam_mm'],1)) for v in pv for c in v['components'] if c['eq_diam_mm']>64])
print(collections.Counter(round(v['spacing'][2],3) for v in pv))
print(collections.Counter(v['axcodes'] for v in pv))
print([(v['i'],v['spacing'][2],v['axcodes']) for v in pv if 48<=v['i']<=52])
print([(v['i'],round(v['spacing'][2],2)) for v in pv if v['spacing'][2]>=3 or v['spacing'][2]<=0.5])
