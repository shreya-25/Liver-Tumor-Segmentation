import json,numpy as np
s=json.load(open('analysis/output/stats.json'))
g=[[c['contrast'] for c in v['components']] for v in s['per_volume'] if v['components']]
g=[x for x in g if x]
k=len(g);N=sum(map(len,g));gm=np.mean(sum(g,[]))
ssb=sum(len(x)*(np.mean(x)-gm)**2 for x in g);ssw=sum(((np.array(x)-np.mean(x))**2).sum() for x in g)
msb=ssb/(k-1);msw=ssw/(N-k);n0=(N-sum(len(x)**2 for x in g)/N)/(k-1)
print(k,N,(msb-msw)/(msb+(n0-1)*msw))
g2=[x for x in g if len(x)>=2];print(len(g2))
