import json, numpy as np, collections
R=json.load(open('/tmp/ds_geom/a2.json')); R.sort(key=lambda r:r['i'])
for r in R:
    if not np.allclose(r['aff'],r['saff']):
        print('MISMATCH',r['i'],r['ax'],r['sax'],np.round(np.array(r['aff'])[:3,3],1),np.round(np.array(r['saff'])[:3,3],1),np.round(np.diag(np.array(r['aff']))[:3],3),np.round(np.diag(np.array(r['saff']))[:3],3))
# RAS subgroup
def rel(r):
    ext=np.array(r['body_bb'][1])-np.array(r['body_bb'][0]); return (np.array(r['liv_c'])-np.array(r['body_c']))/np.maximum(ext,1)
ras=[r for r in R if r['ax']=='RAS']
low=[r['i'] for r in ras if rel(r)[0]<0]; high=[r['i'] for r in ras if rel(r)[0]>0]
print('RAS liver low ax0 (header-inconsistent):',low); print('RAS liver high (consistent):',high)
print('qform codes in low',collections.Counter(tuple(r['qform']) for r in ras if r['i'] in low),'high',collections.Counter(tuple(r['qform']) for r in ras if r['i'] in high))
print('qform by axcode',collections.Counter((r['ax'],tuple(r['qform'])) for r in R))
print('dtype in low vs high',collections.Counter((r['vdt'],r['sdt']) for r in ras if r['i'] in low),collections.Counter((r['vdt'],r['sdt']) for r in ras if r['i'] in high))
# z check: liver z relative to body z bbox
zf=[]
for r in R:
    bb=np.array(r['body_bb']); zf.append((r['liv_c'][2]-bb[0][2])/max(bb[1][2]-bb[0][2],1))
zf=np.array(zf); print('liver z within body-bbox frac: mean',zf.mean().round(3),'min',zf.min().round(3),'max',zf.max().round(3),'n<0.5',(zf<0.5).sum())
# classes
cls=collections.Counter()
for r in R:
    x=rel(r); cls[('ax0 liver '+('low' if x[0]<0 else 'high'),'ax1 anterior '+('low' if x[1]<0 else 'high'))]+=1
print(cls)
cl={}
for r in R:
    x=rel(r); cl[r['i']]=('A' if (x[0]<0 and x[1]<0) else 'B' if (x[0]<0 and x[1]>0) else 'C' if (x[0]>0 and x[1]>0) else 'D')
print('class by index runs'); prev=None
runs=[]
for i in range(131):
    if cl[i]!=prev: runs.append([cl[i],i,i]); prev=cl[i]
    else: runs[-1][2]=i
print(runs)
# mismatch vols data
for i in [r['i'] for r in R if not np.allclose(r['aff'],r['saff'])]:
    print(i,cl[i],R[i]['shape'],R[i]['zooms'])
json.dump(cl,open('/tmp/ds_geom/cls.json','w'))
# cross with thickness
S=json.load(open('analysis/output/stats.json'))['per_volume']
P={int(d['name'].split('-')[1].split('.')[0]):d for d in S}
g=lambda t:'thin' if t<=1 else 'mid' if t<=2.5 else 'thick'
print(collections.Counter((cl[i],g(P[i]['spacing'][2])) for i in range(131)))
ns=collections.Counter()
for i in range(131): ns[cl[i]]+=P[i]['n_slices']
print('slices by class',ns,{k:round(v/58638,3) for k,v in ns.items()})
tsl=collections.Counter()
for i in range(131): tsl[cl[i]]+=P[i]['slices_tumor']
print('tumor slices by class',tsl)
