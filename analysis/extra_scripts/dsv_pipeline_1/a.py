import json,numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
bins=[0,5,10,20,40,64,1e9]
L=[]
for v in pv:
    comps=v['components']
    tot=sum(c['vox'] for c in comps)
    sp=v['spacing']; 
    for c in comps:
        L.append((c['eq_diam_mm'],c['max_extent_mm'],c['ml'],c['vox'],v['name'],tot,c['contrast'],sp[2]))
print(len(L), len(pv))
A=np.array([(l[0],l[1],l[2],l[3],l[5],l[6],l[7]) for l in L])
names=[l[4] for l in L]
# share A: per-lesion min(50,vox) (analyst's rule?) ; B: per-volume sample of 50 voxels (code), share ∝ voxels
bi=np.digitize(A[:,0],bins)-1
# per-volume cap
import collections
vt=collections.defaultdict(float)
for l in L: vt[l[4]]=l[5]
sA=np.minimum(50,A[:,3]) # in native voxels, code uses 1mm resampled -> ml*1000 voxels
vox1=A[:,2]*1000
sA=np.minimum(50,vox1)
sB=np.array([min(50,vt[n]*0+ sum(1 for _ in [0])) for n in names])
# B: per volume, total 1mm tumor voxels
tv=collections.defaultdict(float)
for l in L: tv[l[4]]+=l[2]*1000
sB=np.array([min(50,tv[n])*(l[2]*1000)/tv[n] for l,n in zip(L,names)])
for b in range(len(bins)-1):
    m=bi==b
    print(bins[b],bins[b+1],m.sum(), round(100*m.mean(),1), 'vol%',round(100*A[m,2].sum()/A[:,2].sum(),1),'patchA%',round(100*sA[m].sum()/sA.sum(),1),'patchB%',round(100*sB[m].sum()/sB.sum(),1),'ext>64',round(100*(A[m,1]>64).mean(),0), 'medCNRcontrast',np.median(A[m,5]))
print('n volumes with tumors', len(tv), 'tot patches B',sB.sum())
ex=A[:,1]>64
print('ext>64: lesions%',100*ex.mean(),'vol%',100*A[ex,2].sum()/A[:,2].sum(),'patchA',100*sA[ex].sum()/sA.sum(),'patchB',100*sB[ex].sum()/sB.sum())
# slice thickness
st=np.array([v['spacing'][2] for v in pv]); print(np.percentile(st,[0,25,50,75,100]))
# <5mm smaller than slice thick
m=A[:,0]<5; print((A[m,0]<A[m,6]).mean())
# per-volume concentration: lesions <10mm in how many volumes
