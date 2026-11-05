import json, numpy as np, re, collections
from scipy import stats as st
s=json.load(open('analysis/output/stats.json'))
P=s['per_volume']
idx=[int(re.findall(r'\d+',p['name'])[0]) for p in P]
names=[p['name'] for p in P]
print(names[:5], 'sorted lexi first',sorted(names)[:5])
ax=np.array([p['axcodes'] for p in P])
I=np.array(idx)
print('28-47 axcodes',collections.Counter(ax[(I>=28)&(I<=47)]),'48-52',collections.Counter(ax[(I>=48)&(I<=52)]),'68-82',collections.Counter(ax[(I>=68)&(I<=82)]))
print('RAS idx',sorted(I[ax=='RAS']))
ns=np.array([p['n_slices'] for p in P])
cls=np.where(ax=='LPS','A',np.where(ax=='LAS','B',np.where((I>=68)&(I<=82),'C',np.where((I>=28)&(I<=47),'B','?'))))
print(collections.Counter(cls))
for c in 'ABC': print(c,ns[cls==c].sum(),ns[cls==c].sum()/ns.sum())
try:
    from sklearn.model_selection import train_test_split
    # likely train_test_split of sorted file list
    srt=sorted(names)
    tr,te=train_test_split(srt,test_size=0.2,random_state=42)
    print(len(tr),len(te))
    d={n:i for i,n in enumerate(names)}
    for nm,grp in (('train',tr),('test',te)):
        ii=[d[n] for n in grp]
        print(nm,collections.Counter(cls[ii]), 'slices',ns[ii].sum(),'tumor slices',sum(P[i]['slices_tumor'] for i in ii),sum(P[i]['slices_tumor'] for i in ii)/ns[ii].sum())
        z=np.array([P[i]['spacing'][2] for i in ii]); print(' thin',(z<=1).sum(),'mid',((z>1)&(z<=2.5)).sum(),'thick',(z>2.5).sum(),'thick slices',ns[ii][z>2.5].sum())
        nt=sum(P[i]['slices_tumor'] for i in ii); n_non=ns[ii].sum()-nt; print(' weight',n_non/nt)
        print(' no-tumor vols',sum(P[i]['tumor_vox']==0 for i in ii))
        nl=sum(1 for i in ii if P[i]['slices_liver']==0)
except Exception as e: print('sk',e)
import math
print('ceil(131*.2)',math.ceil(131*.2))
# lesions
C=[(c['eq_diam_mm'],c['ml'],c['vox'],c['contrast'],p['spacing'][2],i) for i,p in zip(idx,P) for c in p['components']]
d=np.array([c[0] for c in C]);v=np.array([c[1] for c in C]);vx=np.array([c[2] for c in C]);ct=np.array([c[3] for c in C]);zz=np.array([c[4] for c in C]);ii=np.array([c[5] for c in C])
print('n',len(C),'median',np.median(d),'<10',(d<10).mean(),'<20',(d<20).mean(),'>64',(d>64).sum(),(d>64).mean())
sv=np.sort(v)[::-1];print('top10% share',sv[:math.ceil(len(v)*.1)].sum()/v.sum(),sv[:int(len(v)*.1)].sum()/v.sum(), 'bottom50',np.sort(v)[:len(v)//2].sum()/v.sum())
bins=[0,5,10,20,40,64,1e9]
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(d>=lo)&(d<hi); print(lo,hi,m.sum(),m.mean(),v[m].sum()/v.sum())
print('hypodense lesions',(ct<0).sum(),'|c|<20',(np.abs(ct)<20).mean(),'hyperdense%',(ct>0).mean(), (ct>0).sum())
m=d<5;print('<5mm',m.sum(),'median vox',np.median(vx[m]),'<=2vox',(vx[m]<=2).mean())
print('comps <5 vox',(vx<5).sum(),'<=4',(vx<=4).sum())
print('under2px at 256 (eq diam<2*inplane*2?)')
# in-plane 
sp=np.array([p['spacing'][0] for p in P]); inpl=np.array([p['spacing'][0] for i,p in zip(idx,P) for c in p['components']])
eff=inpl*2; print('eff pixel',eff.min() if False else np.min(sp*2),np.max(sp*2),np.median(sp*2))
print('<2px', (d<2*eff).mean(),'<3px',(d<3*eff).mean())
# thick vs thin lesion sizes
for nm,m in (('thick>2.5',zz>2.5),('thin<=1',zz<=1),('z>=5',zz>=5),('z>2.5 count',zz>2.5)):
    print(nm,m.sum(),np.median(d[m]))
print('thick under 5mm',(d[zz>2.5]<5).mean())
print('lesions thin share',(zz<=1).mean())
# by 28-47 median eq diam
mm=(ii>=28)&(ii<=47);print('28-47 lesions',mm.sum(),'median d',np.median(d[mm]),'excl',np.median(d[~mm]))
print('max n_tumors',max(p['n_tumors'] for p in P))
print('n comps per vol',np.median([p['n_tumors'] for p in P if p['n_tumors']>0]))
# bbox >64 count? max_extent_mm
me=np.array([c['max_extent_mm'] for p in P for c in p['components']]); print('max_extent>64',(me>64).sum())
# dtype by vol idx
dt=np.array([p['vol_dtype'] for p in P]);sd=np.array([p['seg_dtype'] for p in P])
print('int32 idx',sorted(I[dt=='int32']),'float64 idx',sorted(I[sd=='float64']))
# z by index
z=np.array([p['spacing'][2] for p in P]);
o=np.argsort(I);print('z by idx',[(int(I[k]),round(float(z[k]),2)) for k in o][:60])
print('spearman idx z',st.spearmanr(I,z)[0])
print('not 5mm among 2..130',((z!=5)&(I>=2)&(I<=130)).sum(),'z>=4.99 total',(z>=4.99).sum())
# volumes 31,45,46 liver z extent
bb={i:p['liver_bbox_mm'] for i,p in zip(idx,P)}
print([ (k,bb[k]) for k in (31,45,46)])
# slices_tumor axial share & counts by group
# tumor-bearing patients thickness / 
