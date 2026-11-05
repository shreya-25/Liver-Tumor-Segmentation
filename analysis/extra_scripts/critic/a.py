import json, numpy as np, csv, re, collections
from scipy import stats as st
s=json.load(open('analysis/output/stats.json'))
P=s['per_volume']
rows=list(csv.DictReader(open('analysis/output/per_volume.csv')))
print(len(P),len(rows))
idx=[int(re.findall(r'\d+',p['name'])[0]) for p in P]
A=lambda k:np.array([p[k] for p in P],float)
ns=A('n_slices'); print('slices',ns.sum(),ns.min(),ns.max(),ns.max()/ns.min())
sp=np.array([p['spacing'] for p in P]); 
print('inplane',sp[:,0].min(),sp[:,0].max(),np.median(sp[:,0]),len(set(np.round(sp[:,0],4))), (sp[:,0]==sp[:,1]).all())
print('z',sp[:,2].min(),sp[:,2].max(),sorted(set(np.round(sp[:,2],3))))
print(collections.Counter(p['axcodes'] for p in P))
print(collections.Counter(p['vol_dtype'] for p in P),collections.Counter(p['seg_dtype'] for p in P),collections.Counter(p['seg_shape_match'] for p in P))
print(collections.Counter(tuple(p['shape'][:2]) for p in P))
z=sp[:,2]
g=np.where(z<=1,0,np.where(z<=2.5,1,2))
for k in range(3):
    m=g==k; print('grp',k,m.sum(),m.sum()/131,ns[m].sum(),ns[m].sum()/ns.sum(),np.median(ns[m]))
st_=A('slices_tumor'); sl=A('slices_liver')
for k in range(3):
    m=g==k; print(' tumor slices share',st_[m].sum()/st_.sum(),'tumorvol share',A('tumor_ml')[m].sum()/A('tumor_ml').sum())
print('tumor slices',st_.sum(),st_.sum()/ns.sum(),'liver slices',sl.sum()/ns.sum(), 'no liver',1-sl.sum()/ns.sum())
print('liver only slice share',(sl.sum()-st_.sum())/ns.sum())
print('spearman slices~z',st.spearmanr(ns,z)[0],'tumorburden',st.spearmanr(ns,A('tumor_ml'))[0])
# hist-based
print('top5 slices',np.sort(ns)[-5:].sum()/ns.sum())
# 28-47
m=np.array([28<=i<=47 for i in idx]); print('28-47',m.sum(),ns[m].sum(),ns[m].sum()/ns.sum(),sum(p['n_tumors'] for p,mm in zip(P,m) if mm), sp[m][:,0].min(),sp[m][:,0].max(), sp[m][:,2].min(),sp[m][:,2].max())
print('exactly 1.0 inplane count',(sp[:,0]==1.0).sum(), [i for i,v in zip(idx,sp[:,0]) if v==1.0])
bb=np.array([p['liver_bbox_mm'] for p in P])
print('bbox 28-47 median',np.median(bb[m],0),'others',np.median(bb[~m],0))
lm=A('liver_ml'); print('liver ml 28-47',np.median(lm[m]),np.mean(lm[m]),'others',np.median(lm[~m]),np.mean(lm[~m]))
print('liver median',np.median(lm))
tm=A('tumor_ml'); print('tumor vol median among tumor-bearing',np.median(tm[tm>0]),tm[tm>0].min(),tm.max())
print('no tumor', [i for i,t in zip(idx,tm) if t==0], (tm==0).sum(),'<1',(tm<1).sum(),'<5',(tm<5).sum())
print('top10% volumes share',np.sort(tm)[-13:].sum()/tm.sum(),'largest',tm.max()/tm.sum(), 'argmax',idx[int(tm.argmax())])
lv=A('liver_vox');tv=A('tumor_vox')
tot=(ns*512*512).sum()
print('voxel frac tumor',tv.sum()/tot,'liver',(lv.sum()-0)/tot, 'liver excl tumor',(lv.sum()-tv.sum())/tot)
# is liver_vox incl tumor?
print('tumor frac liver',tv.sum()/lv.sum(), np.median(tv/lv))
print('tumor fraction of pixels in tumor slices @256', tv.sum()/(st_.sum()*512*512))
# HU
for k in ['liver_hu_mean','liver_hu_std','tumor_hu_mean','tumor_hu_std','contrast_hu','cnr']:
    a=np.array([p[k] for p in P if p['tumor_vox']>0 or 'liver' in k],float)
    a=a[~np.isnan(a)]
    print(k,len(a),np.median(a),np.percentile(a,[5,25,75,95]),a.min(),a.max())
lh=A('liver_hu_mean'); print('liver<80',(lh<80).sum(),'>=120',(lh>=120).sum(),'min',lh.min(),np.sort(lh)[:3],'sd',lh.std(ddof=0),lh.std(ddof=1))
ls=A('liver_hu_std'); print('liver sd>35',(ls>35).sum(),'>40',(ls>40).sum(),'median',np.median(ls))
for k in range(3): print('liver sd by grp',np.median(ls[g==k]))
print('spearman ls~z',st.spearmanr(ls,z)[0])
tb=np.array([p['tumor_vox']>0 for p in P])
cn=np.array([p['cnr'] if p['tumor_vox']>0 else np.nan for p in P],float)
co=np.array([p['contrast_hu'] if p['tumor_vox']>0 else np.nan for p in P],float)
print('hypodense',(co[tb]<0).sum(),tb.sum(),[ (i,round(c,1)) for i,c,t in zip(idx,co,tb) if t and c>=-0.5])
print('|cnr|<1',(np.abs(cn[tb])<1).sum())
print('tumor sd median',np.nanmedian([p['tumor_hu_std'] for p in P if p['tumor_vox']>0]))
print('spearman contrast~liver mean',st.spearmanr(co[tb],lh[tb])[0], st.spearmanr(cn[tb],lh[tb])[0])
for lo,hi in [(0,60),(60,80),(80,100),(100,120),(120,999)]:
    mm=tb&(lh>=lo)&(lh<hi); print(lo,hi,mm.sum(),np.median(co[mm]) if mm.sum() else None)
# HU ranges
hmin=A('hu_min');hmax=A('hu_max')
print('min counts',collections.Counter(hmin.astype(int)).most_common(5),'min', hmin.min(),idx[int(hmin.argmin())])
print('max ==3071',(hmax==3071).sum(),'>3071',(hmax>3071).sum(),'>10000',(hmax>10000).sum(),hmax.max(),idx[int(hmax.argmax())])
rng=hmax-hmin; print('range median',np.median(rng),rng.min(),rng.max(),'>=4000',(rng>=4000).sum(),'>=10000',(rng>=10000).sum())
fm=A('frac_at_min')
for v in [-1024,-2048,-3024]: print(v,np.median(fm[hmin==v]),(hmin==v).sum())
print('liver mean 0.20..0.72? minmax position', end=' ')
pos=(lh-hmin)/rng; print(pos.min(),pos.max(),pos.std(),pos.mean())
print('contrast/range*255 median',np.median(np.abs(co[tb])/rng[tb]*255),'liver sd levels',np.median(ls/rng*255))
print('one grey level HU median',np.median(rng/255),(rng/255).min(),(rng/255).max())
print('contrast <3 levels',np.mean((np.abs(co[tb])/rng[tb]*255)<3),'<=1',np.mean((np.abs(co[tb])/rng[tb]*255)<=1))
print('liver sd <1 level',((ls/rng*255)<1).sum())
fx=np.abs(co[tb])/350*255; print('fixed [-200,250] contrast median',np.median(fx))
print('fixed [-100,250] ',np.median(np.abs(co[tb])/350*255), '[-100,200]',np.median(np.abs(co[tb])/300*255),'[0,200]',np.median(np.abs(co[tb])/200*255))
print('ratio',np.median(fx/(np.abs(co[tb])/rng[tb]*255)), np.percentile(fx/(np.abs(co[tb])/rng[tb]*255),95),(fx/(np.abs(co[tb])/rng[tb]*255)).max())
print('liver mean SD/ median contrast',lh.std(ddof=1)/np.median(np.abs(co[tb])))
print('SD of liver means',lh.std(ddof=1))
print('1.4 HU/level: 350/255',350/255)
