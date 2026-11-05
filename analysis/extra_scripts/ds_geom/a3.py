import json, numpy as np
S=json.load(open('analysis/output/stats.json'))
P=S['per_volume']
th=np.array([d['spacing'][2] for d in P]); ipx=np.array([d['spacing'][0] for d in P]); sh=np.array([d['shape'] for d in P])
ns=sh[:,2]
dims=sh*np.array([[d['spacing'][0],d['spacing'][1],d['spacing'][2]] for d in P])
vox1=np.prod(np.round(dims),axis=1)
raw=np.prod(sh,axis=1)
grp=np.where(th<=1,'thin',np.where(th<=2.5,'mid','thick'))
print('raw vox/vol: median',np.median(raw)/1e6,'M; 1mm vox median',np.median(vox1)/1e6,'M; float32 GB median',np.median(vox1)*4/1e9,'max',vox1.max()*4/1e9, 'min',vox1.min()*4/1e9)
for g in ['thin','mid','thick']:
    m=grp==g
    print(g,m.sum(),'z factor (th) range',th[m].min(),th[m].max(),'median',np.median(th[m]),'| inplane factor median',np.median(ipx[m]),
      'raw Mvox med',np.median(raw[m])/1e6,'1mm Mvox med',np.median(vox1[m])/1e6,'f32 GB med',np.median(vox1[m])*4/1e9,'max',vox1[m].max()*4/1e9,
      'z-extent mm med',np.median(dims[m,2]),'1mm slices med',np.median(dims[m,2]),'liver-bbox z mm med', np.median([P[i]['liver_bbox_mm'][2] for i in np.where(m)[0]]))
# fraction of 1mm-z samples that are interpolated (not on source grid) for t>1
for t in [1.25,1.5,2,2.5,3,4,5]: print('t',t,'fraction of output z planes not coincident w/ source ~',round(1-1/t,2), 'in-plane tumor lesion eq diam <t share?')
# lesions: eq diam vs thickness
comps=[]
for i,d in enumerate(P):
    for c in d['components']: comps.append((grp[i],th[i],c['eq_diam_mm'],c['max_extent_mm'],c['vox'],c['ml']))
import collections
C=np.array([(c[1],c[2],c[3],c[4]) for c in comps]); G=np.array([c[0] for c in comps])
for g in ['thin','mid','thick']:
    m=G==g; print(g,'lesions',m.sum(),'median eq diam',np.median(C[m,1]),'share<10mm',(C[m,1]<10).mean(),'share eq diam < thickness',(C[m,1]<C[m,0]).mean(),'median vox',np.median(C[m,3]),'share<=27 vox',(C[m,3]<=27).mean(), 'share <=10 vox',(C[m,3]<=10).mean())
print('all lesions eq diam<thickness',(C[:,1]<C[:,0]).mean(), ' <2*thickness',(C[:,1]<2*C[:,0]).mean())
# slices: lesions that span <=2 slices: max_extent_mm unknown z. 
# 64^3 patch at 1mm: physically 64 mm; lesion >64mm 
# 256 resize: in-plane pixel size after 512->256: 2*spacing
print('effective in-plane mm after 256 resize: median',np.median(2*ipx),'range',2*ipx.min(),2*ipx.max())
# lesions under 2 px at 256 resized
sp=np.array([2*d['spacing'][0] for d in P])
ed=[];spp=[]
for i,d in enumerate(P):
    for c in d['components']: ed.append(c['eq_diam_mm']); spp.append(sp[i])
ed=np.array(ed);spp=np.array(spp)
print('lesions with eq diam < 2 px at 256 res',(ed<2*spp).mean(), '<3px',(ed<3*spp).mean())
# slice-count based: patient weight per slice
w=1/ns
print('slice weight: thin patient mean contribution vs thick: per-patient slice count medians',[ (g,np.median(ns[grp==g])) for g in ['thin','mid','thick']])
# per patient share of slices: gini
s=np.sort(ns)/ns.sum(); print('top 10% of patients share of slices',s[-13:].sum(), 'bottom 50%',s[:65].sum())
# effective number of patients under slice sampling (1/sum p^2)
p=ns/ns.sum(); print('effective N patients slice-uniform',1/np.sum(p**2))
# tumor slices
st=np.array([d['slices_tumor'] for d in P]); pt=st/st.sum(); print('effective N patients tumor-slice-oversampled',1/np.sum(pt**2), 'top 10 vols share tumor slices',np.sort(pt)[-10:].sum())
# thin share of the tumor-slice weighted
print('check cache')
