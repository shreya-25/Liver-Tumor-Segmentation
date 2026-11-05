import nibabel as nib, numpy as np, json
R='data/raw/'
st=json.load(open('analysis/output/stats.json'))['per_volume']
W=[];H=[];Wpx=[];Hpx=[];lbz=[];lbx=[];lby=[]
for i in range(131):
    v=nib.load(R+f'volume-{i}.nii'); z=v.header.get_zooms()
    a=np.asarray(v.dataobj)[::2,::2,::2]
    m=a>-300
    # largest-slice-agnostic: use projection over z, fill; take percentile-free extents
    p=m.any(2)
    xs=np.where(p.any(1))[0]; ys=np.where(p.any(0))[0]
    # robust: use columns with >=5% of slices occupied
    frac=m.mean(2)
    xs=np.where((frac>0.05).any(1))[0]; ys=np.where((frac>0.05).any(0))[0]
    wpx=(xs.max()-xs.min()+1)*2; hpx=(ys.max()-ys.min()+1)*2
    Wpx.append(wpx);Hpx.append(hpx);W.append(wpx*z[0]);H.append(hpx*z[1])
W=np.array(W);H=np.array(H);Wpx=np.array(Wpx);Hpx=np.array(Hpx)
g=np.arange(131); a=(g>=28)&(g<=47); o=~a
print('width mm med',np.median(W[a]),np.median(W[o]),'height',np.median(H[a]),np.median(H[o]))
print('width px med',np.median(Wpx[a]),np.median(Wpx[o]),'height px',np.median(Hpx[a]),np.median(Hpx[o]))
print('ratio',np.median(W[o])/np.median(W[a]),np.median(H[o])/np.median(H[a]))
print('exceed p95 W',(W[a]>np.percentile(W[o],95)).sum(),'max',(W[a]>W[o].max()).sum(),W[o].max())
print('exceed p95 H',(H[a]>np.percentile(H[o],95)).sum(),'max',(H[a]>H[o].max()).sum(),H[o].max())
bb=np.array([s['liver_bbox_mm'] for s in st]); lv=np.array([s['liver_ml'] for s in st])
print('bbox med a',np.median(bb[a],0),'o',np.median(bb[o],0))
print('liver ml med/mean a',np.median(lv[a]),lv[a].mean(),'o',np.median(lv[o]),lv[o].mean())
bp=np.array([s['liver_bbox_vox'] for s in st]); print('bbox vox med',np.median(bp[a],0),np.median(bp[o],0))
print('liver bbox px frac in-plane a',np.median(bp[a][:,0]),'o',np.median(bp[o][:,0]))
# liver vox size in pixels
print('liver vox med a',np.median([s['liver_vox'] for s in np.array(st)[a]]))
