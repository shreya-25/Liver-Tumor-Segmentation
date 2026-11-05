import json,numpy as np,nibabel as nib
from scipy import ndimage as ndi
R='data/raw/'
d=json.load(open('analysis/output/stats.json'))
out=[]
for v in d['per_volume']:
    n=v['name']; 
    k=n.split('-')[-1].replace('.nii','')
    vol=np.asarray(nib.load(R+f'volume-{k}.nii').dataobj).astype(np.float32)
    seg=np.asarray(nib.load(R+f'segmentation-{k}.nii').dataobj)
    lm=vol[seg==1].mean()
    lab,nl=ndi.label(seg==2)
    objs=ndi.find_objects(lab)
    st=ndi.generate_binary_structure(3,1)
    for i,sl in enumerate(objs,1):
        sl2=tuple(slice(max(s.start-1,0),s.stop+1) for s in sl)
        m=lab[sl2]==i
        x=vol[sl2][m]
        er=ndi.binary_erosion(m,st)
        sp=v['spacing']
        eq=(6*m.sum()*sp[0]*sp[1]*sp[2]/np.pi)**(1/3)
        out.append((eq,x.mean()-lm,(x.mean()-vol[sl2][er].mean()) if er.any() else np.nan,m.sum(),er.sum()/m.sum(),x.mean()-lm>0, sp[2]))
o=np.array(out,float);np.save('o.npy',o);print(len(o))
b=[0,5,10,20,40,1e9]
for i in range(5):
    m=(o[:,0]>=b[i])&(o[:,0]<b[i+1])
    print(b[i],m.sum(),np.median(o[m,1]),np.nanmedian(o[m,2]),np.isfinite(o[m,2]).sum(),np.median(o[m,4]),np.median(o[m,3]),o[m,5].mean())
print((o[:,4]==0).sum())
