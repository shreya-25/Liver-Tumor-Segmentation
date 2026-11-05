import nibabel as nib, numpy as np
R='data/raw/'
out=[]
for i in range(131):
    s=nib.load(R+f'segmentation-{i}.nii'); v=nib.load(R+f'volume-{i}.nii')
    m=np.asarray(s.dataobj)>=1
    idx=np.argwhere(m[::2,::2,::2])*2
    cx=idx[:,0].mean()/m.shape[0]; cy=idx[:,1].mean()/m.shape[1]
    # body centroid
    sub=np.asarray(v.dataobj)[::4,::4,::4]>-300
    b=np.argwhere(sub)
    bx=b[:,0].mean()*4/m.shape[0]
    out.append((i,nib.aff2axcodes(v.affine),round(cx,2),round(cy,2),round(bx,2)))
import collections
g=collections.defaultdict(list)
for o in out: g[o[1]].append(o[2])
for k,vv in g.items(): print(k,len(vv),np.mean(vv),np.min(vv),np.max(vv))
print([o for o in out if o[0] in (27,28,29,47,48,68,69,82)])
print('28-47',np.mean([o[2] for o in out[28:48]]),'y',np.mean([o[3] for o in out[28:48]]))
print('LAS others y',np.mean([o[3] for o in out if o[1]==('L','A','S')]))
