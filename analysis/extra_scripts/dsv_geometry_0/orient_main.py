import nibabel as nib, numpy as np, json, os
R='data/raw/'
d=json.load(open('analysis/output/stats.json'))
out=[]
for i,v in enumerate(d['per_volume']):
    n=v['name'].replace('volume-','').replace('.nii','')
    im=nib.load(R+v['name']); sg=nib.load(R+'segmentation-%s.nii'%n)
    aff=im.affine
    seg=np.asanyarray(sg.dataobj)
    vol=np.asanyarray(im.dataobj)[::4,::4,::2]
    body=np.argwhere(vol>-300).mean(0)*np.array([4,4,2])
    li=np.argwhere(seg>0).mean(0)
    sh=np.array(seg.shape)
    rel=(li-body)/sh
    out.append(dict(i=i,n=int(n),ax=''.join(nib.aff2axcodes(aff)),rel=[float(x) for x in rel],diag=bool(np.allclose(aff[:3,:3],np.diag(np.diag(aff[:3,:3])))),shape=[int(x) for x in sh],zfrac=float(li[2]/sh[2]), zb=float(body[2]/sh[2])))
json.dump(out,open('/tmp/dsv_geometry_0/orient_zz/out.json','w'))
