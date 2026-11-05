import json,numpy as np,nibabel as nib,glob,re
from scipy import ndimage as ndi
D=''
d=json.load(open(D+'analysis/output/stats.json'))
ns=[];eds=[]
st=ndi.generate_binary_structure(3,1)
for v in d['per_volume']:
    if not v['components']: continue
    s=np.asarray(nib.load(D+'data/raw/segmentation-'+re.search(r'(\d+)',v['name']).group(1)+'.nii').dataobj)
    ax=int(np.argmax(v['shape'][:3]==np.min(v['shape'][:3]))) if False else 2
    lab,n=ndi.label(s==2,structure=st)
    if n!=len(v['components']): print('mismatch',v['name'],n,len(v['components']))
    objs=ndi.find_objects(lab)
    sl=[]
    for i,o in enumerate(objs):
        zz=np.unique(np.nonzero(lab[o]==i+1)[2]).size
        sl.append((zz,i+1,int((lab[o]==i+1).sum())))
    # match by voxel count
    byv={}
    for zz,i,vx in sl: byv.setdefault(vx,[]).append(zz)
    for c in v['components']:
        ns.append(byv[c['vox']].pop()); eds.append(c['eq_diam_mm'])
ns=np.array(ns);eds=np.array(eds)
bins=[0,5,10,20,40,64,1e9]
for a,b in zip(bins[:-1],bins[1:]):
    m=(eds>=a)&(eds<b);print(a,b,np.median(ns[m]))
print('<=2',(ns<=2).mean())
print(keys+['sz'])
A=Xa[m];B=Xa[~m]
print(np.round(np.abs(np.nanmean(A,0)-np.nanmean(B,0))/np.sqrt((np.nanvar(A,0,ddof=1)+np.nanvar(B,0,ddof=1))/2),3))
