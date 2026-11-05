import json,numpy as np,nibabel as nib
from scipy import ndimage as ndi
D=''
pv=json.load(open(D+'analysis/output/stats.json'))['per_volume']
out=[]
for v in pv:
    if v['n_tumors']==0: continue
    s=nib.load(D+'data/raw/'+v['name'].replace('volume','segmentation'))
    T=np.asarray(s.dataobj)==2; sp=s.header.get_zooms()[:3]
    r={'sz':v['spacing'][2]}
    for k,st in (('6',1),('26',3)):
        lab,n=ndi.label(T,ndi.generate_binary_structure(3,st))
        sl=ndi.find_objects(lab); c=np.bincount(lab.ravel())[1:]
        r[k]=(n,sum(q[2].stop-q[2].start==1 for q in sl),(6*c*np.prod(sp)/np.pi)**(1/3))
    out.append(r)
g=lambda x:0 if x<=1 else (1 if x<=2.5 else 2)
for k in ('6','26'):
  for gg in range(3):
    r=[x for x in out if g(x['sz'])==gg]
    eq=np.concatenate([x[k][2] for x in r])
    print(k,gg,np.median([x[k][0] for x in r]),sum(x[k][1] for x in r)/sum(x[k][0] for x in r),np.median(eq),np.mean(eq<10))
