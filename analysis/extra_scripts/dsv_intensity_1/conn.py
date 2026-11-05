import json,glob,numpy as np,nibabel as nib
from scipy import ndimage as ndi
d=json.load(open('analysis/output/stats.json'))
tot6=tot26=0; small6=0; small_merged=0; small_iso=0; sv=[]
for v in d['per_volume']:
    s=nib.load('data/raw/'+v['name'].replace('volume','segmentation') if not v['name'].startswith('segmentation') else 'data/raw/'+v['name'])
    m=np.asarray(s.dataobj)==2
    sp=np.array(s.header.get_zooms()[:3])
    l6,n6=ndi.label(m)
    l26,n26=ndi.label(m,structure=np.ones((3,3,3)))
    tot6+=n6;tot26+=n26
    sz=ndi.sum(m,l6,range(1,n6+1))
    ed=(6*sz*np.prod(sp)/np.pi)**(1/3) if n6 else []
    for i in range(n6):
        if ed[i]<5:
            small6+=1
            lab26=l26[l6==i+1][0]
            if (l26==lab26).sum()>sz[i]: small_merged+=1
            sv.append(sz[i])
print(tot6,tot26,small6,small_merged)
