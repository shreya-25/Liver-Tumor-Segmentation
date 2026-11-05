import json, numpy as np, nibabel as nib, pickle
from scipy import ndimage as ndi
from multiprocessing import Pool
ROOT='.'
PV=json.load(open(ROOT+'/analysis/output/stats.json'))['per_volume']
S26=ndi.generate_binary_structure(3,3); S6=ndi.generate_binary_structure(3,1)
def w(p):
    num=p['name'].split('-')[1].split('.')[0]
    seg=np.asarray(nib.load(f"{ROOT}/data/raw/segmentation-{num}.nii").dataobj).astype(np.uint8)
    m=seg==2
    out=[]
    if not m.any(): return p['name'],out
    sp=p['spacing']
    for k,S in ((26,S26),(6,S6)):
        lab,n=ndi.label(m,structure=S)
        tot=np.bincount(lab.ravel(),minlength=n+1)[1:]
        # retained at 1::2,1::2 over (H,W)
        ret=np.bincount(lab[1::2,1::2,:].ravel(),minlength=n+1)[1:]
        objs=ndi.find_objects(lab)
        for i in range(n):
            o=objs[i]
            ext=[(o[a].stop-o[a].start)*sp[a] for a in range(3)]
            out.append((k,p['name'],sp[2],int(tot[i]),int(ret[i]),ext))
    return p['name'],out
if __name__=='__main__':
    with Pool(8) as pl: R=pl.map(w,PV)
    pickle.dump(R,open('/tmp/dsv_labels_1/R.pkl','wb'))
