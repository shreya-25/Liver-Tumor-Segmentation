import numpy as np, nibabel as nib, glob, os, pickle, sys
from scipy.ndimage import zoom, label, find_objects
from concurrent.futures import ProcessPoolExecutor
D="data/raw/"
def proc(i):
    img=nib.load(D+f"segmentation-{i}.nii")
    sp=img.header.get_zooms()[:3]
    seg=np.asarray(img.dataobj).astype(np.uint8)
    if (seg==2).sum()==0: return i,None
    z=zoom(seg,np.array(sp),order=0)
    co=np.argwhere(z>0); mn=co.min(0); mx=co.max(0)+1
    r=z[mn[0]:mx[0],mn[1]:mx[1],mn[2]:mx[2]]
    t=(r==2)
    lab,n=label(t,structure=np.ones((3,3,3)))
    objs=find_objects(lab)
    les=[]
    for k,o in enumerate(objs):
        v=int((lab[o]==k+1).sum()); ext=[s.stop-s.start for s in o]
        les.append((v,max(ext)))
    # sample patches
    rng=np.random.default_rng(i)
    cen=np.argwhere(t)
    sel=cen[rng.choice(len(cen),min(50,len(cen)),replace=False)]
    P=[]
    shp=r.shape
    for c in sel:
        st=[int(x-32) for x in c]
        sl=tuple(slice(max(0,s),max(0,s)+64) for s in st)
        reg=r[sl]; lb=lab[sl]
        sh=[st[j]<0 for j in range(3)]
        shift=[max(0,-st[j]) for j in range(3)]
        pad=64**3-reg.size
        P.append((int(lab[tuple(c)]),(reg==2).sum(),(reg==1).sum(),pad,max(shift),float(np.linalg.norm(shift)), lab[tuple(c)]))
    return i,(les,P,len(cen))
if __name__=="__main__":
    idx=sorted(int(os.path.basename(f)[13:-4]) for f in glob.glob(D+"segmentation-*.nii"))
    with ProcessPoolExecutor(6) as ex: res=list(ex.map(proc,idx))
    pickle.dump(res,open("/tmp/dsv_labels_0/res.pkl","wb"))
