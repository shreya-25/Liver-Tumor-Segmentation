import json, os, pickle, numpy as np, nibabel as nib, time
from scipy import ndimage as ndi
from multiprocessing import Pool
R=""
S=json.load(open(R+"analysis/output/stats.json"))
names=sorted(p["name"] for p in S["per_volume"])
byname={p["name"]:p for p in S["per_volume"]}
def work(i):
    n=names[i]; p=byname[n]
    seg_path=R+"data/raw/"+n.replace("volume","segmentation")
    if p["n_tumors"]==0:
        return i, dict(ax0_tumor_slices=0, comps=[], z_tumor=0)
    seg=np.asarray(nib.load(seg_path).dataobj)
    t=(seg==2)
    Z=t.shape[2]
    zi=np.minimum(((np.arange(256)+0.5)*Z/256).astype(int),Z-1)
    sub=t[:,1::2,:][:,:,zi]
    ax0=int(sub.any(axis=(1,2)).sum())
    zt=t.any(axis=(0,1))
    lab,nc=ndi.label(t)
    comps=[]
    objs=ndi.find_objects(lab)
    for k,sl in enumerate(objs,1):
        m=(lab[sl]==k)
        zs=np.nonzero(m.any(axis=(0,1)))[0]+sl[2].start
        comps.append(dict(vox=int(m.sum()), zs=zs.astype(np.int16)))
    return i, dict(ax0_tumor_slices=ax0, comps=comps, z_tumor=int(zt.sum()))
if __name__=="__main__":
    t0=time.time(); out={}
    with Pool(4) as pool:
        for i,r in pool.imap_unordered(work, range(len(names))):
            out[i]=r; print(i, names[i], len(r["comps"]), r["ax0_tumor_slices"], int(time.time()-t0), flush=True)
    pickle.dump((names,out), open("/tmp/ds_impl/raw_lesions.pkl","wb"))
    print("done")
