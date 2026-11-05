import nibabel as nib, numpy as np, json, glob, os
from concurrent.futures import ProcessPoolExecutor
W=[(-200,250),(-100,250),(-100,200),(0,200)]
def f(i):
    v=np.asanyarray(nib.load(f'data/raw/volume-{i}.nii').dataobj)
    s=np.asanyarray(nib.load(f'data/raw/segmentation-{i}.nii').dataobj)
    out={'i':i}
    for nm,m in (('L',s==1),('T',s==2)):
        x=v[m].astype(np.float32)
        out[nm+'n']=int(x.size)
        if x.size==0: continue
        out[nm+'mean']=float(x.mean()); out[nm+'sd']=float(x.std())
        out[nm+'p']=np.percentile(x,[0.1,99.9]).tolist()
        out[nm+'clip']=[float(((x<lo)|(x>hi)).mean()) for lo,hi in W]
    return out
if __name__=='__main__':
    with ProcessPoolExecutor(6) as ex: R=list(ex.map(f,range(131)))
    json.dump(R,open('/tmp/dsv_intensity_0/r.json','w'))
    print(len(R))
