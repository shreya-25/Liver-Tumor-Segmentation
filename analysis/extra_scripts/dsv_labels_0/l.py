import nibabel as nib, numpy as np, glob
from scipy import ndimage as ndi
from multiprocessing import Pool
files=sorted(glob.glob('data/raw/segmentation-*.nii'))
s6=ndi.generate_binary_structure(3,1); s26=np.ones((3,3,3))
def f(p):
    im=nib.load(p); seg=np.asanyarray(im.dataobj).astype(np.uint8); sp=im.header.get_zooms()[:3]
    out={}
    # restrict to liver bbox
    idx=np.argwhere(seg>0); lo=np.maximum(idx.min(0)-2,0); hi=idx.max(0)+3
    seg=seg[tuple(slice(a,b) for a,b in zip(lo,hi))]
    tum=seg==2
    for k,st in (('6',s6),('26',s26)):
        lab,n=ndi.label(tum,structure=st); r=[]
        for i,sl in enumerate(ndi.find_objects(lab),1):
            sl2=tuple(slice(max(s.start-1,0),s.stop+1) for s in sl)
            m=lab[sl2]==i; sg=seg[sl2]
            shell=ndi.binary_dilation(m,structure=s6)&~m
            s=sg[shell]
            eq=(6*m.sum()*np.prod(sp)/np.pi)**(1/3)
            r.append((eq,bool((shell&(sg==0)).any()),float((s==0).mean()),float((s==1).mean())))
        out[k]=r
    return out
if __name__=='__main__':
    with Pool(4) as p: R=p.map(f,files)
    for k in ('6','26'):
        L=[x for r in R for x in r[k]]
        eq=np.array([x[0] for x in L]);t=np.array([x[1] for x in L]);fb=np.array([x[2] for x in L]);f1=np.array([x[3] for x in L])
        print(k,len(L),'touch',t.sum(),t.mean(),'fb>=.5',(fb>=.5).sum(),(fb>=.5).mean(),'>.99',(fb>.99).sum(),'surr',(f1>=.999).mean())
        for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,1e9)]:
            m=(eq>=lo)&(eq<hi); print(' ',lo,m.sum(),round(t[m].mean(),3),round((fb[m]>=.5).mean(),3))
