import glob,json,numpy as np,nibabel as nib
from scipy import ndimage as ndi
from multiprocessing import Pool
D='data/raw/'
s6=ndi.generate_binary_structure(3,1);s26=ndi.generate_binary_structure(3,3)
def f(p):
    im=nib.load(p);m=np.asarray(im.dataobj)==2
    sp=[float(x) for x in im.header.get_zooms()[:3]]
    r={'f':p.split('/')[-1],'sp':sp,'tv':int(m.sum())}
    if r['tv']==0: r['6']=[];r['26']=[];return r
    sl=ndi.find_objects(m.astype(np.uint8))[0]
    sl=tuple(slice(max(s.start-1,0),s.stop+1) for s in sl);mm=m[sl]
    for k,s in(('6',s6),('26',s26)):
        l,n=ndi.label(mm,s);r[k]=np.bincount(l.ravel())[1:].tolist()
    return r
if __name__=='__main__':
    fs=sorted(glob.glob(D+'segmentation-*.nii'))
    with Pool(3) as p: R=p.map(f,fs)
    json.dump(R,open('/tmp/dsv_labels_0/mine/out.json','w'))
    print('done',len(R))
