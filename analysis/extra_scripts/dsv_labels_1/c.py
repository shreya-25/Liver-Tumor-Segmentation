import numpy as np, nibabel as nib
R='.'
names=open(R+'/data/cache/volumes.txt').read().split('\n')
vi=np.load(R+'/data/cache/vol_index.npy'); cm=np.load(R+'/data/cache/masks.npy',mmap_mode='r')
print(len(names),vi.shape,cm.shape,cm.dtype)
st=np.concatenate([[0],np.cumsum(np.bincount(vi))])
bad=0;tot=0
for j in (0,40,77,130):
    num=names[j].split('-')[1].split('.')[0]
    s=np.moveaxis(np.asarray(nib.load(f"{R}/data/raw/segmentation-{num}.nii").dataobj).astype(np.uint8),2,0)
    c=np.array(cm[st[j]:st[j+1]])
    print(names[j],s.shape,c.shape,np.array_equal(s[:,1::2,1::2],c))
