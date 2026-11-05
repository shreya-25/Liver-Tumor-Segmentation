import numpy as np, json
vi=np.load('data/cache/vol_index.npy',mmap_mode='r')
im=np.load('data/cache/images.npy',mmap_mode='r')
print(vi.shape,im.shape,im.dtype,vi[:5])
names=[l.strip() for l in open('data/cache/volumes.txt')]
print(names[:3],len(names))
