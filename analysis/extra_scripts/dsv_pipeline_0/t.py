import numpy as np
R='data/cache/'
vi=np.load(R+'vol_index.npy',allow_pickle=True);print(vi.shape,vi[:3],vi.dtype)
m=np.load(R+'masks.npy',mmap_mode='r');print(m.shape,m.dtype)
names=[l.strip() for l in open(R+'volumes.txt')]
print(names[:3])
