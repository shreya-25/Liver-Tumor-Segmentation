import numpy as np
v=np.load('data/cache/vol_index.npy')
print(v.shape,np.bincount(v).min(),np.bincount(v).max(),len(np.bincount(v)))
