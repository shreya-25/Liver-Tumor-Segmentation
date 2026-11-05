import nibabel as nib, numpy as np
from scipy.ndimage import uniform_filter
res=[]
for i in range(0,131,6):
    v=np.asanyarray(nib.load(f'data/raw/volume-{i}.nii').dataobj).astype(np.float32)
    s=np.asanyarray(nib.load(f'data/raw/segmentation-{i}.nii').dataobj)
    m=(s==1)
    # in-plane noise: diff of adjacent pixels within liver, robust
    d=(v[1:,:,:]-v[:-1,:,:])/np.sqrt(2); mm=m[1:]&m[:-1]
    x=d[mm]; res.append(1.4826*np.median(np.abs(x-np.median(x))))
print(np.median(res),np.round(res,1))
