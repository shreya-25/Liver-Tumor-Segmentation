import numpy as np, json
R='data/cache/'
im=np.load(R+'images.npy',mmap_mode='r'); m=np.load(R+'masks.npy',mmap_mode='r'); vi=np.load(R+'vol_index.npy')
names=open(R+'volumes.txt').read().split('\n')
d={p['name']:p for p in json.load(open('analysis/output/stats.json'))['per_volume']}
for n in ['volume-0.nii','volume-95.nii','volume-10.nii','volume-70.nii','volume-117.nii']:
    k=names.index(n); idx=np.where(vi==k)[0]; a=np.asarray(im[idx[0]:idx[-1]+1]); s=np.asarray(m[idx[0]:idx[-1]+1])
    p=d[n]
    l=a[s==1]; t=a[s==2]
    print(n,a.min(),a.max(),'liver',l.mean().round(1),l.std().round(2),'tumor',t.mean().round(1) if t.size else None,'pred',round(p['contrast_hu']*255/(p['hu_max']-p['hu_min']),2) if p['contrast_hu'] is not None else None, 'obs',round(t.mean()-l.mean(),2) if t.size else None, 'predstd',round(p['liver_hu_std']*255/(p['hu_max']-p['hu_min']),2))
