import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B=''
names=open(B+'data/cache/volumes.txt').read().split('\n'); vi=np.load(B+'data/cache/vol_index.npy')
im=np.load(B+'data/cache/images.npy',mmap_mode='r'); mk=np.load(B+'data/cache/masks.npy',mmap_mode='r')
cl=json.load(open('/tmp/ds_geom/cls.json'))
R={r['i']:r for r in json.load(open('/tmp/ds_geom/a2.json'))}
starts=np.r_[0,np.cumsum(np.bincount(vi))]
acc={}
for k,nm in enumerate(names):
    n=int(nm.split('-')[1].split('.')[0]); 
    key={'A':'A  header LPS (63 vols)','C':'C  header RAS, consistent (15 vols)'}.get(cl[str(n)])
    if cl[str(n)]=='B': key='B1 header LAS (33 vols)' if R[n]['ax']=='LAS' else 'B2 header RAS, vols 28-47 (20 vols)'
    s,e=starts[k],starts[k+1]
    m=np.asarray(mk[s:e])>=1
    sel=np.flatnonzero(m.any((1,2)))
    mi=np.asarray(im[s:e][sel]).astype(np.float32).mean(0)
    lm=m[sel].mean(0)
    acc.setdefault(key,[]).append((mi,lm))
keys=sorted(acc)
fig,ax=plt.subplots(1,4,figsize=(16,4.7))
for a,k in zip(ax,keys):
    mi=np.mean([x[0] for x in acc[k]],0); lm=np.mean([x[1] for x in acc[k]],0)
    a.imshow(mi,cmap='gray'); a.contour(lm,levels=[0.3],colors='#ff7f0e',linewidths=2)
    a.set_title(k,fontsize=10); a.set_xlabel('array axis 1 (image columns)'); a.set_ylabel('array axis 0 (image rows)')
    a.set_xticks([0,128,255]); a.set_yticks([0,128,255])
fig.suptitle('Voxel arrays are not consistently oriented: mean axial image (liver slices) and liver mask (orange, 30% contour) by group',fontsize=12)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(B+'analysis/figures/x_geom_orientation.png',dpi=130)
print(keys)
