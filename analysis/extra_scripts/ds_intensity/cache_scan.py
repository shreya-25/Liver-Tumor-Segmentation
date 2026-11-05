import numpy as np, pickle
R='data/cache/'
a=np.load(R+'images.npy',mmap_mode='r');m=np.load(R+'masks.npy',mmap_mode='r');vi=np.load(R+'vol_index.npy')
names=open(R+'volumes.txt').read().split()
out=[]
for k,nm in enumerate(names):
    idx=np.where(vi==k)[0]; lo,hi=idx[0],idx[-1]+1
    x=np.asarray(a[lo:hi]); y=np.asarray(m[lo:hi])
    lv=x[y==1].astype(np.float64); tv=x[y==2].astype(np.float64)
    h=np.bincount(lv.astype(np.int64),minlength=256)
    out.append(dict(name=nm,k=k,n_sl=hi-lo,xmin=int(x.min()),xmax=int(x.max()),lm=lv.mean(),ls=lv.std(),
      tm=tv.mean() if tv.size else np.nan,ts=tv.std() if tv.size else np.nan,nlev=int((h>0).sum()),
      lev_p1_p99=float(np.percentile(lv,99)-np.percentile(lv,1)),body_mean=float(x.mean()),hist=h))
    print(nm,flush=True)
pickle.dump(out,open('/tmp/ds_intensity/cache_scan.pkl','wb'))
