import json,numpy as np
R=''
d=json.load(open(R+'analysis/output/stats.json'))['per_volume']
im=np.load(R+'data/cache/images.npy',mmap_mode='r');mk=np.load(R+'data/cache/masks.npy',mmap_mode='r')
vi=np.load(R+'data/cache/vol_index.npy'); names=open(R+'data/cache/volumes.txt').read().split('\n')
print(len(names),[v['name'] for v in d[:2]],names[:2])
res=[]
for i in range(len(names)):
    idx=np.flatnonzero(vi==i); a=im[idx[0]:idx[-1]+1].astype(np.float32); m=mk[idx[0]:idx[-1]+1]
    L=a[m==1]; T=a[m==2]
    v=d[i]; r=v['hu_max']-v['hu_min']
    res.append((L.mean(),L.std(),T.mean() if T.size else np.nan, 255*(v['liver_hu_mean']-v['hu_min'])/r, r, T.std() if T.size else np.nan))
res=np.array(res)
print('liver pred-meas r',np.corrcoef(res[:,0],res[:,3])[0,1],np.abs(res[:,0]-res[:,3]).mean())
c=res[:,2]-res[:,0]; ok=~np.isnan(c)
print('contrast med',np.median(c[ok]),np.percentile(c[ok],[25,75]),(np.abs(c[ok])<3).mean(),(np.abs(c[ok])<=1).mean(),(np.abs(c[ok])<1).mean())
print('liver sd',np.median(res[:,1]))
print('liver grey',np.median(res[:,0]),res[:,0].min(),res[:,0].max())
# CNR in cache
cn=c[ok]/res[ok,1]; print('cache cnr median',np.median(cn))
step=res[:,4]/255
print('vols with step>30HU',(step>30).sum(),'>60',(step>60).sum(), 'step<=8.1', (step<=8.1).sum())
hc=np.array([ (v.get('contrast_hu') or np.nan) for v in d])
# contrast in grey vs step: spearman between |contrast grey| and step
print('cnr corr with step', np.corrcoef(np.log(step[ok]), cn)[0,1])
np.save('/tmp/dsv_pipeline_1/res.npy',res)
