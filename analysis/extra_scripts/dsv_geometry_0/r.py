import json, numpy as np
d=json.load(open('analysis/output/stats.json'))
pv=d['per_volume']
sh=np.array([p['shape'] for p in pv],float); sp=np.array([p['spacing'] for p in pv],float)
raw=sh.prod(1)/1e6; g=(sh*sp).prod(1)/1e6
print(len(pv),np.median(raw),np.median(g),np.median(g)*4/1e3,g.max()*4/1e3,g.min()*4/1e3)
sz=sp[:,2]
grp=np.where(sz<1.0001,'thin',np.where(sz<2.6,'mid','thick'))
print({k:(int((grp==k).sum()),np.median(g[grp==k]),np.median(raw[grp==k]),sz[grp==k].min(),sz[grp==k].max(),np.median(sz[grp==k])) for k in ['thin','mid','thick']})
print('thick sz',sorted(sz[grp=='thick']))
print('inplane',sp[:,0].min(),sp[:,0].max(),np.median(sp[:,0]),np.unique(sp[:,0]==sp[:,1]))
print('eff px', sp[:,0].min()*2,sp[:,0].max()*2,np.median(sp[:,0])*2, np.unique(sh[:,0]))
# lesions
allc=[];
for p in pv:
  for c in p['components']:
    allc.append((c['eq_diam_mm'],c['vox'],p['spacing'],grp[pv.index(p)]))
ed=np.array([a[0] for a in allc]);print(len(ed))
effpx=np.array([a[2][0]*sh[0,0]/256 for a in allc])
print((ed/effpx<2).mean(),(ed/effpx<3).mean())
v=np.array([a[1] for a in allc]);gg=np.array([a[3] for a in allc])
for k in ['thin','mid','thick']: print(k,(v[gg==k]<=27).mean(),(gg==k).sum())
print((sz>2.6).sum()/131)
for k in ['thin','mid','thick']: print(k, np.median(sp[grp==k][:,0]))
