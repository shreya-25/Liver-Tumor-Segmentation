import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
import scipy.stats as st
# within-volume check
rows=[]
for r in raw:
    k=r['k']
    if np.isnan(contrast[k]): continue
    for l in r['lesions']:
        rows.append((k,l['eq_diam'],(l['mean']-lm[k])/contrast[k] if abs(contrast[k])>5 else np.nan,(l['mean']-l['core']) if l['core'] is not None else np.nan, l['vox']))
L=np.array(rows); d=L[:,1]
for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,1e9)]:
    m=(d>=lo)&(d<hi); print(lo,hi,'ratio to volume contrast med',round(np.nanmedian(L[m,2]),2),'mean-core HU med',round(np.nanmedian(L[m,3]),1) if (~np.isnan(L[m,3])).any() else None)
# window clip fractions
W=['-200,250','-150,250','-100,250','-100,200','-50,200','-160,240','0,200']
for w in W:
    lo,hi=map(float,w.split(','))
    l_lo=np.mean([r['liver']['winfrac'][w][0] for r in raw]);l_hi=np.mean([r['liver']['winfrac'][w][1] for r in raw])
    lmax=max(r['liver']['winfrac'][w][0]+r['liver']['winfrac'][w][1] for r in raw)
    tl=[r['tumor']['winfrac'][w] for r in raw if 'tumor' in r]
    tw=np.array([x[0]+x[1] for x in tl])
    print(w,'liver clipped mean %.3f%% (lo %.3f hi %.3f) worst-vol %.1f%%'%(100*(l_lo+l_hi),100*l_lo,100*l_hi,100*lmax),'| tumor clipped mean %.3f%% worst %.1f%%'%(100*tw.mean(),100*tw.max()),'width',hi-lo,'|contrast| levels @median %.1f'%(34*255/(hi-lo)))
# liver pct summary
P=np.array([r['liver']['pct'] for r in raw]); print('liver pcts (0.1,1,5,25,50,75,95,99,99.9) median across vols',np.median(P,0), 'p99.9 max',P[:,-1].max(),'p0.1 min',P[:,0].min())
T=np.array([r['tumor']['pct'] for r in raw if 'tumor' in r]); print('tumor pct med',np.median(T,0),'p0.1 min',T[:,0].min(),'p99.9 max',T[:,-1].max())
# body percentile windows
BP=np.array([r['body_pct'] for r in raw]); AP=np.array([r['all_pct'] for r in raw])
print('body pct [0.1,0.5,1,99,99.5,99.9] median',np.median(BP,0),'p5',np.percentile(BP,5,0),'p95',np.percentile(BP,95,0))
print('all pct median',np.median(AP,0),'p95',np.percentile(AP,95,0))
