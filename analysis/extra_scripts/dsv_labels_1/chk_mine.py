import pickle, json, numpy as np
ROOT='.'
PV=json.load(open(ROOT+'/analysis/output/stats.json'))['per_volume']
R=pickle.load(open('/tmp/ds_labels/raw_scan.pkl','rb'))
tv=[r for r in R if 'n6' in r]
comps=[dict(c,thick=PV[r['i']]['spacing'][2]) for r in tv for c in r['comps']]
ed=np.array([c['eq_diam_mm'] for c in comps]); ml=np.array([c['ml'] for c in comps]); vox=np.array([c['vox'] for c in comps])
print(len(ed), np.median(ed), (ed<10).mean(), (ed<5).mean())
# bimodality: histogram of log10 ml / log eq diam
h,b=np.histogram(np.log10(ml),bins=np.arange(-3,3.6,0.25))
for x,y in zip(b,h): print(round(x,2), y, '#'*(y//3))
from scipy import stats
print('skew log', stats.skew(np.log10(ml)), 'kurt', stats.kurtosis(np.log10(ml)))
# bimodality coefficient
g=stats.skew(np.log10(ml)); k=stats.kurtosis(np.log10(ml)); n=len(ml)
print('BC', (g**2+1)/(k+3*(n-1)**2/((n-2)*(n-3))))
o=np.sort(ml)[::-1]; cs=np.cumsum(o)/o.sum(); print(cs[int(.1*len(o))-1], cs[int(.01*len(o))-1], np.sort(ml)[:len(ml)//2].sum()/ml.sum())
# by thickness
for lo,hi in ((0,1),(1,2.5),(2.5,99)):
    s=[(c['thick']>lo)&(c['thick']<=hi) for c in comps]; s=np.array(s)
    print(lo,hi,s.sum(), np.median(ed[s]), (ed[s]<10).mean(), (vox[s]<5).sum())
# vox-based: tiny components in 26 conn : thickness of volumes
import collections
print(collections.Counter(round(PV[r['i']]['spacing'][2],1) for r in tv).most_common(8))
