import json,numpy as np
from scipy.stats import spearmanr
d=json.load(open('analysis/output/stats.json'))['per_volume']
sz=np.array([v['spacing'][2] for v in d]);sx=np.array([v['spacing'][0] for v in d])
sd=np.array([v['liver_hu_std'] for v in d]);m=np.array([v['liver_hu_mean'] for v in d])
print(len(d),np.percentile(sz,[0,25,50,75,100]))
g=[sz<=1,(sz>1)&(sz<=2.5),sz>2.5]
print([x.sum() for x in g])
for n,a in [('sd',sd),('mean',m),('cnr',np.array([v['cnr'] for v in d])),('con',np.array([v['contrast_hu'] for v in d])),('tumorvox',np.array([v['tumor_vox'] for v in d])),('liverml',np.array([v['liver_ml'] for v in d]))]:
    print(n,[round(float(np.nanmedian(a[x])),2) for x in g],spearmanr(sz,a,nan_policy='omit'))
print('sx vs sz',spearmanr(sx,sz),'sx vs sd',spearmanr(sx,sd))
# partial: sd vs sz controlling for mean
from scipy.stats import rankdata
def pr(x,y,z):
    rx,ry,rz=map(rankdata,(x,y,z))
    def res(a,b):
        A=np.c_[b,np.ones_like(b)];return a-A@np.linalg.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(res(rx,rz),res(ry,rz))[0,1]
print('partial sd~sz|mean',pr(sz,sd,m))
# sources? names
print([v['name'] for v in d][:5], [ (v['name'],v['spacing']) for v in d if v['spacing'][2]>2.5][:14])
nm=[int(v['name'].split('-')[-1].split('.')[0]) if v['name'][-1].isdigit() or 'nii' in v['name'] else 0 for v in d]
print(nm[:5])
# sd vs sz within sx
print(np.unique(np.round(sx,2),return_counts=True))
