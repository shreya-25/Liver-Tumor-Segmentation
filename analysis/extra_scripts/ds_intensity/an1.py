import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
import scipy.stats as st
print('slope/inter',set(r['slope_inter'] for r in raw))
print('liver mean diff max',max(abs(r['liver']['mean']-s['liver_hu_mean']) for r,s in zip(raw,S)))
print('vmin/vmax match',all(r['vmin']==s['hu_min'] for r,s in zip(raw,S)),all(r['vmax']==s['hu_max'] for r,s in zip(raw,S)))
print('liver mean',q(lm)); print('liver sd',q(ls))
grp=lambda x,e:np.digitize(x,e)
def table(groups,labels,**cols):
    for g,l in enumerate(labels):
        m=groups==g
        print(l,'n=',m.sum(),{k:(round(float(np.nanmedian(v[m])),2) if m.sum() else None) for k,v in cols.items()},{k+'_iqr':(round(float(np.nanpercentile(v[m],25)),1),round(float(np.nanpercentile(v[m],75)),1)) for k,v in list(cols.items())[:1] if m.sum()})
e=[60,80,100,120]
table(grp(lm,e),['<60','60-80','80-100','100-120','>=120'],sd=ls,contrast=contrast,cnr=cnr,sz=sz)
print('lm<40',(lm<40).sum(),'<60',(lm<60).sum(),'<80',(lm<80).sum(),'>=100',(lm>=100).sum(),'>=120',(lm>=120).sum(),'>=140',(lm>=140).sum())
print('SD>40',(ls>40).sum(),'>35',(ls>35).sum(),'<20',(ls<20).sum())
print('corr mean,sd',np.corrcoef(lm,ls)[0,1],st.spearmanr(lm,ls))
mad=np.array([r['liver']['mad'] for r in raw]); nx=np.array([r['noise_x'] for r in raw]);ny=np.array([r['noise_y'] for r in raw]);nz=np.array([r['noise_z'] for r in raw])
nsx=np.array([r['noiseSD_x'] for r in raw])
soft=np.array([r['soft_median'] for r in raw])
print('mad',q(mad));print('noise x',q(nx),'y',q(ny),'z',q(nz)); print('noiseSD x',q(nsx))
print('soft',q(soft),'corr',np.corrcoef(lm,soft)[0,1],'lm-soft',q(lm-soft))
het=np.sqrt(np.clip(ls**2-nsx**2,0,None)); print('het(SD^2-noise^2)',q(het)); print('noise share of var',q(nsx**2/ls**2))
print('SD>40 count with het',(het>35).sum())
print('slope',np.polyfit(lm,ls,1))
g=np.digitize(sz,[1.0000001,2.5000001])
table(g,['<=1','1-2.5','>2.5'],sd=ls,mad=mad,nx=nx,ny=ny,nz=nz,nsx=nsx,lm=lm,cnr=cnr,ct=contrast)
print('spearman sd~sz',st.spearmanr(sz,ls),'nx~sz',st.spearmanr(sz,nx),'nz~sz',st.spearmanr(sz,nz),'mad~sz',st.spearmanr(sz,mad),'cnr~sz',st.spearmanr(sz,cnr))
table(g,['<=1','1-2.5','>2.5'],r=nz/nx)
print('sd~sx',st.spearmanr(sx,ls),'nx~sx',st.spearmanr(sx,nx))
# also by lm group within thickness
for gg,l in enumerate(['<=1','1-2.5','>2.5']):
    m=g==gg; print(l,'liver mean median',np.median(lm[m]),'SD median',np.median(ls[m]),'het median',np.median(het[m]))
