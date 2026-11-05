import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
import scipy.stats as st
has=~np.isnan(tm)
print('tumor vols',has.sum(),'hypo',(contrast[has]<0).sum(),'hyper',(contrast[has]>0).sum(),'hyper names',[(S[i]['name'],round(contrast[i],1),int(ntum[i]),round(S[i]['tumor_ml'],2)) for i in np.where(contrast>0)[0]])
print('contrast',q(contrast[has]),'cnr',q(cnr[has]))
print('|cnr|<0.5',(abs(cnr[has])<0.5).sum(),'|cnr|<1',(abs(cnr[has])<1).sum(),'>=2',(cnr[has]<=-2).sum(), '|contrast|<10',(abs(contrast[has])<10).sum(),'<=-50',(contrast[has]<=-50).sum())
print('tumor mean',q(tm[has]),'tumor sd',q(ts[has]))
print('corr contrast vs lm',st.spearmanr(lm[has],contrast[has]),' tm vs lm',np.corrcoef(lm[has],tm[has])[0,1], 'tm sd across vols',tm[has].std(),'lm sd',lm.std())
# lesion level
rows=[]
for r in raw:
    k=r['k']
    for l in r.get('lesions',[]): rows.append((k,l['eq_diam'],l['mean']-lm[k],(l['core']-lm[k]) if l['core'] is not None else np.nan,l['n_in']/l['vox'],l['p10']-lm[k],l['vox'],sz[k],l['med']-lm[k]))
L=np.array(rows); print('lesions',len(L))
d=L[:,1];c=L[:,2];core=L[:,3];fi=L[:,4];p10=L[:,5];vox=L[:,6];lz=L[:,7]
print('lesion hyper',(c>0).sum(),'hypo',(c<0).sum(),'|c|<10',(abs(c)<10).sum(),'|c|<20',(abs(c)<20).sum(),'frac',(c>-20).mean())
edges=[0,5,10,20,40,1e9]; labs=['<5','5-10','10-20','20-40','>40']
for i,l in enumerate(labs):
    m=(d>=edges[i])&(d<edges[i+1]); mm=m&~np.isnan(core)
    print(l,'n',m.sum(),'contrast med',round(np.median(c[m]),1),'IQR',np.round(np.percentile(c[m],[25,75]),1),'hyper%',round(100*(c[m]>0).mean(),1),'|c|<10 %',round(100*(abs(c[m])<10).mean(),1),
      '| core n',mm.sum(),'core contrast med',round(np.median(core[mm]),1) if mm.sum() else None,'interior frac med',round(np.median(fi[m]),2),'median vox',int(np.median(vox[m])),'p10c',round(np.median(p10[m]),1))
print('spearman |c| vs d',st.spearmanr(d,abs(c)),'c vs log d',st.spearmanr(np.log(d),c))
# small lesions: thickness effect
small=d<10
g=np.digitize(lz,[1.0000001,2.5000001])
for gg,l in enumerate(['<=1','1-2.5','>2.5']):
    m=small&(g==gg); print('small lesions sz',l,'n',m.sum(),'contrast med',round(np.median(c[m]),1) if m.sum() else None,'IQR',np.round(np.percentile(c[m],[25,75]),1) if m.sum() else None,'med vox',np.median(vox[m]) if m.sum() else None)
    m=(d>=10)&(g==gg); print('   lesions>=10 sz',l,'n',m.sum(),'contrast med',round(np.median(c[m]),1) if m.sum() else None)
# lesions with vox < 5, 27 interior
print('lesions with no interior voxel',(fi==0).sum(),'frac',(fi==0).mean(),' <=5 vox',(vox<=5).sum())
# volume weighted
print('vol-weighted contrast',np.sum(c*vox)/vox.sum(), 'count-weighted median',np.median(c))
# per-volume contrast from stats vs mean over lesions
# expected PV: contrast attenuation model: lesion of diameter D, blur sigma -> check ratio of small to large
big=np.median(c[d>=20]); print('median contrast big',big,'ratio small(<10)/big',np.median(c[d<10])/big,' 5-10/big',np.median(c[(d>=5)&(d<10)])/big,'<5/big',np.median(c[d<5])/big)
# lesion-level contrast vs liver mean? intra-volume
# fraction of lesions with contrast between -10 and 10 
