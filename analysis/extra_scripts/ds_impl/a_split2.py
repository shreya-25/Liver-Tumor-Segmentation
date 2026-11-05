import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
from sklearn.model_selection import train_test_split
import collections
exec(open("/tmp/ds_impl/a_split.py").read().split("# imbalance metric")[0].split("idx=np.arange(N)")[0].replace("from common import *",""))  # nothing
idx=np.arange(N); tr,te=train_test_split(idx,test_size=0.2,random_state=42)
# lesion arrays
les_vol=[];les_d=[];les_ml=[];les_c=[];les_ext=[];les_cnr=[]
for i,p in enumerate(pv):
    for c in p["components"]:
        les_vol.append(i);les_d.append(c["eq_diam_mm"]);les_ml.append(c["ml"]);les_c.append(c["contrast"]);les_ext.append(c["max_extent_mm"]);les_cnr.append(c["contrast"]/p["liver_hu_std"])
les_vol=np.array(les_vol);les_d=np.array(les_d);les_ml=np.array(les_ml);les_c=np.array(les_c);les_ext=np.array(les_ext);les_cnr=np.array(les_cnr)
print("lesions",len(les_d),"sum ml",les_ml.sum().round(1))
intest=np.isin(les_vol,te)
bins=[0,5,10,20,40,64,1e9]; lab=["<5","5-10","10-20","20-40","40-64",">64"]
def tab(mask,l):
    h=np.histogram(les_d[mask],bins)[0]; v=np.array([les_ml[mask][(les_d[mask]>=bins[k])&(les_d[mask]<bins[k+1])].sum() for k in range(6)])
    print(l,"n",mask.sum(),"counts",dict(zip(lab,h)),"pct count",np.round(100*h/h.sum(),1),"pct vol",np.round(100*v/v.sum(),1))
tab(~intest,"train"); tab(intest,"val"); tab(np.ones(len(les_d),bool),"all")
print("val lesions per volume (tb):",sorted(np.bincount(les_vol[intest],minlength=N)[te].tolist()),"top vol share of val lesions",np.sort(np.bincount(les_vol[intest],minlength=N)[te])[-1]/intest.sum())
print("val lesions >=20mm:",(les_d[intest]>=20).sum()," train:",(les_d[~intest]>=20).sum(), "val >64:",(les_d[intest]>64).sum(),"train >64",(les_d[~intest]>64).sum())
print("val patients with >=1 lesion >=20mm",len(set(les_vol[intest&(les_d>=20)])),"train",len(set(les_vol[~intest&(les_d>=20)])))
# KS on lesion eq diam train vs val
from scipy.stats import ks_2samp, mannwhitneyu
print("KS eq_diam train vs val",ks_2samp(les_d[~intest],les_d[intest]))
print("KS cnr",ks_2samp(les_cnr[~intest],les_cnr[intest]))
# 200k random draws percentile
rng=np.random.default_rng(1)
F=np.array([(tg==0).astype(float),(tg==1).astype(float),(tg==2).astype(float),has.astype(float),np.log1p(tml),nt.astype(float),lhu,sx,np.log(sz)]); sdv=F.std(axis=1)
def smd_all(t):
    m=np.ones(N,bool); m[t]=False
    return np.abs(F[:,t].mean(1)-F[:,m].mean(1))/sdv
s42=smd_all(te).mean()
M=np.array([smd_all(rng.permutation(N)[:27]).mean() for _ in range(100000)])
print("100k random splits: min mean-SMD %.3f, frac below seed42 (%.3f): %.5f"%(M.min(),s42,(M<s42).mean()))
# thickness-only
tgc=np.array([ (np.abs(np.bincount(tg[rng.permutation(N)[:27]],minlength=3)-27*np.bincount(tg)/N)).max() for _ in range(20000)])
print("random: max |count dev| thick groups in val median",np.median(tgc),"p90",np.quantile(tgc,.9),"; seed42:",np.abs(np.bincount(tg[te],minlength=3)-27*np.bincount(tg)/N).max())
# sampling variability of val composition under random vs stratified
exec(open("/tmp/ds_impl/a_split.py").read().split("# stratified split")[1].split("for lab,strata")[0].replace("def strat_split","def strat_split") ) if False else None
