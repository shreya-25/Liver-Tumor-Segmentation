import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
import pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
names_raw,raw=pickle.load(open("/tmp/ds_impl/raw_lesions.pkl","rb"))
idx=np.arange(N); tr,te=train_test_split(idx,test_size=0.2,random_state=42)
# axis-0 oversampling
ax0=np.array([raw[i]["ax0_tumor_slices"] for i in range(N)],float)
nT=ax0[tr].sum(); nN=512*len(tr)-nT
q=ax0[tr]; ess=lambda q:(q.sum())**2/(q**2).sum()
print("AXIS-0 train: tumor slices %d non %d weight %.2f draws/tumor slice/epoch %.2f ; tumor-half patient ESS %.1f ; tumor patients %d"%(nT,nN,nN/nT,(nN+nT)/(2*nT),ess(q[q>0]),(q>0).sum()))
print("val thick>2.5 tumor-bearing volumes in seed42 val:",int(((tg[te]==2)&has[te]).sum()),"; thick>2.5 volumes in train",int((tg[tr]==2).sum()))
# figure 1 split balance
F=np.array([(tg==0).astype(float),(tg==1).astype(float),(tg==2).astype(float),has.astype(float),np.log1p(tml),nt.astype(float),lhu,sx,np.log(sz)]); sdv=F.std(axis=1)
def smd(t):
    m=np.ones(N,bool); m[t]=False; return (np.abs(F[:,t].mean(1)-F[:,m].mean(1))/sdv).mean()
rng=np.random.default_rng(0)
rand=[smd(rng.permutation(N)[:27]) for _ in range(20000)]
bt=np.where(~has,0,np.where(tml<=np.quantile(tml[has],1/3),1,np.where(tml<=np.quantile(tml[has],2/3),2,3))); st=tg*4+bt
def strat_split(seed,strata,ntest=27):
    r=np.random.default_rng(seed); te_=[]
    ks=sorted(set(strata)); sizes=np.array([(strata==k).sum() for k in ks]); exact=sizes*ntest/N
    base=np.floor(exact).astype(int); rem=ntest-base.sum()
    order=np.argsort(-(exact-base)+r.random(len(ks))*1e-6)
    for j in order[:rem]: base[j]+=1
    for k,b in zip(ks,base): te_+=list(r.choice(np.where(strata==k)[0],b,replace=False))
    return np.array(te_)
strat=[smd(strat_split(s,st)) for s in range(2000)]
s42=smd(te)
print("mean SMD: random median %.3f p90 %.3f ; stratified median %.3f p90 %.3f ; seed42 %.3f"%(np.median(rand),np.quantile(rand,.9),np.median(strat),np.quantile(strat,.9),s42))
# stratified: variability of val tumor mL total and lesion count
rl=[];rm=[];sl=[];sm=[]
for s in range(3000):
    t=rng.permutation(N)[:27]; rl.append(nt[t].sum()); rm.append(tml[t].sum())
for s in range(2000):
    t=strat_split(s,st); sl.append(nt[t].sum()); sm.append(tml[t].sum())
print("val lesions random p5-p95 %s ; stratified %s ; val tumor mL random %s stratified %s"%(np.percentile(rl,[5,95]),np.percentile(sl,[5,95]),np.round(np.percentile(rm,[5,95])),np.round(np.percentile(sm,[5,95]))))
print("CV val tumor mL random %.2f stratified %.2f ; CV lesions random %.2f stratified %.2f"%(np.std(rm)/np.mean(rm),np.std(sm)/np.mean(sm),np.std(rl)/np.mean(rl),np.std(sl)/np.mean(sl)))
C=dict(a="#2a6f97",b="#e07a1f",c="#555555")
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(8,4.4))
bins=np.linspace(0,0.45,60)
ax.hist(rand,bins,color=C["c"],alpha=.55,label="random 27-volume splits (n=20,000)",density=True)
ax.hist(strat,bins,color=C["a"],alpha=.65,label="stratified splits (thickness x burden, n=2,000)",density=True)
ax.axvline(s42,color=C["b"],lw=2.5); ax.annotate("repo split (seed 42): %.3f"%s42,xy=(s42,10),xytext=(.2,11.5),color=C["b"],arrowprops=dict(arrowstyle="->",color=C["b"]))
ax.set_xlabel("mean standardised difference, validation vs train (9 volume-level covariates)"); ax.set_ylabel("density")
ax.set_title("The seed-42 split is already unusually balanced on volume covariates",loc="left",fontsize=12)
ax.legend(frameon=False,fontsize=9,loc="upper right",bbox_to_anchor=(1,.7)); fig.tight_layout()
fig.savefig(R+"analysis/figures/x_impl_split_balance.png",dpi=130); plt.close()
# figure 2: thickness groups shares (train, seed42)
tcount=np.array([ (np.load("/tmp/ds_impl/tumor_slice_axial.npy")[np.load(R+"data/cache/vol_index.npy")==i]).sum() for i in range(N)],float)
g=["<=1 mm","1-2.5 mm",">2.5 mm"]
pat=[((tg[tr]==k)&has[tr]).sum()/has[tr].sum() for k in range(3)]
les=[nt[tr][tg[tr]==k].sum()/nt[tr].sum() for k in range(3)]
sls=[tcount[tr][tg[tr]==k].sum()/tcount[tr].sum() for k in range(3)]
fig,ax=plt.subplots(figsize=(7.5,4.2)); x=np.arange(3); w=.26
for j,(v,l,c) in enumerate([(pat,"tumor-bearing patients",C["c"]),(les,"lesions",C["a"]),(sls,"tumor slices (what the sampler sees)",C["b"])]):
    b=ax.bar(x+(j-1)*w,np.array(v)*100,w,label=l,color=c)
    for xi,vi_ in zip(x+(j-1)*w,v): ax.text(xi,vi_*100+1,"%.0f%%"%(vi_*100),ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(g); ax.set_xlabel("slice thickness group (train split, seed 42)"); ax.set_ylabel("share of train total (%)")
ax.set_title("Slice-level oversampling over-weights thin-slice scans",loc="left",fontsize=12); ax.legend(frameon=False,fontsize=9); fig.tight_layout()
fig.savefig(R+"analysis/figures/x_impl_oversampling_thickness.png",dpi=130); plt.close()
# figure 3 size regimes
Lm=np.load("/tmp/ds_impl/L.npy"); d=Lm[:,1]; ml=Lm[:,2]; E=Lm[:,6]
b=[0,5,10,20,40,64,1e9]; lab=["<5","5-10","10-20","20-40","40-64",">64"]
sh=lambda w:np.array([w[(d>=b[k])&(d<b[k+1])].sum() for k in range(6)])/w.sum()*100
fig,ax=plt.subplots(figsize=(8,4.4)); x=np.arange(6); w=.26
for j,(v,l,c) in enumerate([(sh(np.ones(len(d))),"share of lesions",C["c"]),(sh(E),"share of GAN training patches (expected)",C["b"]),(sh(ml),"share of tumor volume",C["a"])]):
    ax.bar(x+(j-1)*w,v,w,label=l,color=c)
    for xi,vi_ in zip(x+(j-1)*w,v): ax.text(xi,vi_+1,"%.0f"%vi_ if vi_>=1 else "%.1f"%vi_,ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("lesion equivalent diameter (mm)"); ax.set_ylabel("%")
ax.set_title("GAN patches (centred on random tumor voxels) sit between counts and volume",loc="left",fontsize=11.5); ax.legend(frameon=False,fontsize=9); fig.tight_layout()
fig.savefig(R+"analysis/figures/x_impl_gan_patch_share.png",dpi=130); plt.close()
