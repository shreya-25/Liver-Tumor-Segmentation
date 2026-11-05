import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
from sklearn.model_selection import train_test_split
idx=np.arange(N)
tr,te=train_test_split(idx,test_size=0.2,random_state=42)
print("train",len(tr),"test",len(te))
print("test names",[names[i] for i in te])
def summ(ix,label):
    g=[(tg[ix]==k).sum() for k in range(3)]
    print(f"{label}: n={len(ix)} thick<=1/1-2.5/>2.5={g} tumor-bearing={has[ix].sum()} ({has[ix].mean():.2f}) lesions={nt[ix].sum()} "
          f"tumor_mL total={tml[ix].sum():.0f} median(all)={np.median(tml[ix]):.1f} median(tb)={np.median(tml[ix][has[ix]]):.1f} "
          f"liverHU mean={lhu[ix].mean():.1f} sd={lhu[ix].std():.1f} median sz={np.median(sz[ix]):.2f} slices={nsl[ix].sum()} tumor slices={sltum[ix].sum()} ({sltum[ix].sum()/nsl[ix].sum():.3f})")
summ(tr,"TRAIN"); summ(te,"VAL"); summ(idx,"ALL")
# top tumor volumes
o=np.argsort(-tml)[:5]
print("top tumor volumes:",[(names[i],round(tml[i]),'val' if i in te else 'train') for i in o])
print("no-tumor volumes:",[(names[i],'val' if i in te else 'train',round(sz[i],1)) for i in np.where(~has)[0]])
# burden terciles among tumor-bearing
q=np.quantile(tml[has],[1/3,2/3]); print("tercile cutoffs mL",q)
bt=np.where(~has,0,np.where(tml<=q[0],1,np.where(tml<=q[1],2,3)))
st=tg*4+bt
import collections
cnt=collections.Counter(st); print("strata counts (thick*4+burden):",sorted(cnt.items()))
print("crosstab thick x burden(0=none,1..3 terciles)")
for k in range(3): print(k,[int(((tg==k)&(bt==b)).sum()) for b in range(4)])
# large lesions in val
big=[(names[i],round(tml[i])) for i in te if tml[i]>100]; print("val vols >100 mL",big)
# imbalance metric
def feats():
    F=[(tg==0).astype(float),(tg==1).astype(float),(tg==2).astype(float),has.astype(float),np.log1p(tml),nt.astype(float),lhu,sx,np.log(sz)]
    return np.array(F)
F=feats(); fn=["thick<=1","thick1-2.5","thick>2.5","tumor-bearing","log1p tumor_mL","n_lesions","liver_HU","inplane_sp","log thick"]
sd=F.std(axis=1)
def smd(te_ix):
    m=np.ones(N,bool); m[te_ix]=False
    return np.abs(F[:,te_ix].mean(1)-F[:,m].mean(1))/sd
s42=smd(te); print("SMD seed42:",dict(zip(fn,np.round(s42,2))), "mean",s42.mean().round(3),"max",s42.max().round(3))
rng=np.random.default_rng(0)
R_mean=[];R_max=[];Rall=[]
for _ in range(20000):
    t=rng.permutation(N)[:27]; s=smd(t); R_mean.append(s.mean());R_max.append(s.max());Rall.append(s)
R_mean=np.array(R_mean);R_max=np.array(R_max);Rall=np.array(Rall)
print("random splits: mean-SMD median %.3f p90 %.3f ; seed42 percentile (mean) %.1f ; max-SMD median %.3f p90 %.3f seed42 pct %.1f"%(np.median(R_mean),np.quantile(R_mean,.9),(R_mean<s42.mean()).mean()*100,np.median(R_max),np.quantile(R_max,.9),(R_max<s42.max()).mean()*100))
print("random: per-feature median SMD",dict(zip(fn,np.round(np.median(Rall,0),2))))
# random-split spread of val-set composition
rt=[];rl=[];rtb=[];rml=[]
for _ in range(20000):
    t=rng.permutation(N)[:27]; rtb.append(has[t].sum()); rl.append(nt[t].sum()); rml.append(tml[t].sum())
print("random val: tumor-bearing n p5/50/95",np.percentile(rtb,[5,50,95]),"lesions",np.percentile(rl,[5,50,95]),"tumor mL total",np.percentile(rml,[5,50,95]))
# stratified split
def strat_split(seed,strata,ntest=27):
    r=np.random.default_rng(seed); te_=[]
    ks=sorted(set(strata)); sizes=np.array([(strata==k).sum() for k in ks]); exact=sizes*ntest/N
    base=np.floor(exact).astype(int); rem=ntest-base.sum()
    order=np.argsort(-(exact-base)+r.random(len(ks))*1e-6)
    for j in order[:rem]: base[j]+=1
    for k,b in zip(ks,base):
        mem=np.where(strata==k)[0]; te_+=list(r.choice(mem,b,replace=False))
    return np.array(te_)
for lab,strata in [("thick x burden4 (12 strata)",st),("thick x burden-only-tb (merged none)",np.where(bt==0,100,st))]:
    Sm=[];Sx=[];tbs=[]
    for sd_ in range(2000):
        t=strat_split(sd_,strata); s=smd(t); Sm.append(s.mean());Sx.append(s.max()); tbs.append(has[t].sum())
    Sm=np.array(Sm);Sx=np.array(Sx)
    print(lab,"nstrata",len(set(strata)),"mean-SMD median %.3f p90 %.3f ; max-SMD median %.3f p90 %.3f ; tumor-bearing in val p5/50/95"%(np.median(Sm),np.quantile(Sm,.9),np.median(Sx),np.quantile(Sx,.9)),np.percentile(tbs,[5,50,95]))
# pick stratified seed with best balance & report
best=None
for sd_ in range(2000):
    t=strat_split(sd_,st); s=smd(t)
    if best is None or s.max()<best[0]: best=(s.max(),sd_,t,s)
print("best stratified seed",best[1],"max SMD",best[0].round(3),dict(zip(fn,np.round(best[3],2))))
tb=best[2]; trb=np.setdiff1d(idx,tb); summ(trb,"STRAT TRAIN"); summ(tb,"STRAT VAL")
print("val strata counts",sorted(collections.Counter(st[tb]).items()))
print("strat val vols",[names[i] for i in sorted(tb)])
# per-feature SMD stratified median
Sall=np.array([smd(strat_split(sd_,st)) for sd_ in range(2000)])
print("stratified per-feature median SMD",dict(zip(fn,np.round(np.median(Sall,0),2))))
# segmenter val: trivial predictor
cache=np.load(R+"data/cache/masks.npy",mmap_mode="r"); vi=np.load(R+"data/cache/vol_index.npy")
tumor_slice=np.zeros(len(vi),bool)
for s in range(0,len(vi),4000):
    tumor_slice[s:s+4000]=(np.asarray(cache[s:s+4000])==2).any(axis=(1,2))
np.save("/tmp/ds_impl/tumor_slice_axial.npy",tumor_slice)
print("axial tumor slices total",tumor_slice.sum(),"of",len(vi))
for lab,ix in [("seed42 val",te),("strat val",tb)]:
    order=np.concatenate([np.where(vi==i)[0] for i in ix]) # permuted order as train_test_split returns
    ts=tumor_slice[order]; nb=int(np.ceil(len(ts)/4)); bt_=[ts[b*4:(b+1)*4].any() for b in range(nb)]
    print(lab,"val slices",len(ts),"tumor slices",ts.sum(),f"({ts.mean():.3f}) batches",nb,"batches with no tumor GT",nb-sum(bt_),f"= {1-np.mean(bt_):.3f} (tumor Dice floor of all-background predictor)")
