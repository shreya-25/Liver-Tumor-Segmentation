import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kruskal, ks_2samp
names_raw,raw=pickle.load(open("/tmp/ds_impl/raw_lesions.pkl","rb"))
assert names_raw==names
idx=np.arange(N); tr,te=train_test_split(idx,test_size=0.2,random_state=42)
vi=np.load(R+"data/cache/vol_index.npy"); tsl=np.load("/tmp/ds_impl/tumor_slice_axial.npy")
# check component consistency
bad=0
for i,p in enumerate(pv):
    a=sorted(c["vox"] for c in p["components"]); b=sorted(c["vox"] for c in raw[i]["comps"])
    if a!=b: bad+=1
print("volumes where raw-pass component voxels differ from stats.json:",bad)
# axis-0 vs axial tumor slice counts
ax0=np.array([raw[i]["ax0_tumor_slices"] for i in range(N)])
print("axis-0 (default, non-LITS_AXIAL) tumor slices (post 256 nearest) total",ax0.sum(),"of",512*N,"=",round(ax0.sum()/(512*N),3),"; axial",tsl.sum(),"of",len(vi),"=",round(tsl.mean(),3))
print("train-only ax0 tumor frac",ax0[tr].sum()/(512*len(tr)),"axial",tsl[np.isin(vi,tr)].mean())
# ---- oversampling ESS (axial cache, seed-42 train)
def ess(q): q=np.asarray(q,float); q=q/q.sum(); return 1/np.sum(q**2)
trmask=np.isin(vi,tr); tr_slices=np.where(trmask)[0]
isT=tsl[tr_slices]; nT=isT.sum(); nN=len(isT)-nT; w=np.where(isT,nN/nT,1.0)
print(f"TRAIN axial: slices {len(isT)} tumor {nT} non {nN} weight {nN/nT:.2f}; P(tumor)=",round((w[isT].sum())/w.sum(),3), "expected draws per tumor slice per epoch",round(len(isT)*(nN/nT)/w.sum(),2))
pat=vi[tr_slices]
q_pat=np.bincount(pat,weights=w,minlength=N)[tr]; q_unif=np.bincount(pat,minlength=N)[tr].astype(float)
print("patient-level ESS of draws: oversampled %.1f vs uniform %.1f (of %d train patients)"%(ess(q_pat),ess(q_unif),len(tr)))
tcount=np.bincount(pat[isT],minlength=N)[tr].astype(float)
print("tumor-half patient ESS %.1f ; patients with tumor slices %d ; top1 share %.3f top5 %.3f top10 %.3f"%(ess(tcount),(tcount>0).sum(),np.sort(tcount)[-1]/tcount.sum(),np.sort(tcount)[-5:].sum()/tcount.sum(),np.sort(tcount)[-10:].sum()/tcount.sum()))
print("Kish ESS slices (weights) %.0f of %d"%(ess(w),len(w)))
# lesion-level
lesq={}
for i in tr:
    zsl=np.where(vi==i)[0]; z_t=np.where(tsl[zsl])[0]
    comps=raw[i]["comps"]
    for z in z_t:
        inl=[k for k,c in enumerate(comps) if z in set(c["zs"].tolist())] if False else None
# faster: per comp set
for i in tr:
    comps=raw[i]["comps"]; zsl=np.where(vi==i)[0]; flag=tsl[zsl]
    sets=[set(c["zs"].tolist()) for c in comps]
    for z in np.where(flag)[0]:
        inl=[k for k,s in enumerate(sets) if z in s]
        for k in inl: lesq[(i,k)]=lesq.get((i,k),0)+1/len(inl)
ql=np.array(list(lesq.values()))
nles_tr=sum(len(raw[i]["comps"]) for i in tr)
print("train lesions %d, lesions visible in a flagged slice %d, lesion-level ESS of tumor half %.1f; top 10 lesions share %.3f"%(nles_tr,len(ql),ess(ql),np.sort(ql)[-10:].sum()/ql.sum()))
# thickness-group shares among tumor slices / patients / lesions (train)
tgt=tg[tr]; ts_by=np.array([tcount[tgt==g].sum() for g in range(3)]); pt_by=np.array([((tcount>0)&(tgt==g)).sum() for g in range(3)])
ls_by=np.array([sum(nt[i] for i in tr if tg[i]==g) for g in range(3)])
print("thick groups (<=1,1-2.5,>2.5): tumor-slice share",np.round(ts_by/ts_by.sum(),3),"tumor-patient share",np.round(pt_by/pt_by.sum(),3),"lesion share",np.round(ls_by/ls_by.sum(),3))
print("median tumor slices per tumor-patient by group:",[float(np.median(tcount[(tcount>0)&(tgt==g)])) for g in range(3)])
# physical z-extent
print("median tumor-slice mm per tumor patient (train):",np.median((tcount*sz[tr])[tcount>0]).round(1))
# hard negatives
sl_liver=np.array([p["slices_liver"] for p in pv]); 
print("non-tumor slices that contain liver (train, stats slices_liver - slices_tumor) share:",round((sl_liver[tr]-sltum[tr]).sum()/nN,3))
# ---- lesion arrays
L=[]
for i,p in enumerate(pv):
    for k,c in enumerate(p["components"]):
        L.append((i,c["eq_diam_mm"],c["ml"],c["max_extent_mm"],c["contrast"],c["contrast"]/p["liver_hu_std"],c["hu_mean"]))
L=np.array(L); lv=L[:,0].astype(int); d=L[:,1]; ml=L[:,2]; ext=L[:,3]; con=L[:,4]; cnr=L[:,5]; hum=L[:,6]
print("lesions",len(L))
print("contrast: median %.1f IQR [%.1f,%.1f] p5 %.1f p95 %.1f SD %.1f; hyperdense(>0) %.1f%%"%(np.median(con),*np.percentile(con,[25,75]),*np.percentile(con,[5,95]),con.std(),100*(con>0).mean()))
print("lesion hu_mean median %.1f p5 %.1f p95 %.1f"%(np.median(hum),*np.percentile(hum,[5,95])))
print("CNR median %.2f IQR [%.2f,%.2f]"%(np.median(cnr),*np.percentile(cnr,[25,75])))
print("Spearman size vs contrast",spearmanr(d,con))
print("contrast in GAN clip units (x2/450): median %.3f; liver SD median units %.3f"%(np.median(con)*2/450,np.median([p['liver_hu_std'] for p in pv])*2/450))
# ICC(1) contrast
def icc1(vals,grp):
    g=np.unique(grp); ks=np.array([(grp==x).sum() for x in g]); keep=g[ks>=2]
    m=np.isin(grp,keep); vals=vals[m]; grp=grp[m]; g=np.unique(grp); ks=np.array([(grp==x).sum() for x in g]); n=len(vals)
    gm=np.array([vals[grp==x].mean() for x in g]); mu=vals.mean()
    ssb=(ks*(gm-mu)**2).sum(); ssw=sum(((vals[grp==x]-gm[j])**2).sum() for j,x in enumerate(g))
    msb=ssb/(len(g)-1); msw=ssw/(n-len(g)); n0=(n-(ks**2).sum()/n)/(len(g)-1)
    return (msb-msw)/(msb+(n0-1)*msw), len(g), n
print("ICC(1) lesion contrast across patients (patients with >=2 lesions): %.3f (patients %d, lesions %d)"%icc1(con,lv))
print("ICC(1) log eq-diam: %.3f"%icc1(np.log(d),lv)[0])
ltg=tg[lv]
print("contrast by thick group medians",[round(float(np.median(con[ltg==g])),1) for g in range(3)],"lesion counts",[int((ltg==g).sum()) for g in range(3)],"kruskal p",kruskal(*[con[ltg==g] for g in range(3)]).pvalue)
# lesions per patient
lpp=nt[has]; print("lesions per tumor patient: median %d mean %.1f max %d; top 3 patients hold %.1f%% of lesions; patients with >=20 lesions: %d"%(np.median(lpp),lpp.mean(),lpp.max(),100*np.sort(lpp)[-3:].sum()/lpp.sum(),(lpp>=20).sum()))
print("lesion-level ESS from patient clustering (Kish design effect 1+(m-1)ICC)")
# ---- GAN patch expectations (1 mm iso: tumor voxels ~ ml*1000)
npat_vol=np.minimum(50,np.floor(tml*1000)); 
E=np.array([npat_vol[int(v)]*m/tml[int(v)] for v,m in zip(lv,ml)])
print("expected patches all 131 vols: %.0f (cap 50/volume; 118 tb vols x50 = %d); volumes with <50 tumor voxels at 1mm: %d"%(E.sum(),118*50,(npat_vol[has]<50).sum()))
bins=[0,5,10,20,40,64,1e9]; lab=["<5","5-10","10-20","20-40","40-64",">64"]
rows=[]
print("SIZE REGIME TABLE")
zext=[]
for i in range(N):
    for c in raw[i]["comps"]: zext.append((i,len(c["zs"]),c["vox"]))
# order of raw comps differs from stats comps; match via vox sort per volume
zmap={}
for i in range(N):
    a=sorted(range(len(pv[i]["components"])),key=lambda k:pv[i]["components"][k]["vox"]); b=sorted(raw[i]["comps"],key=lambda c:c["vox"])
    for k,c in zip(a,b): zmap[(i,k)]=len(c["zs"])
zs_n=np.array([zmap[(i,k)] for i,p in enumerate(pv) for k in range(len(p["components"]))])
thick_l=sz[lv]; inplane_px=d/(sx[lv]*2)
for k in range(6):
    m=(d>=bins[k])&(d<bins[k+1])
    print(f"{lab[k]:>6}: n={m.sum():4d} ({100*m.mean():4.1f}%) vol% {100*ml[m].sum()/ml.sum():5.1f} patients {len(set(lv[m])):3d} med contrast {np.median(con[m]):6.1f} med CNR {np.median(cnr[m]):5.2f} hyperdense% {100*(con[m]>0).mean():4.1f} "
          f"d<slice-thickness% {100*(d[m]<thick_l[m]).mean():4.1f} med axial slices {np.median(zs_n[m]):4.0f} med eqdiam-px@256 {np.median(inplane_px[m]):5.1f} "
          f"GAN patch share {100*E[m].sum()/E.sum():5.1f}% ext>64 {100*(ext[m]>64).mean():4.1f}% tumor-slice-share(train)")
print("lesions with max_extent>64mm: %d (%.1f%%), share of expected patches %.1f%%, share of tumor vol %.1f%%"%((ext>64).sum(),100*(ext>64).mean(),100*E[ext>64].sum()/E.sum(),100*ml[ext>64].sum()/ml.sum()))
print("lesions <10mm eqdiam: %.1f%% of lesions; %.1f%% of expected patches; %.2f%% volume"%(100*(d<10).mean(),100*E[d<10].sum()/E.sum(),100*ml[d<10].sum()/ml.sum()))
print("lesions < 3 px eq diam @256: %.1f%% ; <slice thickness: %.1f%%; lesions with 1-2 axial slices: %.1f%%"%(100*(inplane_px<3).mean(),100*(d<thick_l).mean(),100*(zs_n<=2).mean()))
print("lesion-level ESS of expected patches (all): %.0f of %d lesions; patient-level ESS: %.1f"%(ess(E),len(E),ess(np.bincount(lv,weights=E,minlength=N))))
# share of lesion count / volume contributed by volumes with thick>2.5
print("lesions in thick>2.5 volumes: %d (%.1f%%)"%((ltg==2).sum(),100*(ltg==2).mean()))
# ---- 65-volume random subsets
rng=np.random.default_rng(7); out=[]
for _ in range(10000):
    s=rng.choice(N,65,replace=False); m=np.isin(lv,s)
    out.append([has[s].sum(),m.sum(),(d[m]>=10).sum(),(d[m]>=20).sum(),(d[m]>=40).sum(),(d[m]>64).sum(),len(set(lv[m&(d>=20)])),npat_vol[s].sum(),
                ess(E[m]) if m.sum() else 0, 100*E[m&(d<10)].sum()/max(E[m].sum(),1e-9), len(set(lv[m&(d>=10)])), (con[m]>0).sum(), np.std(con[m]) if m.sum()>2 else 0])
out=np.array(out); nm=["tumor-bearing vols","lesions","lesions>=10mm","lesions>=20mm","lesions>=40mm","lesions>64mm","patients w/ lesion>=20mm","patches(<=50/vol)","lesion ESS of patches","% patches from <10mm lesions","patients w/ lesion>=10mm","hyperdense lesions","SD contrast"]
for j,n_ in enumerate(nm): print(f"65-subset {n_:32s} mean {out[:,j].mean():8.1f} p5 {np.percentile(out[:,j],5):8.1f} p95 {np.percentile(out[:,j],95):8.1f}")
# full set equivalents
print("FULL: lesions>=10",(d>=10).sum(),">=20",(d>=20).sum(),">=40",(d>=40).sum(),">64",(d>64).sum(),"patients lesion>=20",len(set(lv[d>=20])),"patients>=10",len(set(lv[d>=10])))
# diversity of large lesions inside 65 subset: heavy-tailed lesion counts: fraction of lesions from top-3 patient in subset
tp=[]
for _ in range(2000):
    s=rng.choice(N,65,replace=False); c=np.bincount(lv[np.isin(lv,s)],minlength=N)[s]; tp.append(np.sort(c)[-3:].sum()/max(c.sum(),1))
print("65-subset share of lesions from top-3 patients: mean %.2f p5 %.2f p95 %.2f"%(np.mean(tp),*np.percentile(tp,[5,95])))
np.save("/tmp/ds_impl/L.npy",np.c_[lv,d,ml,ext,con,cnr,E,zs_n])
# cluster-aware test of CNR difference in seed42 split
intest=np.isin(lv,te); ks0=ks_2samp(cnr[~intest],cnr[intest]).statistic
rr=[]
for _ in range(3000):
    t=rng.permutation(N)[:27]; m=np.isin(lv,t); rr.append(ks_2samp(cnr[~m],cnr[m]).statistic)
print("seed42 val-vs-train lesion CNR KS %.3f ; frac of random patient-level splits >= that: %.3f"%(ks0,(np.array(rr)>=ks0).mean()))
print("median CNR train %.2f val %.2f ; median contrast train %.1f val %.1f"%(np.median(cnr[~intest]),np.median(cnr[intest]),np.median(con[~intest]),np.median(con[intest])))
rr=[]; ksd=ks_2samp(d[~intest],d[intest]).statistic
for _ in range(3000):
    t=rng.permutation(N)[:27]; m=np.isin(lv,t); rr.append(ks_2samp(d[~m],d[m]).statistic)
print("size KS seed42 %.3f; frac random >= : %.3f"%(ksd,(np.array(rr)>=ksd).mean()))
vv=np.bincount(lv[intest],minlength=N); print("val vol with most lesions:",names[int(np.argmax(vv))],vv.max(), "its tumor mL",tml[int(np.argmax(vv))])
