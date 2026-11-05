import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
from scipy.stats import kruskal, spearmanr
im=np.load(R+"data/cache/images.npy",mmap_mode="r"); mk=np.load(R+"data/cache/masks.npy",mmap_mode="r"); vi=np.load(R+"data/cache/vol_index.npy")
starts=np.r_[0,np.cumsum(np.bincount(vi))]
gl=np.full(N,np.nan);gt=np.full(N,np.nan);gsd=np.full(N,np.nan); tv=np.zeros(N); lv_=np.zeros(N)
for i in range(N):
    a=np.asarray(im[starts[i]:starts[i+1]]); m=np.asarray(mk[starts[i]:starts[i+1]])
    L=a[m==1]; T=a[m==2]; tv[i]=len(T); lv_[i]=len(L)
    gl[i]=L.mean(); gsd[i]=L.std()
    if len(T): gt[i]=T.mean()
rng_hu=hmax-hmin
pred_liver=255*(lhu-hmin)/rng_hu
con_hu=np.array([p.get("contrast_hu") if p.get("contrast_hu") is not None else np.nan for p in pv],float)
pred_con=255*con_hu/rng_hu
meas_con=gt-gl
ok=has&~np.isnan(meas_con)
print("corr predicted vs cache liver grey: %.4f ; mean abs diff %.2f levels"%(np.corrcoef(pred_liver,gl)[0,1],np.mean(np.abs(pred_liver-gl))))
print("corr predicted vs measured tumor-liver grey contrast: %.3f ; median abs diff %.2f"%(np.corrcoef(pred_con[ok],meas_con[ok])[0,1],np.median(np.abs(pred_con[ok]-meas_con[ok]))))
print("HU range (max-min) per volume: median %.0f min %.0f max %.0f"%(np.median(rng_hu),rng_hu.min(),rng_hu.max()))
print("levels per HU (255/range): median %.4f ; 1 grey level = %.1f HU median (range %.1f-%.1f)"%(np.median(255/rng_hu),np.median(rng_hu/255),(rng_hu/255).min(),(rng_hu/255).max()))
print("liver grey mean: median %.1f p5 %.1f p95 %.1f min %.1f max %.1f"%(np.median(gl),*np.percentile(gl,[5,95]),gl.min(),gl.max()))
print("liver grey SD (cache): median %.2f levels"%np.median(gsd))
mc=meas_con[ok]; print("tumor-liver contrast in grey levels (cache): median %.2f IQR [%.2f,%.2f] ; |contrast|<3 levels in %.0f%% of volumes; <=1 level %.0f%%"%(np.median(mc),*np.percentile(mc,[25,75]),100*(np.abs(mc)<3).mean(),100*(np.abs(mc)<=1).mean()))
# groups by min HU
grp=np.where(hmin>-1500,0,np.where(hmin>-2500,1,2))
for g,nm in enumerate(["min=-1024","min=-2048","min<=-3024"]):
    m=grp==g; print(nm,"n",m.sum(),"median range",np.median(rng_hu[m]),"liver grey median %.1f"%np.median(gl[m]),"contrast levels median %.2f"%np.nanmedian(np.where(ok,meas_con,np.nan)[m]))
print("kruskal liver grey by min-HU group p=%.2g"%kruskal(*[gl[grp==g] for g in range(3)]).pvalue)
print("range of liver grey across volumes = %.1f levels vs between-volume liver HU sd %.1f HU"%(gl.max()-gl.min(),lhu.std()))
# tumor voxels lost by resize
exp=np.array([p["tumor_vox"]/4 for p in pv]); print("cache tumor voxels total %d vs raw/4 %.0f (ratio %.3f)"%(tv.sum(),exp.sum(),tv.sum()/exp.sum()))
print("volumes with tumor lost entirely in cache:",[(names[i]) for i in range(N) if has[i] and tv[i]==0])
# patient-level contrast by thick groups
vc=np.array([p.get("contrast_hu") if p.get("contrast_hu") is not None else np.nan for p in pv],float)
print("per-volume contrast HU by thick group medians",[round(float(np.nanmedian(vc[(tg==g)&has])),1) for g in range(3)],"kruskal p=%.3g"%kruskal(*[vc[(tg==g)&has] for g in range(3)]).pvalue)
vcnr=np.array([p.get("cnr") if p.get("cnr") is not None else np.nan for p in pv],float)
print("per-volume CNR by thick group medians",[round(float(np.nanmedian(vcnr[(tg==g)&has])),2) for g in range(3)],"kruskal p=%.3g"%kruskal(*[vcnr[(tg==g)&has] for g in range(3)]).pvalue)
print("per-volume liver HU SD by thick",[round(float(np.median([p['liver_hu_std'] for i,p in enumerate(pv) if tg[i]==g])),1) for g in range(3)])
np.save("/tmp/ds_impl/grey.npy",np.c_[gl,gt,gsd,meas_con,rng_hu])
