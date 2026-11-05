import sys; sys.path.insert(0,"/tmp/ds_impl")
from common import *
rng=np.random.default_rng(3)
Lm=np.load("/tmp/ds_impl/L.npy"); lv=Lm[:,0].astype(int); d=Lm[:,1]
rows=[]
for _ in range(10000):
    p=rng.permutation(N); a=p[:65]; b=p[65:]
    val=a[:13]  # if 80/20 within the 65 half (52/13)
    mv=np.isin(lv,val)
    rows.append([has[a].sum(),has[b].sum(),nt[a].sum(),nt[b].sum(),has[val].sum(),nt[val].sum(),(d[mv]>=20).sum(),len(set(lv[mv&(d>=20)])),abs(nt[a].sum()-nt[b].sum()),abs(tml[a].sum()-tml[b].sum())/tml.sum()])
r=np.array(rows)
nm=["tb vols in 65-half","tb vols other 66","lesions in 65-half","lesions other 66","tb vols in 13-val","lesions in 13-val","val lesions>=20mm (13-val)","val patients>=20mm","|lesion diff halves|","|tumor mL diff halves|/total"]
for j,n in enumerate(nm): print("%-30s mean %.2f p5 %.2f p95 %.2f"%(n,r[:,j].mean(),*np.percentile(r[:,j],[5,95])))
print("P(13-val has <=8 tumor-bearing)",(r[:,4]<=10).mean(), "P(13-val has <=10 lesions>=... )")
print("P(13-val has zero lesions>=20mm)",(r[:,6]==0).mean())
