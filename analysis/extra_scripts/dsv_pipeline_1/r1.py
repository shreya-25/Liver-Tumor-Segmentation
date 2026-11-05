import numpy as np, json
from sklearn.model_selection import train_test_split
C="data/cache/"
names=open(C+"volumes.txt").read().split("\n")
vi=np.load(C+"vol_index.npy")
m=np.load(C+"masks.npy",mmap_mode="r")
N=len(vi); print(N,len(names))
t=np.zeros(N,bool); l=np.zeros(N,bool)
for s in range(0,N,2000):
    b=np.asarray(m[s:s+2000]); t[s:s+2000]=(b==2).reshape(len(b),-1).any(1); l[s:s+2000]=(b>=1).reshape(len(b),-1).any(1)
np.savez("/tmp/dsv_pipeline_1/flags.npz",t=t,l=l)
tr,te=train_test_split(np.arange(len(names)),test_size=0.2,random_state=42)
sel=np.isin(vi,tr)
tt=t[sel]; ll=l[sel]; v=vi[sel]
nt=tt.sum(); nn=len(tt)-nt
print(len(tt),nt,nn,nn/nt, 0.5*len(tt)/nt)
w=np.where(tt,nn/nt,1.0); print("ess",w.sum()**2/(w**2).sum())
print("liver frac nontumor", ll[~tt].mean())
# patient ess
p=w/w.sum()
pq=np.bincount(v,weights=p)[tr]; print("patient ess all",1/(pq**2).sum())
pt=np.bincount(v[tt],minlength=len(names))[tr]; q=pt/pt.sum(); print("tumor patients",(pt>0).sum(),"ess",1/(q**2).sum(), np.sort(q)[::-1][:5].sum(),np.sort(q)[::-1][:10].sum())
# test split
st=np.isin(vi,te); print("test tumor slices",t[st].sum(),len(t[st]), "tumor patients test",(np.bincount(vi[st&t],minlength=len(names))[te]>0).sum())
# without-tumor patients in train
print("train patients w/o tumor",(pt==0).sum())
