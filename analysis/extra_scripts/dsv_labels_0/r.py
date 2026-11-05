import json,numpy as np,nibabel as nib
from scipy import ndimage as ndi
D=''
pv=json.load(open(D+'analysis/output/stats.json'))['per_volume']
def dice(a,b):
    s=a.sum()+b.sum()
    return 2*(a&b).sum()/s if s else np.nan
rows=[]
for v in pv:
    n=v['name'].replace('volume','segmentation')
    s=np.asarray(nib.load(D+'data/raw/'+n).dataobj)
    z=np.rint(s).astype(np.int8)
    ax=int(np.argmin(s.shape)) 
    L=(z>=1);T=(z==2)
    # z axis = third
    ld=[];fr=[];td=[];ext1=0;nl=0
    for i in range(s.shape[2]-1):
        a,b=L[:,:,i],L[:,:,i+1]
        if a.any() and b.any():
            ld.append(dice(a,b)); A,B=a.sum(),b.sum(); fr.append(abs(A-B)/max(A,B)>0.30)
        a,b=T[:,:,i],T[:,:,i+1]
        if a.any() and b.any(): td.append(dice(a,b))
    lab,n_=ndi.label(T,structure=ndi.generate_binary_structure(3,1))
    if n_:
        sl=ndi.find_objects(lab)
        ext1=sum(1 for q in sl if q[2].stop-q[2].start==1)
    rows.append(dict(name=v['name'],sz=v['spacing'][2],ld=np.mean(ld),fr=np.mean(fr),td=td,n=n_,e1=ext1))
import pickle;pickle.dump(rows,open('rows.pkl','wb'))
g=lambda x:0 if x<=1 else (1 if x<=2.5 else 2)
for k in range(3):
    r=[x for x in rows if g(x['sz'])==k]
    td=np.concatenate([x['td'] for x in r if x['td']])
    print(k,len(r),np.mean([x['ld'] for x in r]),np.mean([x['fr'] for x in r]),np.median(td),np.mean(td<.5),sum(x['e1'] for x in r)/sum(x['n'] for x in r), np.median([x['n'] for x in r if x['n']]))
