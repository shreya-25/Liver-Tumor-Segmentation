import pickle, numpy as np, json
R=pickle.load(open('/tmp/dsv_labels_1/R.pkl','rb'))
PV={p['name']:p for p in json.load(open('analysis/output/stats.json'))['per_volume']}
rows=[r for _,o in R for r in o]
for k in (26,6):
    rr=[r for r in rows if r[0]==k]
    tot=np.array([r[3] for r in rr]);ret=np.array([r[4] for r in rr])
    print(k,len(rr),(ret==0).sum(),tot[ret==0].sum())
rr=[r for r in rows if r[0]==26]
tot=np.array([r[3] for r in rr]);ret=np.array([r[4] for r in rr])
sp=np.array([PV[r[1]]['spacing'] for r in rr]); vm=sp.prod(1)
ed=(6*tot*vm/np.pi)**(1/3)
ext=np.array([r[5] for r in rr]); inpl=ext[:,:2].max(1); mx=ext.max(1); zext=ext[:,2]
for lo,hi in ((0,3),(3,5),(5,8),(8,12),(12,1e9)):
    s=(ed>=lo)&(ed<hi); print('ed',lo,hi,s.sum(),(ret[s]==0).sum())
for lo,hi in ((0,1.5),(1.5,3),(3,5),(5,1e9)):
    s=(inpl>=lo)&(inpl<hi); print('inplane extent mm',lo,hi,s.sum(),(ret[s]==0).sum())
lost=ret==0
print('lost: ed max',ed[lost].max(),'inplane max',inpl[lost].max(),'maxext',mx[lost].max(),'zext max',zext[lost].max())
# lost lesions by thickness
th=sp[:,2]
print('lost thick', np.unique(np.round(th[lost],1),return_counts=True))
# lesions with maxext>=5mm lost
print('lost with max extent >=3mm', (mx[lost]>=3).sum(), ' >=5', (mx[lost]>=5).sum())
# per-volume
import collections
c=collections.Counter(r[1] for r in rr if r[4]==0); print('vols losing',len(c), 'tumor vols',len(set(r[1] for r in rr)))
# fraction of lesions <3mm overall
print('lesions <3mm eqdiam',(ed<3).sum(),'of',len(ed))
# number of tumor lesions by dataset (stats n_tumors, 6 conn)
print(sum(p['n_tumors'] for p in PV.values()))
# phase sensitivity: lost lesions if phase differs not computed. 
# small-lesion supervision: retained pixel counts
print('<=10 px',(ret<=10).sum()/len(ret), 'ed<10 n',(ed<10).sum())
# tumor-bearing volumes where lesions<5mm present: how many lesions lost as share of lesions<5mm
