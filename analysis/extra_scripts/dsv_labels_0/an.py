import json,numpy as np
R=json.load(open('/tmp/dsv_labels_0/mine/out.json'))
st=json.load(open('analysis/output/stats.json'))
tv={d['name']:d['tumor_vox'] for d in st['per_volume']}
print('tv match',all(sum(r['6'])==r['tv'] for r in R), sum(r['tv'] for r in R), sum(tv.values()))
for k in('6','26'):
    c=[];d=[];ml=[]
    for r in R:
        vv=np.prod(r['sp'])
        for v in r[k]:
            c.append(v);ml.append(v*vv/1000);d.append((6*v*vv/np.pi)**(1/3))
    c=np.array(c);d=np.array(d);ml=np.array(ml);n=len(c)
    print(k,n,'<5',(c<5).sum(),(c<5).mean(),'==1',(c==1).sum(),'vox<5',c[c<5].sum(),c[c<5].sum()/c.sum())
    print(' med',np.median(d),'<10',(d<10).mean(),'<5',(d<5).mean(),'<20',(d<20).mean(),'>64',(d>64).sum(),(d>64).mean(),'<1mL',(ml<1).mean())
    s=np.sort(ml)[::-1];t=s.sum()
    print(' top10',s[:round(n*.1)].sum()/t,s[:int(n*.1)].sum()/t,'top1',s[:round(n*.01)].sum()/t,s[:int(n*.01)].sum()/t,'bot50',s[n//2:].sum()/t, s[-round(n*.5):].sum()/t)
    pv=np.array([len(r[k]) for r in R]);print(' per vol med',np.median(pv),pv.max(),(pv==1).sum(),(pv>=10).sum(),(pv==0).sum())
    if k=='26':
        sz=np.array([r['sp'][2] for r in R])
        tiny=[(len(r['26']) and sum(1 for v in r['26'] if v<5), r['sp'][2]) for r in R]
        print(' tiny in <=1mm vols',sum(t for t,z in tiny if z<=1), sum(t for t,z in tiny))
n6=[len(r['6']) for r in R];n26=[len(r['26']) for r in R]
print('vols changed',sum(a!=b for a,b in zip(n6,n26)),'max merge',max(a-b+1 for a,b in zip(n6,n26)))
print('----')
sp={d['name'].replace('volume','segmentation'):d['spacing'] for d in st['per_volume']}
print(list(sp)[:2],R[0]['f'],R[0]['sp'],sp.get(R[0]['f']))
d=[];ml=[]
for r in R:
    s=sp[r['f'].replace('.nii','.nii')] if r['f'] in sp else r['sp']
    vv=np.prod(s)
    for v in r['26']: d.append((6*v*vv/np.pi)**(1/3));ml.append(v*vv/1000)
d=np.array(d);ml=np.array(ml);n=len(d)
print(n,np.median(d),(d<10).mean(),(d<5).mean(),(d>64).sum(),(ml<1).mean())
s=np.sort(ml)[::-1];print(s[:round(n*.01)].sum()/s.sum(), s[:9].sum()/s.sum(),s[:10].sum()/s.sum())
pv=np.array([len(r['26']) for r in R]);print(np.median(pv[pv>0]))
mx=0
for r in R:
    pass
