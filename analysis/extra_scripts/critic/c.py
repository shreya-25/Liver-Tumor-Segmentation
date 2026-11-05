import json, numpy as np, re, collections, csv, math
from scipy import stats as st
from sklearn.model_selection import train_test_split
s=json.load(open('analysis/output/stats.json'))
P=s['per_volume']; names=[p['name'] for p in P]; I=np.array([int(re.findall(r'\d+',n)[0]) for n in names])
rows=list(csv.DictReader(open('analysis/output/per_volume.csv')))
print('csv n_tumors sum',sum(int(r['n_tumors']) for r in rows),'json',sum(p['n_tumors'] for p in P),'csv dtype',collections.Counter(r['vol_dtype'] for r in rows))
# csv-json consistency
bad=0
for r,p in zip(rows,P):
    for k in ['n_slices','tumor_vox','liver_vox','hu_min','hu_max']:
        if abs(float(r[k])-float(p[k]))>1e-6: bad+=1
print('csv/json mismatch',bad)
ns=np.array([p['n_slices'] for p in P],float); t=np.array([p['slices_tumor'] for p in P],float); z=np.array([p['spacing'][2] for p in P])
print('Neff slices',ns.sum()**2/(ns**2).sum(),'Neff tumor slices',t.sum()**2/(t**2).sum())
tr,te=train_test_split(sorted(names),test_size=0.2,random_state=42); d={n:i for i,n in enumerate(names)}
ti=np.array([d[n] for n in tr])
tt=t[ti]; print('train Neff tumor slices',tt.sum()**2/(tt**2).sum(),'train Neff slices',ns[ti].sum()**2/(ns[ti]**2).sum())
nl=np.array([p['n_tumors'] for p in P],float)
print('train neff lesions', nl[ti].sum()**2/(nl[ti]**2).sum())
tb=ti[t[ti]>0]; zt=z[tb]
print('train tumor-bearing',len(tb),'thin',(zt<=1).mean(),'thick',(zt>2.5).mean(),'mid',((zt>1)&(zt<=2.5)).mean())
print('train lesions thin',nl[tb][zt<=1].sum()/nl[tb].sum(),'tumor slices thin',t[tb][zt<=1].sum()/t[tb].sum(),'thick',t[tb][zt>2.5].sum()/t[tb].sum())
# draws
n=ns[ti].sum(); T=t[ti].sum(); print('draws',n*0.5/T,'P(drawn)',1-math.exp(-n*0.5/T), 'with 5724',n*.5/5724, 1-math.exp(-n*.5/5724))
print('46910/7/3600',46910/7/3600,'46783/7/3600')
print('91/36', 624/91,251/36, 'hours',46910/7.0/3600, 46910/6.5/3600, 46910/7.3/3600)
# contrast levels variants
rng=np.array([p['hu_max']-p['hu_min'] for p in P],float); co=np.array([p.get('contrast_hu',np.nan) for p in P],float); tbm=np.array([p['tumor_vox']>0 for p in P])
lv=np.abs(co)/rng*255
for den in (255,256):
    l=np.nan_to_num(np.abs(co)/rng*den,nan=1e9); print(den,'<3 of118',np.mean(l[tbm]<3),'<=1',np.mean(l[tbm]<=1),'<3 of 131',(l[tbm]<3).sum()/131)
l=np.abs(co[tbm])/rng[tbm]*255; print('<3',(l<3).sum(),'<=1',(l<=1).sum(),'<2',(l<2).sum())
print('ratio',[np.median(rng[tbm]/450),np.percentile(rng[tbm]/450,95),(rng[tbm]/450).max(),np.median(rng/450),np.percentile(rng/450,95),(rng/450).max()])
cn=np.array([p.get('cnr',np.nan) for p in P],float);print('spearman cnr z',st.spearmanr(cn[tbm],z[tbm])[0])
print('median z',np.median(z),'q',np.percentile(z,[25,75]))
# hist-based clipping
b=np.array(s['hu_bins']); H=s['hu_hist']; print(len(b),{k:len(v) for k,v in H.items()},b[:3],b[-3:])
for k in ('liver','tumor'):
    h=np.array(H[k],float); print(k,h.sum())
    for lo,hi in [(-200,250),(-100,250),(-100,200),(0,200)]:
        # bins: edges b[i]..b[i+1]
        c=0
        for i in range(len(h)):
            e0,e1=b[i],b[i+1]
            if e1<=lo or e0>=hi: c+=h[i]  # fully outside
        print(' ',lo,hi,'clipped pooled %.3f%%'%(100*c/h.sum()))
    cs=np.cumsum(h)/h.sum(); print(' p0.1',b[np.searchsorted(cs,0.001)],'p99.9',b[np.searchsorted(cs,0.999)+1])
# volumes 48-52 liver HU
print([ (int(i),round(p['liver_hu_mean'],1)) for i,p in zip(I,P) if 48<=i<=52])
# CNR / contrasts 28-47 etc: n lesions per group
