import json, sys, numpy as np, nibabel as nib, pickle
from multiprocessing import Pool
from scipy import ndimage as ndi
R=''
S=json.load(open(R+'analysis/output/stats.json'))['per_volume']
WINS=[(-200,250),(-150,250),(-100,250),(-100,200),(-50,200),(-160,240),(0,200)]
PCT=[0.1,1,5,25,50,75,95,99,99.9]
def mad(x): return 1.4826*np.median(np.abs(x-np.median(x)))
def work(k):
    name=S[k]['name']; n=int(name.split('-')[1].split('.')[0])
    im=nib.load(R+f'data/raw/volume-{n}.nii'); sg=nib.load(R+f'data/raw/segmentation-{n}.nii')
    sl=(im.dataobj.slope, im.dataobj.inter)
    v=np.asarray(im.dataobj); s=np.rint(np.asarray(sg.dataobj)).astype(np.int8)
    sp=S[k]['spacing']; out={'k':k,'name':name,'slope_inter':sl}
    if v.dtype.kind=='f': v=np.rint(v)
    v=v.astype(np.int32)
    L=s==1; T=s==2
    lv=v[L]; tv=v[T]
    out['liver']=dict(n=int(lv.size),mean=float(lv.mean()),sd=float(lv.std()),mad=float(mad(lv)),pct=np.percentile(lv,PCT).tolist(),
        winfrac={f'{a},{b}':(float((lv<a).mean()),float((lv>b).mean())) for a,b in WINS})
    if tv.size:
        out['tumor']=dict(n=int(tv.size),mean=float(tv.mean()),sd=float(tv.std()),pct=np.percentile(tv,PCT).tolist(),
        winfrac={f'{a},{b}':(float((tv<a).mean()),float((tv>b).mean())) for a,b in WINS})
    # noise
    for ax,nm in enumerate(['x','y','z']):
        a=[slice(None)]*3;b=[slice(None)]*3;a[ax]=slice(1,None);b[ax]=slice(None,-1)
        m=L[tuple(a)]&L[tuple(b)]
        d=(v[tuple(a)][m]-v[tuple(b)][m]).astype(np.float32)
        out['noise_'+nm]=float(mad(d)/np.sqrt(2)); out['noiseSD_'+nm]=float(d.std()/np.sqrt(2))
    # volume percentiles
    sub=v.ravel()[::7]
    out['vmin']=int(v.min()); out['vmax']=int(v.max())
    out['all_pct']=np.percentile(sub,[0.1,0.5,1,99,99.5,99.9]).tolist()
    body=sub[sub>-500]
    out['body_pct']=np.percentile(body,[0.1,0.5,1,99,99.5,99.9]).tolist()
    st=sub[(sub>-20)&(sub<150)]
    out['soft_median']=float(np.median(st)); out['soft_n']=int(st.size)
    out['frac_gt3071']=float((sub>3071).mean()); out['frac_le_min']=float((sub==out['vmin']).mean())
    # distinct value count in liver (quantisation)
    out['liver_nuniq']=int(np.unique(lv).size)
    # lesions
    lab,nl=ndi.label(T)
    objs=ndi.find_objects(lab); les=[]
    vox_ml=sp[0]*sp[1]*sp[2]
    for i,o in enumerate(objs):
        o2=tuple(slice(max(x.start-1,0),x.stop+1) for x in o)
        m=lab[o2]==(i+1); vv=v[o2][m]
        inter=ndi.binary_erosion(m)  # 6-conn, border_value 0
        ic=v[o2][inter]
        les.append(dict(vox=int(m.sum()),eq_diam=float(2*(3*m.sum()*vox_ml/(4*np.pi))**(1/3)),mean=float(vv.mean()),med=float(np.median(vv)),
            p10=float(np.percentile(vv,10)),p90=float(np.percentile(vv,90)),n_in=int(inter.sum()),core=float(ic.mean()) if ic.size else None))
    out['lesions']=les
    return out
if __name__=='__main__':
    ks=list(range(len(S)))
    with Pool(6) as p:
        res=[]
        for r in p.imap_unordered(work,ks):
            res.append(r); print(r['name'],len(res),flush=True)
    res.sort(key=lambda r:r['k'])
    pickle.dump(res,open('/tmp/ds_intensity/raw_scan.pkl','wb'))
