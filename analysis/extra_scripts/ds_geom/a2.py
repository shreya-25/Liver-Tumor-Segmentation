import numpy as np, nibabel as nib, json, sys, time
from multiprocessing import Pool
R='data/raw/'
def work(i):
    try:
        v=nib.load(R+f'volume-{i}.nii'); s=nib.load(R+f'segmentation-{i}.nii')
        out={'i':i}
        out['aff']=v.affine.tolist(); out['saff']=s.affine.tolist()
        out['ax']=''.join(nib.aff2axcodes(v.affine)); out['sax']=''.join(nib.aff2axcodes(s.affine))
        hv,hs=v.header,s.header
        out['vslope']=[float(hv['scl_slope']),float(hv['scl_inter'])]; out['sslope']=[float(hs['scl_slope']),float(hs['scl_inter'])]
        out['vdt']=str(v.get_data_dtype()); out['sdt']=str(s.get_data_dtype())
        out['qform']=int(hv['qform_code']),int(hv['sform_code'])
        out['zooms']=[float(z) for z in hv.get_zooms()]
        out['file_sz']=None
        sd=np.asanyarray(s.dataobj)
        out['s_in_dtype']=str(sd.dtype)
        u,c=np.unique(sd,return_counts=True)
        out['ulab']=[float(x) for x in u]; out['ucnt']=[int(x) for x in c]
        out['s_nonint']=int((sd!=np.round(sd)).sum()) if sd.dtype.kind=='f' else 0
        sh=sd.shape
        # centroids in voxel idx
        def cen(m):
            if m.sum()==0: return None
            idx=np.argwhere(m[::2,::2,:])  # subsample xy
            idx=idx*np.array([2,2,1])
            return idx.mean(0).tolist()
        out['shape']=list(sh)
        out['liv_c']=cen(sd>=1); out['tum_c']=cen(sd==2)
        # liver bbox
        m=sd>=1; nz=np.argwhere(m[::4,::4,::2])
        out['liv_bb']=[(nz.min(0)*np.array([4,4,2])).tolist(),(nz.max(0)*np.array([4,4,2])).tolist()]
        # image (subsample)
        im=np.asanyarray(v.dataobj)[::4,::4,::4].astype(np.float32)
        b=im>-300
        nzb=np.argwhere(b)
        out['body_c']=(nzb.mean(0)*4).tolist()
        out['body_bb']=[(nzb.min(0)*4).tolist(),(nzb.max(0)*4).tolist()]
        # stats on raw image values: fractional?
        out['v_in_dtype']=str(np.asanyarray(v.dataobj).dtype) if False else None
        # table detection: posterior half mean HU of rows? skip; compute mean of body mask along axis1 
        out['bodyfrac_ax0']=b.mean(axis=(1,2)).tolist()[::1]
        out['bodyfrac_ax1']=b.mean(axis=(0,2)).tolist()
        return out
    except Exception as e:
        return {'i':i,'err':repr(e)}
if __name__=='__main__':
    t=time.time()
    with Pool(6) as p: res=p.map(work,range(131),chunksize=1)
    json.dump(res,open('/tmp/ds_geom/a2.json','w'))
    print('done',time.time()-t, [r for r in res if 'err' in r])
