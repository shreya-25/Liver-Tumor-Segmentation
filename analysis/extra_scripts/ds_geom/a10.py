import json, numpy as np
S=json.load(open('analysis/output/stats.json'))['per_volume']
P={int(d['name'].split('-')[1].split('.')[0]):d for d in S}
R={r['i']:r for r in json.load(open('/tmp/ds_geom/a2.json'))}
G=list(range(28,48)); O=[i for i in range(131) if i not in G]
def bodyext(i):
    bb=np.array(R[i]['body_bb']); return (bb[1]-bb[0])
vx=lambda L:np.array([bodyext(i)[0] for i in L]); vy=lambda L:np.array([bodyext(i)[1] for i in L])
# body width in voxels: G vs O (voxel counts)
print('body width vox G',np.median(vx(G)),'O',np.median(vx(O)),' height vox G',np.median(vy(G)),'O',np.median(vy(O)))
mmx=np.array([bodyext(i)[0]*P[i]['spacing'][0] for i in O]); mmy=np.array([bodyext(i)[1]*P[i]['spacing'][1] for i in O])
gx=vx(G)*1.0; gy=vy(G)*1.0
print('body mm O: w med',np.median(mmx),'IQR',np.percentile(mmx,[25,75]),'h med',np.median(mmy),np.percentile(mmy,[25,75]))
print('body mm G (as labelled): w',np.median(gx),'min',gx.min(),'max',gx.max(),'h',np.median(gy),'max',gy.max(), 'n with width > O max',(gx>mmx.max()).sum(),'O max',mmx.max())
sx=np.median(mmx)/np.median(gx); sy=np.median(mmy)/np.median(gy); print('implied inplane spacing ratio',sx,sy)
# percent of G whose width mm exceeds 99th pct of O
print('G wider than 95th pct of O',(gx>np.percentile(mmx,95)).sum(),'/20')
# liver volume
lg=np.array([P[i]['liver_ml'] for i in G]); lo=np.array([P[i]['liver_ml'] for i in O])
print('liver ml G',np.median(lg),'O',np.median(lo))
# liver bbox mm
bg=np.array([P[i]['liver_bbox_mm'] for i in G]); bo=np.array([P[i]['liver_bbox_mm'] for i in O])
print('liver bbox mm med G',np.median(bg,0),'O',np.median(bo,0))
# implied z spacing from liver volume: lg_true = lg * s^2 * z => z = lo_med/(lg_med*s^2)
s=(sx+sy)/2
print('implied in-plane s',s,'implied z thickness from liver volume ratio',np.median(lo)/(np.median(lg)*s*s),'from bbox z',np.median(bo[:,2])/np.median(bg[:,2]))
# liver bbox x,y ratio
print('bbox x ratio',np.median(bo[:,0])/np.median(bg[:,0]),'y',np.median(bo[:,1])/np.median(bg[:,1]))
# per-vol implied thickness using body-based s
# effect on headline stats
comps=[(i,c) for i in range(131) for c in P[i]['components']]
ed_all=np.array([c['eq_diam_mm'] for i,c in comps]); ed_G=np.array([c['eq_diam_mm'] for i,c in comps if i in G]); ed_O=np.array([c['eq_diam_mm'] for i,c in comps if i not in G])
print('lesions G',len(ed_G),'of',len(ed_all),'med eq diam G',np.median(ed_G),'O',np.median(ed_O),'share>64 G',(ed_G>64).sum(),'of 18 total; O',(ed_O>64).sum(), 'share<10 G',(ed_G<10).mean(),'O',(ed_O<10).mean())
# apply scale correction: true eq diam = ed * (s^2*z)^(1/3)
z=np.median(lo)/(np.median(lg)*s*s); f=(s*s*z)**(1/3); print('linear correction factor for eq diam',f)
ed_Gc=ed_G*f; print('corrected G median',np.median(ed_Gc),'all median corrected',np.median(np.concatenate([ed_Gc,ed_O])),'orig all',np.median(ed_all), 'share<10 corrected',(np.concatenate([ed_Gc,ed_O])<10).mean(),'share>64',(np.concatenate([ed_Gc,ed_O])>64).sum())
tm=np.array([P[i]['tumor_ml'] for i in range(131)]); top=np.argsort(-tm)[:8]; print('top tumor ml vols',[(int(i),round(tm[i])) for i in top])
print('tumor ml G median',np.median(tm[G]),'O',np.median(tm[O]),'sum share G',tm[G].sum()/tm.sum(), 'corrected sum share', tm[G].sum()*s*s*z/(tm[G].sum()*s*s*z+tm[O].sum()))
print('slices G',sum(P[i]['n_slices'] for i in G),'share',sum(P[i]['n_slices'] for i in G)/58638)
print('n_tumors G',sum(P[i]['n_tumors'] for i in G),'no tumor vols in G',[i for i in G if P[i]['n_tumors']==0])
print('no tumor vols overall',[i for i in range(131) if P[i]['n_tumors']==0])
