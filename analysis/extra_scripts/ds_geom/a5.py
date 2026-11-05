import json, numpy as np, collections
R=json.load(open('/tmp/ds_geom/a2.json'))
R.sort(key=lambda r:r['i'])
# affine checks
print('aff==saff',sum(np.allclose(r['aff'],r['saff']) for r in R),'ax==sax',sum(r['ax']==r['sax'] for r in R))
def diag(a): a=np.array(a)[:3,:3]; return np.abs(a-np.diag(np.diag(a))).max()
print('max off-diagonal in affine', max(diag(r['aff']) for r in R))
print('qform/sform codes',collections.Counter(tuple(r['qform']) for r in R))
print('scl_slope v',collections.Counter(tuple(r['vslope']) for r in R),'seg',collections.Counter(tuple(r['sslope']) for r in R))
print('dtype stored',collections.Counter((r['vdt'],r['sdt']) for r in R)); print('in-mem seg dtype',collections.Counter(r['s_in_dtype'] for r in R))
print('labels',collections.Counter(tuple(r['ulab']) for r in R)); print('nonint',sum(r['s_nonint'] for r in R))
# diag sign of affine
sg=collections.Counter(tuple(np.sign(np.diag(np.array(r['aff'])[:3,:3])).astype(int)) for r in R); print('affine diag signs',sg)
# array-space centroids normalised by shape
rows=[]
for r in R:
    sh=np.array(r['shape']); lc=np.array(r['liv_c'])/sh; bc=np.array(r['body_c'])/sh
    # liver centroid relative to body centroid in voxel units normalised by body bbox extent
    bb=np.array(r['body_bb']); ext=bb[1]-bb[0]
    rel=(np.array(r['liv_c'])-np.array(r['body_c']))/np.maximum(ext,1)
    rows.append((r['ax'],lc,bc,rel,r['i']))
for ax in ['LPS','RAS','LAS']:
    sel=[x for x in rows if x[0]==ax]
    L=np.array([x[1] for x in sel]); Rl=np.array([x[3] for x in sel]); Bc=np.array([x[2] for x in sel])
    print(ax,len(sel),'liver centroid frac (ax0,ax1,ax2) mean',L.mean(0).round(3),'sd',L.std(0).round(3),'| liver-body rel (ax0,ax1)',Rl[:,:2].mean(0).round(3),'sd',Rl[:,:2].std(0).round(3),'min/max ax0',Rl[:,0].min().round(3),Rl[:,0].max().round(3),'ax1',Rl[:,1].min().round(3),Rl[:,1].max().round(3))
# sign in array space
a0=np.array([x[3][0] for x in rows]); a1=np.array([x[3][1] for x in rows])
axs=np.array([x[0] for x in rows])
for ax in ['LPS','RAS','LAS']:
    m=axs==ax; print(ax,'liver at lower ax0 idx than body centroid:',(a0[m]<0).sum(),'/',m.sum(),'; lower ax1 idx:',(a1[m]<0).sum(),'/',m.sum())
print('overall ax0 lower',(a0<0).sum(),'ax1 lower',(a1<0).sum())
# world space: patient-right coordinate
res=[]
for r,x in zip(R,rows):
    A=np.array(r['aff']); 
    lw=A[:3,:3]@np.array(r['liv_c'])+A[:3,3]; bw=A[:3,:3]@np.array(r['body_c'])+A[:3,3]
    d=lw-bw  # world RAS-ish (nibabel world is RAS+, +x=R? in nibabel world RAS+: x increases to Right? yes)
    res.append((r['ax'],d))
for ax in ['LPS','RAS','LAS']:
    D=np.array([d for a,d in res if a==ax]); print(ax,'world dx(R+) mean',D[:,0].mean().round(1),'frac dx>0',(D[:,0]>0).mean().round(3),'dy(A+) mean',D[:,1].mean().round(1),'frac dy>0',(D[:,1]>0).mean().round(3),'dz mean',D[:,2].mean().round(1))
# which volumes have liver on 'wrong' side in array ax0 vs group majority
print('LPS ax0 rel sorted tail',sorted([(round(x[3][0],3),x[4]) for x in rows if x[0]=='LPS'])[-3:], sorted([(round(x[3][0],3),x[4]) for x in rows if x[0]=='LPS'])[:3])
for ax in ['LPS','RAS','LAS']:
    print(ax,'ax0 rel sorted ends',sorted([(round(x[3][0],3),x[4]) for x in rows if x[0]==ax])[:2],sorted([(round(x[3][0],3),x[4]) for x in rows if x[0]==ax])[-2:])
    print(ax,'ax1 rel sorted ends',sorted([(round(x[3][1],3),x[4]) for x in rows if x[0]==ax])[:2],sorted([(round(x[3][1],3),x[4]) for x in rows if x[0]==ax])[-2:])
# z: liver centroid frac along ax2 (s-i) and whether slices ascend toward S
z=np.array([x[1][2] for x in rows]); print('liver z frac mean',z.mean().round(3),'sd',z.std().round(3))
# does z order consistent? liver vs body bbox
# tumor centroid x frac
