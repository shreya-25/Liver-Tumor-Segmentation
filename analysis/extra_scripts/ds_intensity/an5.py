import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
has=~np.isnan(tm); rng=vmax-vmin
BP=np.array([r['body_pct'] for r in raw]); AP=np.array([r['all_pct'] for r in raw])
sch={}
def pos(x,lo,hi,clip): 
    if clip: x=np.clip(x,lo,hi)
    return (x-lo)/(hi-lo)
for nm,lo,hi,cl in [('A',vmin,vmax,False),('B',np.full(131,-200.),np.full(131,250.),True),('C',BP[:,1],BP[:,4],True),('C2',AP[:,1],AP[:,4],True)]:
    w=hi-lo
    sch[nm]=dict(w=w,lm=pos(lm,lo,hi,cl),tm=pos(tm,lo,hi,cl),sd=ls/w,con=(tm-lm)/w)
print('window width C (body p0.5-p99.5)',q(sch['C']['w']),'C2 (all p0.5-p99.5)',q(sch['C2']['w']))
print('window width A',q(sch['A']['w']))
for nm in sch:
    s=sch[nm]; print('---',nm)
    print(' liver mean pos',q(s['lm']),'CV',round(s['lm'].std()/s['lm'].mean(),3),'sd',round(s['lm'].std(),4))
    print(' tumor mean pos',q(s['tm'][has]),'sd',round(np.nanstd(s['tm']),4))
    print(' liver SD [0,1]',q(s['sd']))
    cl=np.abs(s['con'][has])*255; print(' |contrast| in 8-bit levels',q(cl),'<1 lvl',(cl<1).sum(),'<2',(cl<2).sum(),'<5',(cl<5).sum(),'of',has.sum())
    sl=s['sd']*255; print(' liver SD in levels',q(sl),'<2 lvl',(sl<2).sum(),'<1',(sl<1).sum())
    if nm!='B':
        f=s['w']/450; print(' contrast loss factor vs fixed (width/450)',q(f),'>5x',(f>5).sum(),'>10x',(f>10).sum())
    # spread: ratio of contrast/lm spread
    print(' median|contrast| / SD(liver mean pos) =', round(float(np.median(np.abs(s['con'][has]))/s['lm'].std()),2), ' ratio of CV contrast', round(float(np.nanstd(np.abs(s['con']))/np.nanmean(np.abs(s['con']))),2))
    print(' max/min |contrast|',np.nanmax(np.abs(s['con']))/np.nanmin(np.abs(s['con'])[np.abs(s['con'])>0]))
# cache realised
cl=np.array([cn[s['name']]['lm'] for s in S]);csd=np.array([cn[s['name']]['ls'] for s in S]);ctm=np.array([cn[s['name']]['tm'] for s in S]);nlev=np.array([cn[s['name']]['nlev'] for s in S]);lp=np.array([cn[s['name']]['lev_p1_p99'] for s in S])
print('CACHE realised: liver mean level',q(cl),'liver SD levels',q(csd),'tumor-liver levels',q((ctm-cl)[has]),'|.|<1',(abs((ctm-cl)[has])<1).sum(),'<2',(abs((ctm-cl)[has])<2).sum())
print(' distinct liver levels',q(nlev),'p1-p99 span levels',q(lp), 'nlev<=20',(nlev<=20).sum(),' nlev<=10',(nlev<=10).sum())
print(' cache SD / analytic SD (levels)',q(csd/(sch['A']['sd']*255)))
# groups by padding value
for v in [-1024,-2048,-3024]:
    m=vmin==v; print('minval',v,'n',m.sum(),'range med',np.median(rng[m]),'liver mean pos A med',np.median(sch['A']['lm'][m]),'contrast lvls med',np.nanmedian(np.abs(sch['A']['con'][m])*255))
m=~np.isin(vmin,[-1024,-2048,-3024]); print('other min',m.sum(),'range med',np.median(rng[m]))
# range extreme volumes
ex=np.where(rng>10000)[0]; print('range>10000 vols',[(S[i]['name'],int(vmin[i]),int(vmax[i]),round(abs(sch['A']['con'][i])*255,2) if has[i] else None, round(sch['A']['sd'][i]*255,2)) for i in ex])
# correlation between range and what? 
# percent of volume voxels: frac_gt3071
print('liver mean pos A is spread: min/max',sch['A']['lm'].min(),sch['A']['lm'].max())
# corr of raw rng with lm pos
# figure
C={'A':'#2a78d6','B':'#eb6834','C':'#1baf7a'}; INK='#0b0b0b';SEC='#52514e'
plt.rcParams.update({'font.size':10,'axes.edgecolor':SEC,'axes.labelcolor':INK,'xtick.color':SEC,'ytick.color':SEC,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})
fig,ax=plt.subplots(2,2,figsize=(12.5,9.2)); rs=np.random.RandomState(0)
names={'A':'A  segmenter: (v-min)/(max-min)','B':'B  fixed clip -200..250','C':'C  percentile window (body p0.5-p99.5)'}
# (a)
a=ax[0,0]
for i,n in enumerate('ABC'):
    s=sch[n]
    a.scatter(i-0.18+rs.uniform(-.1,.1,131),s['lm'],s=14,color=C[n],alpha=.8,lw=0)
    a.scatter(i+0.18+rs.uniform(-.1,.1,has.sum()),s['tm'][has],s=14,facecolor='none',edgecolor=C[n],alpha=.9,lw=.9)
a.set_xticks(range(3));a.set_xticklabels(['A segmenter','B fixed clip','C percentile'])
a.set_ylim(-0.02,1.0);a.set_ylabel('mean position in [0,1]')
a.set_title('(a) Where liver (filled) and tumor (open) land',loc='left',fontsize=11,color=INK)
a.text(-.45,.99,'A: liver spans 0.20-0.72 (SD 0.13), tumor ~0.001-0.01 below\nB: liver 0.67 (SD 0.04), tumor ~0.59   C: liver 0.45 (SD 0.06)',fontsize=8.5,color=SEC,va='top')
a.text(2.45,0.02,'filled = liver mean, open = tumor mean (one dot per volume)',ha='right',va='bottom',fontsize=8.5,color=SEC)
# (b)
b=ax[0,1]
for i,n in enumerate('ABC'):
    y=np.abs(sch[n]['con'][has])*255; y=np.clip(y,0.05,None)
    b.scatter(i+rs.uniform(-.2,.2,len(y)),y,s=14,color=C[n],alpha=.8,lw=0)
    b.hlines(np.median(y),i-.3,i+.3,color=INK,lw=1.5)
    b.text(i+.33,np.median(y),f'{np.median(y):.1f}',va='center',fontsize=9,color=INK)
b.set_yscale('log');b.axhline(1,color=SEC,ls='--',lw=.8);b.text(2.45,1.1,'1 gray level',ha='right',fontsize=8.5,color=SEC)
b.set_xticks(range(3));b.set_xticklabels(['A segmenter','B fixed clip','C percentile']);b.set_ylabel('|tumor - liver| in 8-bit levels (log)')
b.set_title('(b) Tumor-liver contrast the network sees',loc='left',fontsize=11,color=INK);b.set_xlim(-.5,2.6);b.set_ylim(0.2,80)
# (c)
c=ax[1,0]
for i,n in enumerate('ABC'):
    y=sch[n]['sd']*255
    c.scatter(i+rs.uniform(-.2,.2,131),y,s=14,color=C[n],alpha=.8,lw=0); c.hlines(np.median(y),i-.3,i+.3,color=INK,lw=1.5); c.text(i+.33,np.median(y),f'{np.median(y):.1f}',va='center',fontsize=9,color=INK)
c.set_yscale('log');c.axhline(1,color=SEC,ls='--',lw=.8)
c.set_xticks(range(3));c.set_xticklabels(['A segmenter','B fixed clip','C percentile']);c.set_ylabel('within-liver SD in 8-bit levels (log)');c.set_xlim(-.5,2.6)
c.set_title('(c) Liver texture/noise amplitude',loc='left',fontsize=11,color=INK)
# (d)
d=ax[1,1]; o=np.argsort(rng); x=np.arange(131)
for n in 'ABC':
    y=np.abs(sch[n]['con'])*255
    ok=has[o]; d.plot(x[ok],np.clip(y[o][ok],0.05,None),'o',ms=3.5,color=C[n],label=names[n],alpha=.85)
d.set_yscale('log');d.axhline(1,color=SEC,ls='--',lw=.8)
d.set_xlabel('volumes (tumor-bearing) ordered by min-max HU range, narrow -> wide');d.set_ylabel('|tumor - liver| in 8-bit levels (log)')
d.set_title('(d) Scheme A contrast collapses as the HU range grows',loc='left',fontsize=11,color=INK)
d.legend(frameon=False,fontsize=8.5,loc='lower left')
d.set_xticks([]);d.set_ylim(0.2,80)
fig.suptitle('Intensity normalisation: effective liver/tumor signal under three schemes (131 LiTS volumes)',x=0.01,ha='left',fontsize=13,color=INK,fontweight='bold')
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(R+'analysis/figures/x_intensity_normalisation_schemes.png',dpi=130)
