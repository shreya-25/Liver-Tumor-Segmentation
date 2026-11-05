import json,numpy as np
R=json.load(open('/tmp/dsv_intensity_0/r.json'))
W=[(-200,250),(-100,250),(-100,200),(0,200)]
L=[r for r in R if r['Ln']>0]; T=[r for r in R if r.get('Tn',0)>0]
print(len(L),len(T))
lp=np.array([r['Lp'] for r in L]); tp=np.array([r['Tp'] for r in T])
print('liver p med',np.median(lp,0),'tumor',np.median(tp,0),'min tumor p0.1',tp[:,0].min(), 'liver min',lp[:,0].min())
for k,w in enumerate(W):
    lc=np.array([r['Lclip'][k] for r in L])*100; tc=np.array([r['Tclip'][k] for r in T])*100
    # pooled
    lw=sum(r['Lclip'][k]*r['Ln'] for r in L)/sum(r['Ln'] for r in L)*100
    tw=sum(r['Tclip'][k]*r['Tn'] for r in T)/sum(r['Tn'] for r in T)*100
    print(w,'liver mean %.3f max %.2f pooled %.3f | tumor mean %.3f max %.2f pooled %.3f'%(lc.mean(),lc.max(),lw,tc.mean(),tc.max(),tw))
both=[r for r in R if r['Ln']>0 and r.get('Tn',0)>0]
c=np.array([r['Tmean']-r['Lmean'] for r in both]);print('contrast med',np.median(c),len(both))
lm=np.array([r['Lmean'] for r in L]);print('liver mean SD',lm.std(),lm.std(ddof=1),lm.std()/350, lm.std()/abs(np.median(c)))
lmb=np.array([r['Lmean'] for r in both]);print('both-only SD',lmb.std())
for w in W: print(w,abs(np.median(c))*255/(w[1]-w[0]))
print('step',350/255)
