import json, numpy as np, collections
S=json.load(open('analysis/output/stats.json'))['per_volume']
P={int(d['name'].split('-')[1].split('.')[0]):d for d in S}
R={r['i']:r for r in json.load(open('/tmp/ds_geom/a2.json'))}
rows=[]
for i in range(131):
    d=P[i]; sp=d['spacing']; bb=np.array(R[i]['body_bb']); ext=(bb[1]-bb[0])*np.array(sp)
    lb=d['liver_bbox_mm']; rows.append((i,sp[0],sp[2],d['n_slices'],ext[0],ext[1],ext[2],lb[0],lb[1],lb[2],d['liver_ml']))
A=np.array(rows)
print('liver bbox z mm: all median',np.median(A[:,9]),'min',A[:,9].min(),'max',A[:,9].max())
grp1=A[(A[:,2]==1.0)]
print('thickness 1.0 vols',len(grp1))
for r in A:
    if r[2]==1.0: print(int(r[0]),'ip',round(r[1],3),'ns',int(r[3]),'body w,h mm',round(r[4]),round(r[5]),'liver bbox mm',np.round(r[7:10],0),'liverml',round(r[10]))
# group stats of body extents for ip==1.0 vs not
m=A[:,1]>=0.999
print('ip==1.0 n',m.sum(),'body ext ax0 mm med',np.median(A[m,4]),'ax1',np.median(A[m,5]),'| others',np.median(A[~m,4]),np.median(A[~m,5]))
print('ip==1.0 idx',[int(x) for x in A[m,0]])
print('liver ml ip1.0',np.median(A[m,10]),'others',np.median(A[~m,10]))
