import json
from sklearn.model_selection import train_test_split
d=json.load(open('analysis/output/stats.json'))
byn={p['name']:p for p in d['per_volume']};names=sorted(byn)
tr,te=train_test_split(range(131),test_size=0.2,random_state=42)
for i in te[:0]:pass
print(sum(byn[names[i]]['n_tumors'] for i in te), sum(len(byn[names[i]]['components']) for i in te))
print(sum(1 for i in te for c in byn[names[i]]['components'] if c['vox']>=1))
tt=sorted([byn[names[i]]['slices_tumor'] for i in tr],reverse=True)
for k in (10,11):print(k,sum(tt[:k])/sum(tt))
print(sum(byn[names[i]]['tumor_vox']==0 for i in te))
