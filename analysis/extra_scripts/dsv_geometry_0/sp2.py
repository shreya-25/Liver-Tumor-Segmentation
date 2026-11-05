import json,collections
from sklearn.model_selection import train_test_split
o=json.load(open('/tmp/dsv_geometry_0/orient_zz/out.json'))
cls={}
for x in o:
    cls['volume-%d.nii'%x['n']]=('A' if x['rel'][1]<0 else 'B') if x['rel'][0]<0 else 'C'
names=[l.strip() for l in open('data/cache/volumes.txt')]
tr,te=train_test_split(names,test_size=0.2,random_state=42)
print(collections.Counter(cls[n] for n in tr),collections.Counter(cls[n] for n in te))
