import json,numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
cl=json.load(open('/tmp/ds_geom/cls.json'))
names=[n for n in open('data/cache/volumes.txt').read().split('\n') if n]
tr,te=train_test_split(np.arange(len(names)),test_size=0.2,random_state=42)
num=lambda k:int(names[k].split('-')[1].split('.')[0])
print(Counter(cl[str(num(k))] for k in tr),Counter(cl[str(num(k))] for k in te))
