import json, numpy as np, os
from sklearn.model_selection import train_test_split
R=""
S=json.load(open(R+"analysis/output/stats.json"))
pv=S["per_volume"]; print(len(pv), pv[0]["name"], list(pv[0].keys())[:5])
names=sorted(os.listdir(R+"data/raw")); vols=[n for n in names if 'vol' in n.lower()]; print(len(vols), vols[:4])
byname={p["name"]:p for p in pv}
print(list(byname)[:3])
