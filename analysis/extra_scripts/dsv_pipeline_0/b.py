import numpy as np,os
from sklearn.model_selection import train_test_split
R=''
names=sorted(f for f in os.listdir(R+'data/raw') if 'vol' in f.lower())
segs=sorted(f for f in os.listdir(R+'data/raw') if 'seg' in f.lower())
vt,vv,_,_=train_test_split(names,segs,test_size=0.2,random_state=42)
cn=[l.strip() for l in open(R+'data/cache/volumes.txt')]
assert cn==names
vi=np.load(R+'data/cache/vol_index.npy'); m=np.load(R+'data/cache/masks.npy',mmap_mem:=None) if False else np.load(R+'data/cache/masks.npy',mmap_mode='r')
tum=np.array([ (m[i]==2).any() for i in range(len(m))])
def frac(order):
    t=np.concatenate([tum[vi==cn.index(n)] for n in order])
    nb=(len(t)+3)//4; e=0;N=0;
    for k in range(nb):
        N+=1; e+= (not t[k*4:k*4+4].any())
    return e,N,e/N, t.mean()
print('val',frac(vv)); print('train',frac(vt))
print('sorted val',frac(sorted(vv)))
