import sys; sys.path.insert(0,"/tmp/ds_intensity")
from common import *
print(q(contrast))
print(np.sum(sz<=1.0001),np.sum(ls[sz<=1.0001]>40), np.sum(ls[sz>1.0001]>40), np.sum(ls[sz<=1.0001]>35), np.sum(ls[sz>1.0001]>35))
import scipy.stats as st
print(st.spearmanr(lm,contrast))
