import pickle,numpy as np
from scipy import stats
R=pickle.load(open('/tmp/ds_labels/raw_scan.pkl','rb'))
print(len(R))
def g(z): return 0 if z<=1.0 else (1 if z<=2.5 else 2)
G={0:[],1:[],2:[]}
for r in R: G[g(r['spacing'][2])].append(r)
for k,L in G.items():
    ld=[r['liver_adj_dice_mean'] for r in L if 'liver_adj_dice_mean' in r]
    j=sum(r['liver_area_jumps30'] for r in L if 'liver_adj_pairs' in r); p=sum(r['liver_adj_pairs'] for r in L if 'liver_adj_pairs' in r)
    td=[np.asarray(r['tumor_adj_dice']) for r in L if 'tumor_adj_dice' in r]
    allt=np.concatenate(td)
    nl=[len(r['comps']) for r in L if 'comps' in r]
    one=[c['zext_slices']==1 for r in L if 'comps' in r for c in r['comps']]
    print(k,len(L),'liverdice',np.mean(ld),'jump',j/p,'tumdice med',np.median(allt),'<.5',(allt<.5).mean(),'n pairs',len(allt),'single',np.mean(one),len(one),'nles med',np.median(nl))
    # per-volume tumour dice
    print('   per-vol tumor median-of-medians',np.median([np.median(t) for t in td if len(t)]), 'n vols',len(td))
    print('   ', np.round(sorted(ld)[:5],3), np.round(sorted(ld)[-3:],3))
# within thick: 3 vs 5 mm
th=[(r['spacing'][2],r['liver_adj_dice_mean']) for r in G[2]]; print(sorted(th))
# nonmonotonic: spearman liver dice vs sz
a=[(r['spacing'][2],r['liver_adj_dice_mean']) for r in R if 'liver_adj_dice_mean' in r]
print(stats.spearmanr(*zip(*a)))
# tests
x=[r['liver_adj_dice_mean'] for r in G[0] if 'liver_adj_dice_mean' in r];y=[r['liver_adj_dice_mean'] for r in G[2] if 'liver_adj_dice_mean' in r]
print(stats.mannwhitneyu(x,y))
print('fisher notumor thin vs rest',stats.fisher_exact([[12,72],[1,46]]),'thin vs thick',stats.fisher_exact([[12,72],[0,14]]))
# slice-count / sampling: 1 mm vs 0.7-0.8
for lo,hi in [(0,0.75),(0.75,0.9),(0.9,1.01),(1.1,1.6),(1.9,2.6),(2.9,6)]:
    L=[r for r in R if lo<r['spacing'][2]<=hi and 'liver_adj_dice_mean' in r]
    print(lo,hi,len(L),np.mean([r['liver_adj_dice_mean'] for r in L]),sum('no_tumor' in r for r in L))
