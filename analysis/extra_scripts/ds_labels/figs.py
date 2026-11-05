import pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '.'
OUT = ROOT + '/analysis/figures/'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'axes.axisbelow': True})
BL, OR, GR = '#2a6fbb', '#d9822b', '#6b7280'
S = pickle.load(open('/tmp/ds_labels/comps.pkl', 'rb'))['comps']
G = pickle.load(open('/tmp/ds_labels/gan_patch.pkl', 'rb'))
C = pickle.load(open('/tmp/ds_labels/cache_cmp.pkl', 'rb'))
R = pickle.load(open('/tmp/ds_labels/raw_scan.pkl', 'rb'))
PV = json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']

# ---- fig 1: lesion size vs tumour volume vs GAN sampling probability (1 mm grid)
sizes = np.concatenate([g['sizes1mm'] for g in G]); pk = np.concatenate([g['pk'] for g in G])
ed = (6 * sizes / np.pi) ** (1 / 3)
bins = [0, 5, 10, 20, 40, 64, 1e9]; lab = ['<5', '5-10', '10-20', '20-40', '40-64', '>64']
n = np.array([((ed >= a) & (ed < b)).sum() for a, b in zip(bins[:-1], bins[1:])]); vs = np.array([sizes[(ed >= a) & (ed < b)].sum() for a, b in zip(bins[:-1], bins[1:])])
pm = np.array([pk[(ed >= a) & (ed < b)].mean() for a, b in zip(bins[:-1], bins[1:])])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
x = np.arange(len(lab)); w = .38
ax[0].bar(x - w / 2, 100 * n / n.sum(), w, color=BL, label='share of lesions')
ax[0].bar(x + w / 2, 100 * vs / vs.sum(), w, color=OR, label='share of tumour volume')
for xi, v in zip(x, 100 * vs / vs.sum()): ax[0].text(xi + w / 2, v + 1, f'{v:.1f}', ha='center', fontsize=8)
for xi, v in zip(x, 100 * n / n.sum()): ax[0].text(xi - w / 2, v + 1, f'{v:.0f}', ha='center', fontsize=8)
ax[0].set_xticks(x, lab); ax[0].set_xlabel('lesion equivalent diameter (mm)'); ax[0].set_ylabel('%'); ax[0].legend(frameon=False)
ax[0].set_title('Half the lesions are <10 mm; they hold <1% of volume', fontsize=10, loc='left')
ax[1].bar(x, 100 * pm, color=BL)
for xi, v in zip(x, 100 * pm): ax[1].text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9)
ax[1].set_xticks(x, lab); ax[1].set_xlabel('lesion equivalent diameter (mm)'); ax[1].set_ylabel('P(lesion gets >=1 of the 50 patches), %'); ax[1].set_ylim(0, 112)
ax[1].set_title('GAN patch sampler (50 random tumour voxels / volume)\nrarely visits small lesions', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(OUT + 'x_labels_lesion_size_vs_gan_sampling.png', dpi=130); plt.close(fig)

# ---- fig 2: downsample loss
tv = [c for c in C if c['raw_T'] > 0]
tot = np.concatenate([c['lesions']['26'][0] for c in tv]); ret = np.concatenate([c['lesions']['26'][1] for c in tv])
sp = np.concatenate([[c['spacing']] * len(c['lesions']['26'][0]) for c in tv])
edn = (6 * tot * sp[:, 0] * sp[:, 1] * sp[:, 2] / np.pi) ** (1 / 3)
b2 = [0, 3, 5, 8, 10, 15, 20, 1e9]; l2 = ['<3', '3-5', '5-8', '8-10', '10-15', '15-20', '>20']
lost = [100 * (ret[(edn >= a) & (edn < b)] == 0).mean() for a, b in zip(b2[:-1], b2[1:])]
cnt = [((edn >= a) & (edn < b)).sum() for a, b in zip(b2[:-1], b2[1:])]
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.bar(range(len(l2)), lost, color=BL)
for i_, (v, c_) in enumerate(zip(lost, cnt)): ax.text(i_, v + 1, f'{v:.0f}%\n(n={c_})', ha='center', fontsize=8)
ax.set_xticks(range(len(l2)), l2); ax.set_xlabel('lesion equivalent diameter (mm, 26-connectivity)'); ax.set_ylabel('lesions with no surviving pixel, %'); ax.set_ylim(0, 50)
ax.set_title('512->256 nearest-neighbour mask resize removes only lesions <3 mm\n(21 of 908 lesions, 38 of 18.5 M tumour voxels)', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(OUT + 'x_labels_downsample_loss.png', dpi=130); plt.close(fig)

# ---- fig 3: annotation continuity by thickness
def grp(t): return 0 if t <= 1 else (1 if t <= 2.5 else 2)
gn = ['<=1 mm\n(84 vols)', '1-2.5 mm\n(33)', '>2.5 mm\n(14)']
ld = np.array([r['liver_adj_dice_mean'] for r in R if 'liver_adj_dice_mean' in r]); gl = np.array([grp(PV[r['i']]['spacing'][2]) for r in R if 'liver_adj_dice_mean' in r])
td = [np.concatenate([r['tumor_adj_dice'] for r in R if 'n6' in r and grp(PV[r['i']]['spacing'][2]) == k]) for k in range(3)]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].boxplot([ld[gl == k] for k in range(3)], tick_labels=gn, showfliers=True, medianprops=dict(color=OR), boxprops=dict(color=BL), whiskerprops=dict(color=BL), capprops=dict(color=BL), flierprops=dict(markeredgecolor=GR, markersize=3))
ax[0].set_ylabel('mean Dice of adjacent liver slices (per volume)'); ax[0].set_title('Liver mask', fontsize=10, loc='left')
ax[1].boxplot(td, tick_labels=gn, showfliers=False, medianprops=dict(color=OR), boxprops=dict(color=BL), whiskerprops=dict(color=BL), capprops=dict(color=BL))
ax[1].set_ylabel('Dice of adjacent tumour slices (per slice pair)'); ax[1].set_title('Tumour mask (pairs where both slices have tumour)', fontsize=10, loc='left')
fig.suptitle('Thick-slice scans change much more between neighbouring slices (expected from sampling, not necessarily worse labels)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT + 'x_labels_slice_continuity_by_thickness.png', dpi=130); plt.close(fig)
print('ok')
