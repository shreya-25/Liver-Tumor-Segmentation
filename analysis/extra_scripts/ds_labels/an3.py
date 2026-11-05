import pickle, json
import numpy as np
G = pickle.load(open('/tmp/ds_labels/gan_patch.pkl', 'rb'))
pr = print
pr('volumes', len(G))
sizes = np.concatenate([g['sizes1mm'] for g in G]); ext = np.concatenate([g['ext1mm'] for g in G]); pk = np.concatenate([g['pk'] for g in G])
ed = (6 * sizes / np.pi) ** (1 / 3)
pr('lesions at 1 mm (26-conn):', len(sizes), 'eq diam median', np.median(ed), '<10:', 100 * (ed < 10).mean(), '<5', 100 * (ed < 5).mean(), '<20', 100 * (ed < 20).mean(), '>64 eq', (ed > 64).sum(), 100 * (ed > 64).mean())
mx = ext.max(1)
pr('bbox max side >64mm:', (mx > 64).sum(), 100 * (mx > 64).mean(), '  >32mm', 100 * (mx > 32).mean(), ' any side>64 -> cannot be wholly inside a 64^3 patch;  all sides <=64:', 100 * (mx <= 64).mean())
pr('lesion vol > 64^3 (262144 mm3)', (sizes > 64 ** 3).sum(), ' vol > half patch', (sizes > 64 ** 3 / 2).sum())
pr('bbox max side >48mm', 100 * (mx > 48).mean(), ' >64 lesions by volume share', sizes[mx > 64].sum() / sizes.sum(), ' >32', sizes[mx > 32].sum() / sizes.sum())
# patches
rows = [dict(r, vol=g['i']) for g in G for r in g['rows']]
pr('patches', len(rows), 'per vol median', np.median([len(g['rows']) for g in G]), 'vols with <50 patches', sum(len(g['rows']) < 50 for g in G))
Nv = np.array([g['Nvox1mm'] for g in G]); pr('tumor voxel count at 1mm per volume: min', Nv.min(), 'median', np.median(Nv), ' vols with Nvox<=50:', (Nv <= 50).sum())
pr('fraction of tumor voxels (per vol) sampled by cap: median', np.median(np.minimum(50, Nv) / Nv), ' tumor vols where cap samples <0.1% of tumor voxels', (50 / Nv < .001).sum())
# per-lesion probability
pr('lesions with P(sampled>=1) <0.5:', 100 * (pk < .5).mean(), ' <0.9', 100 * (pk < .9).mean(), ' >=0.99', 100 * (pk >= .99).mean(), ' expected lesions never sampled', (1 - pk).sum(), 'of', len(pk), 100 * (1 - pk).sum() / len(pk))
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 64), (64, 1e9)):
    s = (ed >= lo) & (ed < hi)
    pr(f'  lesion eq diam [{lo},{hi}) n={s.sum()} ({100 * s.mean():.1f}%) mean P(sampled)={pk[s].mean():.3f}  expected never sampled={(1 - pk[s]).sum():.1f}')
# patch-level: size of centre lesion
led = np.array([(6 * r['les_vox'] / np.pi) ** (1 / 3) for r in rows])
pr('share of patches centred in lesions: <10mm', 100 * (led < 10).mean(), ' <20', 100 * (led < 20).mean(), ' >=40', 100 * (led >= 40).mean(), ' >64', 100 * (led > 64).mean(), ' (vs lesion-level <10:', 100 * (ed < 10).mean(), ')')
tf = np.array([r['tumor_frac'] for r in rows]); lf = np.array([r['liver_frac'] for r in rows]); bf = np.array([r['bg_frac'] for r in rows]); pf = np.array([r['pad_frac'] for r in rows])
pr('all patches tumor frac median', np.median(tf), 'mean', tf.mean(), ' liver', np.median(lf), 'bg(0 incl pad)', np.median(bf), ' pad_frac mean', pf.mean(), 'patches with pad>0:', 100 * (pf > 0).mean(), ' pad>10%', 100 * (pf > .1).mean())
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 64), (64, 1e9)):
    s = (led >= lo) & (led < hi)
    if not s.any(): continue
    fin = np.array([r['frac_les_in_patch'] for r in rows])[s]
    nl = np.array([r['n_lesions_in_patch'] for r in rows])[s]
    pr(f'  centre lesion eq diam [{lo},{hi}) n={s.sum()}: tumor frac of patch median {np.median(tf[s]) * 100:.2f}% (p10 {np.percentile(tf[s], 10) * 100:.2f}, p90 {np.percentile(tf[s], 90) * 100:.1f}); liver frac {np.median(lf[s]) * 100:.1f}%; bg+pad {np.median(bf[s]) * 100:.1f}%; lesion fully in patch {100 * (fin > .999).mean():.0f}%; lesion share inside median {np.median(fin):.2f}; other lesions in patch >1: {100 * (nl > 1).mean():.0f}%')
# physical: mm3 of tumor
pr('tumor <1% of patch voxels:', 100 * (tf < .01).mean(), ' <5%', 100 * (tf < .05).mean(), ' >50%', 100 * (tf > .5).mean(), ' >=90%', 100 * (tf >= .9).mean(), '  100% (all tumor)', (tf >= .999).sum())
# off-centre
pos = np.array([r['pos'] for r in rows]); off = np.abs(pos - 32)
pr('patches where centre voxel is NOT at 32 in some axis (window clipped at ROI lower edge):', 100 * (off.max(1) > 0).mean(), ' mean abs offset (vox) when off', off[off.max(1) > 0].max(1).mean(), ' >16 vox off', 100 * (off.max(1) > 16).mean())
# lesion-sampled patch counts
pr('patches from volumes: top 10% volumes lesions share?')
# lesion count vs patches: patch allocation among lesions in same volume
big = 0
for g in G:
    r = g['rows']
    ls = [x['lesion'] for x in r]
    u, c = np.unique(ls, return_counts=True)
    big += (c.max() / len(r))
pr('mean over volumes: share of the volume\'s patches centred in its single most-sampled lesion', big / len(G))
# per-lesion count in volumes with multiple lesions
mult = [g for g in G if g['n_les_1mm'] > 1]
pr('vols with >1 lesion', len(mult), ' share of patches landing in the largest lesion (mean)', np.mean([max(np.unique([x['lesion'] for x in g['rows']], return_counts=True)[1]) / len(g['rows']) for g in mult]), ' vs largest-lesion volume share', np.mean([g['sizes1mm'].max() / g['sizes1mm'].sum() for g in mult]))
# smallest lesions (<10mm): total patches they get vs lesions
pr('lesions<10mm: share of lesions', 100 * (ed < 10).mean(), ' share of tumor voxels', 100 * sizes[ed < 10].sum() / sizes.sum(), ' share of patches', 100 * (led < 10).mean())
# 1mm resampling effect on count: native 26 count vs 1mm 26 count
R = pickle.load(open('/tmp/ds_labels/raw_scan.pkl', 'rb'))
nat = {r['i']: r['n26'] for r in R if 'n26' in r}
n1 = {g['i']: g['n_les_1mm'] for g in G}
pr('lesion count native(26) total', sum(nat.values()), ' at 1mm(26)', sum(n1.values()), ' vols differing', sum(nat[i] != n1[i] for i in nat))
