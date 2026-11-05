import pickle, json
import numpy as np
from sklearn.model_selection import train_test_split
ROOT = '.'
PV = {p['name']: p for p in json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']}
C = pickle.load(open('/tmp/ds_labels/cache_cmp.pkl', 'rb'))
S = pickle.load(open('/tmp/ds_labels/comps.pkl', 'rb'))['comps']
R = pickle.load(open('/tmp/ds_labels/raw_scan.pkl', 'rb'))
pr = print
pr('aligned (same n slices):', sum(c.get('aligned', True) for c in C), 'of', len(C))
pr('cache == raw[:,1::2,1::2] exactly:', sum(c['eq_odd'] for c in C), ' == even:', sum(c['eq_even'] for c in C), 'total mismatch px', sum(c['mismatch_odd'] for c in C))
rT = sum(c['raw_T'] for c in C); cT = sum(c['cache_T'] for c in C)
pr('tumor vox raw', rT, 'cache', cT, 'cache*4/raw', cT * 4 / rT, '  liver1 raw', sum(c['raw_L1'] for c in C), 'cache*4/raw', 4 * sum(c['cache_L1'] for c in C) / sum(c['raw_L1'] for c in C))
# per volume volume error
tv = [c for c in C if c['raw_T'] > 0]
err = np.array([4 * c['cache_T'] / c['raw_T'] - 1 for c in tv]); pr('per-volume tumor vol rel error: median', np.median(err), 'p5/p95', np.percentile(err, [5, 95]), 'min/max', err.min(), err.max())
# slices
rs = np.concatenate([c['raw_T_slice'] for c in C]); cs = np.concatenate([c['cache_T_slice'] for c in C])
pr('slices', len(rs), 'raw tumor slices', (rs > 0).sum(), 'cache', (cs > 0).sum(), 'lost', ((rs > 0) & (cs == 0)).sum(), 100 * ((rs > 0) & (cs == 0)).sum() / (rs > 0).sum(), '%  gained', ((rs == 0) & (cs > 0)).sum())
lost = (rs > 0) & (cs == 0)
pr('lost slice raw area px: median', np.median(rs[lost]), 'max', rs[lost].max(), ' tumor px in lost slices', rs[lost].sum(), 100 * rs[lost].sum() / rs.sum())
for lo, hi in ((1, 4), (4, 10), (10, 50), (50, 200), (200, 1e9)):
    s = (rs >= lo) & (rs < hi)
    pr(f' raw slice area [{lo},{hi}) n={s.sum()} lost%={100 * (cs[s] == 0).mean():.1f}  px share={100 * rs[s].sum() / rs.sum():.2f}%')
# cache per-slice area distribution
pr('cache tumor slice area px: median', np.median(cs[cs > 0]), '<=3 px %', 100 * (cs[cs > 0] <= 3).mean(), '<=10', 100 * (cs[cs > 0] <= 10).mean(), '<=25', 100 * (cs[cs > 0] <= 25).mean(), '<=100', 100 * (cs[cs > 0] <= 100).mean())
# lesions
for tag in ('26', '6'):
    tot = np.concatenate([c['lesions'][tag][0] for c in tv]); ret = np.concatenate([c['lesions'][tag][1] for c in tv])
    pr(f'[{tag}-conn] lesions', len(tot), 'lost (0 retained px):', (ret == 0).sum(), 100 * (ret == 0).mean(), '%; retained 1-3 px:', ((ret >= 1) & (ret <= 3)).sum(), '; <=10 px:', (ret <= 10).sum(), ' voxels in lost lesions', tot[ret == 0].sum())
tot = np.concatenate([c['lesions']['26'][0] for c in tv]); ret = np.concatenate([c['lesions']['26'][1] for c in tv])
ev = np.concatenate([c['lesions']['26even'] for c in tv]); mx = np.concatenate([c['lesions']['26mixed'] for c in tv])
pr('26-conn lost with even phase:', (ev == 0).sum(), ' mixed phase:', (mx == 0).sum())
sp = np.concatenate([[c['spacing']] * len(c['lesions']['26'][0]) for c in tv])
vox_ml = sp[:, 0] * sp[:, 1] * sp[:, 2] / 1000
ed = (6 * tot * vox_ml * 1000 / np.pi) ** (1 / 3)
zext = np.concatenate([c['lesions']['zext'] for c in tv]); ipx = np.concatenate([c['lesions']['inplane_ext_vox'] for c in tv])
pr('lost lesion: eq diam median', np.median(ed[ret == 0]), 'max', ed[ret == 0].max(), ' native vox median', np.median(tot[ret == 0]), 'max', tot[ret == 0].max())
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 10), (10, 15), (15, 20), (20, 1e9)):
    s = (ed >= lo) & (ed < hi)
    pr(f' eq diam [{lo},{hi}) n={s.sum()} lost={100 * (ret[s] == 0).mean():.1f}%  <=4px retained {100 * (ret[s] <= 4).mean():.1f}%')
for lo, hi in ((1, 5), (5, 20), (20, 100), (100, 1e9)):
    s = (tot >= lo) & (tot < hi)
    pr(f' native vox [{lo},{hi}) n={s.sum()} lost={100 * (ret[s] == 0).mean():.1f}%')
# lesion-wise vol estimation: for surviving lesions, 4*ret/tot
s = ret > 0
rat = 4 * ret[s] / tot[s]
pr('surviving lesion volume est ratio 4*ret/tot: median', np.median(rat), 'p5/p95', np.percentile(rat, [5, 95]))
sm = s & (ed < 10); rr = 4 * ret[sm] / tot[sm]
pr(' lesions <10mm surviving: n', sm.sum(), 'ratio median', np.median(rr), 'p5/p95', np.percentile(rr, [5, 95]), 'overestimated >2x', (rr > 2).mean(), 'under <0.5x', (rr < .5).mean())
# lesions <10mm: how many cache pixels
pr('lesions <10 mm eq diam:', (ed < 10).sum(), 'cache px median', np.median(ret[ed < 10]), 'has <=4 cache px', (ret[ed < 10] <= 4).mean())
# lost by volume
lv = [(c['name'], int((c['lesions']['26'][1] == 0).sum()), len(c['lesions']['26'][0])) for c in tv]
lvl = [x for x in lv if x[1] > 0]
pr('volumes losing >=1 lesion:', len(lvl), 'of', len(tv), ' volumes losing all lesions:', sum(x[1] == x[2] for x in lv), [x for x in lv if x[1] == x[2]])
# volumes where tumor slices lost
vs = [(c['name'], int(((c['raw_T_slice'] > 0) & (c['cache_T_slice'] == 0)).sum()), int((c['raw_T_slice'] > 0).sum())) for c in tv]
pr('volumes with any lost tumor slice:', sum(x[1] > 0 for x in vs), ' max lost', max(vs, key=lambda x: x[1]))
# a volume whose entire tumor disappears after downsample?
pr('volumes with tumor in raw and zero tumor px in cache:', [c['name'] for c in tv if c['cache_T'] == 0])
# lesions lost -> volume rel error by lesion size for smaller (<20mm) lesion population: total volume represented
pr('tumor vox in lesions with eq diam<10: share', tot[ed < 10].sum() / tot.sum(), ' <20:', tot[ed < 20].sum() / tot.sum())
# by thickness: lost lesion %
th = np.array([c['spacing'][2] for c in tv for _ in range(len(c['lesions']['26'][0]))])
for nm, s in (('<=1', th <= 1), ('1-2.5', (th > 1) & (th <= 2.5)), ('>2.5', th > 2.5)):
    pr(' thickness', nm, 'lesions', s.sum(), 'lost %', 100 * (ret[s] == 0).mean())
# split
names = open(ROOT + '/data/cache/volumes.txt').read().split('\n')
idx = list(range(len(names)))
tr, te = train_test_split(idx, test_size=0.2, random_state=42)
pr('split: train', len(tr), 'test', len(te))
vi = np.load(ROOT + '/data/cache/vol_index.npy')
nT = {c['j']: c for c in C}
for nm, ids in (('train', tr), ('test', te)):
    sl = np.isin(vi, ids)
    ts = np.concatenate([nT[j]['raw_T_slice'] for j in sorted(ids)]) if False else None
    cs_ = np.concatenate([nT[j]['cache_T_slice'] for j in sorted(ids)]); rs_ = np.concatenate([nT[j]['raw_T_slice'] for j in sorted(ids)])
    les = sum(len(nT[j]['lesions']['26'][0]) for j in ids if 'lesions' in nT[j] and nT[j]['lesions'])
    notum = sum(1 for j in ids if nT[j]['raw_T'] == 0)
    pr(nm, 'vols', len(ids), 'slices', len(cs_), 'cache tumor slices', (cs_ > 0).sum(), 'frac', (cs_ > 0).mean(), 'no-tumor vols', notum, 'lesions(26)', les, 'tumor vox raw', int(sum(nT[j]['raw_T'] for j in ids)), 'median thick', np.median([nT[j]['spacing'][2] for j in ids]),
       'n thick>2.5', sum(nT[j]['spacing'][2] > 2.5 for j in ids))
    # test set no-tumor
    pr('   no-tumor volume names', [names[j] for j in ids if nT[j]['raw_T'] == 0])
    # lesions sizes in that split
    e = []
    for j in ids:
        L = nT[j].get('lesions')
        if L:
            t = L['26'][0]; sp_ = nT[j]['spacing']
            e += list((6 * t * sp_[0] * sp_[1] * sp_[2] / np.pi) ** (1 / 3))
    e = np.array(e); pr('   eq diam median', np.median(e), '<10mm', np.mean(e < 10), '>64', (e > 64).sum(), 'max', e.max())
# oversampling: train
trsl = np.concatenate([nT[j]['cache_T_slice'] for j in sorted(tr)])
hasT = trsl > 0; nt_, nn_ = hasT.sum(), (~hasT).sum()
pr('train slices', len(trsl), 'tumor', nt_, 'non', nn_, 'weight on tumor slice', nn_ / nt_, ' expected tumor slice share after oversample', 0.5)
# unique-tumor-slice repetition
pr('expected draws per tumor slice per epoch', nn_ / nt_, ' per non-tumor slice', 1.0)
# which fraction of the tumor-slice mass comes from top volumes
per_vol_ts = np.array([(nT[j]['cache_T_slice'] > 0).sum() for j in sorted(tr)])
o = np.sort(per_vol_ts)[::-1]
pr('train: top 10% of volumes contribute tumor slices', o[:int(np.ceil(.1 * len(o)))].sum() / o.sum(), 'top 5 vols', o[:5].sum() / o.sum(), ' vols with >0 tumor slices', (per_vol_ts > 0).sum())
# tumor slices with tiny area in train oversampled
trA = trsl[hasT]
pr('train tumor slice area (cache px) <=10: %', 100 * (trA <= 10).mean(), ' <=25 %', 100 * (trA <= 25).mean(), ' median', np.median(trA))
# voxel-level imbalance in train: fraction of pixels in tumor slices that are tumor
pr('tumor px fraction inside tumor slices (cache)', trA.sum() / (hasT.sum() * 65536), ' overall', trsl.sum() / (len(trsl) * 65536))
# overall slice-level in all volumes
pr('all slices: cache pix tumor frac', cs.sum() / (len(cs) * 65536), ' liver(1|2) frac', sum(c['cache_L_slice'].sum() for c in C) / (len(cs) * 65536))
# patient-level imbalance: tumor ml distributions and tumor volume Gini
tm = np.array([PV[n]['tumor_ml'] for n in names])
srt = np.sort(tm)[::-1]; pr('top 10% volumes (13) hold of tumor volume', srt[:13].sum() / tm.sum(), 'top 5', srt[:5].sum() / tm.sum(), 'top 1', srt[0] / tm.sum(), ' vols tumor<1 mL', (tm < 1).sum(), ' <5mL', (tm < 5).sum(), ' <=0', (tm == 0).sum())
ntum = np.array([PV[n]['n_tumors'] for n in names]); pr('lesions held by top10% volumes by count', np.sort(ntum)[::-1][:13].sum() / ntum.sum())
ts = np.array([PV[n]['slices_tumor'] for n in names]); tn = np.array([PV[n]['n_slices'] for n in names])
pr('per-volume tumor-slice fraction: median (among tumor vols)', np.median((ts / tn)[ts > 0]), 'range', (ts / tn)[ts > 0].min(), (ts / tn)[ts > 0].max())
pr('volume-level: tumor slices share from thin(<=1) vols', ts[[PV[n]['spacing'][2] <= 1 for n in names]].sum() / ts.sum(), ' slices share thin', tn[[PV[n]['spacing'][2] <= 1 for n in names]].sum() / tn.sum())
# slice-level patients: tumor slice count by group
pr('slice counts', tn.sum(), 'tumor slices (raw stats)', ts.sum(), ts.sum() / tn.sum(), 'liver slices', sum(PV[n]['slices_liver'] for n in names) / tn.sum())
# voxel imbalance: bg/liver/tumor
nvox = sum(np.prod(PV[n]['shape']) for n in names)
lv_ = sum(PV[n]['liver_vox'] for n in names); tv_ = sum(PV[n]['tumor_vox'] for n in names)
pr('voxels total', nvox, 'tumor frac', tv_ / nvox, 'liver incl frac', lv_ / nvox, ' tumor/liver ratio', tv_ / lv_, 'bg:liver:tumor', (nvox - lv_) / tv_, (lv_ - tv_) / tv_, 1)
# within liver bbox
# cache slice-level: tumor vs liver area ratio in tumor slices
pr('in tumor slices cache: tumor px / liver px', trA.sum() / max(1, sum(c['cache_L_slice'][c['cache_T_slice'] > 0].sum() for c in C if c['j'] in tr)))
