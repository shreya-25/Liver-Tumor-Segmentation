import pickle, json
import numpy as np
ROOT = '.'
PV = json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']
R = pickle.load(open('/tmp/ds_labels/raw_scan.pkl', 'rb'))
pr = lambda *a: print(*a)
pr('n', len(R), 'nonint', sum(r['nonint'] for r in R), 'empty', sum(r.get('empty', False) for r in R))
# consistency with stats.json
bad = [r['name'] for r, p in zip(R, PV) if r['tumor_vox'] != p['tumor_vox']]
pr('tumor_vox mismatch vs stats', len(bad))
tv = [r for r in R if 'n6' in r]
pr('vols with tumor', len(tv), 'no tumor', sum(r.get('no_tumor', False) for r in R))
n6 = np.array([r['n6'] for r in tv]); n26 = np.array([r['n26'] for r in tv])
pr('n comps 6:', n6.sum(), '26:', n26.sum(), 'vols changed', (n6 != n26).sum(), 'reduction %', 100 * (1 - n26.sum() / n6.sum()))
pr('per-vol n6 median', np.median(n6), 'n26 median', np.median(n26), 'max', n6.max(), n26.max())
# compare with stats n_tumors
st = np.array([PV[r['i']]['n_tumors'] for r in tv]); pr('stats n_tumors sum', st.sum(), 'match n6', (st == n6).all())
sz6 = np.concatenate([r['sz6'] for r in tv]); sz26 = np.concatenate([r['sz26'] for r in tv])
for k in (1, 2, 4, 5, 10, 27, 100):
    pr(f'comps < {k}+ vox: 6:', int((sz6 < k).sum()) if k > 1 else int((sz6 == 1).sum()), ' 26:', int((sz26 < k).sum()) if k > 1 else int((sz26 == 1).sum()))
pr('single voxel 6/26', int((sz6 == 1).sum()), int((sz26 == 1).sum()), 'of', len(sz6), len(sz26))
pr('<5 vox 6/26:', int((sz6 < 5).sum()), int((sz26 < 5).sum()), ' pct', 100 * (sz6 < 5).mean(), 100 * (sz26 < 5).mean())
pr('tumor vox in <5vox comps (6conn):', sz6[sz6 < 5].sum(), 'of', sz6.sum(), 100 * sz6[sz6 < 5].sum() / sz6.sum())
# per-volume tiny comps
tiny6 = np.array([(r['sz6'] < 5).sum() for r in tv]); pr('vols with >=1 comp<5vox (6):', (tiny6 > 0).sum(), 'vols with >=1 single voxel:', sum((r['sz6'] == 1).any() for r in tv))
# which volumes are the tiny comps in, by thickness
thick = np.array([PV[r['i']]['spacing'][2] for r in tv])
tiny26 = np.array([(r['sz26'] < 5).sum() for r in tv])
pr('tiny26 by thickness group: <=1:', tiny26[thick <= 1].sum(), ' (1,2.5]:', tiny26[(thick > 1) & (thick <= 2.5)].sum(), ' >2.5:', tiny26[thick > 2.5].sum())
# merging
comps = [dict(c, vol=r['i'], thick=PV[r['i']]['spacing'][2], sx=PV[r['i']]['spacing'][0]) for r in tv for c in r['comps']]
pr('26-comps', len(comps), 'comps containing >1 six-conn comps', sum(c['n6'] > 1 for c in comps), ' total six-comps absorbed', sum(c['n6'] - 1 for c in comps))
m = [c for c in comps if c['n6'] > 1]
pr('merged comps eq diam median', np.median([c['eq_diam_mm'] for c in m]), ' max n6 in one', max(c['n6'] for c in comps))
ed = np.array([c['eq_diam_mm'] for c in comps]); ml = np.array([c['ml'] for c in comps])
for lab, a in (('6', None), ('26', ed)):
    pass
pr('26conn eq diam: median', np.median(ed), 'mean', ed.mean(), '<5:', 100 * (ed < 5).mean(), '<10:', 100 * (ed < 10).mean(), '<20', 100 * (ed < 20).mean(), '>64', (ed > 64).sum(), 100 * (ed > 64).mean(), '>=30', 100 * (ed >= 30).mean())
pr('percentiles eq diam 5,25,50,75,95', np.percentile(ed, [5, 25, 50, 75, 95]))
pr('volume ml percentiles 5,25,50,75,95,max', np.percentile(ml, [5, 25, 50, 75, 95]), ml.max())
o = np.argsort(-ml); cs = np.cumsum(ml[o]) / ml.sum()
k10 = int(np.ceil(0.1 * len(ml))); pr('top 10% lesions hold', cs[k10 - 1], 'top 1%', cs[int(0.01 * len(ml))], 'top 3 comps', cs[2], ' bottom 50% hold', ml[np.argsort(ml)][:len(ml) // 2].sum() / ml.sum())
pr('lesions <1 mL', 100 * (ml < 1).mean(), '< 0.1 mL', 100 * (ml < 0.1).mean())
# per-patient lesion count dist
pr('n26 per vol percentiles 10,25,50,75,90', np.percentile(n26, [10, 25, 50, 75, 90]), 'vols with 1 lesion', (n26 == 1).sum(), '>=10 lesions', (n26 >= 10).sum(), '>=30', (n26 >= 30).sum(), 'max', n26.max())
# lesion sizes by thickness group
def grp(t): return 0 if t <= 1 else (1 if t <= 2.5 else 2)
g = np.array([grp(c['thick']) for c in comps]); gv = np.array([grp(t) for t in thick])
gn = ['<=1mm', '1-2.5mm', '>2.5mm']
pr('--- thickness groups (volumes with tumor)')
for k in range(3):
    s = g == k; v = gv == k
    cc = [c for c in comps if grp(c['thick']) == k]
    pr(gn[k], 'vols', v.sum(), 'lesions', s.sum(), 'lesions/vol median', np.median(n26[v]), 'mean', n26[v].mean(), 'eq diam median', np.median(ed[s]), '<10mm %', 100 * (ed[s] < 10).mean(),
       '<5 vox %', 100 * np.mean([c['vox'] < 5 for c in cc]), 'min ml', ml[s].min(), 'p10 ml', np.percentile(ml[s], 10),
       'zext=1slice %', 100 * np.mean([c['zext_slices'] == 1 for c in cc]), 'median zext_mm', np.median([c['zext_slices'] * c['thick'] for c in cc]), 'tumor ml median/vol', np.median([PV[r['i']]['tumor_ml'] for r in tv if grp(PV[r['i']]['spacing'][2]) == k]))
# no-tumor volumes by group
nt = [r for r in R if r.get('no_tumor')]
pr('no-tumor vols', [(r['name'], PV[r['i']]['spacing'][2]) for r in nt])
pr('no-tumor by group', np.bincount([grp(PV[r['i']]['spacing'][2]) for r in nt], minlength=3), 'of', np.bincount([grp(p['spacing'][2]) for p in PV], minlength=3))
# --- boundary
pr('--- boundary')
tb = np.array([c['touch_bg'] for c in comps]); tbi = np.array([c['touch_bg_ip'] for c in comps])
pr('lesions with any 6-nbr bg voxel:', tb.sum(), 100 * tb.mean(), ' in-plane:', tbi.sum(), 100 * tbi.mean())
sbg = np.array([c['shell_bg'] / c['shell_n'] for c in comps])
pr('shell bg fraction: >0', 100 * (sbg > 0).mean(), '>=0.25', 100 * (sbg >= .25).mean(), '>=0.5', 100 * (sbg >= .5).mean(), '>=0.9', 100 * (sbg >= .9).mean(), '>0.99 (fully outside liver label)', (sbg > .99).sum())
fb = np.array([c['frac_vox_bgadj'] for c in comps])
pr('tumor vox adjacent to bg total frac', sum(r['tumor_bgadj_vox'] for r in tv) / sum(r['tumor_vox'] for r in tv))
for lo_, hi_ in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
    s = (ed >= lo_) & (ed < hi_)
    pr(f'eq diam [{lo_},{hi_}) n={s.sum()} touch_bg%={100 * tb[s].mean():.1f}  shell_bg>=.5 % {100 * (sbg[s] >= .5).mean():.1f}  shell_bg median {np.median(sbg[s]):.3f}')
# mix of shell labels (liver-surrounded) summary
sl_ = np.array([c['shell_liv'] / c['shell_n'] for c in comps])
pr('lesions fully surrounded by label 1 (shell_liv==1):', (sl_ > .999).sum(), 100 * (sl_ > .999).mean())
# touch by group
for k in range(3):
    pr(gn[k], 'touch_bg %', 100 * tb[g == k].mean(), 'shell_bg>=.5 %', 100 * (sbg[g == k] >= .5).mean())
# --- holes etc
pr('--- holes / fragments')
ls = [r for r in R if 'liver_holes2d_vox' in r]
h2 = np.array([r['liver_holes2d_vox'] for r in ls]); h3 = np.array([r['liver_holes3d_vox'] for r in ls]); lv = np.array([r['liver_total_vox'] for r in ls])
pr('liver 2D holes: vols with >0', (h2 > 0).sum(), 'total vox', h2.sum(), 'frac of liver', h2.sum() / lv.sum(), 'max per vol', h2.max())
pr('liver 3D cavities: vols with >0', (h3 > 0).sum(), 'total', h3.sum(), 'max', h3.max())
nc = np.array([r['liver_ncc26'] for r in ls]); fv = np.array([r['liver_frag_vox'] for r in ls])
pr('liver (1|2) comps 26: vols with >1:', (nc > 1).sum(), 'max comps', nc.max(), 'median', np.median(nc), 'frag vox frac', fv.sum() / lv.sum(), ' vols frag frac>1%:', ((fv / lv) > .01).sum())
pr('liver z-gap slices: vols with>0', sum(r['liver_z_gap_slices'] > 0 for r in ls))
th = np.array([r['tumor_holes2d_vox'] for r in tv]); tvx = np.array([r['tumor_vox'] for r in tv])
pr('tumor 2D holes(label1 enclosed in tumour): vols>0', (th > 0).sum(), 'total', th.sum(), 'frac of tumor vox', th.sum() / tvx.sum(), 'max', th.max())
zg = np.array([r['tumor_zgap_vox'] for r in tv]); zgl = np.array([r['tumor_zgap_in_liver_vox'] for r in tv])
pr('tumor single-slice z-gaps: vols>0', (zg > 0).sum(), 'vox', zg.sum(), 'frac of tumor', zg.sum() / tvx.sum(), 'in liver label', zgl.sum())
# --- slice-to-slice
pr('--- slice discontinuity by thickness group')
ld = np.array([r['liver_adj_dice_mean'] for r in ls]); lp = np.array([r['liver_adj_dice_p05'] for r in ls]); lj = np.array([r['liver_area_jumps30'] / max(r['liver_adj_pairs'], 1) for r in ls])
tl = np.array([PV[r['i']]['spacing'][2] for r in ls]); gl = np.array([grp(t) for t in tl])
for k in range(3):
    s = gl == k
    pr(gn[k], 'n', s.sum(), 'liver adj dice mean of vols', np.nanmean(ld[s]), 'p05', np.nanmedian(lp[s]), 'frac pairs area jump>30%', np.mean(lj[s]))
# tumor adj dice
for k in range(3):
    arr = np.concatenate([r['tumor_adj_dice'] for r in tv if grp(PV[r['i']]['spacing'][2]) == k])
    pr(gn[k], 'tumor adj dice median', np.nanmedian(arr), 'mean', np.nanmean(arr), 'p10', np.nanpercentile(arr, 10), 'n pairs', len(arr), 'frac <0.5', np.mean(arr < .5))
# z-gap by group
for k in range(3):
    s = gv == k
    pr(gn[k], 'zgap vox/tumor vox', zg[s].sum() / tvx[s].sum(), 'tumor holes frac', th[s].sum() / tvx[s].sum(), 'liver holes frac', h2[gl == k].sum() / lv[gl == k].sum())
# roughness
pr('--- 2D boundary roughness C=P^2/(4 pi A), median by area bin and group')
bins = [(50, 200), (200, 1000), (1000, 5000), (5000, 1e9)]
for name, key, vols in (('tumor', 'tumor_regions', tv), ('liver', 'liver_regions', ls)):
    for k in range(3):
        arrs = [r[key] for r in vols if grp(PV[r['i']]['spacing'][2]) == k and len(r[key])]
        if not arrs: continue
        a = np.concatenate(arrs); C = a[:, 1] ** 2 / (4 * np.pi * a[:, 0])
        row = []
        for lo_, hi_ in bins:
            s = (a[:, 0] >= lo_) & (a[:, 0] < hi_)
            row.append(f'[{lo_},{hi_}):n={s.sum()} med={np.median(C[s]) if s.any() else float("nan"):.2f}')
        pr(name, gn[k], ' | '.join(row))
# in-plane spacing by group
pr('inplane spacing median by group', [np.median([p['spacing'][0] for p in PV if grp(p['spacing'][2]) == k]) for k in range(3)])
# tumour slice area (native vox) distribution
ta = np.concatenate([r['tumor_slice_area_vox'] for r in tv])
pr('tumor slice native area px: n', len(ta), 'median', np.median(ta), '<=10 px %', 100 * (ta <= 10).mean(), '<=50 px %', 100 * (ta <= 50).mean(), '<=200 %', 100 * (ta <= 200).mean())
pickle.dump(dict(comps=comps), open('/tmp/ds_labels/comps.pkl', 'wb'))
