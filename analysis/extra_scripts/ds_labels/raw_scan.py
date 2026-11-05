import sys, json, pickle, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from multiprocessing import Pool

ROOT = '.'
stats = json.load(open(ROOT + '/analysis/output/stats.json'))
PV = stats['per_volume']
S6 = ndi.generate_binary_structure(3, 1)
S26 = ndi.generate_binary_structure(3, 3)
S2D = np.zeros((3, 3, 3), bool); S2D[:, :, 1] = ndi.generate_binary_structure(2, 1)  # 4-conn in-plane, no z link
S4 = S2D.copy()


def region_perimeter(mask, px):
    """2D 4-connected regions per axial slice. returns area_mm2, perim_mm arrays."""
    lab, n = ndi.label(mask, structure=S2D)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
    per = np.zeros(n + 1)
    for ax in (0, 1):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(1, None); sl_b[ax] = slice(None, -1)
        a = lab[tuple(sl_a)]; b = lab[tuple(sl_b)]
        d = a != b
        per += np.bincount(a[d], minlength=n + 1) + np.bincount(b[d], minlength=n + 1)
    return area * px * px, per[1:] * px


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def work(i):
    t0 = time.time()
    p = PV[i]
    sx, sy, sz = p['spacing']
    num = p['name'].split('-')[1].split('.')[0]
    img = nib.load(f"{ROOT}/data/raw/segmentation-{num}.nii")
    raw = np.asarray(img.dataobj)
    nonint = bool(np.any(raw != np.round(raw))) if raw.dtype.kind == 'f' else False
    seg = raw.astype(np.uint8)
    del raw
    out = dict(i=i, name=p['name'], spacing=p['spacing'], shape=list(seg.shape), nonint=nonint,
               tumor_vox=int((seg == 2).sum()), liver_vox_only=int((seg == 1).sum()))
    L = seg > 0
    if not L.any():
        out['empty'] = True
        return out
    co = np.argwhere(L)
    lo = np.maximum(co.min(0) - 2, 0); hi = np.minimum(co.max(0) + 3, seg.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    seg = seg[sl]
    L = seg > 0
    T = seg == 2
    out['crop_lo'] = lo.tolist()
    vox_ml = sx * sy * sz / 1000.0
    holes2d = 0
    for z in range(L.shape[2]):
        s = L[:, :, z]
        if s.any():
            holes2d += int(ndi.binary_fill_holes(s).sum() - s.sum())
    holes3d = int(ndi.binary_fill_holes(L).sum() - L.sum())
    labL, nL = ndi.label(L, structure=S26)
    szL = np.bincount(labL.ravel())[1:]
    out.update(liver_holes2d_vox=holes2d, liver_holes3d_vox=holes3d, liver_ncc26=int(nL),
               liver_frag_vox=int(szL.sum() - szL.max()), liver_total_vox=int(szL.sum()),
               liver_frag_small=int(nL - 1))
    del labL
    A = L.sum((0, 1))
    zs = np.where(A > 0)[0]
    out['liver_z_gap_slices'] = int((A[zs.min():zs.max() + 1] == 0).sum())
    ld = []
    jumps = 0
    for z in range(zs.min(), zs.max()):
        d = dice(L[:, :, z], L[:, :, z + 1])
        ld.append(d)
        if A[z] > 0 and A[z + 1] > 0 and abs(A[z + 1] - A[z]) / max(A[z], A[z + 1]) > 0.3:
            jumps += 1
    ld = np.array(ld)
    out.update(liver_adj_dice_mean=float(np.nanmean(ld)) if len(ld) else np.nan,
               liver_adj_dice_p05=float(np.nanpercentile(ld, 5)) if len(ld) else np.nan,
               liver_area_jumps30=jumps, liver_adj_pairs=len(ld))
    areaL, perL = region_perimeter(L, (sx + sy) / 2)
    keep = areaL > 500
    out['liver_regions'] = np.stack([areaL[keep], perL[keep]], 1).astype(np.float32)
    if not T.any():
        out['no_tumor'] = True
        return out
    lab6, n6 = ndi.label(T, structure=S6)
    lab26, n26 = ndi.label(T, structure=S26)
    sz6 = np.bincount(lab6.ravel())[1:]; sz26 = np.bincount(lab26.ravel())[1:]
    out.update(n6=int(n6), n26=int(n26), sz6=sz6.astype(np.int64), sz26=sz26.astype(np.int64))
    objs = ndi.find_objects(lab26)
    bg = ~L
    bgadj = ndi.binary_dilation(bg, structure=S6) & T
    bgadj_ip = ndi.binary_dilation(bg, structure=S4) & T
    ones = seg == 1
    comps = []
    for k, o in enumerate(objs, start=1):
        o2 = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in o)
        m = lab26[o2] == k
        v = int(m.sum())
        shell = ndi.binary_dilation(m, structure=S6) & ~m
        sh_n = int(shell.sum())
        sh_bg = int((shell & bg[o2]).sum())
        sh_liv = int((shell & ones[o2]).sum())
        ext = [(s.stop - s.start) for s in o]
        n6in = int(np.unique(lab6[o2][m]).size)
        comps.append(dict(vox=v, ml=v * vox_ml, eq_diam_mm=(6 * v * vox_ml * 1000 / np.pi) ** (1 / 3),
                          bbox_mm=[ext[0] * sx, ext[1] * sy, ext[2] * sz], zext_slices=ext[2],
                          touch_bg=bool((bgadj[o2] & m).any()), touch_bg_ip=bool((bgadj_ip[o2] & m).any()),
                          frac_vox_bgadj=float((bgadj[o2] & m).sum() / v),
                          shell_n=sh_n, shell_bg=sh_bg, shell_liv=sh_liv, n6=n6in))
    out['comps'] = comps
    gap = T[:, :, :-2] & T[:, :, 2:] & ~T[:, :, 1:-1]
    out['tumor_zgap_vox'] = int(gap.sum())
    out['tumor_zgap_in_liver_vox'] = int((gap & ones[:, :, 1:-1]).sum())
    th = 0
    for z in np.where(T.any((0, 1)))[0]:
        s = T[:, :, z]
        th += int((ndi.binary_fill_holes(s) & ~s).sum())
    out['tumor_holes2d_vox'] = th
    At = T.sum((0, 1))
    td = []
    for z in range(T.shape[2] - 1):
        if At[z] > 0 and At[z + 1] > 0:
            td.append(dice(T[:, :, z], T[:, :, z + 1]))
    out['tumor_adj_dice'] = np.array(td, dtype=np.float32)
    out['tumor_slice_area_vox'] = At[At > 0].astype(np.int32)
    areaT, perT = region_perimeter(T, (sx + sy) / 2)
    out['tumor_regions'] = np.stack([areaT, perT], 1).astype(np.float32)
    out['tumor_bgadj_vox'] = int(bgadj.sum())
    out['sec'] = time.time() - t0
    return out


if __name__ == '__main__':
    idx = list(range(len(PV)))
    if len(sys.argv) > 1:
        idx = [int(a) for a in sys.argv[1:]]
    t0 = time.time()
    res = []
    with Pool(8) as pool:
        for k, r in enumerate(pool.imap(work, idx)):
            res.append(r)
            print(k, r['name'], round(time.time() - t0), flush=True)
    pickle.dump(res, open('/tmp/ds_labels/raw_scan.pkl' if len(sys.argv) == 1 else '/tmp/ds_labels/raw_scan_test.pkl', 'wb'))
    print('done', time.time() - t0)
