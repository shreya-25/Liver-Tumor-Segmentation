import json, pickle, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from multiprocessing import Pool

ROOT = '.'
PV = json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']
names = open(ROOT + '/data/cache/volumes.txt').read().split('\n')
vi = np.load(ROOT + '/data/cache/vol_index.npy')
S6 = ndi.generate_binary_structure(3, 1)
S26 = ndi.generate_binary_structure(3, 3)
starts = np.concatenate([[0], np.cumsum(np.bincount(vi))])


def work(j):
    name = names[j]
    num = name.split('-')[1].split('.')[0]
    cm = np.load(ROOT + '/data/cache/masks.npy', mmap_mode='r')
    c = np.array(cm[starts[j]:starts[j + 1]])
    seg = np.asarray(nib.load(f"{ROOT}/data/raw/segmentation-{num}.nii").dataobj).astype(np.uint8)
    seg = np.moveaxis(seg, 2, 0)
    pv = next(p for p in PV if p['name'] == name)
    r = dict(j=j, name=name, spacing=pv['spacing'], nz=seg.shape[0], cache_nz=c.shape[0])
    if seg.shape[0] != c.shape[0]:
        r['aligned'] = False
        return r
    odd = seg[:, 1::2, 1::2]; even = seg[:, 0::2, 0::2]
    r['eq_odd'] = bool(np.array_equal(odd, c)); r['eq_even'] = bool(np.array_equal(even, c))
    r['mismatch_odd'] = int((odd != c).sum())
    # also slice flip test
    r['eq_flip_z'] = bool(np.array_equal(odd[::-1], c)) if not r['eq_odd'] else False
    r['raw_T'] = int((seg == 2).sum()); r['cache_T'] = int((c == 2).sum())
    r['raw_L1'] = int((seg == 1).sum()); r['cache_L1'] = int((c == 1).sum())
    rT = (seg == 2).sum((1, 2)); cT = (c == 2).sum((1, 2))
    r['raw_T_slice'] = rT.astype(np.int32); r['cache_T_slice'] = cT.astype(np.int32)
    r['raw_L_slice'] = (seg > 0).sum((1, 2)).astype(np.int32); r['cache_L_slice'] = (c > 0).sum((1, 2)).astype(np.int32)
    T = seg == 2
    out = {}
    if T.any():
        for tag, S in (('26', S26), ('6', S6)):
            lab, n = ndi.label(T, structure=S)
            kept = lab[:, 1::2, 1::2]
            ret = np.bincount(kept.ravel(), minlength=n + 1)[1:]
            tot = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            out[tag] = (tot, ret)
        # also even-phase retention and a hypothetical (all-4-offset) for the 26 label
        lab, n = ndi.label(T, structure=S26)
        for ph, (a, b) in {'even': (0, 0), 'mixed': (0, 1)}.items():
            k = lab[:, a::2, b::2]
            out['26' + ph] = np.bincount(k.ravel(), minlength=n + 1)[1:]
        # per-lesion z-extent and eq diam
        objs = ndi.find_objects(lab)
        out['zext'] = np.array([o[0].stop - o[0].start for o in objs])
        out['inplane_ext_vox'] = np.array([max(o[1].stop - o[1].start, o[2].stop - o[2].start) for o in objs])
    r['lesions'] = out
    return r


if __name__ == '__main__':
    t0 = time.time()
    with Pool(8) as pool:
        res = []
        for k, r in enumerate(pool.imap(work, range(len(names)))):
            res.append(r)
            print(k, r['name'], r.get('eq_odd'), round(time.time() - t0), flush=True)
    pickle.dump(res, open('/tmp/ds_labels/cache_cmp.pkl', 'wb'))
