import json, pickle, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from multiprocessing import Pool

ROOT = '.'
PV = json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']
S26 = ndi.generate_binary_structure(3, 3)
P = (64, 64, 64)


def extract_patch_with_info(vol, center, size=P):
    # exact copy of the pipeline's extract_patch logic, plus bookkeeping
    start = [int(c - s // 2) for c, s in zip(center, size)]
    slices = tuple(slice(max(0, start[i]), max(0, start[i]) + size[i]) for i in range(3))
    patch = np.zeros(size, np.uint8)
    region = vol[slices]
    pad = [region.shape[i] for i in range(3)]
    patch[tuple(slice(0, pad[i]) for i in range(3))] = region
    # position of centre voxel inside patch
    pos = [int(center[i]) - max(0, start[i]) for i in range(3)]
    win0 = [max(0, start[i]) for i in range(3)]
    return patch, pos, pad, win0


def work(i):
    p = PV[i]
    if p['tumor_vox'] == 0:
        return None
    num = p['name'].split('-')[1].split('.')[0]
    seg = np.asarray(nib.load(f"{ROOT}/data/raw/segmentation-{num}.nii").dataobj).astype(np.uint8)
    sp = np.array(p['spacing'])
    seg1 = zoom(seg, sp / 1.0, order=0)
    del seg
    co = np.argwhere(seg1 > 0)
    mins, maxs = co.min(0), co.max(0) + 1
    roi = seg1[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    T = roi == 2
    lab, n = ndi.label(T, structure=S26)
    sizes = np.bincount(lab.ravel())[1:]
    objs = ndi.find_objects(lab)
    ext = np.array([[s.stop - s.start for s in o] for o in objs])
    Nvox = int(T.sum())
    k = min(50, Nvox)
    rng = np.random.default_rng(1000 + i)
    centers = np.argwhere(T)
    sel = centers[rng.choice(len(centers), k, replace=False)]
    rows = []
    for c in sel:
        patch, pos, pad, win0 = extract_patch_with_info(roi, c)
        L = lab[c[0], c[1], c[2]]
        # lesion voxels inside window
        w = tuple(slice(win0[a], win0[a] + pad[a]) for a in range(3))
        sub = lab[w]
        inwin = int((sub == L).sum())
        ids = np.unique(sub[sub > 0])
        rows.append(dict(lesion=int(L), les_vox=int(sizes[L - 1]), les_ext=ext[L - 1].tolist(), tumor_frac=float((patch == 2).mean()),
                         liver_frac=float((patch == 1).mean()), bg_frac=float((patch == 0).mean()),
                         pad_frac=float(1 - np.prod(pad) / 64 ** 3), pos=pos,
                         frac_les_in_patch=inwin / sizes[L - 1], n_lesions_in_patch=int(len(ids)),
                         other_tumor_frac=float((patch == 2).mean() - inwin / 64 ** 3)))
    # expected sampling probability per lesion
    pk = 1 - (1 - sizes / Nvox) ** k if Nvox > k else np.ones(len(sizes))
    # native comps for resampling effect
    return dict(i=i, name=p['name'], Nvox1mm=Nvox, n_les_1mm=int(n), sizes1mm=sizes, ext1mm=ext, pk=pk, rows=rows, roi_shape=list(roi.shape),
                scale=sp.tolist())


if __name__ == '__main__':
    t0 = time.time()
    res = []
    with Pool(8) as pool:
        for k, r in enumerate(pool.imap(work, range(len(PV)))):
            if r is not None:
                res.append(r)
            print(k, round(time.time() - t0), flush=True)
    pickle.dump(res, open('/tmp/ds_labels/gan_patch.pkl', 'wb'))
