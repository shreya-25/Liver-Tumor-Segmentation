import json, pickle
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from multiprocessing import Pool
ROOT = '.'
PV = json.load(open(ROOT + '/analysis/output/stats.json'))['per_volume']
S26 = ndi.generate_binary_structure(3, 3)


def work(i):
    p = PV[i]
    if p['tumor_vox'] == 0:
        return None
    num = p['name'].split('-')[1].split('.')[0]
    seg = np.asarray(nib.load(f"{ROOT}/data/raw/segmentation-{num}.nii").dataobj).astype(np.uint8)
    H, W, Z = seg.shape
    zsel = np.floor((np.arange(256) + 0.5) * Z / 256).astype(int)
    zsel = np.unique(np.minimum(zsel, Z - 1))
    wsel = np.arange(1, W, 2)
    T = seg == 2
    keep = np.zeros_like(T)
    keep[:, wsel[:, None], zsel[None, :]] = True if False else False
    kept = T[:, 1::2, :][:, :, zsel]
    lab, n = ndi.label(T, structure=S26)
    ret = np.bincount(lab[:, 1::2, :][:, :, zsel].ravel(), minlength=n + 1)[1:]
    tot = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    # slices along H with tumor
    rs = T.sum((1, 2)) > 0
    # after resize: slice h kept (all h kept), its tumor px = count among retained (w odd, z in zsel)
    cs = kept.sum((1, 2)) > 0
    return dict(i=i, tot=tot, ret=ret, raw_slices=int(rs.sum()), kept_slices=int(cs.sum()), kept_vox=int(kept.sum()), T=int(T.sum()), Z=Z)


if __name__ == '__main__':
    with Pool(8) as pool:
        res = [r for r in pool.map(work, range(len(PV))) if r]
    tot = np.concatenate([r['tot'] for r in res]); ret = np.concatenate([r['ret'] for r in res])
    print('default-axis pipeline (slice along axis0=H, resize (W,Z)->256x256, NN)')
    print('lesions', len(tot), 'lost', (ret == 0).sum(), 100 * (ret == 0).mean(), '%')
    print('unique tumor vox retained fraction', sum(r['kept_vox'] for r in res) / sum(r['T'] for r in res))
    print('raw tumor slices (H-axis)', sum(r['raw_slices'] for r in res), 'kept', sum(r['kept_slices'] for r in res))
    big = [r for r in res if r['Z'] > 256]
    print('volumes with Z>256:', sum(PV[r['i']]['n_slices'] > 256 for r in res), ' lesions lost there', sum((r['ret'] == 0).sum() for r in big), 'of', sum(len(r['tot']) for r in big),
          ' tumor vox retained frac there', sum(r['kept_vox'] for r in big) / sum(r['T'] for r in big))
    print('slice count total for default axis', sum(PV[i]['shape'][0] for i in range(len(PV))))
    for lo, hi in ((1, 5), (5, 100), (100, 1000), (1000, 1e12)):
        s = (tot >= lo) & (tot < hi)
        print(f' native vox [{lo},{hi}) n={s.sum()} lost={100 * (ret[s] == 0).mean():.1f}%')
