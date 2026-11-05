import nibabel as nib, numpy as np, json
B=''
S=json.load(open(B+'analysis/output/stats.json'))['per_volume']; P={int(d['name'].split('-')[1].split('.')[0]):d for d in S}
for i in range(46,54):
    v=nib.load(B+f'data/raw/volume-{i}.nii'); s=nib.load(B+f'data/raw/segmentation-{i}.nii')
    print(i,'vol zooms',np.round(v.header.get_zooms(),3),'seg zooms',np.round(s.header.get_zooms(),3),'vol codes',v.header['qform_code'],v.header['sform_code'],'seg',s.header['qform_code'],s.header['sform_code'],'liver HU mean',round(P[i]['liver_hu_mean'],1),'tumor n',P[i]['n_tumors'])
# global seg zooms vs vol
bad=[]
for i in range(131):
    v=nib.load(B+f'data/raw/volume-{i}.nii'); s=nib.load(B+f'data/raw/segmentation-{i}.nii')
    if not np.allclose(v.header.get_zooms(),s.header.get_zooms(),atol=1e-3): bad.append(i)
print('zoom mismatch vols',bad)
import os
sz=[os.path.getsize(B+f'data/raw/volume-{i}.nii') for i in range(131)]
print('file size total GB',sum(sz)/1e9,'max MB',max(sz)/1e6,'min MB',min(sz)/1e6)
