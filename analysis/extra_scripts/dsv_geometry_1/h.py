import nibabel as nib,numpy as np,sys
R='data/raw/'
for i in [0,2,27,28,29,40,47,48,52,53,68]:
    for p in ['volume','segmentation']:
        im=nib.load(f'{R}{p}-{i}.nii'); h=im.header
        print(i,p,nib.aff2axcodes(im.affine),h.get_zooms(),int(h['qform_code']),int(h['sform_code']),h['descrip'][()][:40], np.round(im.affine[:3],2).tolist())
