import nibabel as nib, glob, collections
fs=sorted(glob.glob('data/**/volume-*.nii*',recursive=True))
c=collections.Counter()
for f in fs:
    im=nib.load(f)
    c[(im.header.get_slope_inter(), float(im.dataobj.slope), float(im.dataobj.inter))]+=1
print(c)
