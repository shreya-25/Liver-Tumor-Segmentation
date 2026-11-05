# legacy: the original thesis scripts

| File | What it is |
|---|---|
| `train_orig_AIMUNetALF.py` | original AIM-UNet training script |
| `test_aimunet_orig_65.py` | original test script (per-slice Dice, batch size 1) |
| `Syn_Img_Pipeline_Code.py` | original multi-GAN synthetic tumor script |

These are the **unchanged original scripts**, except that the author's Windows directory prefix was replaced by the
placeholder `<PATH_TO>\LTS_V1`. They are **not maintained** and will not run as they are: set the data and output paths first.
Use the maintained code in `Image Segmentation/` and `Image Synthesis/` instead.

Known problems:

- `Syn_Img_Pipeline_Code.py`: the GAN training loop is a stub (no loss, backward or optimizer step), tensor shapes and value
  ranges are inconsistent, and the insertion step can corrupt labels. A line-referenced audit is in
  the GAN analysis report (kept locally); the corrected version is `Image Synthesis/Syn_Img_Pipeline_fixed.py`.
- `test_aimunet_orig_65.py`: Dice is computed per slice and averaged over all slices, so slices with no tumor in the label and none
  predicted score 1.0. The resulting numbers are not comparable to the patient-level 3D Dice used elsewhere in this repo; see the metric
  note in the [main README](../README.md) and the report (the research report, kept locally).
- For the data handling of these scripts and the other issues found, see the report.
