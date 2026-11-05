# Experiments

Code only. Every script writes its outputs to `results/<name>/` (or, for the Colab notebooks, to the Google Drive
`OUT_DIR` set in their first cell), so nothing here needs to be moved to re-run it.

```
aimunet_colab/                    Google Colab notebooks: legacy AIM-UNet, trained on LiTS, tested on synthetic + LiTS
  AIMUNet_Train_LiTS0-100_Test_Synthetic_LiTS101-130.ipynb
                                  original: 2-fold CV on LiTS 0-79, validation 80-100, 20 epochs;
                                  Part A train, Part B test on synthetic images, Part C test on LiTS 101-130
  AIMUNet_v2_SmallTumour_Train_LiTS0-100_Test_Synthetic_LiTS101-130.ipynb
                                  v2, same split, changes for (small) tumour detection: tumour-weighted loss + Tversky,
                                  2.5D input, small-tumour oversampling, cosine LR, flip TTA, tumour threshold / hole
                                  filling tuned on validation, tumours-found-by-size metrics. Parts B / C can also
                                  re-test models trained with the original notebook.
  builders/                       build_original.py, build_v2.py: generate the two notebooks above (edit the builder,
                                  then `python builders/build_v2.py`, or edit the .ipynb directly)
  older/                          earlier, superseded notebooks (LiTS 10 -> synthetic 4; Training Batch 1 + 10 synthetic)
aimunet_local/
  train_aimunet_cv.py             local (CPU/GPU) version of the LiTS 10 -> synthetic 4 run -> results/aimunet_lits10_syn4/
gan_lits130/                      tumour GAN on LiTS (Image Synthesis/syn_training_improved.py) -> results/gan_lits130/
  prepare_lits.py                 extract tumour patches from the LiTS volumes
  run_lits.py                     train the GAN (50 epochs) and evaluate it
  generate_100.py                 100 tumour patches (50 small + 50 medium) from checkpoints/gan_best.pt
synthetic_healthy40/
  make_and_evaluate.py            40 synthetic images: GAN tumours inserted into healthy livers -> results/synthetic_healthy40/
                                  (its images/ folder is SYN_ROOT for the Colab notebooks)
```

Paths: the local scripts use the absolute repo path in their `REPO = ...` line; change it if the repo moves.
