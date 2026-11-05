# Liver Tumor Segmentation on LiTS CT: a corrected, reproducible pipeline

A 2D AIM-UNet pipeline that segments liver and liver tumors in CT volumes from the [LiTS](https://arxiv.org/abs/1901.04056)
training set (131 volumes), plus a corrected 3D multi-GAN pipeline for synthetic tumor insertion and a research report.
The project started from a thesis codebase. A dataset audit found that the original slice cache normalised each volume
by its own min-max range and ignored inconsistent array orientation, so this repo rebuilds the cache with a fixed
HU window and canonical orientation, evaluates per patient in 3D, and documents what was found and what is still open.

**Status in one line:** the data-pipeline fixes show a large gain in a small, single-seed experiment; absolute tumor
Dice is still low; no synthetic data has been used to train a segmenter yet.

## Key results (baseline A / B / C, patient-level on a held-out test fold)

| Arm | Cache and augmentation | Liver Dice (mean) | Tumor Dice (mean) | Lesions found (pooled) |
|---|---|---|---|---|
| A | per-volume min-max, flips | 0.537 | 0.060 | 20 / 192 |
| B | fixed -100..250 HU window + canonical orientation, flips | 0.840 | 0.214 | 65 / 192 |
| C | as B, no flips | 0.924 | 0.277 | 81 / 192 |

**Read this before quoting the table.** Test fold of 26 patients (23 with tumor), 16 training volumes, 6 validation
volumes, 6 epochs, seed 1234, **one training run per arm**. The A to B step changes the window and the orientation
handling together, so their effects cannot be separated; B to C isolates flip augmentation only. Differences are
indications, not established effects, and absolute tumor Dice is still low. The runs came from an earlier,
manifest-based split and older code (archived locally, not in this repo), so they are not what the
current scripts print. The B and C test volumes were also oriented with their ground-truth liver masks (the cache is built
that way for all 131 volumes), so these scores assume the correct orientation is known; an inference-time orientation rule
has not been evaluated, and the A to B gain is not what a new patient would see without one.

This repository holds **code only**. Results, figures, tables, the research report and the showcase notebook are kept
locally (see `.gitignore`); `analysis/make_showcase.py` and `analysis/build_notebook.py` rebuild them from saved results.

## What is in this repo

```
README.md
setup.sh, run_pipeline.sh         environment setup and the check / quick / full pipeline
requirements.txt, requirements-notebook.txt
Image Segmentation/
  prepare_data.py                 NIfTI -> uint8 slice cache, per-volume min-max (v1, arm A)
  prepare_data_v2.py              fixed HU window + orientation canonicalisation + metadata (v2)
  train_fast.py                   AIM-UNet trainer (same model and loss as the original script)
  eval_fast.py                    per-patient 3D Dice, lesion recall, size / thickness breakdowns
  test_train_eval_fast.py         tests for train_fast.py and eval_fast.py
Image Synthesis/
  Syn_Img_Pipeline_fixed.py       corrected 3D multi-GAN synthesis pipeline
  test_syn_pipeline_fixed.py      its tests
analysis/
  dataset_analysis.py, make_figures.py   dataset statistics and figures
  make_showcase.py, build_notebook.py    rebuild tables / figures / notebook from saved results
  run_baseline.sh                        re-run design of the A/B/C comparison
  slice_vs_patient_dice.py               the metric demo described below
  extra_scripts/                         provenance scripts for the report's [tmp] numbers
docs/
  RUNNING.md                      full setup and run guide (the former START_HERE.md)
legacy/                           the original thesis scripts, unmaintained (see legacy/README.md)
experiments/                      code of the later experiments (Colab AIM-UNet notebooks, GAN / synthetic-image runs);
                                  their outputs go to results/ (see experiments/README.md)
```

Not included: the LiTS CT data (about 50 GB, in `data/raw/`), `results/` (logs, metrics, figures, checkpoints, synthetic
images), and the reports, figures and tables; see Quick start for the data.

## Quick start

```bash
bash setup.sh                    # creates .venv, installs requirements.txt, prints the training device
# place the LiTS files, uncompressed, in data/raw/: volume-0.nii ... volume-130.nii, segmentation-0.nii ... segmentation-130.nii
bash run_pipeline.sh check       # verifies the data and shows the device (seconds)
bash run_pipeline.sh quick       # smoke run: builds the cache, trains 1 epoch on 4 volumes
bash run_pipeline.sh full        # 104 train volumes, 10 epochs, evaluation on the 27 held-out volumes (hours)
```

`full` trains with flip augmentation by default (the arm B setting); set `LITS_FLIP_AUG=0` for the arm C setting (no flips).
For tumor Dice the paired bootstrap interval of the mean difference C minus B is [-0.044, +0.166], which includes zero;
both runs are single-seed.

Python 3.11+ (3.12 tested). A GPU (CUDA or Apple `mps`) is strongly recommended; on CPU only the smoke tests are
practical. Details, environment variables and the data layout are in [docs/RUNNING.md](docs/RUNNING.md).
Note that `full` uses the original 104 / 27 split, where the held-out part is both validation and test set, so its
scores are validation-selected and optimistic.

## Notebook and report (kept locally, not in this repo)

- `LiTS_Pipeline_Showcase.ipynb` (built by `analysis/build_notebook.py`): the baseline, preprocessing effect, per-patient
  results and GAN status. Re-running it needs `requirements-notebook.txt`.
- `docs/Liver_Tumor_Research_Report.pdf`: dataset analysis, pipeline audit, findings and their confidence, and caveats.
  Numbers tagged `[tmp]` come from ad hoc scripts kept under `analysis/extra_scripts/`.

## Synthetic tumor generation: honest status

`Image Synthesis/Syn_Img_Pipeline_fixed.py` fixes the problems found in the original script (documented line by line in
the local GAN analysis report): the original training loop was a stub, tensor shapes and
value ranges were inconsistent, and the insertion step could corrupt labels. The corrected version passes its tests. But:

- generated tumor patches are blurry and their realism was not verified;
- the generator's mask head does not generalise on this little data (Dice 0.567 on training patches versus 0.058 on 2 held-out volumes);
- the insertion step was checked visually only;
- **no synthetic data has been used to train any segmenter.** There is no evidence here that synthetic augmentation helps.

Treat the pipeline as a tested starting point, not a trained synthetic-data generator.

## A note on the Dice metric (why thesis numbers are not comparable)

The original test script computes Dice **per slice** with a small smoothing term and averages over all slices. A slice with no
tumor in the label and none predicted scores 1.0, and most slices contain no tumor, so the mean is dominated by
empty-vs-empty slices. This repo reports **patient-level 3D Dice** (one Dice over all voxels of a patient), which is not inflated by empty slices.
The two numbers measure different things, so the thesis figures are not comparable to the tables above.
They are not necessarily wrong for what they compute.

Demonstration with identical predictions (arm C model, 8 test patients with tumor, `analysis/slice_vs_patient_dice.py`):

| Tumor Dice definition | Value |
|---|---|
| per-slice mean over all slices (original script's metric) | 0.858 |
| per-slice, only slices that contain tumor | 0.215 |
| patient-level 3D (this repo) | 0.281 |

In these 8 patients, 85.4% of the 5,200 slices have no tumor in the label, and the model predicts no tumor on 96.9% of those. Reproduce with
`.venv/bin/python -I analysis/slice_vs_patient_dice.py` (needs `data/cache_v2` and the arm C checkpoint, which are not in the repo).

## Limitations

- The baseline is small: 16 training volumes, 6 epochs, one seed per arm; absolute tumor Dice is low (0.06 / 0.21 / 0.28).
- The window and orientation fixes are confounded in A to B; arm C is a flips-only change from B on a single seed.
- The baseline numbers come from an archived split and code (see above); the current `run_pipeline.sh` uses the original split
  in which validation and test sets coincide, so a clean estimate needs a separate untouched test set.
- Orientation canonicalisation uses the ground-truth liver mask, so it is not an inference-time rule, and the reported B and C test
  scores already use ground-truth-based orientation of the test volumes; no inference-time orientation rule was evaluated.
- Volumes 28-47 carry a placeholder 1 mm header; their true spacing is unknown.
- Synthetic-data results: none (see above).

## License and data terms

No license has been chosen yet: the repository owner has to choose one before others may reuse the code. The LiTS data is not
part of this repo and has its own terms; obtain it from the LiTS challenge and follow them.

## Acknowledgements and AI assistance

The thesis work this repo builds on is by Shreya Agarwal; the thesis discloses ChatGPT editing. The dataset analysis,
code refactoring, experiments write-up and report in this repo were produced with assistance from Claude (Anthropic)
and should be reviewed before being relied on.
