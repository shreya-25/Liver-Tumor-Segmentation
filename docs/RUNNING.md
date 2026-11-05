# Liver tumor segmentation: how to run this project

This folder contains (1) a 2D AIM-UNet liver/tumor segmentation pipeline on the LiTS CT dataset, with a
fixed-window, orientation-corrected data cache, the original volume-level (= patient-level) train/held-out split and a per-patient evaluation script, (2) a corrected
3D multi-GAN synthetic-tumor pipeline, and (3) a research report with an in-depth analysis of the dataset.
It does **not** contain the CT data (about 50 GB). You need to download it yourself (step 2).

Read first: `docs/Liver_Tumor_Research_Report.pdf` (dataset analysis, pipeline audit, findings, caveats).

## 1. Requirements
- Python 3.12 (tested). macOS or Linux; on Windows use the commands in `setup.sh` and run the Python scripts directly.
- A GPU is strongly recommended: Apple Silicon (`mps`) or NVIDIA (`cuda`) is detected automatically. On CPU only
  the smoke tests are practical. Measured on an Apple M4 Pro (48 GB): the segmentation model trains at about
  7 slices/s (not measured on CUDA).
- Disk: about 50 GB for the raw data, plus 7.7 GB for the slice cache. RAM: 16 GB or more (the training slices are held in memory).

```bash
bash setup.sh          # creates .venv and installs requirements.txt, prints the training device
```

## 2. Get the data
Download the LiTS training set (131 CT volumes with liver and tumor labels) from the LiTS challenge
(benchmark paper: Bilic et al., https://arxiv.org/abs/1901.04056) and place the files, **uncompressed**, as

```
data/raw/volume-0.nii ... volume-130.nii
data/raw/segmentation-0.nii ... segmentation-130.nii
```
If your files are `.nii.gz`, run `gunzip data/raw/*.gz` first: the statistics in `analysis/output/` are keyed
by the `.nii` file names, and the split is taken over the sorted volume names, so the numbering and names must match.
Check it with `bash run_pipeline.sh check`.

## 3. Run the segmentation pipeline
```bash
bash run_pipeline.sh check   # verifies the data and shows the training device (seconds)
bash run_pipeline.sh quick   # smoke run: builds the cache (about 1-2 min), trains 1 epoch on 4 train volumes (about 2 min incl. validation; measured on an M4 Pro)
bash run_pipeline.sh full    # all 104 train volumes, 10 epochs, then evaluation on the 27 held-out volumes
```
What happens: `prepare_data_v2.py` builds `data/cache_v2` (fixed -100..250 HU window, orientation
canonicalised, uint8 256x256 axial slices), `train_fast.py` trains on the original split (sklearn
`train_test_split` over the sorted volume names, `test_size=0.2`, `random_state=42`: 104 train / 27 held-out
volumes, each volume is one patient), `eval_fast.py` reports per-patient 3D Dice, lesion-wise recall and false
positives on the held-out volumes (`results/full/eval/eval.md`). **The held-out part serves as both the
validation set (the best epoch is picked on it) and the test set, as in the original script**, so the evaluation
scores are validation-selected and there is no untouched test set; read them as optimistic.
`quick` uses a deterministic stratified subset (`LITS_SUBSET_N`, balanced over slice thickness x tumor/no-tumor)
of the train part and of the held-out part for validation; it is a convenience, not a different split.
Expect a `full` epoch to take roughly 40-45 minutes on an M4 Pro (about 18,000 sampled slices per epoch with
`LITS_EMPTY_KEEP=0.1` at 7 slices/s; this is an estimate, not a measurement), so budget accordingly or lower `LITS_EPOCHS`.

## 4. Run the GAN pipeline (corrected version)
```bash
LITS_TEST_SKIP_E2E=1 .venv/bin/python -I "Image Synthesis/test_syn_pipeline_fixed.py"   # unit tests, seconds
.venv/bin/python -I "Image Synthesis/test_syn_pipeline_fixed.py"                         # plus end-to-end plumbing test (~3 min)
LITS_GAN_STAGE=all LITS_GAN_EPOCHS=25 .venv/bin/python -I "Image Synthesis/Syn_Img_Pipeline_fixed.py"
```
Stages (`LITS_GAN_STAGE`): `prepare`, `train`, `generate`, `insert`, or `all`. Output goes to `results/gan_fixed/`
(augmented volumes in `aug/`, saved in HU, never written into `data/raw`). The GAN is a CPU-by-default
pipeline (`LITS_DEVICE=cpu|mps|cuda`); on an M4 Pro the CPU was faster than mps.

## 5. What is where
| Path | What |
|---|---|
| `Image Segmentation/prepare_data_v2.py` | raw NIfTI -> uint8 slice cache (fixed window, canonical orientation, metadata) |
| `Image Segmentation/train_fast.py` | optimized trainer (same model, loss and split as the original); writes `split_used.json` |
| `Image Segmentation/eval_fast.py` | per-patient evaluation on the held-out volumes, with a leakage guard against the checkpoint's train list |
| `Image Synthesis/Syn_Img_Pipeline_fixed.py` + `test_syn_pipeline_fixed.py` | corrected GAN pipeline and its tests |
| `analysis/output/per_volume.csv` | per-volume statistics of the 131 LiTS volumes (read by the subset picker and the evaluator) |
| `analysis/run_baseline.sh` | re-run design of the A/B/C data-pipeline comparison on the final code (hours; writes `results/baseline_original_split/`) |
| `docs/baseline_results/` | the reported A/B/C baseline, small files only (logs, `eval/`, `split_used.json`, no checkpoints) plus `archive/` with the as-run code and split manifest; produced with older code and a different split, see `docs/baseline_results/archive/README.md` |
| `LiTS_Pipeline_Showcase.ipynb` | notebook that reads saved results and shows the baseline (needs `requirements-notebook.txt` on top of `requirements.txt`) |
| `docs/Liver_Tumor_Research_Report.pdf` | dataset analysis, pipeline audit, findings and caveats |

## 6. Settings you may want (environment variables)
`LITS_EPOCHS`, `LITS_SEED`, `LITS_FLIP_AUG=0` (no h/v flips, keeps the canonical orientation),
`LITS_EMPTY_KEEP=0.1` (keep 10% of the 67% of slices that contain no liver; cuts epoch time about 2.5x),
`LITS_WIN_LO/LITS_WIN_HI` (HU window), `LITS_SUBSET_N` / `LITS_SUBSET_VAL_N` (stratified subset of the train /
held-out part), `LITS_MAX_VOLUMES` (training smoke test only: first N volumes before the split), `LITS_EVAL_MAX_VOLUMES`
(evaluation smoke test: only the first N of the 27 held-out volumes; evaluation ignores `LITS_MAX_VOLUMES`), `LITS_DATA_DIR`, `LITS_CACHE_DIR`, `LITS_RESULTS_DIR`. Each script documents its own
variables at the top of the file.

## 7. Honest status (read before trusting results)
- The corrected GAN passes its tests, but sample realism is **poor** after the short runs done here, and its mask
  head does not generalise on that little data. Treat it as a correct, tested starting point, not a trained
  synthetic-data generator. No segmenter has been trained on its output.
- The held-out part (27 volumes) is both validation and test set (original-script behaviour). The best epoch is chosen
  on it, so scores on it are optimistic; a clean estimate needs a separate untouched test set.
- The reported baseline in `docs/baseline_results` (3 arms: v1 cache, v2 cache with flips, v2 cache without flips; 6 epochs,
  16 train volumes, one seed each) is tiny and undertrained: absolute tumor Dice is still low (0.06 / 0.21 / 0.28 mean
  over the 23 tumor-bearing of 26 test patients). It was produced with an earlier stratified 5-fold split and older
  code that were later removed from the main code (archived in `docs/baseline_results/archive/`), so it is not directly
  comparable to what the current scripts print. The window/orientation fix changed together in A to B, so their
  effects cannot be separated; B to C isolates flip augmentation only. Single seeds: treat differences as indications.
- Orientation canonicalisation uses the ground-truth liver mask, so it is not an inference-time rule. The B and C test volumes of the reported baseline were also oriented with their ground-truth liver masks, so those scores assume the correct orientation is known; an inference-time orientation rule has not been evaluated.
- Volumes 28-47 carry a placeholder 1.0 x 1.0 x 1.0 mm header; their true spacing is unknown (`spacing_reliable` flag).
- All scripts use paths relative to this folder, so it can live anywhere.
