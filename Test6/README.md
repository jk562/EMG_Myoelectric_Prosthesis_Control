# Test 6 — SSL Evaluation on Real Ninapro DB2

Predicts continuous hand kinematics (CyberGlove joint values) from forearm EMG, using masked-reconstruction SSL pretraining — the project's first evaluation on **real** Ninapro DB2 data (`Test1`–`Test5` used synthetic or single-subject data). This is the **algorithm + evaluation** stage — the graded core of the project brief (predict hand kinematics from muscle activity, and evaluate it). It is not a physics simulator; `visualize.py` gives one supporting figure.

## Install

```bash
pip install torch numpy scipy matplotlib
```

## Get the data

Register at https://ninapro.hevs.ch and download **DB2**. Put the per-subject `.mat` files (e.g. `S1_E1_A1.mat`, `S2_E1_A1.mat`, …) in `data/` next to these scripts. DB2 gives 12 EMG channels @ 2 kHz and a 22-channel glove. `data/` here is the actual source directory later stages (`Test7`–`Test9`) symlink to.

## Run order

```bash
# 1. self-supervised pretraining on unlabeled EMG
python pretrain_ssl.py                                  # -> encoder_ssl.pt

# 2a. baseline: train regressor FROM SCRATCH
python train_regression.py --out scratch.pt

# 2b. SSL: train regressor from the pretrained encoder
python train_regression.py --pretrained encoder_ssl.pt --out ssl.pt

# 3. evaluate BOTH on held-out subjects + noise sweep
python evaluate.py --model scratch.pt
python evaluate.py --model ssl.pt

# 4. optional figure for the report
python visualize.py --model ssl.pt
```

## What each metric means

- **RMSE** — average joint-angle error (lower better).
- **R²** — variance explained across joints (1.0 = perfect).
- **corr** — mean Pearson correlation between predicted and true joint tracks.
- **SNR sweep** — how those degrade as noise rises; the robustness result, and where SSL is expected to help.

## The experiment that answers the brief

Compare `scratch.pt` vs. `ssl.pt` on the held-out subjects. `test_subjects` in `config.py` is excluded from all training, so this measures **cross-subject generalisation** — the hard, meaningful case for real prosthesis control. The gap between the two models is the main result of this stage, and the cross-subject collapse it demonstrates is what motivates the calibration-first redesign from `Test7` onward — see the root [`README.md`](../README.md).

## Honest notes

- You must supply the Ninapro data; nothing here ships with it.
- Real training needs a GPU and will take a while; start with fewer epochs to smoke-test the pipeline end to end, then scale up.
- The target is the glove signal as-is. If your DB version needs calibration or you want specific named joints, map/scale `glove` columns in `data.py`.
