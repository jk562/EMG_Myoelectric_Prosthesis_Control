# EMG → Hand Kinematics

Calibration-first deep learning pipeline that predicts continuous hand kinematics from forearm surface EMG, for prosthetic hand control. MSc final project.

**Final deliverable: [`Test9/`](Test9/)** — one backend, one frontend, trained, calibrated, seeded for reproducibility, and verified end-to-end. Start there to run the project. This root README is a map of the whole repository; see `Test9/README.md`'s content (now merged below) for setup and usage.

## Why calibration-first

Pooled models trained across many subjects do not generalize zero-shot to a new person — EMG varies too much between individuals (electrode placement, subcutaneous fat, muscle-fiber orientation, movement style). Three independent zero-shot cross-subject attempts across this project's development (Random Forest, CNN, DANN-adversarial CNN) all failed, with R² as low as −1.76. Rather than chase an unrealistic zero-shot number, the final system treats a pooled model as a warm start and reports accuracy only after a fast, per-subject calibration step, evaluated on leakage-safe held-out data the calibration step never saw.

## Results (seeded, reproducible — seed=0)

| Dataset | Subject | Calibrated R² (joint-averaged) |
|---|---|---|
| Ninapro DB2 | S13 | **0.732** |
| Ninapro DB2 | S22 | **0.600** |
| RAW TEST (self-collected) | Subject 01 | 0.527 (not yet re-seeded) |

A controlled protocol-matching test shows this same frozen model reaches **R² = 0.840** when re-evaluated at a comparable published SOTA method's window-overlap density — evidence the raw gap against the literature is substantially an evaluation-protocol difference, not a capability gap. Full write-up, figures, and literature comparison: [`Report/Report_final.docx`](Report/Report_final.docx).

## Project structure

- **[`Test9/`](Test9/)** — **the final deliverable.** `backend.py` (train/finetune/predict/pretrain CLI) + `frontend.py` (Streamlit demo). See below for setup/usage.
- **[`Report/`](Report/)** — the full project report (`Report_final.docx`, all figures, frontend screenshots, architecture diagram) and earlier stage-specific reports.
- **[`visual representation/`](visual%20representation/)** — `visualize_pipeline.ipynb`, a self-contained notebook building all 12 figures (raw EMG → pipeline → predictions → hand pose → per-joint accuracy → inter-subject variability) directly from the trained Test9 model.
- **`Test1`–`Test8`** — the nine-stage development history. Each stage is a real, independent experiment (baseline models, SSL pretraining attempts, cross-subject generalization diagnostics, architecture ablations, DANN, dense-training/evaluation-protocol investigations) that led to Test9's final design. Kept for provenance; not required to run the final system.
- **`RAW TEST/`** — the self-collected 8-channel EMG / 15-angle-glove device recordings, the second dataset Test9 supports.
- **`Raw data/`, `Subject Data/`** — source Ninapro DB2 recordings.
- **`DS5_Cross-User/`, `bionic-arm/`** — earlier/adjacent exploratory work.

## Setup

```bash
pip install torch numpy scipy pandas streamlit plotly mne
```

`mne` is only required for RAW TEST's `.fif` files.

## Usage

### CLI (`Test9/backend.py`)

```bash
cd Test9
python backend.py <pretrain|train|predict|finetune> <db2|rawtest> [args...]
```

- **`pretrain`** — masked-reconstruction self-supervised pretraining on unlabeled EMG. Produces `<prefix>-SSL.pt`.
- **`train`** — supervised training, pools all subjects (no held-out-subject split). Produces `<prefix>.pt` — the base, "cold" model. Not the reported deliverable.
- **`finetune`** — per-subject calibration: fine-tunes the base model on 70% of one subject's recording, evaluates on the leakage-safe held-out 30%. Produces `<prefix>_finetuned_<stem>.pt`. **This is the real deliverable.**
- **`predict`** — batched inference over a file's windows → `predicted_kinematics.csv` + printed metrics.

### Frontend (`Test9/frontend.py`)

```bash
cd Test9
streamlit run frontend.py
```

Single unified Streamlit app for both datasets: dataset/checkpoint selection, EMG upload, per-subject calibration with a leakage-aware accuracy warning (auto-detects when a displayed R² would be inflated by evaluating a checkpoint on data it already trained on), predicted-vs-true kinematics (table + time series), accuracy metrics (pooled and joint-averaged R²), and a 3D reach-trajectory visualization (DB2 only).

## Architecture

One dataset-agnostic `Net`, parameterized by channel count / kinematics dimensionality / whether the cosine-similarity branch applies, instead of hard-coded per-dataset modules. Three branches feed one regression head:

1. **1D CNN encoder** — 4 conv blocks over the raw 200 ms EMG window, multi-channel mixing, 256-dim embedding.
2. **Hand-crafted features** — 16 per-channel statistics (RMS, waveform length, log-variance, zero crossings, slope-sign changes, MAV, Willison amplitude, Hjorth mobility/complexity, mean/median power frequency, 5 FFT-band sums), KinEMbed-inspired.
3. **Cross-channel cosine similarity** (DB2 only) — pairwise cosine similarity between every pair of channels' feature vectors, ROCKET-paper-inspired; verified in ablation to give a real accuracy gain.

## Checkpoints (in `Test9/`)

| File | What it is |
|---|---|
| `EMG-KinNet.pt` | Base pooled DB2 model (cold, not the reported number) |
| `EMG-KinNet-SSL.pt` | SSL-pretrained encoder used to initialize the base model |
| `EMG-KinNet_finetuned_S13_E1_A1.pt` | Calibrated on Subject 13 — R² 0.732 |
| `EMG-KinNet_finetuned_S22_E1_A1.pt` | Calibrated on Subject 22 — R² 0.600 |
| `RawTest-KinNet.pt` | Base pooled RAW TEST model |
| `RawTest-KinNet_finetuned_Subject_01.pt` | Calibrated on RAW TEST Subject 01 — R² 0.527 |

Other `*finetuned_tmp*` checkpoints in `Test9/` are user-generated from ad-hoc frontend sessions, not official results.

## Reproducibility

`seed_everything(seed=0)` is called at the start of every training entry point in `backend.py`, seeding `random`, NumPy, PyTorch, and the MPS/CUDA backends. Earlier stages of this project were not seeded, which is why the same architecture and protocol could produce calibrated R² anywhere from 0.585 to 0.790 across different retrains — the numbers reported above are deliberately the reproducible ones, not the highest ever observed. Independently verified deterministic: two isolated reruns of the full pipeline produced bit-identical per-epoch training losses.

## Known limitations

- Cross-subject zero-shot generalization remains negative even with the best-attempted fix (DANN, R² = −0.019) — calibration is required, not optional.
- Thumb (MCP/IP) and wrist abduction/adduction remain the weakest-predicted joints even after SSL pretraining, plausibly due to anatomical/electrode-placement limits.
- The RAW TEST checkpoint has not yet been re-run under the seeding fix.
- Hand-pose / reach-trajectory visualizations use generic anthropometric finger-segment lengths and an unverified channel-to-joint mapping — illustrative geometry, not measured physical values.

### Ninapro DB2 (main benchmark)

1. Register and download **DB2** at [ninapro.hevs.ch](https://ninapro.hevs.ch) (free, academic use).
2. You need the per-subject `.mat` exercise files (e.g. `S1_E1_A1.mat`, `S2_E1_A1.mat`, …) — 12-channel EMG @ 2 kHz + 22-channel CyberGlove kinematics.
3. Place them in **`Test6/data/`**. `Test7/data`, `Test8/data`, and `Test9/data` are all symlinks to this one folder, so downloading once covers every stage.
4. This project used 19 subjects (S13 and S22 are the two examined in detail throughout the report/notebook) — any subset of DB2 subjects will work for a smoke test; the full 19 for reproducing the reported ablations.


