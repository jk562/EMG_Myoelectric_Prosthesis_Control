# Test 9 — Final Consolidated Deliverable

The project's final system: exactly two files, `backend.py` and `frontend.py`, consolidating `Test7`/`Test8`'s two separate pipelines (DB2 + RAW TEST) into one dataset-parameterized backend and one unified Streamlit frontend. Uses the proven `Test7`/`Test8` architecture as-is — deliberately excludes the three `Test8` architecture variants that were tried and reverted (attention pooling, a Transformer block, per-joint loss weighting), none of which beat the plain CNN baseline.

See the root [`README.md`](../README.md) for the full project overview, results table, architecture summary, and literature comparison. This file covers just what's specific to running this folder.

## Setup

```bash
pip install torch numpy scipy pandas streamlit plotly mne
```

## Usage

```bash
python backend.py <pretrain|train|predict|finetune> <db2|rawtest> [args...]
streamlit run frontend.py
```

## What's new here vs. Test7/Test8

- **One `Net` class** parameterized by `max_channels`/`n_kin`/`use_sim` instead of separate hard-coded modules per dataset.
- **`seed_everything(seed=0)`** — added at the start of every training entry point (`pretrain_ssl`, `train`, `finetune`). Nothing before this stage was seeded, which is why identical architecture/protocol produced calibrated R² anywhere from 0.585 to 0.790 across retrains in `Test7`/`Test8`. Independently verified deterministic (two isolated reruns → bit-identical per-epoch losses).
- **SSL pretraining kept, with an honest ablation.** `pretrain_ssl` (unused in `Test7`/`Test8`'s deployed system) was tested here specifically to target this project's three weakest joints. Real, verified gain on both tested subjects, though far more modest on S22 than S13 — kept, not oversold.
- **Deployed checkpoints are the seeded numbers**, not the best number ever observed. S13: 0.732 (vs. an earlier unseeded 0.790). This was a deliberate choice, made explicitly to prioritize reproducibility over a marginally higher but unreproducible number.

## Checkpoints in this folder

| File | What it is |
|---|---|
| `EMG-KinNet.pt` | Base pooled DB2 model (cold, not the reported number) |
| `EMG-KinNet-SSL.pt` | SSL-pretrained encoder used to initialize the base model |
| `EMG-KinNet_finetuned_S13_E1_A1.pt` | Calibrated on Subject 13 — R² 0.732 |
| `EMG-KinNet_finetuned_S22_E1_A1.pt` | Calibrated on Subject 22 — R² 0.600 |
| `RawTest-KinNet.pt` | Base pooled RAW TEST model |
| `RawTest-KinNet_finetuned_Subject_01.pt` | Calibrated on RAW TEST Subject 01 — R² 0.527 (not yet re-seeded) |

Other `*finetuned_tmp*` checkpoints are user-generated from ad-hoc frontend sessions, not official results.

## Reports and figures

`Test9_Final_Report.docx` (in this folder) and `../Report/Report_final.docx` (the fuller version, with all 12 figures, the architecture flowchart, and frontend screenshots) both cover this stage's full results and literature comparison.
