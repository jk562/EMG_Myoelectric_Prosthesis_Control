# Test 7 — Calibration-First Pipeline (DB2 + RAW TEST) + Frontend

The design turning point of the project. In response to `Test6`'s cross-subject collapse, this stage introduces the **calibration-first** architecture and workflow that `Test8` and `Test9` build on: a pooled base model as a warm start, plus a fast per-subject calibration step whose accuracy is measured only on leakage-safe held-out data.

## What's here

- `quick_emg_to_kinematics.py` — the DB2 pipeline: CNN encoder + hand-crafted features + cross-channel cosine-similarity branch + regression head, with `train`/`finetune`/`predict` CLI modes.
- `rawtest_emg_to_kinematics.py` — the equivalent pipeline for the self-collected RAW TEST device (`.fif` files via MNE).
- `EMG_frontend.py` — a Streamlit app covering both datasets: dataset selector, calibration panel (with a **leakage-aware accuracy warning** — auto-detects and flags when a displayed R² would be inflated by evaluating a checkpoint on data it already trained on), predicted-vs-true visualization, and a 3D reach-trajectory plot (DB2 only).
- `benchmark_repetition_split.py` — an alternative, repetition-held-out evaluation split, for comparing against literature that evaluates that way.
- Checkpoints: `EMG-KinNet*.pt` (DB2 base/SSL/finetuned) and `RawTest-KinNet*.pt` (RAW TEST base/SSL/finetuned).
- `Hand_Joint_Motion_Diagram.png` — reference diagram for the frontend's anatomy section.
- `data` — symlink to `Test6/data`.

## This stage's role

`quick_emg_to_kinematics.py` + `rawtest_emg_to_kinematics.py` + `EMG_frontend.py` are two separate pipelines sharing the same design philosophy. `Test9` later consolidates both into a single, dataset-parameterized `backend.py` + `frontend.py` — see the root [`README.md`](../README.md) for the final, unified version and its reproducible results.

Best calibrated result from this stage (unseeded): S13 R² = 0.790 — later superseded by `Test9`'s seeded, reproducible 0.732 (see root README's reproducibility note for why the seeded number is lower but preferred).
