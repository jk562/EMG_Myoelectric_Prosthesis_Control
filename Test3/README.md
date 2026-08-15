# Test 3 — Streamlined Contrastive-SSL Pipeline

**Part 3 — Simulation, Real EMG Signal, Contrastive-SSL Pretraining, and Kinematics Output.**

`Test1` and `Test2` explored a Random Forest baseline, a from-scratch CNN, and two competing SSL pretraining methods (contrastive and masked-reconstruction). Having established that contrastive (SimCLR-style) pretraining gives the best all-round encoder, this stage is the streamlined, self-contained pipeline built on that choice — from synthetic signal simulation through to a working kinematic prediction driving a virtual prosthetic hand.

## What's here

`EMG_Project3.ipynb` — the end-to-end streamlined pipeline. Key outputs:

- `contrastive_ssl_model.pt` — the trained, saved contrastive-SSL encoder + regressor (unlike Test2, this checkpoint is actually saved to disk).
- `movement_catalog.json` — the movement/task label set used from here through `Test4`/`Test5`.
- `fig1_pipeline_stages.png`, `fig3_real_emg_channels.png` — pipeline and signal figures.
- `fig2_snr_comparison.png`, `fig4_contrastive_loss.png` — training/robustness figures.
- `fig5_kinematics_per_joint.png`, `fig6_kinematics_timeseries.png` — prediction quality.
- `fig6_prosthetic_hand_demo.gif`, `fig6b_closing_fingers_test.png`, `fig6c_relaxed_vs_closing.png`, `fig7_prosthetic_hand_demo.gif` — qualitative hand-motion demos, including a specific "closing all fingers" test case.

## Where this leads

`contrastive_ssl_model.pt` and `movement_catalog.json` are the checkpoint/label-set this project's simulation/visualization stages (`Test4`, `Test5`) are built around, before the project moved to real Ninapro DB2 evaluation from `Test6` onward.
