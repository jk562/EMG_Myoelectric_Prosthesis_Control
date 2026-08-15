# Test 8 — SOTA Comparison and Cross-Subject Investigation

Investigates the apparent gap between `Test7`'s calibrated results and published SOTA (sMAPEN, 2024), and makes further genuine attempts at cross-subject generalization. Same core pipeline as `Test7`; this stage adds diagnostic and ablation scripts rather than changing the base architecture.

## What's here

- `hop_sweep.py` — the key finding of this stage: evaluates the same frozen `Test7` model at different analysis-window overlap densities (hop sizes). Shows the reported R² is heavily dependent on evaluation-window overlap — at sMAPEN's own 99.5%-overlap convention, this project's model scores **0.840**, exceeding sMAPEN's published 0.8163. Most of the apparent "SOTA gap" is a metric artifact, not a capability gap.
- `subject_count_ablation.py` — tests whether DB2 subject-pool size (19 here vs. sMAPEN's 40) affects the comparison. Finds smaller pools score *higher* — an effect that, if anything, favours sMAPEN's larger pool, not this project.
- `dense_train_eval.py` — tests whether *training* density (not just evaluation density) matters. Finds a real additional gain at partial density, leaving the fully-matched number unresolved (likely above 0.840).
- `dense_train_streaming.py` — an attempt at literal hop=1 training via a memory-safe streaming pipeline. **Not trustworthy** — three sequential bugs (index misalignment, validation-set resampling causing spurious early stop, and a "fixed" training sample that didn't preserve true hop=1 density) meant the literal hop=1 number was never reliably measured. Kept for transparency, not cited as a result.
- `cross_subject_dann.py` — a domain-adversarial (DANN, Ganin & Lempitsky 2015) attempt at zero-shot cross-subject generalization. Best result: R² = −0.019 — the best cross-subject attempt in the whole project, but still negative.
- `benchmark_repetition_split.py` — repetition-held-out evaluation split (carried over from `Test7`).
- `quick_emg_to_kinematics.py`, `rawtest_emg_to_kinematics.py`, `EMG_frontend.py` — same pipeline/frontend as `Test7`, used as the base for the above investigations.

## This stage's role

The evaluation-protocol findings here (`hop_sweep.py` especially) are the basis for this project's honest literature-comparison framing, used throughout `Test9`'s report: "comparable to published SOTA once evaluation protocol is matched," not "behind." See the root [`README.md`](../README.md) and `Report/Report_final.docx` for the full writeup.
