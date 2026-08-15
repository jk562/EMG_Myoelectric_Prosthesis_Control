# Test 1 — Baseline: Random Forest vs. CNN

**Improving Myoelectric Prosthesis Control: Machine Learning Prediction of Hand Kinematics from EMG Signals.**

The project's starting point. Establishes the baseline the rest of the project (`Test2`–`Test9`) builds on and compares against.

## What's here

`EMG_Project_Notebook.ipynb` — one self-contained notebook, in order:

1. **Setup** — imports, paths, constants.
2. **EMG simulation** — synthetic signal generation pipeline (no real device data yet).
3. **Visualisations** — signal plots, SNR comparison, feature heatmap.
4. **Random Forest baseline** — classical ML on hand-engineered features.
5. **CNN baseline** — deep learning directly on raw EMG windows.
6. **Results comparison** — RF vs. CNN.

`FIG/` — output figures from the notebook.

## Key results (referenced throughout later stages)

- Random Forest, within-subject: R² = 0.708
- Random Forest, cross-subject (zero-shot): R² = −0.42
- CNN, within-subject: R² = 0.659

The cross-subject collapse seen here (−0.42) is the first appearance of the problem this whole project is ultimately about — see the root [`README.md`](../README.md) and `Test9/` for how it gets diagnosed and addressed.
