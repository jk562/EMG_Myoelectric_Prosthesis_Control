"""
Optional supporting figure for the report: predicted vs. true kinematics for a
few joints over time. This is the ONLY visualisation the brief needs -- a plot
showing your algorithm's predicted hand motion against ground truth.

  python visualize.py --model regressor.pt

(If you also want to pose a 3D MuJoCo hand from these predictions, feed `pred`
row-by-row into the hand's qpos with mj_forward -- but that is a demo extra,
not part of the graded algorithm/evaluation.)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import cfg
from data import load_files, build_arrays
from evaluate import load_model, predict
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="regressor.pt")
    ap.add_argument("--joints", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n", type=int, default=500, help="timesteps to plot")
    args = ap.parse_args()

    device = get_device(cfg.device)
    model, stats = load_model(args.model, device)

    recordings = load_files(cfg.data_dir, n_emg=cfg.n_emg)
    Xraw, Y = build_arrays(recordings, cfg, subjects=set(cfg.test_subjects))
    Xn = ((Xraw - stats["x_mean"]) / stats["x_std"]).astype(np.float32)

    pred_n = predict(model, Xn[:args.n], device)
    pred = pred_n * stats["y_std"] + stats["y_mean"]   # back to real units
    true = Y[:args.n]

    fig, axes = plt.subplots(len(args.joints), 1, figsize=(9, 2 * len(args.joints)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, args.joints):
        ax.plot(true[:, j], label="true", lw=1.2)
        ax.plot(pred[:, j], label="predicted", lw=1.2, alpha=0.8)
        ax.set_ylabel(f"joint {j}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("window index (time)")
    fig.suptitle("Predicted vs. true hand kinematics (held-out subject)")
    fig.tight_layout()
    fig.savefig("kinematics_prediction.png", dpi=130)
    print("saved -> kinematics_prediction.png")


if __name__ == "__main__":
    main()
