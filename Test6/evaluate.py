"""
Evaluate a trained regressor on the held-out (cross-subject) test set and
sweep noise robustness. This is objective 4.

  python evaluate.py --model regressor.pt

Run it on both your from-scratch and SSL models and put the two tables
side by side in your report.
"""

import argparse

import numpy as np
import torch

from config import cfg
from data import load_files, build_arrays, Normalizer
from models import RegressionModel
from utils import get_device, rmse, r2, mean_corr, add_noise_snr, set_seed


def load_model(path, device):
    # weights_only=False: PyTorch 2.6+ defaults to True, which rejects the numpy
    # normalisation stats (y_mean/y_std/x_mean/x_std) saved alongside the model
    # weights in train_regression.py. Safe here -- these are checkpoints this
    # same pipeline just created, not files from an untrusted source.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = RegressionModel(cfg.n_emg, cfg.n_kin, cfg.feat_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    stats = {k: ckpt[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    return model, stats


def predict(model, X, device):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), cfg.batch_size):
            xb = torch.from_numpy(X[i:i + cfg.batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="regressor.pt")
    args = ap.parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    model, stats = load_model(args.model, device)

    recordings = load_files(cfg.data_dir, n_emg=cfg.n_emg)
    Xte_raw, Yte = build_arrays(recordings, cfg, subjects=set(cfg.test_subjects))

    # apply the SAME normalisation the model was trained with
    Xte = ((Xte_raw - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    Yte_n = (Yte - stats["y_mean"]) / stats["y_std"]

    # ---- clean performance ----
    pred = predict(model, Xte, device)
    print("=== clean (held-out subjects) ===")
    print(f"RMSE {rmse(Yte_n, pred):.4f}   "
          f"R2 {r2(Yte_n, pred):.3f}   corr {mean_corr(Yte_n, pred):.3f}")

    # ---- noise-robustness sweep ----
    print("\n=== SNR robustness ===")
    print(f"{'SNR(dB)':>8} {'RMSE':>8} {'R2':>8} {'corr':>8}")
    for snr in (20, 15, 10, 5, 0):
        Xn = add_noise_snr(Xte_raw, snr)
        Xn = ((Xn - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
        p = predict(model, Xn, device)
        print(f"{snr:>8} {rmse(Yte_n, p):>8.4f} {r2(Yte_n, p):>8.3f} "
              f"{mean_corr(Yte_n, p):>8.3f}")


if __name__ == "__main__":
    main()
