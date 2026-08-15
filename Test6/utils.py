"""
Shared helpers used across the scripts: reproducibility, device selection, the
three evaluation metrics referenced in the README, and the SNR noise sweep.

This file was missing from the original drop (every other script imports from
it) -- reconstructed to match exactly how each function is called elsewhere:
  set_seed(cfg.seed)
  get_device(cfg.device)
  rmse(y_true, y_pred) / r2(...) / mean_corr(...)          -- all (N, n_kin) arrays
  add_noise_snr(X, snr_db)                                  -- X is (N, C, T)
"""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested):
    """Falls back automatically: requested device -> mps (Apple Silicon) -> cpu."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    """Mean R^2 across kinematic channels (matches the per-joint-then-average
    convention used throughout this project's other notebooks)."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    ss_tot = np.where(ss_tot < 1e-12, 1e-12, ss_tot)
    return float(np.mean(1.0 - ss_res / ss_tot))


def mean_corr(y_true, y_pred):
    """Mean Pearson correlation across kinematic channels."""
    n_kin = y_true.shape[1]
    corrs = []
    for j in range(n_kin):
        yt, yp = y_true[:, j], y_pred[:, j]
        if yt.std() < 1e-8 or yp.std() < 1e-8:
            continue   # constant channel -- correlation undefined, skip rather than NaN
        corrs.append(np.corrcoef(yt, yp)[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def add_noise_snr(X, snr_db, seed=0):
    """Adds Gaussian noise to EMG windows (N, C, T) at the given SNR in dB."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), X.shape).astype(np.float32)
    return X + noise
