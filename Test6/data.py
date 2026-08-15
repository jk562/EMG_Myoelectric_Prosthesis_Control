"""
Load Ninapro DB2, window it, and build datasets.

DB2 .mat fields used:
    emg    : (n_samples, 12)   raw EMG      -> model INPUT
    glove  : (n_samples, 22)   CyberGlove   -> the KINEMATICS we predict (TARGET)
    subject: scalar            subject id   -> used for cross-subject splitting

The task is REGRESSION: given a window of EMG, predict the hand kinematics
(glove vector) at the end of that window. This is objective 3 of the brief.
"""

import glob
import os
import re

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def _subject_id(mat, path):
    if "subject" in mat:
        return int(np.asarray(mat["subject"]).ravel()[0])
    m = re.search(r"[Ss](\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_files(data_dir, n_emg=12):
    """Read every .mat file. Returns list of (emg, glove, subject).

    Skips files whose EMG channel count doesn't match n_emg -- NinaPro DB1
    (10 channels) and DB2 (12 channels) subjects are sometimes mixed in the
    same folder (e.g. DB1 subject 1 uses a different naming convention,
    "S1_A1_E1.mat" vs DB2's "S1_E1_A1.mat"), and concatenating windows of
    different channel counts crashes at np.concatenate with no useful message.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not paths:
        raise FileNotFoundError(
            f"No .mat files in '{data_dir}'. Download Ninapro DB2 first.")
    recordings = []
    for p in paths:
        m = loadmat(p)
        if "emg" not in m or m["emg"].shape[1] != n_emg:
            print(f"  skipping {os.path.basename(p)}: "
                  f"{m['emg'].shape[1] if 'emg' in m else '?'} EMG channels (expected {n_emg})")
            continue
        emg = m["emg"].astype(np.float32)
        glove = m["glove"].astype(np.float32)
        recordings.append((emg, glove, _subject_id(m, p)))
    return recordings


def window_recording(emg, glove, fs, win_ms, stride_ms):
    """Slide windows; target = kinematics at the window's last sample."""
    win = int(fs * win_ms / 1000)
    hop = int(fs * stride_ms / 1000)
    X, Y = [], []
    for s in range(0, len(emg) - win, hop):
        X.append(emg[s:s + win].T)        # (channels, time)
        Y.append(glove[s + win - 1])      # kinematics at window end
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def build_arrays(recordings, cfg, subjects=None):
    """Window a set of recordings (optionally filtered to given subjects)."""
    Xs, Ys = [], []
    for emg, glove, subj in recordings:
        if subjects is not None and subj not in subjects:
            continue
        X, Y = window_recording(emg, glove, cfg.fs, cfg.win_ms, cfg.stride_ms)
        if len(X):
            Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


class Normalizer:
    """Per-channel z-score. Fit on train only, then apply everywhere."""

    def fit(self, X, Y):
        self.x_mean = X.mean((0, 2), keepdims=True)
        self.x_std = X.std((0, 2), keepdims=True) + 1e-8
        self.y_mean = Y.mean(0, keepdims=True)
        self.y_std = Y.std(0, keepdims=True) + 1e-8
        return self

    def x(self, X):
        return (X - self.x_mean) / self.x_std

    def y(self, Y):
        return (Y - self.y_mean) / self.y_std

    def y_inv(self, Yn):
        return Yn * self.y_std + self.y_mean


class EMGKinematicsDataset(Dataset):
    """Returns (emg_window, kinematics_target) pairs for regression."""

    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class EMGOnlyDataset(Dataset):
    """EMG windows only, for self-supervised pretraining (no labels)."""

    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]
