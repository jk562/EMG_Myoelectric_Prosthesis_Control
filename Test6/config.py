"""Central configuration. Edit paths/params here, not inside the scripts."""

from dataclasses import dataclass


@dataclass
class Config:
    # ---- data ----
    data_dir: str = "data"      # folder containing Ninapro DB2 .mat files
    fs: int = 2000              # DB2 sampling rate (Hz)
    n_emg: int = 12             # DB2 EMG channels
    n_kin: int = 22             # CyberGlove kinematic channels (the targets)
    win_ms: int = 200           # analysis window length
    stride_ms: int = 50         # hop between windows

    # ---- model ----
    feat_dim: int = 128

    # ---- SSL pretraining (masked reconstruction) ----
    mask_ratio: float = 0.4
    ssl_epochs: int = 40
    ssl_lr: float = 1e-3

    # ---- supervised regression ----
    reg_epochs: int = 40
    reg_lr: float = 1e-3
    batch_size: int = 128

    # ---- cross-subject split ----
    # subjects whose files are held out entirely for testing generalisation
    # (1, 2) in the original default aren't in this dataset -- and subject 1 in
    # particular is NinaPro DB1 (10 EMG channels), not DB2, so it would have
    # been silently dropped by load_files's channel filter anyway. Using
    # subjects actually present in data/: train on 10/11/12/16, test on 13/15.
    test_subjects: tuple = (13, 15)

    seed: int = 0
    device: str = "cuda"        # code falls back to cpu automatically


cfg = Config()
