"""
Models for EMG -> kinematics.

Shared encoder (1D CNN over raw EMG). Two heads:
  - MaskedAutoencoder : SSL pretraining (reconstruct masked EMG, no labels)
  - RegressionModel   : supervised EMG -> 22 kinematic values

Pretrain the encoder with the autoencoder, then load those weights into the
regression model. Training the regression model WITHOUT loading them is your
"from scratch" baseline for the comparison in objective 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGEncoder(nn.Module):
    """(B, n_emg, T) -> feature map (B, feat_dim, T/4)."""

    def __init__(self, n_emg=12, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_emg, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, feat_dim, 3, padding=1), nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.net(x)

    def pooled(self, x):
        return F.adaptive_avg_pool1d(self.net(x), 1).flatten(1)  # (B, feat_dim)


class _Decoder(nn.Module):
    """Feature map -> reconstructed EMG (B, n_emg, T)."""

    def __init__(self, feat_dim, n_emg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, n_emg, 3, padding=1),
        )

    def forward(self, feat, out_len):
        y = self.net(feat)
        return F.interpolate(y, size=out_len, mode="linear",
                             align_corners=False)


class MaskedAutoencoder(nn.Module):
    def __init__(self, n_emg=12, feat_dim=128):
        super().__init__()
        self.encoder = EMGEncoder(n_emg, feat_dim)
        self.decoder = _Decoder(feat_dim, n_emg)

    def forward(self, x_masked, out_len):
        return self.decoder(self.encoder(x_masked), out_len)


class RegressionModel(nn.Module):
    def __init__(self, n_emg=12, n_kin=22, feat_dim=128):
        super().__init__()
        self.encoder = EMGEncoder(n_emg, feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, n_kin),
        )

    def forward(self, x):
        return self.head(self.encoder.pooled(x))
