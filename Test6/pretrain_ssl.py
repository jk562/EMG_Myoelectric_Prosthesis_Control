"""
Self-supervised pretraining by masked reconstruction.

Randomly zero out a fraction of each EMG window's timesteps, then train the
encoder+decoder to reconstruct the masked parts. No kinematics labels are used,
so this runs on ALL your EMG (every subject). Saves the encoder for fine-tuning.

Run:  python pretrain_ssl.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from data import load_files, build_arrays, EMGOnlyDataset, Normalizer
from models import MaskedAutoencoder
from utils import set_seed, get_device


def random_mask(x, mask_ratio):
    """x: (B, C, T). Returns masked input and a keep-mask (1=kept, 0=masked)."""
    B, C, T = x.shape
    keep = (torch.rand(B, 1, T, device=x.device) > mask_ratio).float()
    return x * keep, keep


def main():
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    recordings = load_files(cfg.data_dir, n_emg=cfg.n_emg)
    # pretrain on training subjects only (never touch held-out test subjects)
    train_subjects = [s for (_, _, s) in recordings
                      if s not in cfg.test_subjects]
    X, _ = build_arrays(recordings, cfg, subjects=set(train_subjects))

    norm = Normalizer().fit(X, np.zeros((1, cfg.n_kin), np.float32))
    X = norm.x(X).astype(np.float32)
    np.savez("norm_stats.npz", x_mean=norm.x_mean, x_std=norm.x_std)

    loader = DataLoader(EMGOnlyDataset(X), batch_size=cfg.batch_size,
                        shuffle=True, drop_last=True)

    model = MaskedAutoencoder(cfg.n_emg, cfg.feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ssl_lr)

    for ep in range(cfg.ssl_epochs):
        total = 0.0
        for xb in loader:
            xb = xb.to(device)
            x_masked, keep = random_mask(xb, cfg.mask_ratio)
            recon = model(x_masked, out_len=xb.shape[2])
            masked = 1.0 - keep
            # reconstruction loss on masked positions only
            loss = (F.mse_loss(recon, xb, reduction="none") * masked).sum() \
                / (masked.sum() * xb.shape[1] + 1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[ssl] epoch {ep + 1:3d}/{cfg.ssl_epochs}  "
              f"loss {total / len(loader):.5f}")

    torch.save(model.encoder.state_dict(), "encoder_ssl.pt")
    print("saved -> encoder_ssl.pt")


if __name__ == "__main__":
    main()
