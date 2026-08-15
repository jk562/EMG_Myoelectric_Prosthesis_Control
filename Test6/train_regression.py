"""
Train the EMG -> kinematics regressor.

  python train_regression.py                    # from scratch (baseline)
  python train_regression.py --pretrained encoder_ssl.pt   # SSL-initialised

Running both and comparing them is the core experiment for objective 4: does
SSL pretraining improve kinematics prediction? Saves the trained model and the
normalizer stats so evaluate.py can reproduce the exact preprocessing.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data import (load_files, build_arrays, EMGKinematicsDataset, Normalizer)
from models import RegressionModel
from utils import set_seed, get_device, rmse, r2, mean_corr


def prepare_data():
    recordings = load_files(cfg.data_dir, n_emg=cfg.n_emg)
    all_subjects = {s for (_, _, s) in recordings}
    train_subjects = all_subjects - set(cfg.test_subjects)

    Xtr, Ytr = build_arrays(recordings, cfg, subjects=train_subjects)
    Xte, Yte = build_arrays(recordings, cfg, subjects=set(cfg.test_subjects))

    norm = Normalizer().fit(Xtr, Ytr)          # fit on TRAIN only
    Xtr, Ytr = norm.x(Xtr).astype(np.float32), norm.y(Ytr).astype(np.float32)
    Xte, Yte = norm.x(Xte).astype(np.float32), norm.y(Yte).astype(np.float32)
    return (Xtr, Ytr), (Xte, Yte), norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default=None,
                    help="path to SSL encoder weights; omit for from-scratch")
    ap.add_argument("--out", default="regressor.pt")
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    (Xtr, Ytr), (Xte, Yte), norm = prepare_data()
    tl = DataLoader(EMGKinematicsDataset(Xtr, Ytr), batch_size=cfg.batch_size,
                    shuffle=True)
    te = DataLoader(EMGKinematicsDataset(Xte, Yte), batch_size=cfg.batch_size)

    model = RegressionModel(cfg.n_emg, cfg.n_kin, cfg.feat_dim).to(device)
    if args.pretrained:
        model.encoder.load_state_dict(torch.load(args.pretrained,
                                                  map_location=device))
        print(f"loaded SSL encoder from {args.pretrained}")
    else:
        print("training from scratch (no SSL)")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.reg_lr)
    lossf = nn.MSELoss()

    for ep in range(cfg.reg_epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # quick val each epoch (in normalised space)
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in te:
                preds.append(model(xb.to(device)).cpu().numpy())
                gts.append(yb.numpy())
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        print(f"[reg] epoch {ep + 1:3d}/{cfg.reg_epochs}  "
              f"RMSE {rmse(gts, preds):.4f}  R2 {r2(gts, preds):.3f}  "
              f"corr {mean_corr(gts, preds):.3f}")

    torch.save({"state_dict": model.state_dict(),
                "y_mean": norm.y_mean, "y_std": norm.y_std,
                "x_mean": norm.x_mean, "x_std": norm.x_std}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
