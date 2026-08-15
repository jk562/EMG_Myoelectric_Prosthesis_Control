"""
Repetition-held-out benchmark for quick_emg_to_kinematics.py's Net --
matches the evaluation protocol used by sMAPEN (Springer/JNER 2024,
R^2 = 0.8163 on Ninapro DB2) and similar published continuous-kinematics
papers: ONE model trained across ALL subjects at once, with the train/test
split done by REPETITION, not by subject or by calibration -- reps 1-4
train, reps 5-6 test, using each file's `rerepetition` field (Atzori et
al.'s corrected repetition boundaries; falls back to `repetition` if a
file doesn't have it). The SAME subjects appear in both train and test.

This is a fundamentally different, easier question than finetune()'s
per-subject calibration split (a model that has NEVER seen this subject
at all, evaluated on a leakage-safe held-out slice of one recording) --
this script exists purely so we can compare against published numbers on
their own terms, not to replace the calibration-first workflow that's
still this project's real deliverable (see quick_emg_to_kinematics.py's
own docstring). Reuses that script's Net/features/constants directly so
architecture and hyperparameters match the rest of the project exactly --
only the split methodology differs.

  python benchmark_repetition_split.py [data_dir]
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

import quick_emg_to_kinematics as qk

TRAIN_REPS = {1, 2, 3, 4}
TEST_REPS = {5, 6}


def windows_with_repetition(path):
    """Same windowing as qk.windows_from(), plus the majority `rerepetition`
    (or `repetition`) label for each window -- read directly from the .mat
    file since qk.load_emg_file() doesn't surface this field. Returns None
    if the file has no usable ground truth or no repetition field at all."""
    m = loadmat(path)
    rep_key = "rerepetition" if "rerepetition" in m else ("repetition" if "repetition" in m else None)
    if rep_key is None:
        return None
    reps = m[rep_key].astype(int).ravel()

    X, Y, M, warns = qk.windows_from(path)
    if Y is None or len(X) == 0:
        return None
    starts = list(range(0, len(reps) - qk.WIN, qk.HOP))[:len(X)]
    rep_per_window = np.asarray([np.bincount(reps[s:s + qk.WIN]).argmax() for s in starts])
    return X, Y, M, rep_per_window


def main(data_dir="data"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:qk.MAX_FILES]
    if not files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    # The full dataset's raw EMG windows alone run ~7.5GB (float32, all 57
    # files) on a machine with only ~17GB total RAM -- concatenating that
    # much before capping (the way train() does, relying on the rest of
    # its memory budget being lighter) risks an OOM kill here. So windows
    # are capped PER FILE, separately for train-reps and test-reps, BEFORE
    # concatenation or feature computation -- the same per-file-cap
    # discipline already established for the RAW TEST pipeline (30-min
    # recordings), just applied here too since this script's per-file
    # handcrafted_features()/channel_cosine_similarity() calls and the
    # cross-file concatenation both scale with total window count.
    PER_FILE_TRAIN_CAP = 1200
    PER_FILE_TEST_CAP = 600
    cap_rng = np.random.default_rng(0)

    Xs, Ys, Ms, Fs, Ss, tr_masks, te_masks = [], [], [], [], [], [], []
    n_used = n_skipped = 0
    for f in files:
        res = windows_with_repetition(f)
        if res is None:
            n_skipped += 1
            continue
        X, Y, M, reps = res
        tr_idx_f = np.where(np.isin(reps, list(TRAIN_REPS)))[0]
        te_idx_f = np.where(np.isin(reps, list(TEST_REPS)))[0]
        if len(tr_idx_f) == 0 or len(te_idx_f) == 0:
            n_skipped += 1
            continue
        if len(tr_idx_f) > PER_FILE_TRAIN_CAP:
            tr_idx_f = cap_rng.choice(tr_idx_f, PER_FILE_TRAIN_CAP, replace=False)
        if len(te_idx_f) > PER_FILE_TEST_CAP:
            te_idx_f = cap_rng.choice(te_idx_f, PER_FILE_TEST_CAP, replace=False)
        keep_f = np.concatenate([tr_idx_f, te_idx_f])
        Xf, Yf, Mf = X[keep_f], Y[keep_f], M[keep_f]
        tr_mask_f = np.zeros(len(keep_f), dtype=bool)
        tr_mask_f[:len(tr_idx_f)] = True

        Xs.append(Xf); Ys.append(Yf); Ms.append(Mf)
        Fs.append(qk.handcrafted_features(Xf, Mf)); Ss.append(qk.channel_cosine_similarity(Xf, Mf))
        tr_masks.append(tr_mask_f); te_masks.append(~tr_mask_f)
        n_used += 1
    if not Xs:
        sys.exit(f"None of the {len(files)} file(s) in '{data_dir}' had both usable ground truth and a "
                  f"repetition field.")

    X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)
    Feat, Sim = np.concatenate(Fs), np.concatenate(Ss)
    tr_mask, te_mask = np.concatenate(tr_masks), np.concatenate(te_masks)

    subs = sorted({qk.subject_of(f) for f in files})
    print(f"files: {len(files)} total -- {n_used} used, {n_skipped} skipped (no ground truth / no "
          f"repetition field), subjects {subs}")
    print(f"reps {sorted(TRAIN_REPS)} -> train ({tr_mask.sum()} windows), "
          f"reps {sorted(TEST_REPS)} -> test ({te_mask.sum()} windows), device={qk.dev}")

    rng = np.random.default_rng(0)
    train_idx = np.where(tr_mask)[0]
    test_idx = np.where(te_mask)[0]

    # early-stopping val slice carved OUT OF the train-repetition windows only
    # (same leakage-safe block-split the rest of the project uses) -- test_idx
    # (the held-out repetitions) is never touched by this, same discipline as
    # finetune()'s calib-train/calib-val split.
    tr_pos, val_pos = qk.leakage_safe_split(len(train_idx), 1 - qk.VAL_FRAC, rng, gap=1)
    tr_idx, val_idx = train_idx[tr_pos], train_idx[val_pos]

    max_val = max(1, int(qk.MAX_WINDOWS * qk.VAL_FRAC))
    max_tr = max(1, qk.MAX_WINDOWS - max_val)
    if len(tr_idx) > max_tr:
        tr_idx = rng.choice(tr_idx, max_tr, replace=False)
    if len(val_idx) > max_val:
        val_idx = rng.choice(val_idx, max_val, replace=False)
    print(f"{len(tr_idx)} train / {len(val_idx)} calib-val / {len(test_idx)} held-out-repetition test windows")

    xm, xs = X[tr_idx].mean((0, 2), keepdims=True), X[tr_idx].std((0, 2), keepdims=True) + 1e-8
    ym, ys = Y[tr_idx].mean(0, keepdims=True), Y[tr_idx].std(0, keepdims=True) + 1e-8
    fm, fs = Feat[tr_idx].mean(0, keepdims=True), Feat[tr_idx].std(0, keepdims=True) + 1e-8
    sm, ss = Sim[tr_idx].mean(0, keepdims=True), Sim[tr_idx].std(0, keepdims=True) + 1e-8
    Xn, Yn, Fn, Simn = (X - xm) / xs, (Y - ym) / ys, (Feat - fm) / fs, (Sim - sm) / ss

    def make_dl(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xn[idx].astype(np.float32)),
                                        torch.from_numpy(Yn[idx].astype(np.float32)),
                                        torch.from_numpy(M[idx].astype(np.float32)),
                                        torch.from_numpy(Fn[idx].astype(np.float32)),
                                        torch.from_numpy(Simn[idx].astype(np.float32))),
                          batch_size=qk.BATCH, shuffle=shuffle)

    tr_dl, val_dl = make_dl(tr_idx, True), make_dl(val_idx, False)

    net = qk.Net().to(qk.dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=qk.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for ep in range(qk.EPOCHS):
        net.train()
        tot = 0.0
        for xb, yb, mb, fb, sb in tr_dl:
            xb, yb, mb, fb, sb = xb.to(qk.dev), yb.to(qk.dev), mb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev)
            if qk.NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * qk.NOISE_STD
            out = net(xb, fb, sb, mb)
            loss = lf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        net.eval()
        with torch.no_grad():
            val_loss = sum(lf(net(xb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev), mb.to(qk.dev)), yb.to(qk.dev)).item()
                           for xb, yb, mb, fb, sb in val_dl) / len(val_dl)
        sched.step(val_loss)

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1
        print(f"epoch {ep+1}/{qk.EPOCHS}  train {tr_loss:.4f}  calib-val {val_loss:.4f}{marker}")
        if bad_epochs >= qk.PATIENCE:
            print(f"no calib-val improvement for {qk.PATIENCE} epochs, stopping early")
            break

    net.load_state_dict(best_state)
    net.eval()
    Xe = (X[test_idx] - xm) / xs
    Fe = (Feat[test_idx] - fm) / fs
    Se = (Sim[test_idx] - sm) / ss
    out = []
    with torch.no_grad():
        for i in range(0, len(Xe), qk.BATCH):
            xb = torch.from_numpy(Xe[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            mb = torch.from_numpy(M[test_idx][i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            fb = torch.from_numpy(Fe[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            sb = torch.from_numpy(Se[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            out.append(net(xb, fb, sb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym

    print(f"\n=== repetition-held-out benchmark (reps {sorted(TRAIN_REPS)} train / "
          f"{sorted(TEST_REPS)} test, {len(subs)} subjects, single pooled model) ===")
    mean_r2 = qk.report_metrics(pred, Y[test_idx])
    print(f"\nfor comparison: sMAPEN reports R^2 = 0.8163 +/- 0.0398 on this same dataset/protocol "
          f"(Springer JNER 2024, 40 subjects, 4-of-6-reps train / 2-of-6 test).")
    return mean_r2


if __name__ == "__main__":
    main(*sys.argv[1:])
