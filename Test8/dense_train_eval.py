"""
Trains (not just evaluates) at a denser window step, to check whether
training on more densely-overlapping windows adds anything BEYOND the
evaluation-density effect hop_sweep.py already established (fair R^2
0.611 @ hop=200 -> 0.840 @ hop=1, same frozen model, eval-only).

hop_sweep.py proved the EVAL side of the density effect is real and
large, but never retrained -- sMAPEN also TRAINS at their dense 0.5ms
step, so their model saw far more (redundant, but more) training
windows than this project's baseline ever has. This script trains a
fresh model with windows generated at DENSE_HOP (parameterizable, not
hardcoded to qk.HOP like windows_from()/windows_with_repetition() are),
for BOTH the calibration/train side and the eval side, then compares
against hop_sweep.py's eval-only number at the same hop level. A literal
DENSE_HOP=1 full retrain isn't attempted here for the same memory reason
noted in hop_sweep.py's own docstring (a ~15-minute DB2 file has ~1.8M
hop=1 candidate positions) -- DENSE_HOP=20 was chosen since it's one of
hop_sweep.py's own tested points (eval-only fair R^2=0.706 there),
giving a direct, fair before/after comparison at identical density.

Per-file caps here (3000 train / 1500 test) are larger than
benchmark_repetition_split.py's baseline caps (1200/600) specifically so
the DENSER candidate pool at DENSE_HOP=20 still gets meaningfully wider
temporal coverage of each file, not just more redundant near-duplicates
crammed into the same narrow time range a same-sized cap would produce.

  python dense_train_eval.py [data_dir]
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
from benchmark_repetition_split import TRAIN_REPS, TEST_REPS

DENSE_HOP = 20   # samples @ 2000Hz = 10ms step -- matches hop_sweep.py's own hop=20 eval point
                # (eval-only fair R^2 there was 0.706, off a hop=200-trained model)
PER_FILE_TRAIN_CAP = 3000
PER_FILE_TEST_CAP = 1500


def dense_windows_with_repetition(path, hop):
    m = loadmat(path)
    rep_key = "rerepetition" if "rerepetition" in m else ("repetition" if "repetition" in m else None)
    if rep_key is None:
        return None
    reps = m[rep_key].astype(int).ravel()

    emg, glove, channel_mask, warns = qk.load_emg_file(path)
    if glove is None:
        return None
    n = len(emg)
    starts = np.arange(0, n - qk.WIN, hop)
    if len(starts) == 0:
        return None
    rep_at_start = reps[starts]
    X = np.asarray([emg[s:s + qk.WIN].T for s in starts], np.float32)
    Y = np.asarray([glove[s + qk.WIN - 1] for s in starts], np.float32)
    M = np.tile(channel_mask, (len(X), 1))
    return X, Y, M, rep_at_start


def main(data_dir="data", save_model=None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:qk.MAX_FILES]
    if not files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    cap_rng = np.random.default_rng(0)
    Xs, Ys, Ms, Fs, Ss, tr_masks, te_masks = [], [], [], [], [], [], []
    n_used = n_skipped = 0
    for f in files:
        res = dense_windows_with_repetition(f, DENSE_HOP)
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
        sys.exit(f"None of the {len(files)} file(s) in '{data_dir}' had usable ground truth/repetition field.")

    X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)
    Feat, Sim = np.concatenate(Fs), np.concatenate(Ss)
    tr_mask, te_mask = np.concatenate(tr_masks), np.concatenate(te_masks)

    subs = sorted({qk.subject_of(f) for f in files})
    print(f"DENSE_HOP={DENSE_HOP} ({DENSE_HOP/2}ms step) -- files: {len(files)} total -- {n_used} used, "
          f"{n_skipped} skipped, subjects {subs}")

    rng = np.random.default_rng(0)
    train_idx = np.where(tr_mask)[0]
    test_idx = np.where(te_mask)[0]

    tr_pos, val_pos = qk.leakage_safe_split(len(train_idx), 1 - qk.VAL_FRAC, rng, gap=1)
    tr_idx, val_idx = train_idx[tr_pos], train_idx[val_pos]

    max_val = max(1, int(qk.MAX_WINDOWS * qk.VAL_FRAC))
    max_tr = max(1, qk.MAX_WINDOWS - max_val)
    if len(tr_idx) > max_tr:
        tr_idx = rng.choice(tr_idx, max_tr, replace=False)
    if len(val_idx) > max_val:
        val_idx = rng.choice(val_idx, max_val, replace=False)
    print(f"{len(tr_idx)} train / {len(val_idx)} calib-val / {len(test_idx)} held-out-repetition test windows "
          f"(all at DENSE_HOP={DENSE_HOP})")

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
    if save_model:
        torch.save({"sd": best_state, "xm": xm, "xs": xs, "ym": ym, "ys": ys,
                   "fm": fm, "fs": fs, "sm": sm, "ss": ss}, save_model)
        print(f"saved trained checkpoint -> {save_model}")

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

    print(f"\n=== dense-trained ({DENSE_HOP/2}ms step) benchmark, same held-out reps as baseline ===")
    mean_r2 = qk.report_metrics(pred, Y[test_idx])
    print(f"\nfor comparison: hop_sweep.py's EVAL-ONLY hop=20 point (coarse-trained model, "
          f"just evaluated at this density) got fair R^2=0.706. If this dense-TRAINED number is "
          f"meaningfully higher, training density adds something beyond the eval-density effect.")
    return mean_r2


if __name__ == "__main__":
    main(*sys.argv[1:])
