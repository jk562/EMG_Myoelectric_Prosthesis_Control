"""
RAW TEST -> hand-kinematics pipeline: EMG (.fif, 8 channels) -> 15 continuous
joint angles. A SEPARATE, self-contained pipeline from
quick_emg_to_kinematics.py -- different device (8-channel Myo-style
armband, ~500 Hz native), different kinematics representation (15 angle
channels, not the 22-channel CyberGlove mapping), so it gets its own model
rather than being merged into the DB2 pipeline's fixed
MAX_CHANNELS=12/N_KIN=22 output space. Mirrors that file's proven design
(joint multi-channel CNN + hand-crafted-feature branch, leakage-safe
splits, early-stopping finetune) but is deliberately duplicated rather than
imported from it, for the same reason Test7 doesn't reference Test6.

  python rawtest_emg_to_kinematics.py pretrain                              # SSL on unlabeled EMG
  python rawtest_emg_to_kinematics.py train [data_dir] [pretrained.pt]      # trains, saves RawTest-KinNet.pt
  python rawtest_emg_to_kinematics.py predict FILE.fif                      # prints kinematics
  python rawtest_emg_to_kinematics.py finetune FILE.fif [base_model]        # calibrate to one subject

`data_dir` defaults to '../RAW TEST' (relative to this file's directory),
matching the folder the 15 Subject_XX.fif recordings were dropped into.
"""

import os
import re
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset

FS, MAX_CHANNELS, N_KIN = 2000, 8, 15   # 8 real EMG channels (no padding needed for this device), 15 angle outputs
FEAT_DIM = 256
N_HC_FEATS = 16
FEAT_HC_DIM = 64
WIN, HOP = 400, 200          # 200 ms window, 100 ms hop (same convention as quick_emg_to_kinematics.py)
MAX_FILES = 15
MAX_WINDOWS = 60000
EPOCHS = 80
PATIENCE = 8
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
BATCH = 256
NOISE_STD = 0.1
MASK_RATIO = 0.4
SSL_EPOCHS = 40
SSL_LR = 1e-3

dev = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")


def subject_of(path):
    m = re.search(r"Subject_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def handcrafted_features(X, M, thresh=0.01):
    """Classic per-channel EMG features (RMS, waveform length, log-variance,
    zero crossings, slope sign changes, MAV, Willison amplitude, Hjorth
    mobility/complexity, mean power frequency, median frequency, 5 FFT-band
    magnitude sums) -- identical to quick_emg_to_kinematics.py's version,
    duplicated here to keep this pipeline self-contained. X: (n, C, WIN)
    raw EMG windows (already per-file z-scored). M: (n, C) channel mask."""
    n, C, W = X.shape
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)

    rms = np.sqrt(np.mean(X ** 2, axis=-1))
    wl = np.sum(np.abs(d1), axis=-1)
    logvar = np.log(np.var(X, axis=-1) + 1e-8)
    zc = np.sum((X[..., :-1] * X[..., 1:] < 0) & (np.abs(d1) > thresh), axis=-1).astype(np.float32)
    ssc = np.sum((d1[..., :-1] * d1[..., 1:] < 0) & (np.abs(d2) > thresh), axis=-1).astype(np.float32)
    mav = np.mean(np.abs(X), axis=-1)
    wamp = np.sum(np.abs(d1) > thresh, axis=-1).astype(np.float32)
    var_x, var_d1, var_d2 = np.var(X, axis=-1) + 1e-8, np.var(d1, axis=-1) + 1e-8, np.var(d2, axis=-1) + 1e-8
    mobility = np.sqrt(var_d1 / var_x)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-8)

    spec = np.abs(np.fft.rfft(X, axis=-1))
    freqs = np.fft.rfftfreq(W)
    power = spec ** 2
    total_power = np.sum(power, axis=-1) + 1e-8
    mpf = np.sum(freqs[None, None, :] * power, axis=-1) / total_power
    cum_power = np.cumsum(power, axis=-1)
    mdf = freqs[np.argmax(cum_power >= (total_power[..., None] / 2), axis=-1)]

    n_bands = 5
    edges = np.linspace(0, spec.shape[-1], n_bands + 1).astype(int)
    bands = [spec[..., edges[i]:edges[i + 1]].sum(axis=-1) for i in range(n_bands)]

    feats = np.stack([rms, wl, logvar, zc, ssc, mav, wamp, mobility, complexity, mpf, mdf, *bands], axis=-1)
    feats = feats * M[..., None]
    return feats.reshape(n, C * N_HC_FEATS).astype(np.float32)


def leakage_safe_split(n, keep_frac, rng, gap, block=3):
    """Block-shuffle n windows into a majority "keep" group (~keep_frac of
    blocks) and a minority "held_out" group (the rest, with anything within
    `gap` windows of a keep-group member additionally dropped) -- so
    held_out shares no raw EMG samples with keep across the 50% WIN/HOP
    overlap. Identical to quick_emg_to_kinematics.py's version."""
    n_blocks = (n + block - 1) // block
    order = rng.permutation(n_blocks)
    n_keep_blocks = max(1, int(n_blocks * keep_frac))
    keep_blocks = set(order[:n_keep_blocks].tolist())
    keep_mask = np.zeros(n, dtype=bool)
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, n)
        if b in keep_blocks:
            keep_mask[s:e] = True
    keep_idx = np.where(keep_mask)[0]
    keep_set = set(keep_idx.tolist())
    held_out_idx = np.array([i for i in range(n) if not keep_mask[i]
                             and not any((i + d) in keep_set for d in range(-gap, gap + 1))], dtype=int)
    return keep_idx, held_out_idx


class Encoder(nn.Module):
    """1D CNN over all EMG channels jointly -- same design as
    quick_emg_to_kinematics.py's Encoder (cross-channel mixing matters more
    than channel count), sized for MAX_CHANNELS=8 here."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(MAX_CHANNELS, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, FEAT_DIM, 5, padding=2), nn.BatchNorm1d(FEAT_DIM), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(FEAT_DIM, FEAT_DIM, 3, padding=1), nn.BatchNorm1d(FEAT_DIM), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    def pooled(self, x, mask=None):
        return F.adaptive_avg_pool1d(self.net(x), 1).flatten(1)


class Decoder(nn.Module):
    """features -> reconstructed EMG. Only used during SSL pretraining."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(FEAT_DIM, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, MAX_CHANNELS, 3, padding=1),
        )

    def forward(self, feat, out_len):
        return F.interpolate(self.net(feat), size=out_len, mode="linear", align_corners=False)


class Net(nn.Module):
    """encoder + hand-crafted-feature branch + regression head:
    (B, MAX_CHANNELS, WIN) EMG window + (B, MAX_CHANNELS*N_HC_FEATS)
    hand-crafted features -> (B, N_KIN) kinematics."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.feat_proj = nn.Sequential(
            nn.Linear(MAX_CHANNELS * N_HC_FEATS, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, FEAT_HC_DIM), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM + FEAT_HC_DIM, FEAT_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_KIN),
        )

    def forward(self, x, feat, mask=None):
        emb = self.encoder.pooled(x, mask)
        femb = self.feat_proj(feat)
        return self.head(torch.cat([emb, femb], dim=-1))


def load_fif_file(path, max_channels=MAX_CHANNELS, expected_fs=FS):
    """RAW TEST's loader: .fif (MNE) recordings with 'EMG n' and 'Angle n'
    channels -- 8 real EMG channels and 15 continuous joint-angle channels
    (degrees). Mirrors quick_emg_to_kinematics.load_emg_file's contract:
    returns (emg, glove_or_None, channel_mask, warnings)."""
    warns = []
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    data = raw.get_data()
    ch_names = raw.ch_names
    emg_idx = [i for i, n in enumerate(ch_names) if n.startswith("EMG")]
    angle_idx = [i for i, n in enumerate(ch_names) if n.startswith("Angle")]
    if not emg_idx:
        raise ValueError(f"{path}: no 'EMG *' channels found.")
    emg = data[emg_idx].T.astype(np.float32)
    glove = data[angle_idx].T.astype(np.float32) if angle_idx else None
    if glove is not None and glove.shape[1] != N_KIN:
        warns.append(f"found {glove.shape[1]} angle channels, expected {N_KIN} -- dropping, predictions only")
        glove = None
    if glove is None:
        warns.append("no angle channels found -- predictions only, no RMSE/R^2")

    fs = raw.info["sfreq"]
    if fs != expected_fs:
        n_new = max(1, int(round(emg.shape[0] * expected_fs / fs)))
        emg = resample(emg, n_new, axis=0).astype(np.float32)
        if glove is not None:
            glove = resample(glove, n_new, axis=0).astype(np.float32)
        warns.append(f"resampled {fs:.0f} Hz -> {expected_fs} Hz")

    n_real = emg.shape[1]
    if n_real > max_channels:
        warns.append(f"{n_real} channels found, this build only pools up to {max_channels} -- "
                     f"using first {max_channels}")
        emg = emg[:, :max_channels]
        n_real = max_channels
    elif n_real < max_channels:
        pad = np.zeros((len(emg), max_channels - n_real), dtype=np.float32)
        emg = np.concatenate([emg, pad], axis=1)
        warns.append(f"{n_real} channels found -- the model pools only over those {n_real} real channels")

    channel_mask = np.zeros(max_channels, dtype=np.float32)
    channel_mask[:n_real] = 1.0

    real = emg[:, :n_real]
    emg[:, :n_real] = (real - real.mean(0, keepdims=True)) / (real.std(0, keepdims=True) + 1e-8)

    return emg, glove, channel_mask, warns


def windows_from(path):
    emg, glove, channel_mask, warns = load_fif_file(path)
    X = np.asarray([emg[s:s + WIN].T for s in range(0, len(emg) - WIN, HOP)], np.float32)
    M = np.tile(channel_mask, (len(X), 1)) if len(X) else np.zeros((0, len(channel_mask)), np.float32)
    Y = None
    if glove is not None:
        Y = np.asarray([glove[s + WIN - 1] for s in range(0, len(emg) - WIN, HOP)], np.float32)
    return X, Y, M, warns


def mask_windows(xb):
    xb_masked = xb.clone()
    T = xb.shape[-1]
    span = max(1, int(T * MASK_RATIO))
    for i in range(xb.shape[0]):
        s = np.random.randint(0, T - span + 1)
        xb_masked[i, :, s:s + span] = 0.0
    return xb_masked


def masked_mse(recon, target, chan_mask):
    err = (recon - target) ** 2 * chan_mask.unsqueeze(-1)
    return err.sum() / (chan_mask.sum() * target.shape[-1]).clamp_min(1e-8)


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAW TEST")


def pretrain_ssl(data_dir=DEFAULT_DATA_DIR, on_epoch=None):
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.fif")))[:MAX_FILES]
    if not all_files:
        sys.exit(f"No .fif files in '{data_dir}'.")

    # cap PER FILE before concatenating -- these recordings are ~30 min each,
    # concatenating all of them at full resolution first (then capping) would
    # transiently need far more memory than the final MAX_WINDOWS budget
    per_file_cap = max(1, MAX_WINDOWS // len(all_files))
    rng = np.random.default_rng(0)
    Xs, Ms = [], []
    for f in all_files:
        emg, glove, channel_mask, warns = load_fif_file(f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        Xf = np.asarray([emg[s:s + WIN].T for s in range(0, len(emg) - WIN, HOP)], np.float32)
        if len(Xf) > per_file_cap:
            keep = rng.choice(len(Xf), per_file_cap, replace=False)
            Xf = Xf[keep]
        Xs.append(Xf)
        Ms.append(np.tile(channel_mask, (len(Xf), 1)))
    X, M = np.concatenate(Xs), np.concatenate(Ms)

    subs = sorted({subject_of(f) for f in all_files})
    print(f"SSL pretrain: subjects {subs}, {len(X)} windows, device={dev}")

    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * VAL_FRAC))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    xm, xs = X[tr_idx].mean((0, 2), keepdims=True), X[tr_idx].std((0, 2), keepdims=True) + 1e-8
    Xn = (X - xm) / xs

    def make_dl(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xn[idx].astype(np.float32)),
                                        torch.from_numpy(M[idx].astype(np.float32))),
                          batch_size=BATCH, shuffle=shuffle)

    tr_dl, val_dl = make_dl(tr_idx, True), make_dl(val_idx, False)

    enc, dec = Encoder().to(dev), Decoder().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(SSL_EPOCHS):
        enc.train(); dec.train()
        tot = 0.0
        for xb, mb in tr_dl:
            xb, mb = xb.to(dev), mb.to(dev)
            recon = dec(enc(mask_windows(xb)), xb.shape[-1])
            loss = masked_mse(recon, xb, mb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        enc.eval(); dec.eval()
        with torch.no_grad():
            val_loss = sum(masked_mse(dec(enc(mask_windows(xb.to(dev))), xb.shape[-1]), xb.to(dev), mb.to(dev)).item()
                           for xb, mb in val_dl) / len(val_dl)
        sched.step(val_loss)
        history.append((ep + 1, tr_loss, val_loss))

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in enc.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1

        print(f"ssl epoch {ep+1}/{SSL_EPOCHS}  train {tr_loss:.4f}  val {val_loss:.4f}{marker}")
        if on_epoch:
            on_epoch(ep + 1, SSL_EPOCHS, tr_loss, val_loss, marker)
        if bad_epochs >= PATIENCE:
            print(f"no val improvement for {PATIENCE} epochs, stopping early")
            break

    torch.save({"sd": best_state, "history": history}, "RawTest-KinNet-SSL.pt")
    print(f"saved -> RawTest-KinNet-SSL.pt (best val reconstruction loss {best_val:.4f})")
    print("use it with:\n  python rawtest_emg_to_kinematics.py train \"../RAW TEST\" RawTest-KinNet-SSL.pt")
    return best_val


def train(data_dir=DEFAULT_DATA_DIR, pretrained="", on_epoch=None):
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.fif")))[:MAX_FILES]
    if not all_files:
        sys.exit(f"No .fif files in '{data_dir}'.")

    per_file_cap = max(1, MAX_WINDOWS // len(all_files))
    rng = np.random.default_rng(0)
    Xs, Ys, Ms, Fs, tr_parts, val_parts = [], [], [], [], [], []
    n_skipped = 0
    for f in all_files:
        Xf, Yf, Mf, warns = windows_from(f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        if Yf is None:
            n_skipped += 1
            continue
        if len(Xf) == 0:
            continue
        if len(Xf) > per_file_cap:
            keep = rng.choice(len(Xf), per_file_cap, replace=False)
            Xf, Yf, Mf = Xf[keep], Yf[keep], Mf[keep]
        base = sum(len(x) for x in Xs)
        tr_idx_f, val_idx_f = leakage_safe_split(len(Xf), 1 - VAL_FRAC, rng, gap=1)
        Xs.append(Xf); Ys.append(Yf); Ms.append(Mf); Fs.append(handcrafted_features(Xf, Mf))
        tr_parts.append(tr_idx_f + base); val_parts.append(val_idx_f + base)
    if not Xs:
        sys.exit(f"None of the {len(all_files)} file(s) in '{data_dir}' have usable ground-truth kinematics.")
    if n_skipped:
        print(f"files: {len(all_files)} total -- {n_skipped} skipped (no ground truth), "
              f"{len(all_files) - n_skipped} used for training")
    X, Y, M, Feat = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms), np.concatenate(Fs)
    tr_idx, val_idx = np.concatenate(tr_parts), np.concatenate(val_parts)
    if len(val_idx) == 0:
        val_idx, tr_idx = tr_idx[-1:], tr_idx[:-1]

    subs = sorted({subject_of(f) for f in all_files})
    print(f"{len(all_files)} file(s), subjects {subs}, {len(tr_idx)} train / {len(val_idx)} val windows, device={dev}")

    xm = X[tr_idx].mean((0, 2), keepdims=True)
    xs = X[tr_idx].std((0, 2), keepdims=True) + 1e-8
    ym = Y[tr_idx].mean(0, keepdims=True)
    ys = Y[tr_idx].std(0, keepdims=True) + 1e-8
    fm = Feat[tr_idx].mean(0, keepdims=True)
    fs = Feat[tr_idx].std(0, keepdims=True) + 1e-8
    Xn, Yn, Fn = (X - xm) / xs, (Y - ym) / ys, (Feat - fm) / fs

    def make_dl(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xn[idx].astype(np.float32)),
                                        torch.from_numpy(Yn[idx].astype(np.float32)),
                                        torch.from_numpy(M[idx].astype(np.float32)),
                                        torch.from_numpy(Fn[idx].astype(np.float32))),
                          batch_size=BATCH, shuffle=shuffle)

    tr_dl, val_dl = make_dl(tr_idx, True), make_dl(val_idx, False)

    net = Net().to(dev)
    if pretrained:
        ck = torch.load(pretrained, map_location=dev, weights_only=False)
        net.encoder.load_state_dict(ck["sd"])
        print(f"loaded SSL-pretrained encoder from {pretrained}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for xb, yb, mb, fb in tr_dl:
            xb, yb, mb, fb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev)
            if NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * NOISE_STD
            out = net(xb, fb, mb)
            loss = lf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        net.eval()
        with torch.no_grad():
            val_loss = sum(lf(net(xb.to(dev), fb.to(dev), mb.to(dev)), yb.to(dev)).item()
                           for xb, yb, mb, fb in val_dl) / len(val_dl)
        sched.step(val_loss)
        history.append((ep + 1, tr_loss, val_loss))

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
            marker = " *"
        else:
            bad_epochs += 1

        print(f"epoch {ep+1}/{EPOCHS}  train {tr_loss:.4f}  val {val_loss:.4f}{marker}")
        if on_epoch:
            on_epoch(ep + 1, EPOCHS, tr_loss, val_loss, marker)
        if bad_epochs >= PATIENCE:
            print(f"no val improvement for {PATIENCE} epochs, stopping early")
            break

    torch.save({"sd": best_state, "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs, "history": history},
               "RawTest-KinNet.pt")
    print(f"saved -> RawTest-KinNet.pt (best val loss {best_val:.4f})")
    print("trained on all available subjects -- for an honest per-subject accuracy check, use:\n"
          "  python rawtest_emg_to_kinematics.py finetune FILE.fif")
    return best_val


def report_metrics(pred, Y):
    joint_ss_res = np.sum((pred - Y) ** 2, axis=0)
    joint_ss_tot = np.sum((Y - Y.mean(0, keepdims=True)) ** 2, axis=0)
    joint_r2 = 1 - joint_ss_res / joint_ss_tot
    pooled_rmse = np.sqrt(np.mean((pred - Y) ** 2))
    pooled_r2 = 1 - joint_ss_res.sum() / joint_ss_tot.sum()
    mean_r2 = joint_r2.mean()

    print(f"\nRMSE vs ground truth (pooled, raw units): {pooled_rmse:.3f}")
    print(f"R^2 vs ground truth (pooled, raw units):   {pooled_r2:.3f}")
    dominant = int(np.argmax(joint_ss_tot))
    dominant_share = joint_ss_tot[dominant] / joint_ss_tot.sum()
    if dominant_share > 0.5:
        print(f"  note: joint {dominant} alone accounts for {dominant_share*100:.0f}% of the pooled "
              f"variance -- the two numbers above mostly measure that one joint, not overall accuracy")
    best, worst = int(np.argmax(joint_r2)), int(np.argmin(joint_r2))
    print(f"R^2 averaged across all {len(joint_r2)} joints (fair, scale-independent): {mean_r2:.3f}")
    print(f"  best joint: {best} (R^2 {joint_r2[best]:.3f})   worst joint: {worst} (R^2 {joint_r2[worst]:.3f})")
    return mean_r2


def predict(path, model="RawTest-KinNet.pt"):
    ck = torch.load(model, map_location=dev, weights_only=False)
    net = Net().to(dev); net.load_state_dict(ck["sd"]); net.eval()

    X, Y, M, warns = windows_from(path)
    for w in warns:
        print(f"note: {w}")

    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    fm, fs = ck["fm"], ck["fs"]
    Xn = ((X - xm) / xs).astype(np.float32)
    Fn = ((handcrafted_features(X, M) - fm) / fs).astype(np.float32)

    import time; t0 = time.time()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), BATCH):
            xb = torch.from_numpy(Xn[i:i + BATCH]).to(dev)
            mb = torch.from_numpy(M[i:i + BATCH].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fn[i:i + BATCH]).to(dev)
            out.append(net(xb, fb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym
    print(f"predicted {len(pred)} timesteps in {time.time()-t0:.2f}s")

    np.savetxt("rawtest_predicted_kinematics.csv", pred, delimiter=",", fmt="%.4f")
    print("saved -> rawtest_predicted_kinematics.csv")
    print(f"\nfirst 5 timesteps ({N_KIN} joint values each):")
    for r in pred[:5]:
        print("  " + "  ".join(f"{v:7.2f}" for v in r))
    if Y is None:
        print("\nno ground truth available for this file -- predictions only, no RMSE/R^2")
        return
    report_metrics(pred, Y)


def finetune(path, base_model="RawTest-KinNet.pt", epochs=60, lr=1e-4, out_name=None):
    """Calibrate an already-trained model to ONE specific subject's own
    recording -- same design as quick_emg_to_kinematics.finetune: leakage-
    safe calibrate/eval split, plus an internal calib-train/calib-val split
    for early stopping so eval_idx never influences model selection."""
    ck = torch.load(base_model, map_location=dev, weights_only=False)
    net = Net().to(dev)
    net.load_state_dict(ck["sd"])

    X, Y, M, warns = windows_from(path)
    for w in warns:
        print(f"  {w}")
    if Y is None:
        sys.exit(f"{path}: no ground-truth kinematics -- can't calibrate without labels.")

    n = len(X)
    if n < 4:
        sys.exit(f"{path}: only {n} window(s) -- too short to calibrate.")

    rng = np.random.default_rng(0)
    calib_idx, eval_idx = leakage_safe_split(n, 0.7, rng, gap=1)
    if len(eval_idx) == 0:
        sys.exit(f"{path}: no eval windows left after the leakage gap -- try a longer recording.")

    ct_pos, cv_pos = leakage_safe_split(len(calib_idx), 1 - VAL_FRAC, rng, gap=1)
    if len(cv_pos) == 0:
        ct_pos, cv_pos = np.arange(len(calib_idx) - 1), np.arange(len(calib_idx) - 1, len(calib_idx))
    calib_tr_idx, calib_val_idx = calib_idx[ct_pos], calib_idx[cv_pos]

    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    fm, fs = ck["fm"], ck["fs"]
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    Fn = (handcrafted_features(X, M) - fm) / fs

    dl = DataLoader(TensorDataset(torch.from_numpy(Xn[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(Yn[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(M[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(Fn[calib_tr_idx].astype(np.float32))),
                    batch_size=BATCH, shuffle=True)

    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    def calib_val_loss():
        net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xn[calib_val_idx].astype(np.float32)).to(dev)
            yb = torch.from_numpy(Yn[calib_val_idx].astype(np.float32)).to(dev)
            mb = torch.from_numpy(M[calib_val_idx].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fn[calib_val_idx].astype(np.float32)).to(dev)
            return lf(net(xb, fb, mb), yb).item()

    print(f"calibrating on {len(calib_tr_idx)} windows ({len(calib_val_idx)} held out for early "
          f"stopping) from {path}, evaluating on {len(eval_idx)} held-out windows")
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(int(epochs)):
        net.train()
        tot = 0.0
        for xb, yb, mb, fb in dl:
            xb, yb, mb, fb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev)
            out = net(xb, fb, mb)
            loss = lf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(dl)
        val_loss = calib_val_loss()
        sched.step(val_loss)
        history.append((ep + 1, tr_loss, val_loss))

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
            marker = " *"
        else:
            bad_epochs += 1
        print(f"finetune epoch {ep+1}/{epochs}  train {tr_loss:.4f}  calib-val {val_loss:.4f}{marker}")
        if bad_epochs >= PATIENCE:
            print(f"no calib-val improvement for {PATIENCE} epochs, stopping early")
            break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = []
        Xe, Fe = Xn[eval_idx], Fn[eval_idx]
        for i in range(0, len(Xe), BATCH):
            xb = torch.from_numpy(Xe[i:i + BATCH].astype(np.float32)).to(dev)
            mb = torch.from_numpy(M[eval_idx][i:i + BATCH].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fe[i:i + BATCH].astype(np.float32)).to(dev)
            out.append(net(xb, fb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym
    print("\nafter calibration, on held-out windows from the SAME subject:")
    report_metrics(pred, Y[eval_idx])

    out_name = out_name or f"RawTest-KinNet_finetuned_{os.path.splitext(os.path.basename(path))[0]}.pt"
    torch.save({"sd": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs,
               "history": history}, out_name)
    print(f"\nsaved -> {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        if sys.argv[1] == "pretrain":
            pretrain_ssl(*sys.argv[2:])
        elif sys.argv[1] == "train":
            train(*sys.argv[2:])
        elif sys.argv[1] == "predict":
            predict(*sys.argv[2:])
        elif sys.argv[1] == "finetune":
            finetune(*sys.argv[2:])
        else:
            sys.exit(__doc__)
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"Error: {e}")
