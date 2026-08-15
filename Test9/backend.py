"""
Test9: unified EMG -> hand-kinematics backend, combining Test7/Test8's two
separate pipelines (quick_emg_to_kinematics.py for Ninapro DB2,
rawtest_emg_to_kinematics.py for the RAW TEST device) into ONE
parameterized file. Uses the proven, verified Test7/Test8 architecture --
a joint multi-channel 1D CNN encoder (plain average pooling; see
"architecture note" below for why), a hand-crafted per-channel feature
branch, and (DB2 only) a cross-channel cosine-similarity branch -- NOT any
of the experimental Test8 architecture changes (single-head attention
pooling, a Transformer block, per-joint loss weighting), all three of
which were tried and reverted after measuring real regressions.

Both datasets share the same Net/Encoder/Decoder classes, feature
functions, leakage-safe splitting, and training/predict/finetune logic;
only per-dataset specifics (channel count, kinematics dimensionality,
file loader, whether the similarity branch applies) are parameterized via
the DATASETS config below, instead of duplicating the whole file the way
Test7/Test8 did (a deliberate choice there, to keep each project stage
isolated -- Test9 is the final consolidation once the architecture is
settled, not a new experiment, so duplication no longer earns its keep).

  python backend.py pretrain DATASET [data_dir]
  python backend.py train DATASET [data_dir] [pretrained.pt]
  python backend.py predict DATASET FILE
  python backend.py finetune DATASET FILE [base_model]

DATASET is "db2" or "rawtest".

Architecture note: Encoder.pooled() uses plain average pooling over time,
not learned attention. Two attention-based alternatives (single-head
attention pooling, a 2-layer/8-head Transformer block) were tried in
Test8 and both measured WORSE on real retrains (fair R^2 0.695->0.681
and ->0.674 respectively, with an earlier and worse validation-loss
plateau each time -- the signature of added capacity overfitting this
project's moderate dataset size, not a missing mechanism). Per-joint
adaptive loss weighting was also tried and reverted (no effect at all --
the worst-predicted joint's own R^2 was unchanged to three decimal
places). See Test8/quick_emg_to_kinematics.py's Encoder docstring and
Test8/benchmark_repetition_split.py's WeightedMSE docstring for the full
numbers. Don't re-attempt any of these three without new evidence.
"""

import os
import re
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset

SEED = 0   # reproducibility fix: Test7/Test8/Test9 never seeded torch's global RNG, so network
          # weight init, DataLoader shuffle order, dropout masks, and NOISE_STD's gaussian draws
          # all varied run-to-run even though the data SPLITS themselves were already seeded
          # (np.random.default_rng(0) throughout) -- this is why the same architecture/protocol
          # produced calibrated fair R^2 anywhere from 0.585 to 0.790 across different retrains
          # in this project's history. seed_everything() is called at the start of pretrain_ssl(),
          # train(), and finetune() so each is independently, fully reproducible.


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)          # global numpy state -- mask_windows() uses np.random.randint
                                  # directly (not a seeded Generator), so this is required too
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Shared constants (identical across both datasets)
# ============================================================
FEAT_DIM = 256                # encoder feature width
N_HC_FEATS = 16               # hand-crafted features per channel
FEAT_HC_DIM = 64              # projected width of the hand-crafted-feature branch
FEAT_SIM_DIM = 32             # projected width of the cross-channel cosine-similarity branch (DB2 only)
WIN, HOP = 400, 200          # 200 ms window, 100 ms hop @ 2000 Hz -- same convention both datasets
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

EMG_KEYS = ("emg", "EMG", "signal", "data", "x", "X")
GLOVE_KEYS = ("glove", "kinematics", "y", "Y", "labels")
FS_KEYS = ("fs", "Fs", "sampling_rate", "sample_rate")

dev = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
RAWTEST_DIR = os.path.join(HERE, "..", "RAW TEST")


# ============================================================
# Per-dataset loaders
# ============================================================
def _find_array(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k])
    return None


def _as_time_by_channels(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < a.shape[1]:
        a = a.T
    return a


def _finish_load(emg, glove, fs, max_channels, n_kin, expected_fs, warns):
    """Shared tail end of both loaders: shape/resample/pad/normalise --
    identical logic to what Test7/Test8's two separate loaders each did."""
    emg = _as_time_by_channels(emg)
    if glove is not None:
        glove = _as_time_by_channels(glove)
        if glove.shape[1] != n_kin:
            warns.append(f"ground truth has {glove.shape[1]} channels, model predicts {n_kin} -- "
                         f"dropping it, predictions only")
            glove = None
    if glove is None:
        warns.append("no ground-truth kinematics found -- predictions only, no RMSE/R^2")

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


def load_db2_file(path, max_channels, n_kin, expected_fs):
    """Ninapro DB2 loader -- .mat (usual keys), .csv, .npy."""
    ext = os.path.splitext(path)[1].lower()
    warns = []
    glove, fs = None, expected_fs
    if ext == ".mat":
        m = loadmat(path)
        emg = _find_array(m, EMG_KEYS)
        if emg is None:
            raise ValueError(f"{path}: no EMG-like array found (looked for {EMG_KEYS}).")
        glove = _find_array(m, GLOVE_KEYS)
        fs_arr = _find_array(m, FS_KEYS)
        if fs_arr is not None:
            fs = float(np.ravel(fs_arr)[0])
    elif ext == ".csv":
        emg = np.loadtxt(path, delimiter=",", skiprows=1)
        warns.append("csv input: assuming every column is an EMG channel")
    elif ext == ".npy":
        emg = np.load(path)
        warns.append("npy input: assuming array is raw EMG (time x channels)")
    else:
        raise ValueError(f"unsupported file type '{ext}' -- supported: .mat, .csv, .npy")
    return _finish_load(emg, glove, fs, max_channels, n_kin, expected_fs, warns)


def load_rawtest_file(path, max_channels, n_kin, expected_fs):
    """RAW TEST loader -- .fif (MNE), 'EMG n' / 'Angle n' channels."""
    import mne
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
    fs = raw.info["sfreq"]
    return _finish_load(emg, glove, fs, max_channels, n_kin, expected_fs, warns)


def subject_of_db2(path):
    m = re.search(r"S(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def subject_of_rawtest(path):
    m = re.search(r"Subject_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


# ============================================================
# Dataset configuration -- everything that differs between DB2 and RAW TEST
# ============================================================
DATASETS = {
    "db2": dict(
        max_channels=12, n_kin=22, fs=2000, use_sim=True,
        data_dir=os.path.join(HERE, "data"),
        ext=("*.mat", "*.csv", "*.npy"),
        load_fn=load_db2_file,
        subject_of=subject_of_db2,
        max_files=57,
        ckpt_prefix="EMG-KinNet",
    ),
    "rawtest": dict(
        max_channels=8, n_kin=15, fs=2000, use_sim=False,
        data_dir=RAWTEST_DIR,
        ext=("*.fif",),
        load_fn=load_rawtest_file,
        subject_of=subject_of_rawtest,
        max_files=15,
        ckpt_prefix="RawTest-KinNet",
    ),
}


def _ds(name):
    if name not in DATASETS:
        sys.exit(f"Unknown dataset '{name}' -- choose one of {list(DATASETS)}.")
    return DATASETS[name]


def load_emg_file(ds, path):
    return ds["load_fn"](path, ds["max_channels"], ds["n_kin"], ds["fs"])


def list_files(ds, data_dir=None):
    data_dir = data_dir or ds["data_dir"]
    files = []
    for pattern in ds["ext"]:
        files += glob.glob(os.path.join(data_dir, pattern))
    return sorted(files)[:ds["max_files"]]


# ============================================================
# Shared feature engineering (dataset-agnostic -- operates on arrays only)
# ============================================================
def _channel_features(X, M, thresh=0.01):
    """Per-channel hand-crafted EMG features, UNFLATTENED -- (n, C, N_HC_FEATS)."""
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
    return feats * M[..., None]


def handcrafted_features(X, M, thresh=0.01):
    """16 classic per-channel EMG features (RMS, waveform length, log-
    variance, zero crossings, slope sign changes, MAV, Willison amplitude,
    Hjorth mobility/complexity, mean/median power frequency, 5 FFT-band
    sums) -- KinEMbed-inspired (Gilardini et al., arXiv:2607.04820)."""
    n, C = X.shape[0], X.shape[1]
    feats = _channel_features(X, M, thresh)
    return feats.reshape(n, C * N_HC_FEATS).astype(np.float32)


def channel_cosine_similarity(X, M, thresh=0.01):
    """Pairwise cosine similarity between channels' hand-crafted feature
    vectors (ROCKET-paper-inspired, Ovadia/Segal/Rabin 2024, Sci Rep
    14:4134) -- DB2 only, see Net's use_sim flag."""
    n, C, _ = X.shape
    feats = _channel_features(X, M, thresh)
    norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    sim = np.einsum("ncf,ndf->ncd", norm, norm)
    return sim.reshape(n, C * C).astype(np.float32)


def leakage_safe_split(n, keep_frac, rng, gap, block=3):
    """Block-shuffled split so held-out windows share no raw samples with
    kept windows, despite 50% WIN/HOP overlap."""
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


# ============================================================
# Model -- ONE Net/Encoder/Decoder serving both datasets
# ============================================================
class Encoder(nn.Module):
    """1D CNN over all EMG channels jointly -- channels mixed at every
    layer (not processed independently), which is what's been verified to
    matter: an earlier per-channel-independent variant cost ~2.3x worse
    val loss by losing cross-channel muscle-coordination patterns.
    Pooling is plain average pooling over time -- see module docstring
    for why (two attention-based alternatives were tried and reverted)."""
    def __init__(self, max_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(max_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
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
    def __init__(self, max_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(FEAT_DIM, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, max_channels, 3, padding=1),
        )

    def forward(self, feat, out_len):
        return F.interpolate(self.net(feat), size=out_len, mode="linear", align_corners=False)


class Net(nn.Module):
    """encoder + hand-crafted-feature branch + (optional) cross-channel
    similarity branch + regression head. use_sim=True adds the third
    branch (DB2 only -- verified real gain there, S13 calibrated R^2
    0.764->0.790; not separately verified for RAW TEST, so left off there
    per the same scoped-decision Test7/Test8 made)."""
    def __init__(self, max_channels, n_kin, use_sim=True):
        super().__init__()
        self.max_channels, self.n_kin, self.use_sim = max_channels, n_kin, use_sim
        self.encoder = Encoder(max_channels)
        self.feat_proj = nn.Sequential(
            nn.Linear(max_channels * N_HC_FEATS, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, FEAT_HC_DIM), nn.ReLU(),
        )
        head_in = FEAT_DIM + FEAT_HC_DIM
        if use_sim:
            self.sim_proj = nn.Sequential(
                nn.Linear(max_channels * max_channels, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, FEAT_SIM_DIM), nn.ReLU(),
            )
            head_in += FEAT_SIM_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, FEAT_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_kin),
        )

    def forward(self, x, feat, sim=None, mask=None):
        emb = self.encoder.pooled(x, mask)
        femb = self.feat_proj(feat)
        parts = [emb, femb]
        if self.use_sim:
            parts.append(self.sim_proj(sim))
        return self.head(torch.cat(parts, dim=-1))


def new_net(ds):
    return Net(ds["max_channels"], ds["n_kin"], ds["use_sim"]).to(dev)


# ============================================================
# Windowing
# ============================================================
def windows_from(ds, path):
    emg, glove, channel_mask, warns = load_emg_file(ds, path)
    n = len(emg)
    starts = list(range(0, n - WIN, HOP))
    X = np.asarray([emg[s:s + WIN].T for s in starts], np.float32)
    M = np.tile(channel_mask, (len(X), 1)) if len(X) else np.zeros((0, len(channel_mask)), np.float32)
    Y = None
    if glove is not None:
        Y = np.asarray([glove[s + WIN - 1] for s in starts], np.float32)
    return X, Y, M, warns


def _features(ds, X, M):
    """Returns (Feat, Sim_or_None) for a dataset's use_sim setting."""
    Feat = handcrafted_features(X, M)
    Sim = channel_cosine_similarity(X, M) if ds["use_sim"] else None
    return Feat, Sim


def _forward(net, ds, xb, fb, sb, mb):
    return net(xb, fb, sb, mb) if ds["use_sim"] else net(xb, fb, mask=mb)


# ============================================================
# report_metrics
# ============================================================
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
    return float(mean_r2)


# ============================================================
# pretrain / train / predict / finetune
# ============================================================
def pretrain_ssl(dataset, data_dir=None, on_epoch=None):
    seed_everything()
    ds = _ds(dataset)
    files = list_files(ds, data_dir)
    if not files:
        sys.exit(f"No files for dataset '{dataset}' in '{data_dir or ds['data_dir']}'.")

    per_file_cap = max(1, MAX_WINDOWS // len(files))
    rng = np.random.default_rng(0)
    Xs, Ms = [], []
    for f in files:
        emg, glove, channel_mask, warns = load_emg_file(ds, f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        Xf = np.asarray([emg[s:s + WIN].T for s in range(0, len(emg) - WIN, HOP)], np.float32)
        if len(Xf) > per_file_cap:
            keep = rng.choice(len(Xf), per_file_cap, replace=False)
            Xf = Xf[keep]
        Xs.append(Xf)
        Ms.append(np.tile(channel_mask, (len(Xf), 1)))
    X, M = np.concatenate(Xs), np.concatenate(Ms)

    subs = sorted({ds["subject_of"](f) for f in files})
    print(f"SSL pretrain [{dataset}]: subjects {subs}, {len(X)} windows, device={dev}")

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

    enc, dec = Encoder(ds["max_channels"]).to(dev), Decoder(ds["max_channels"]).to(dev)
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

    out_name = f"{ds['ckpt_prefix']}-SSL.pt"
    torch.save({"sd": best_state, "history": history}, out_name)
    print(f"saved -> {out_name} (best val reconstruction loss {best_val:.4f})")
    return best_val


def train(dataset, data_dir=None, pretrained="", on_epoch=None):
    seed_everything()
    ds = _ds(dataset)
    files = list_files(ds, data_dir)
    if not files:
        sys.exit(f"No files for dataset '{dataset}' in '{data_dir or ds['data_dir']}'.")

    rng = np.random.default_rng(0)
    per_file_cap = max(1, MAX_WINDOWS // len(files))
    Xs, Ys, Ms, Fs, Ss, tr_parts, val_parts = [], [], [], [], [], [], []
    n_skipped = 0
    for f in files:
        Xf, Yf, Mf, warns = windows_from(ds, f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        if Yf is None or len(Xf) == 0:
            n_skipped += 1
            continue
        if len(Xf) > per_file_cap:
            keep = rng.choice(len(Xf), per_file_cap, replace=False)
            Xf, Yf, Mf = Xf[keep], Yf[keep], Mf[keep]
        base = sum(len(x) for x in Xs)
        tr_idx_f, val_idx_f = leakage_safe_split(len(Xf), 1 - VAL_FRAC, rng, gap=1)
        Ff, Sf = _features(ds, Xf, Mf)
        Xs.append(Xf); Ys.append(Yf); Ms.append(Mf); Fs.append(Ff)
        if ds["use_sim"]:
            Ss.append(Sf)
        tr_parts.append(tr_idx_f + base); val_parts.append(val_idx_f + base)
    if not Xs:
        sys.exit(f"None of the {len(files)} file(s) for dataset '{dataset}' have usable ground truth.")
    if n_skipped:
        print(f"files: {len(files)} total -- {n_skipped} skipped (no ground truth), "
              f"{len(files) - n_skipped} used for training")

    X, Y, M, Feat = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms), np.concatenate(Fs)
    Sim = np.concatenate(Ss) if ds["use_sim"] else None
    tr_idx, val_idx = np.concatenate(tr_parts), np.concatenate(val_parts)
    if len(val_idx) == 0:
        val_idx, tr_idx = tr_idx[-1:], tr_idx[:-1]

    subs = sorted({ds["subject_of"](f) for f in files})
    print(f"[{dataset}] {len(files)} file(s), subjects {subs}, {len(tr_idx)} train / "
          f"{len(val_idx)} val windows, device={dev}")

    xm, xs = X[tr_idx].mean((0, 2), keepdims=True), X[tr_idx].std((0, 2), keepdims=True) + 1e-8
    ym, ys = Y[tr_idx].mean(0, keepdims=True), Y[tr_idx].std(0, keepdims=True) + 1e-8
    fm, fs = Feat[tr_idx].mean(0, keepdims=True), Feat[tr_idx].std(0, keepdims=True) + 1e-8
    Xn, Yn, Fn = (X - xm) / xs, (Y - ym) / ys, (Feat - fm) / fs
    sm = ss = Simn = None
    if ds["use_sim"]:
        sm, ss = Sim[tr_idx].mean(0, keepdims=True), Sim[tr_idx].std(0, keepdims=True) + 1e-8
        Simn = (Sim - sm) / ss

    def make_dl(idx, shuffle):
        tensors = [torch.from_numpy(Xn[idx].astype(np.float32)),
                   torch.from_numpy(Yn[idx].astype(np.float32)),
                   torch.from_numpy(M[idx].astype(np.float32)),
                   torch.from_numpy(Fn[idx].astype(np.float32))]
        if ds["use_sim"]:
            tensors.append(torch.from_numpy(Simn[idx].astype(np.float32)))
        return DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=shuffle)

    tr_dl, val_dl = make_dl(tr_idx, True), make_dl(val_idx, False)

    net = new_net(ds)
    if pretrained:
        ck = torch.load(pretrained, map_location=dev, weights_only=False)
        net.encoder.load_state_dict(ck["sd"])
        print(f"loaded SSL-pretrained encoder from {pretrained}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    def step_batch(batch, train_mode):
        if ds["use_sim"]:
            xb, yb, mb, fb, sb = batch
        else:
            xb, yb, mb, fb = batch
            sb = None
        xb, yb, mb, fb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev)
        if sb is not None:
            sb = sb.to(dev)
        if train_mode and NOISE_STD > 0:
            xb = xb + torch.randn_like(xb) * NOISE_STD
        out = _forward(net, ds, xb, fb, sb, mb)
        return lf(out, yb)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for batch in tr_dl:
            loss = step_batch(batch, True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        net.eval()
        with torch.no_grad():
            val_loss = sum(step_batch(b, False).item() for b in val_dl) / len(val_dl)
        sched.step(val_loss)
        history.append((ep + 1, tr_loss, val_loss))

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1
        print(f"epoch {ep+1}/{EPOCHS}  train {tr_loss:.4f}  val {val_loss:.4f}{marker}")
        if on_epoch:
            on_epoch(ep + 1, EPOCHS, tr_loss, val_loss, marker)
        if bad_epochs >= PATIENCE:
            print(f"no val improvement for {PATIENCE} epochs, stopping early")
            break

    ck_out = {"sd": best_state, "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs, "history": history}
    if ds["use_sim"]:
        ck_out["sm"], ck_out["ss"] = sm, ss
    out_name = f"{ds['ckpt_prefix']}.pt"
    torch.save(ck_out, out_name)
    print(f"saved -> {out_name} (best val loss {best_val:.4f})")
    print(f"trained on all available subjects -- for an honest per-subject accuracy check, use:\n"
          f"  python backend.py finetune {dataset} FILE")
    return best_val


def predict_batches(net, ds, ck, X, M):
    xm, xs = ck["xm"], ck["xs"]
    fm, fs = ck["fm"], ck["fs"]
    Xn = ((X - xm) / xs).astype(np.float32)
    Feat, Sim = _features(ds, X, M)
    Fn = ((Feat - fm) / fs).astype(np.float32)
    Sn = None
    if ds["use_sim"]:
        sm, ss = ck["sm"], ck["ss"]
        Sn = ((Sim - sm) / ss).astype(np.float32)

    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), BATCH):
            xb = torch.from_numpy(Xn[i:i + BATCH]).to(dev)
            mb = torch.from_numpy(M[i:i + BATCH].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fn[i:i + BATCH]).to(dev)
            sb = torch.from_numpy(Sn[i:i + BATCH]).to(dev) if Sn is not None else None
            out.append(_forward(net, ds, xb, fb, sb, mb).cpu().numpy())
    return np.concatenate(out) * ck["ys"] + ck["ym"]


def predict(dataset, path, model=None):
    ds = _ds(dataset)
    model = model or f"{ds['ckpt_prefix']}.pt"
    ck = torch.load(model, map_location=dev, weights_only=False)
    net = new_net(ds)
    net.load_state_dict(ck["sd"]); net.eval()

    X, Y, M, warns = windows_from(ds, path)
    for w in warns:
        print(f"note: {w}")

    import time; t0 = time.time()
    pred = predict_batches(net, ds, ck, X, M)
    print(f"predicted {len(pred)} timesteps in {time.time()-t0:.2f}s")

    out_csv = f"{dataset}_predicted_kinematics.csv"
    np.savetxt(out_csv, pred, delimiter=",", fmt="%.4f")
    print(f"saved -> {out_csv}")
    print(f"\nfirst 5 timesteps ({ds['n_kin']} joint values each):")
    for r in pred[:5]:
        print("  " + "  ".join(f"{v:7.2f}" for v in r))
    if Y is None:
        print("\nno ground truth available for this file -- predictions only, no RMSE/R^2")
        return
    report_metrics(pred, Y)


def finetune(dataset, path, base_model=None, epochs=60, lr=1e-4, out_name=None):
    seed_everything()
    ds = _ds(dataset)
    base_model = base_model or f"{ds['ckpt_prefix']}.pt"
    ck = torch.load(base_model, map_location=dev, weights_only=False)
    net = new_net(ds)
    net.load_state_dict(ck["sd"])

    X, Y, M, warns = windows_from(ds, path)
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
    Feat, Sim = _features(ds, X, M)
    Fn = (Feat - fm) / fs
    Sn = None
    if ds["use_sim"]:
        sm, ss = ck["sm"], ck["ss"]
        Sn = (Sim - sm) / ss

    def make_dl(idx):
        tensors = [torch.from_numpy(Xn[idx].astype(np.float32)),
                   torch.from_numpy(Yn[idx].astype(np.float32)),
                   torch.from_numpy(M[idx].astype(np.float32)),
                   torch.from_numpy(Fn[idx].astype(np.float32))]
        if ds["use_sim"]:
            tensors.append(torch.from_numpy(Sn[idx].astype(np.float32)))
        return DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True)

    dl = make_dl(calib_tr_idx)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    def batch_loss(batch):
        if ds["use_sim"]:
            xb, yb, mb, fb, sb = batch
        else:
            xb, yb, mb, fb = batch
            sb = None
        xb, yb, mb, fb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev)
        if sb is not None:
            sb = sb.to(dev)
        out = _forward(net, ds, xb, fb, sb, mb)
        return lf(out, yb)

    def calib_val_loss():
        net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xn[calib_val_idx].astype(np.float32)).to(dev)
            yb = torch.from_numpy(Yn[calib_val_idx].astype(np.float32)).to(dev)
            mb = torch.from_numpy(M[calib_val_idx].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fn[calib_val_idx].astype(np.float32)).to(dev)
            sb = torch.from_numpy(Sn[calib_val_idx].astype(np.float32)).to(dev) if ds["use_sim"] else None
            return lf(_forward(net, ds, xb, fb, sb, mb), yb).item()

    print(f"calibrating [{dataset}] on {len(calib_tr_idx)} windows ({len(calib_val_idx)} held out for "
          f"early stopping) from {path}, evaluating on {len(eval_idx)} held-out windows")
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(int(epochs)):
        net.train()
        tot = 0.0
        for batch in dl:
            loss = batch_loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(dl)
        val_loss = calib_val_loss()
        sched.step(val_loss)
        history.append((ep + 1, tr_loss, val_loss))

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1
        print(f"finetune epoch {ep+1}/{epochs}  train {tr_loss:.4f}  calib-val {val_loss:.4f}{marker}")
        if bad_epochs >= PATIENCE:
            print(f"no calib-val improvement for {PATIENCE} epochs, stopping early")
            break

    net.load_state_dict(best_state)
    net.eval()
    pred = predict_batches(net, ds, ck, X[eval_idx], M[eval_idx])
    print("\nafter calibration, on held-out windows from the SAME subject:")
    report_metrics(pred, Y[eval_idx])

    out_name = out_name or f"{ds['ckpt_prefix']}_finetuned_{os.path.splitext(os.path.basename(path))[0]}.pt"
    ck_out = {"sd": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs, "history": history}
    if ds["use_sim"]:
        ck_out["sm"], ck_out["ss"] = ck["sm"], ck["ss"]
    torch.save(ck_out, out_name)
    print(f"\nsaved -> {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, dataset = sys.argv[1], sys.argv[2]
    try:
        if cmd == "pretrain":
            pretrain_ssl(dataset, *sys.argv[3:])
        elif cmd == "train":
            train(dataset, *sys.argv[3:])
        elif cmd == "predict":
            predict(dataset, *sys.argv[3:])
        elif cmd == "finetune":
            finetune(dataset, *sys.argv[3:])
        else:
            sys.exit(__doc__)
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"Error: {e}")
