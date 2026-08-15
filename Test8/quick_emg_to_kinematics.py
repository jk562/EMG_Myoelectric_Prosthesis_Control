"""
EMG -> hand-kinematics pipeline: (optionally) SSL-pretrain an encoder, train
a CNN regressor, then predict. Self-contained -- everything it needs is in
this file.

  python quick_emg_to_kinematics.py pretrain                          # SSL on unlabeled EMG, saves EMG-KinNet-SSL.pt
  python quick_emg_to_kinematics.py train [data_dir] [pretrained.pt]  # trains, saves EMG-KinNet.pt
  python quick_emg_to_kinematics.py predict FILE.mat                  # seconds, prints kinematics
  python quick_emg_to_kinematics.py finetune FILE.mat [base_model]    # calibrate to one subject's own data

`train` uses every available subject -- there's no held-out set, so there's
no "unseen subject" cross-population number to report. If you need an
honest accuracy check for a specific person, use `finetune`: it evaluates
on a held-out 30% *of that person's own recording* (a leakage-free block
split, not just re-testing on what it trained on) regardless of whether
that subject's data was in the training pool.

`pretrain` does masked-reconstruction SSL: it masks a random span of each
EMG window and trains an encoder+decoder to reconstruct it, using NO labels
at all. Feed the resulting `EMG-KinNet-SSL.pt` into `train` to initialise the
regressor's encoder instead of starting from scratch:

  python quick_emg_to_kinematics.py pretrain
  python quick_emg_to_kinematics.py train data EMG-KinNet-SSL.pt

Files with fewer than MAX_CHANNELS real EMG channels (e.g. 6 or 10) are
zero-padded up to MAX_CHANNELS -- the encoder still mixes all channels
together jointly (standard multi-channel conv), so it keeps learning
cross-channel muscle-coordination patterns regardless of how many of those
channels are real vs. padding. A per-channel mask is still tracked and used
to (a) normalise only the real channels per file and (b) exclude padded
channels from the SSL reconstruction loss -- but it does NOT drive pooling;
an earlier version that processed channels independently and pooled only
the real ones measured ~2.3x worse held-out loss on identical data, because
it lost the ability to learn those cross-channel patterns at all.
"""

import os
import re
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset

FS, MAX_CHANNELS, N_KIN = 2000, 12, 22   # MAX_CHANNELS is a batch-tensor cap, not a requirement --
                                          # files with fewer real channels work fine, see load_emg_file
FEAT_DIM = 256                # encoder feature width -- sized for real cross-subject accuracy
N_HC_FEATS = 16               # hand-crafted features per channel, see handcrafted_features()
FEAT_HC_DIM = 64              # projected width of the hand-crafted-feature branch
FEAT_SIM_DIM = 32             # projected width of the cross-channel cosine-similarity branch
WIN, HOP = 400, 200          # 200 ms window, 100 ms hop
POOL_T = WIN // 8             # conv stack has 3x MaxPool1d(2) -- length of the token sequence the
                              # Encoder's transformer block operates over, see Encoder
LABEL_LAG_MS = 0              # electromechanical delay: TRIED 200ms (Guo et al. 2025, Sci Data 12:445,
                              # backward-shifted predictions by 200ms for their best RMS/MU-based
                              # kinematics regression) -- measured worse on real data here: val loss
                              # 0.2985->0.3151, S22 calibrated R^2 0.674->0.654 (S13 barely moved,
                              # 0.769->0.773). Reverted to 0 (label pinned to the window's last sample).
                              # Their setup differs enough (RMS+MLR, not a CNN; 300ms window; different
                              # device) that the same fixed delay doesn't transfer here -- don't
                              # re-enable without new evidence.
LABEL_LAG = int(round(LABEL_LAG_MS * FS / 1000))
MAX_FILES = 57                # all files from the training subjects (test subjects excluded below)
MAX_WINDOWS = 60000          # cap on TOTAL windows, sampled evenly across files (not truncated)
EPOCHS = 80                  # early stopping usually ends this well before the ceiling
PATIENCE = 8                 # stop if val loss hasn't improved in this many epochs
WEIGHT_DECAY = 1e-4      # L2 regularisation -- FEAT_DIM=256 has more capacity to overfit with
VAL_FRAC = 0.1
BATCH = 256
NOISE_STD = 0.1              # gaussian noise added to normalised EMG during training only,
                              # so the model doesn't just learn clean lab signal
MASK_RATIO = 0.4             # fraction of each window blanked out for SSL reconstruction
SSL_EPOCHS = 40
SSL_LR = 1e-3

# predict() accepts files that don't match the training data exactly -- these
# are the key names it'll look for, and what it does when the shape differs.
EMG_KEYS = ("emg", "EMG", "signal", "data", "x", "X")
GLOVE_KEYS = ("glove", "kinematics", "y", "Y", "labels")
FS_KEYS = ("fs", "Fs", "sampling_rate", "sample_rate")

dev = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")


def subject_of(path):
    m = re.search(r"S(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def _channel_features(X, M, thresh=0.01):
    """Per-channel hand-crafted EMG features, UNFLATTENED -- (n, C,
    N_HC_FEATS). See handcrafted_features() for the feature list and
    rationale; split out so channel_cosine_similarity() can reuse it without
    recomputing or needing to un-reshape."""
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
    return feats * M[..., None]   # zero out padded channels


def handcrafted_features(X, M, thresh=0.01):
    """Classic hand-crafted EMG features per channel window -- inspired by
    the feature set used in KinEMbed (Gilardini et al., arXiv:2607.04820)
    and the broader sEMG feature literature (Phinyomark & Scheme, 2018).
    Fed to the model as an ADDITIONAL input alongside the CNN's learned
    embedding, not a replacement -- these are independent per-channel
    descriptive statistics with no cross-channel mixing, and cross-channel
    mixing (done by the CNN) is what's been shown to matter most here.

    X: (n, C, WIN) raw EMG windows (already per-file z-scored).
    M: (n, C) channel mask -- padded channels are zeroed out afterwards so
    they can't contribute spurious values (e.g. log-variance of an
    all-zero signal). Returns (n, C * N_HC_FEATS) float32.

    16 features per channel: RMS, waveform length, log-variance, zero
    crossings, slope sign changes, MAV, Willison amplitude, Hjorth
    mobility, Hjorth complexity, mean power frequency, median frequency,
    and 5 FFT-magnitude band sums (a simplified stand-in for the paper's
    per-band STFT magnitude, since our window is short enough that a full
    sub-windowed STFT adds little beyond a single FFT's band split)."""
    n, C = X.shape[0], X.shape[1]
    feats = _channel_features(X, M, thresh)
    return feats.reshape(n, C * N_HC_FEATS).astype(np.float32)


def channel_cosine_similarity(X, M, thresh=0.01):
    """Pairwise cosine similarity between each pair of channels' hand-crafted
    feature vectors -- an explicit cross-channel coordination/synchronization
    feature, complementary to handcrafted_features()'s per-channel-only
    statistics. Inspired by Ovadia, Segal & Rabin (2024, Sci Rep 14:4134),
    who found that concatenating a channel-similarity matrix with their
    ROCKET features measurably improved sEMG classification accuracy,
    especially when the channel/feature budget is limited (our case: 8-12
    channels) -- their features differ (random-kernel PPV, not hand-crafted
    stats) but the underlying idea (cross-channel correlation carries signal
    beyond per-channel stats) is architecture-agnostic.

    X: (n, C, WIN), M: (n, C). Returns (n, C*C) float32 -- padded channels'
    rows/columns are exactly 0 (zero feature vectors have 0 norm, guarded
    below), so they contribute no spurious similarity."""
    n, C, _ = X.shape
    feats = _channel_features(X, M, thresh)                    # (n, C, N_HC_FEATS)
    norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    sim = np.einsum("ncf,ndf->ncd", norm, norm)                 # (n, C, C)
    return sim.reshape(n, C * C).astype(np.float32)


class Encoder(nn.Module):
    """1D CNN over all EMG channels jointly (standard multi-channel conv) --
    channels are mixed together at every layer, so the model can learn
    cross-channel muscle-coordination patterns, which is most of the actual
    signal in EMG. Channel-count flexibility comes from zero-padding missing
    channels (see load_emg_file), not from processing channels independently
    -- an earlier per-channel-independent version of this class measurably
    hurt accuracy (~2.3x worse val loss on identical data) by losing the
    ability to learn cross-channel interactions.

    Four conv blocks / FEAT_DIM=256 -- sized for real cross-subject accuracy,
    not just a fast pipeline smoke test.

    Two attention-based pooling/encoding variants were TRIED here and
    REVERTED, both motivated by a real gap against published attention/
    transformer architectures on the same dataset/protocol (sMAPEN
    R^2=0.8163 vs. this project's 0.695 fair-averaged R^2, matched
    repetition-held-out benchmark, see benchmark_repetition_split.py):

    (1) Single-head attention pooling (weighted sum over time via a
    per-timestep Conv1d(FEAT_DIM,1,1) scorer + softmax, instead of plain
    average pooling): fair R^2 0.695->0.681, pooled R^2 0.608->0.576,
    best calib-val loss reached earlier and worse (0.2826 @ epoch 28 vs.
    0.2684 @ epoch 48).

    (2) A real small Transformer block over the conv stack's POOL_T=50-
    timestep output sequence (2 layers, 8 heads, dim_feedforward=512,
    dropout=0.1, learned positional embedding, then attention pooling
    over the transformer's OUTPUT): WORSE STILL -- fair R^2 0.674,
    pooled R^2 0.569, best calib-val 0.2904 @ epoch 27.

    Both real retrains, same data/subjects/protocol/seed as the baseline.
    The trend across (1)->(2) is MONOTONIC: more attention capacity ->
    worse held-out accuracy AND an earlier, worse validation-loss plateau
    -- the signature of added capacity overfitting a moderate-sized
    dataset (43k train windows) rather than a missing mechanism being
    supplied. Conclusion: the sMAPEN/RoFormer gap is NOT closed by
    bolting attention onto this encoder, at this data scale -- don't
    re-attempt attention-based pooling/encoding here without either much
    more training data or a regularization strategy specifically aimed
    at controlling it (their papers' larger reported gains may also rest
    on regularization, preprocessing, or evaluation details not
    replicated here, which is unverified, not asserted)."""
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
        """(B, MAX_CHANNELS, T) -> (B, FEAT_DIM). `mask` is accepted for
        call-site compatibility with train/predict but unused here --
        channels are already mixed by the conv layers, so there's nothing
        left to mask at pooling time."""
        return F.adaptive_avg_pool1d(self.net(x), 1).flatten(1)


class Decoder(nn.Module):
    """features -> reconstructed EMG across all MAX_CHANNELS, jointly (mirrors
    Encoder). Only used during SSL pretraining, discarded afterwards."""
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
    """encoder + hand-crafted-feature branch + cross-channel-similarity
    branch + regression head: (B, MAX_CHANNELS, WIN) EMG window +
    (B, MAX_CHANNELS*N_HC_FEATS) hand-crafted features +
    (B, MAX_CHANNELS*MAX_CHANNELS) channel-similarity matrix -> (B, N_KIN)
    kinematics. Each window is predicted independently. The three inputs
    are complementary, not redundant: the CNN embedding is cross-channel
    but learned/opaque, the hand-crafted features are per-channel with no
    cross-channel information, and the similarity branch is explicitly
    cross-channel but derived from the (already per-channel) hand-crafted
    features rather than the raw signal -- concatenated before the head."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.feat_proj = nn.Sequential(
            nn.Linear(MAX_CHANNELS * N_HC_FEATS, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, FEAT_HC_DIM), nn.ReLU(),
        )
        self.sim_proj = nn.Sequential(
            nn.Linear(MAX_CHANNELS * MAX_CHANNELS, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, FEAT_SIM_DIM), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM + FEAT_HC_DIM + FEAT_SIM_DIM, FEAT_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_KIN),
        )

    def forward(self, x, feat, sim, mask=None):
        emb = self.encoder.pooled(x, mask)
        femb = self.feat_proj(feat)
        semb = self.sim_proj(sim)
        return self.head(torch.cat([emb, femb, semb], dim=-1))


def _find_array(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k])
    return None


def _as_time_by_channels(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < a.shape[1]:      # heuristic: recordings are far longer than n_channels
        a = a.T
    return a


def load_emg_file(path, max_channels=MAX_CHANNELS, expected_fs=FS):
    """Best-effort loader for whatever EMG file you hand it -- 6 channels, 10
    channels, 12, doesn't matter, it works for any count up to max_channels.

    Ninapro-style .mat with the usual keys works as before. Beyond that it
    tries a few common key names, accepts .csv/.npy, and resamples to the
    rate the model was trained at. Returns (emg, glove_or_None, channel_mask,
    warnings) -- channel_mask marks which of the max_channels columns are
    real EMG vs. padding added only to fit the batch tensor.
    """
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

    emg = _as_time_by_channels(emg)
    if glove is not None:
        glove = _as_time_by_channels(glove)
        if glove.shape[1] != N_KIN:
            warns.append(f"ground truth has {glove.shape[1]} channels, model predicts {N_KIN} -- "
                         f"dropping it, predictions only")
            glove = None
    if glove is None and ext == ".mat":
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
        warns.append(f"{n_real} channels found -- the model pools only over those {n_real} real "
                     f"channels, the rest is just padding to fit the batch tensor")

    channel_mask = np.zeros(max_channels, dtype=np.float32)
    channel_mask[:n_real] = 1.0

    # per-file z-score of the REAL channels only: puts every subject/recording
    # on the same EMG amplitude scale before pooling, so one global mean/std
    # (and BatchNorm's running stats) don't have to paper over subject-to-
    # subject amplitude differences
    real = emg[:, :n_real]
    emg[:, :n_real] = (real - real.mean(0, keepdims=True)) / (real.std(0, keepdims=True) + 1e-8)

    return emg, glove, channel_mask, warns


def windows_from(path, label_lag=LABEL_LAG):
    """label_lag (samples) shifts which glove sample is treated as ground
    truth for a window -- 0 pins it to the window's last sample (assumes
    synchrony); label_lag > 0 aligns it to a sample that many samples LATER,
    modeling electromechanical delay (EMG precedes the resulting joint
    movement). Windows near the end of the recording that would need a
    glove sample past the end are dropped (trims X/M to match)."""
    emg, glove, channel_mask, warns = load_emg_file(path)
    n = len(emg)
    starts = list(range(0, n - WIN, HOP))
    X = np.asarray([emg[s:s + WIN].T for s in starts], np.float32)
    M = np.tile(channel_mask, (len(X), 1)) if len(X) else np.zeros((0, len(channel_mask)), np.float32)
    Y = None
    if glove is not None:
        valid = [s for s in starts if s + WIN - 1 + label_lag < len(glove)]
        if len(valid) < len(starts):
            X, M = X[:len(valid)], M[:len(valid)]
        Y = np.asarray([glove[s + WIN - 1 + label_lag] for s in valid], np.float32)
    return X, Y, M, warns


def leakage_safe_split(n, keep_frac, rng, gap, block=3):
    """Block-shuffle n windows into a majority "keep" group (~keep_frac of
    blocks, used as-is) and a minority "held_out" group (the rest, with
    anything within `gap` windows of a keep-group member additionally
    dropped) -- so held_out shares no raw EMG samples with keep, even
    though windows overlap 50% via WIN/HOP. Shared by train (train vs val)
    and finetune (calibrate vs eval)."""
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
    """Zero out a random contiguous time span per sample -- the reconstruction target."""
    xb_masked = xb.clone()
    T = xb.shape[-1]
    span = max(1, int(T * MASK_RATIO))
    for i in range(xb.shape[0]):
        s = np.random.randint(0, T - span + 1)
        xb_masked[i, :, s:s + span] = 0.0
    return xb_masked


def masked_mse(recon, target, chan_mask):
    """MSE over only the real channels (chan_mask == 1), so padded channels
    (always zero, trivially "reconstructed") don't dilute the loss signal."""
    err = (recon - target) ** 2 * chan_mask.unsqueeze(-1)
    return err.sum() / (chan_mask.sum() * target.shape[-1]).clamp_min(1e-8)


def load_epn612_emg(pkl_path, max_trials=2000, source_fs=200):
    """Extra UNLABELED EMG for SSL pretraining ONLY, from the public
    EMG-EPN612 dataset (Myo armband, 8 channels, discrete gesture trials).
    NOT usable for the supervised kinematics train/finetune -- this dataset
    has no continuous joint-angle ground truth, only per-trial gesture
    labels, which SSL doesn't need anyway (it never looks at labels).
    source_fs=200 is EMG-EPN612's documented native EMG sampling rate (the
    Myo armband's fixed rate) -- there's no rate header in this raw-array
    pickle to read it from directly, so this is a literature-based
    assumption, not detected from the file.

    Expected structure: a 4-item list [emg, imu, label_a, label_b], each a
    dict with 'training'/'testing' keys holding parallel per-trial lists;
    only emg (item 0) is used here. Returns (X, M) windowed the same way as
    the rest of the pipeline, zero-padded from 8 to MAX_CHANNELS."""
    import pickle
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    trials = obj[0]["training"] + obj[0]["testing"]
    rng = np.random.default_rng(0)
    if len(trials) > max_trials:
        idx = rng.choice(len(trials), max_trials, replace=False)
        trials = [trials[i] for i in idx]

    Xs = []
    for trial in trials:
        t = trial.astype(np.float32)
        t = (t - t.mean(0, keepdims=True)) / (t.std(0, keepdims=True) + 1e-8)
        n_new = max(1, int(round(len(t) * FS / source_fs)))
        t = resample(t, n_new, axis=0).astype(np.float32)
        n_real = t.shape[1]
        if n_real < MAX_CHANNELS:
            t = np.concatenate([t, np.zeros((len(t), MAX_CHANNELS - n_real), np.float32)], axis=1)
        if len(t) > WIN:
            Xs.append(np.asarray([t[s:s + WIN].T for s in range(0, len(t) - WIN, HOP)], np.float32))
    X = np.concatenate(Xs) if Xs else np.zeros((0, MAX_CHANNELS, WIN), np.float32)
    channel_mask = np.zeros(MAX_CHANNELS, np.float32)
    channel_mask[:8] = 1.0
    M = np.tile(channel_mask, (len(X), 1))
    return X, M


def pretrain_ssl(data_dir="data", extra_pkl=None, on_epoch=None):
    """on_epoch(ep, total, tr_loss, val_loss, marker), if given, is called
    after every epoch -- lets a caller (e.g. a UI) show live progress
    without needing to parse stdout.

    extra_pkl, if given, is a path to EMGEPN612.pkl (or a similarly-shaped
    pickle) -- its EMG is pooled in as EXTRA unlabeled windows for THIS SSL
    pass only. It's never used for supervised train/finetune, which need
    continuous kinematics labels this dataset doesn't have."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:MAX_FILES]
    if not all_files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    Xs, Ms = [], []
    n_labeled = n_channel_adapted = 0
    for f in all_files:
        emg, glove, channel_mask, warns = load_emg_file(f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        n_labeled += glove is not None
        n_channel_adapted += channel_mask.sum() < MAX_CHANNELS
        Xf = np.asarray([emg[s:s + WIN].T for s in range(0, len(emg) - WIN, HOP)], np.float32)
        Xs.append(Xf)
        Ms.append(np.tile(channel_mask, (len(Xf), 1)))
    X, M = np.concatenate(Xs), np.concatenate(Ms)

    if extra_pkl:
        Xe, Me = load_epn612_emg(extra_pkl)
        print(f"extra unlabeled EMG from {extra_pkl}: {len(Xe)} windows (8/{MAX_CHANNELS} real channels)")
        X, M = np.concatenate([X, Xe]), np.concatenate([M, Me])

    rng = np.random.default_rng(0)
    if len(X) > MAX_WINDOWS:
        keep = rng.choice(len(X), MAX_WINDOWS, replace=False)
        X, M = X[keep], M[keep]

    subs = sorted({subject_of(f) for f in all_files})
    print(f"files: {len(all_files)} total -- {n_labeled} with ground truth (irrelevant here, SSL is "
          f"unlabeled), {len(all_files) - n_labeled} without, {n_channel_adapted} channel-adapted")
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
            print(f"no val improvement for {PATIENCE} epochs, stopping early -- more data/epochs "
                  f"past this point is the diminishing-returns zone")
            break

    torch.save({"sd": best_state, "history": history}, "EMG-KinNet-SSL.pt")
    print(f"saved -> EMG-KinNet-SSL.pt (best val reconstruction loss {best_val:.4f})")
    print("use it with:\n  python quick_emg_to_kinematics.py train data EMG-KinNet-SSL.pt")
    return best_val


def train(data_dir="data", pretrained="", on_epoch=None):
    """on_epoch(ep, total, tr_loss, val_loss, marker), if given, is called
    after every epoch -- lets a caller (e.g. a UI) show live progress
    without needing to parse stdout.

    Trains on independent windows, pooled across files. Each file's own
    windows are split into train/val separately (via leakage_safe_split,
    gap=1 to cover the 50% WIN/HOP overlap between adjacent windows) before
    pooling, so overlapping windows never cross the train/val boundary and
    a window never crosses a file boundary."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    files = all_files[:MAX_FILES]
    if not files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    rng = np.random.default_rng(0)
    Xs, Ys, Ms, Fs, Ss, tr_parts, val_parts = [], [], [], [], [], [], []
    n_skipped = 0
    for f in files:
        Xf, Yf, Mf, warns = windows_from(f)
        for w in warns:
            print(f"  [{os.path.basename(f)}] {w}")
        if Yf is None:
            n_skipped += 1
            continue
        if len(Xf) == 0:
            continue
        base = sum(len(x) for x in Xs)
        tr_idx_f, val_idx_f = leakage_safe_split(len(Xf), 1 - VAL_FRAC, rng, gap=1)
        Xs.append(Xf); Ys.append(Yf); Ms.append(Mf)
        Fs.append(handcrafted_features(Xf, Mf)); Ss.append(channel_cosine_similarity(Xf, Mf))
        tr_parts.append(tr_idx_f + base); val_parts.append(val_idx_f + base)
    if not Xs:
        sys.exit(f"None of the {len(files)} file(s) in '{data_dir}' have usable ground-truth kinematics.")
    if n_skipped:
        print(f"files: {len(files)} total -- {n_skipped} skipped (no ground truth), "
              f"{len(files) - n_skipped} used for training")
    X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)
    Feat, Sim = np.concatenate(Fs), np.concatenate(Ss)
    tr_idx, val_idx = np.concatenate(tr_parts), np.concatenate(val_parts)
    if len(val_idx) == 0:
        # only happens with very sparse/short data_dir -- borrow one so training can still run
        val_idx, tr_idx = tr_idx[-1:], tr_idx[:-1]

    max_val = max(1, int(MAX_WINDOWS * VAL_FRAC))
    max_tr = max(1, MAX_WINDOWS - max_val)
    if len(tr_idx) > max_tr:
        tr_idx = rng.choice(tr_idx, max_tr, replace=False)
    if len(val_idx) > max_val:
        val_idx = rng.choice(val_idx, max_val, replace=False)

    subs = sorted({subject_of(f) for f in files})
    print(f"{len(files)} file(s), subjects {subs}, {len(tr_idx)} train / {len(val_idx)} val windows, device={dev}")

    # normalisation stats from TRAIN split only (saved with the model so predict/ reuses them exactly)
    xm = X[tr_idx].mean((0, 2), keepdims=True)
    xs = X[tr_idx].std((0, 2), keepdims=True) + 1e-8
    ym = Y[tr_idx].mean(0, keepdims=True)
    ys = Y[tr_idx].std(0, keepdims=True) + 1e-8
    fm = Feat[tr_idx].mean(0, keepdims=True)
    fs = Feat[tr_idx].std(0, keepdims=True) + 1e-8
    sm = Sim[tr_idx].mean(0, keepdims=True)
    ss = Sim[tr_idx].std(0, keepdims=True) + 1e-8
    Xn, Yn, Fn, Simn = (X - xm) / xs, (Y - ym) / ys, (Feat - fm) / fs, (Sim - sm) / ss

    def make_dl(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xn[idx].astype(np.float32)),
                                        torch.from_numpy(Yn[idx].astype(np.float32)),
                                        torch.from_numpy(M[idx].astype(np.float32)),
                                        torch.from_numpy(Fn[idx].astype(np.float32)),
                                        torch.from_numpy(Simn[idx].astype(np.float32))),
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
        for xb, yb, mb, fb, sb in tr_dl:
            xb, yb, mb, fb, sb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev), sb.to(dev)
            if NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * NOISE_STD
            out = net(xb, fb, sb, mb)
            loss = lf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        net.eval()
        with torch.no_grad():
            val_loss = sum(lf(net(xb.to(dev), fb.to(dev), sb.to(dev), mb.to(dev)), yb.to(dev)).item()
                           for xb, yb, mb, fb, sb in val_dl) / len(val_dl)
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

    torch.save({"sd": best_state, "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs,
               "sm": sm, "ss": ss, "history": history}, "EMG-KinNet.pt")
    print(f"saved -> EMG-KinNet.pt (best val loss {best_val:.4f})")
    print("trained on all available subjects -- for an honest per-subject accuracy check, use:\n"
          "  python quick_emg_to_kinematics.py finetune FILE.mat")
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


def predict(path, model="EMG-KinNet.pt"):
    ck = torch.load(model, map_location=dev, weights_only=False)
    net = Net().to(dev); net.load_state_dict(ck["sd"]); net.eval()

    X, Y, M, warns = windows_from(path)
    for w in warns:
        print(f"note: {w}")

    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    fm, fs = ck["fm"], ck["fs"]
    sm, ss = ck["sm"], ck["ss"]
    Xn = ((X - xm) / xs).astype(np.float32)
    Fn = ((handcrafted_features(X, M) - fm) / fs).astype(np.float32)
    Sn = ((channel_cosine_similarity(X, M) - sm) / ss).astype(np.float32)

    import time; t0 = time.time()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), BATCH):
            xb = torch.from_numpy(Xn[i:i + BATCH]).to(dev)
            mb = torch.from_numpy(M[i:i + BATCH].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fn[i:i + BATCH]).to(dev)
            sb = torch.from_numpy(Sn[i:i + BATCH]).to(dev)
            out.append(net(xb, fb, sb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym     # back to real units
    print(f"predicted {len(pred)} timesteps in {time.time()-t0:.2f}s")

    np.savetxt("predicted_kinematics.csv", pred, delimiter=",", fmt="%.4f")
    print("saved -> predicted_kinematics.csv")
    print("\nfirst 5 timesteps (22 joint values each):")
    for r in pred[:5]:
        print("  " + "  ".join(f"{v:7.2f}" for v in r))
    if Y is None:
        print("\nno ground truth available for this file -- predictions only, no RMSE/R^2")
        return
    report_metrics(pred, Y)


def finetune(path, base_model="EMG-KinNet.pt", epochs=60, lr=1e-4, out_name=None):
    """Calibrate an already-trained model to ONE specific subject's own
    data -- the realistic deployment scenario for a prosthesis fitted to a
    single person, as opposed to the harder cold-start cross-subject case
    `train` tests. Splits `path`'s own WINDOWS into a calibration set
    (fine-tuned on) and a held-out set (evaluated on), so the reported
    accuracy is still honest -- not just re-testing on what it just saw.

    The calibration set is further split internally into calib-train /
    calib-val purely to pick the best epoch via early stopping (previously
    this ran a fixed 15 epochs blind, with no way to tell if that under- or
    over-fit). eval_idx itself is NEVER used for model selection, only for
    the final reported number, so that number stays an honest, untouched
    held-out estimate.
    """
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

    # windows overlap 50% (WIN=400, HOP=200), so shuffle small GROUPS of
    # windows (both sets still sample the whole recording's movement
    # variety) and drop any eval window within `gap` of a calibration one --
    # zero raw-sample overlap between calibrate and eval.
    rng = np.random.default_rng(0)
    calib_idx, eval_idx = leakage_safe_split(n, 0.7, rng, gap=1)
    if len(eval_idx) == 0:
        sys.exit(f"{path}: no eval windows left after the leakage gap -- try a longer recording.")

    # carve an early-stopping val slice OUT OF the calibration windows (same
    # leakage-safe logic) -- eval_idx is deliberately left untouched by this
    # split so it never influences which epoch's weights get chosen
    ct_pos, cv_pos = leakage_safe_split(len(calib_idx), 1 - VAL_FRAC, rng, gap=1)
    if len(cv_pos) == 0:
        ct_pos, cv_pos = np.arange(len(calib_idx) - 1), np.arange(len(calib_idx) - 1, len(calib_idx))
    calib_tr_idx, calib_val_idx = calib_idx[ct_pos], calib_idx[cv_pos]

    # reuse the base model's own normalisation stats -- no need to re-derive
    # them from a single subject's small sample
    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    fm, fs = ck["fm"], ck["fs"]
    sm, ss = ck["sm"], ck["ss"]
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    Fn = (handcrafted_features(X, M) - fm) / fs
    Sn = (channel_cosine_similarity(X, M) - sm) / ss

    dl = DataLoader(TensorDataset(torch.from_numpy(Xn[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(Yn[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(M[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(Fn[calib_tr_idx].astype(np.float32)),
                                  torch.from_numpy(Sn[calib_tr_idx].astype(np.float32))),
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
            sb = torch.from_numpy(Sn[calib_val_idx].astype(np.float32)).to(dev)
            return lf(net(xb, fb, sb, mb), yb).item()

    print(f"calibrating on {len(calib_tr_idx)} windows ({len(calib_val_idx)} held out for early "
          f"stopping) from {path}, evaluating on {len(eval_idx)} held-out windows")
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for ep in range(int(epochs)):
        net.train()
        tot = 0.0
        for xb, yb, mb, fb, sb in dl:
            xb, yb, mb, fb, sb = xb.to(dev), yb.to(dev), mb.to(dev), fb.to(dev), sb.to(dev)
            out = net(xb, fb, sb, mb)
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
        Xe, Fe, Se = Xn[eval_idx], Fn[eval_idx], Sn[eval_idx]
        for i in range(0, len(Xe), BATCH):
            xb = torch.from_numpy(Xe[i:i + BATCH].astype(np.float32)).to(dev)
            mb = torch.from_numpy(M[eval_idx][i:i + BATCH].astype(np.float32)).to(dev)
            fb = torch.from_numpy(Fe[i:i + BATCH].astype(np.float32)).to(dev)
            sb = torch.from_numpy(Se[i:i + BATCH].astype(np.float32)).to(dev)
            out.append(net(xb, fb, sb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym
    print("\nafter calibration, on held-out windows from the SAME subject:")
    report_metrics(pred, Y[eval_idx])

    out_name = out_name or f"EMG-KinNet_finetuned_{os.path.splitext(os.path.basename(path))[0]}.pt"
    torch.save({"sd": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys, "fm": fm, "fs": fs,
               "sm": sm, "ss": ss, "history": history}, out_name)
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
