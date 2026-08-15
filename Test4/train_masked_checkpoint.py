"""
Standalone script -- reproduces just the Masked-Reconstruction SSL pipeline from Test2
(pretrain -> fine-tune) and saves a standalone checkpoint, in the same format Test3 uses
for its Contrastive checkpoint. This exists because Test2's trained model only lived inside
that notebook's kernel process (already exited), so there is nothing on disk to load from --
re-running the relevant slice of training is the only way to get a checkpoint, and this
script only does the Masked slice (skips Scratch, Contrastive, label-efficiency, and
robustness sections that Test2 also runs), to avoid repeating ~30+ minutes of unrelated work.

Same hyperparameters as the Test2 run that produced the numbers already reported
(MASKED_EPOCHS=100, FT_EPOCHS=80) -- not a fresh tune, just an exact reproduction for the
purpose of getting a saved checkpoint.
"""
import os
import glob
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

DATA_DIR = '/Users/kailashjram/Desktop/MSC FINAL PROJECT/Subject Data'
FS, BANDPASS_LOW, BANDPASS_HIGH = 2000, 20, 450
WINDOW_MS, STRIDE_MS = 200, 50
N_CHANNELS, N_JOINTS = 12, 22
WINDOW_SIZE, STRIDE = int(FS * WINDOW_MS / 1000), int(FS * STRIDE_MS / 1000)
DB2_TRAIN_REPS, DB2_TEST_REPS = [1, 2, 3, 4], [5, 6]
FINE_TUNE_SUBJECT = 13
MAX_WINDOWS_PER_FILE = 600
MASKED_EPOCHS, MASKED_BATCH, MASKED_LR = 100, 128, 1e-3
FT_EPOCHS = 80
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'masked_ssl_model.pt')

DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')


def bandpass_filter(emg, fs=FS, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, emg, axis=0)

def subject_id_from_path(path):
    m = re.search(r'S(\d+)[_]', os.path.basename(path))
    return int(m.group(1)) if m else None


class EMGEncoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)

class EMGRegressor(nn.Module):
    def __init__(self, encoder, n_joints):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_joints))
    def forward(self, x):
        return self.head(self.pool(self.encoder(x)).squeeze(-1))

class EMGDecoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, n_channels, kernel_size=7, padding=3))
    def forward(self, feat): return self.net(feat)

class EMGWindowDataset(Dataset):
    def __init__(self, X, y):
        Xt = np.transpose(X, (0, 2, 1)).astype(np.float32)
        self.X = torch.from_numpy(Xt)
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MaskedEMGDataset(Dataset):
    def __init__(self, X, patch_size=20, mask_ratio=0.4, seed=1):
        self.X, self.patch_size, self.mask_ratio = X, patch_size, mask_ratio
        self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        w = self.X[idx].astype(np.float32)
        T = w.shape[0]
        n_patches = T // self.patch_size
        n_mask = max(1, int(n_patches * self.mask_ratio))
        mask_patches = self.rng.choice(n_patches, n_mask, replace=False)
        mask = np.zeros(T, dtype=np.float32)
        masked = w.copy()
        for p in mask_patches:
            s, e = p * self.patch_size, (p + 1) * self.patch_size
            masked[s:e] = 0.0
            mask[s:e] = 1.0
        return torch.from_numpy(masked.T.copy()), torch.from_numpy(w.T.copy()), torch.from_numpy(mask)

def masked_mse_loss(recon, target, mask):
    mask = mask.unsqueeze(1)
    return ((recon - target) ** 2 * mask).sum() / (mask.sum() * target.shape[1] + 1e-8)

def process_file_unlabeled(mat_path, max_windows=600, exclude_subject=FINE_TUNE_SUBJECT,
                            exclude_reps=DB2_TEST_REPS, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    data = loadmat(mat_path)
    if 'emg' not in data or data['emg'].shape[1] != N_CHANNELS:
        return None
    emg = bandpass_filter(data['emg']).astype(np.float32)
    repetition = data['rerepetition'].flatten() if 'rerepetition' in data else None
    subj = subject_id_from_path(mat_path)
    filter_reps = (exclude_subject is not None and subj == exclude_subject and repetition is not None)
    starts = list(range(0, len(emg) - WINDOW_SIZE + 1, STRIDE))
    if filter_reps:
        kept = []
        for s in starts:
            e = s + WINDOW_SIZE
            rep = np.bincount(repetition[s:e].astype(int)).argmax()
            if rep not in exclude_reps: kept.append(s)
        starts = kept
    if not starts: return None
    if max_windows and len(starts) > max_windows:
        idx = rng.choice(len(starts), max_windows, replace=False)
        starts = [starts[i] for i in idx]
    return np.stack([emg[s:s + WINDOW_SIZE] for s in starts])

def process_file_raw(mat_path):
    data = loadmat(mat_path)
    if 'glove' not in data: return None
    emg = bandpass_filter(data['emg']).astype(np.float32)
    glove = data['glove'].astype(np.float32)
    repetition = data['rerepetition'].flatten()
    X, y, reps = [], [], []
    for start in range(0, len(emg) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        X.append(emg[start:end]); y.append(glove[end - 1])
        reps.append(np.bincount(repetition[start:end].astype(int)).argmax())
    return np.array(X), np.array(y), np.array(reps)

def fit_normalizer_3d(X):
    mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    std = X.reshape(-1, X.shape[2]).std(axis=0)
    return mean, np.where(std < 1e-9, 1e-9, std)

def fit_normalizer(arr):
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    return mean, np.where(std < 1e-9, 1e-9, std)


def main():
    print(f'Device: {DEVICE}')
    all_files = (sorted(glob.glob(os.path.join(DATA_DIR, '*E1*.mat'))) +
                 sorted(glob.glob(os.path.join(DATA_DIR, '*E2*.mat'))) +
                 sorted(glob.glob(os.path.join(DATA_DIR, '*E3*.mat'))))

    print('Building pretraining pool...')
    t0 = time.time()
    rng = np.random.default_rng(0)
    pool_parts = [w for w in (process_file_unlabeled(f, MAX_WINDOWS_PER_FILE, rng=rng) for f in all_files) if w is not None]
    X_pretrain = np.vstack(pool_parts).astype(np.float32)
    print(f'  {X_pretrain.shape[0]} windows ({time.time() - t0:.0f}s)')

    print('Loading fine-tuning data...')
    finetune_files = [os.path.join(DATA_DIR, f'S{FINE_TUNE_SUBJECT}_E1_A1.mat'),
                       os.path.join(DATA_DIR, f'S{FINE_TUNE_SUBJECT}_E2_A1.mat')]
    X_parts, y_parts, rep_parts = [], [], []
    for f in finetune_files:
        result = process_file_raw(f)
        X_parts.append(result[0]); y_parts.append(result[1]); rep_parts.append(result[2])
    X_ft, y_ft, reps_ft = np.vstack(X_parts), np.vstack(y_parts), np.concatenate(rep_parts)
    train_mask, test_mask = np.isin(reps_ft, DB2_TRAIN_REPS), np.isin(reps_ft, DB2_TEST_REPS)

    xm, xs = fit_normalizer_3d(X_ft[train_mask])
    ym, ys = fit_normalizer(y_ft[train_mask])
    X_train = (X_ft[train_mask] - xm) / xs
    y_train = (y_ft[train_mask] - ym) / ys
    X_pretrain_norm = ((X_pretrain - xm) / xs).astype(np.float32)
    print(f'  Train: {X_train.shape[0]} windows')

    print(f'Masked-reconstruction pretraining ({MASKED_EPOCHS} epochs)...')
    masked_ds = MaskedEMGDataset(X_pretrain_norm, patch_size=20, mask_ratio=0.4, seed=1)
    masked_loader = DataLoader(masked_ds, batch_size=MASKED_BATCH, shuffle=True)
    encoder_masked = EMGEncoder(N_CHANNELS).to(DEVICE)
    decoder = EMGDecoder(N_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder_masked.parameters()) + list(decoder.parameters()), lr=MASKED_LR)
    t0 = time.time()
    for epoch in range(MASKED_EPOCHS):
        total_loss = 0.0
        for masked_x, target_x, mask in masked_loader:
            masked_x, target_x, mask = masked_x.to(DEVICE), target_x.to(DEVICE), mask.to(DEVICE)
            recon = decoder(encoder_masked(masked_x))
            loss = masked_mse_loss(recon, target_x, mask)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * masked_x.size(0)
        if epoch % 10 == 0 or epoch == MASKED_EPOCHS - 1:
            print(f'  epoch {epoch:>3}  loss {total_loss / len(masked_ds):.4f}  ({time.time() - t0:.0f}s)')
    masked_state = {k: v.clone() for k, v in encoder_masked.state_dict().items()}

    print(f'Fine-tuning ({FT_EPOCHS} epochs)...')
    enc = EMGEncoder(N_CHANNELS)
    enc.load_state_dict(masked_state)
    model = EMGRegressor(enc, N_JOINTS).to(DEVICE)
    train_ds = EMGWindowDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for epoch in range(FT_EPOCHS):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        epoch_loss = total_loss / len(train_ds)
        scheduler.step(epoch_loss)
        if epoch % 10 == 0 or epoch == FT_EPOCHS - 1:
            print(f'  epoch {epoch:>3}  loss {epoch_loss:.4f}  ({time.time() - t0:.0f}s)')

    torch.save({
        'model_state_dict': model.state_dict(),
        'xm': xm, 'xs': xs, 'ym': ym, 'ys': ys,
        'n_channels': N_CHANNELS, 'n_joints': N_JOINTS,
        'window_ms': WINDOW_MS, 'stride_ms': STRIDE_MS, 'fs': FS,
        'fine_tune_subject': FINE_TUNE_SUBJECT,
    }, CHECKPOINT_PATH)
    print(f'Saved: {CHECKPOINT_PATH}')


if __name__ == '__main__':
    main()
