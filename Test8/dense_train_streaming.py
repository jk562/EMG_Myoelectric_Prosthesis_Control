"""
Streaming (lazy-windowed) training at DENSE_HOP=1 (0.5ms @ 2000Hz), matching
sMAPEN/RoFormer's exact reported window step -- the extreme case
dense_train_eval.py's own docstring flagged as infeasible with that
script's "materialize-then-cap" approach (a full DB2 file has ~1.8M
hop=1 candidate positions; capping per file the way that script does
would either explode memory or badly undersample most of the file).

The fix: never materialize the windowed dataset at all. Each file's RAW
(un-windowed) EMG + glove arrays are kept in memory once (~a few GB
total for the whole 39-file pool -- verified empirically below, not
just estimated, since the earlier per-file-cap fix was based on an
estimate that undershot reality once). A PyTorch Dataset holds only
(file_idx, start_sample) index pairs -- cheap even at hop=1 density
(tens of millions of int entries, not float window arrays) -- and
slices the actual WIN=400-sample window lazily in __getitem__, on
demand, per item. handcrafted_features()/channel_cosine_similarity()
(the two functions whose FFT/einsum intermediates caused this project's
very first OOM, see benchmark_repetition_split.py's own history) are
computed per BATCH inside the training loop, never on the full dataset,
so peak memory stays bounded by BATCH regardless of how many total
candidate windows exist.

A fixed number of windows is still SAMPLED per epoch (RANDOM_PER_EPOCH)
-- not literally every hop=1 window, which would make each epoch
enormous -- but now sampled from the FULL hop=1-dense candidate pool,
not a per-file-capped, narrow-time-range subset like dense_train_eval.py
had to use for hop=20. This is the direct test of whether training
density gains continue all the way to sMAPEN's own reported extreme, or
plateau before it (dense_train_eval.py only established the gain exists
at hop=20, not that it continues to hop=1).

  python dense_train_streaming.py [data_dir]
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

import quick_emg_to_kinematics as qk
from benchmark_repetition_split import TRAIN_REPS, TEST_REPS

DENSE_HOP = 1                 # samples @ 2000Hz = 0.5ms -- sMAPEN/RoFormer's exact step
FIXED_TRAIN_SAMPLE = 60000    # drawn ONCE from the hop=1 candidate pool, reused every epoch --
                              # matches qk.MAX_WINDOWS, keeps this comparable to every other
                              # run in this project despite the far denser source pool
RANDOM_PER_EPOCH_VAL = 4000
EVAL_TARGET_TOTAL = 21786     # matches hop_sweep.py's test-set size, for a directly comparable number


class LazyWindowDataset(Dataset):
    """indices: (n, 2) array of (file_idx, start_sample). Slices the WIN-
    sample window and its label lazily, per item -- never materializes
    more than one window at a time per worker."""
    def __init__(self, emg_list, glove_list, mask_list, indices):
        self.emg_list, self.glove_list, self.mask_list = emg_list, glove_list, mask_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        fidx, s = self.indices[i]
        emg, glove, mask = self.emg_list[fidx], self.glove_list[fidx], self.mask_list[fidx]
        x = emg[s:s + qk.WIN].T.astype(np.float32)
        y = glove[s + qk.WIN - 1].astype(np.float32)
        return x, y, mask.astype(np.float32)


def load_pool(files, hop):
    """Loads RAW (un-windowed) emg/glove/mask per file (kept in memory once),
    plus (file_idx, start) candidate index arrays at `hop`, split by
    TRAIN_REPS/TEST_REPS via each file's rerepetition/repetition field."""
    emg_list, glove_list, mask_list = [], [], []
    train_idx_parts, test_idx_parts = [], []
    subs_used = []
    for f in files:
        m = loadmat(f)
        rep_key = "rerepetition" if "rerepetition" in m else ("repetition" if "repetition" in m else None)
        if rep_key is None:
            continue
        reps = m[rep_key].astype(int).ravel()
        emg, glove, mask, warns = qk.load_emg_file(f)
        if glove is None:
            continue
        n = len(emg)
        starts = np.arange(0, n - qk.WIN, hop)
        rep_at_start = reps[starts]
        tr_starts = starts[np.isin(rep_at_start, list(TRAIN_REPS))]
        te_starts = starts[np.isin(rep_at_start, list(TEST_REPS))]
        if len(tr_starts) == 0 or len(te_starts) == 0:
            continue

        fidx = len(emg_list)   # position WITHIN emg_list, not the enumerate() position over
                               # `files` -- some files get skipped above, so those would silently
                               # misalign (or index out of range) otherwise
        emg_list.append(emg); glove_list.append(glove); mask_list.append(mask)
        train_idx_parts.append(np.stack([np.full(len(tr_starts), fidx, dtype=np.int32), tr_starts], axis=1))
        test_idx_parts.append(np.stack([np.full(len(te_starts), fidx, dtype=np.int32), te_starts], axis=1))
        subs_used.append(qk.subject_of(f))

    train_idx = np.concatenate(train_idx_parts).astype(np.int64)
    test_idx = np.concatenate(test_idx_parts).astype(np.int64)
    return emg_list, glove_list, mask_list, train_idx, test_idx, subs_used


def collate_features(batch, xm, xs, fm, fs, sm, ss):
    """Turns a batch of raw (x, y, mask) triples into normalised tensors
    PLUS handcrafted/cosine-similarity features -- computed HERE, per
    batch, not on the full dataset, so this is the only place those two
    functions' memory-hungry intermediates ever get materialized, and
    only at BATCH size."""
    X = np.stack([b[0] for b in batch])
    Y = np.stack([b[1] for b in batch])
    M = np.stack([b[2] for b in batch])
    Feat = qk.handcrafted_features(X, M)
    Sim = qk.channel_cosine_similarity(X, M)
    Xn = ((X - xm) / xs).astype(np.float32)
    Fn = ((Feat - fm) / fs).astype(np.float32)
    Sn = ((Sim - sm) / ss).astype(np.float32)
    return (torch.from_numpy(Xn), torch.from_numpy(Y), torch.from_numpy(M.astype(np.float32)),
            torch.from_numpy(Fn), torch.from_numpy(Sn))


def main(data_dir="data"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:qk.MAX_FILES]
    if not files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    print(f"loading raw (un-windowed) EMG/glove for all files, building hop={DENSE_HOP} "
          f"({DENSE_HOP/2}ms step) candidate index pool (indices only, not window data)...")
    emg_list, glove_list, mask_list, train_idx, test_idx, subs = load_pool(files, DENSE_HOP)
    print(f"subjects used: {sorted(set(subs))}")
    print(f"candidate pool: {len(train_idx)} train-rep windows, {len(test_idx)} test-rep windows "
          f"(at hop={DENSE_HOP} -- NOT materialized as float arrays, just (file,start) index pairs)")

    import psutil
    proc = psutil.Process()
    print(f"resident memory after loading raw per-file arrays: {proc.memory_info().rss / 1e9:.2f} GB")

    rng = np.random.default_rng(0)
    # small held-out slice of train_idx for early stopping (leakage-safe not needed here in the
    # same block-shuffle sense -- at hop=1 essentially every window overlaps its neighbours, so
    # instead validation windows are drawn from a DISJOINT set of (file, start-range) blocks
    n_val_files = max(1, len(emg_list) // 8)
    val_file_ids = set(rng.choice(len(emg_list), n_val_files, replace=False).tolist())
    is_val = np.isin(train_idx[:, 0], list(val_file_ids))
    val_pool, tr_pool = train_idx[is_val], train_idx[~is_val]
    print(f"{len(tr_pool)} train-file candidates ({len(emg_list) - n_val_files} files), "
          f"{len(val_pool)} val-file candidates ({n_val_files} files held out of training entirely, "
          f"file-level split so val is never contaminated by adjacent overlapping windows)")

    # normalisation stats from a large random SAMPLE of the train pool (don't need every window)
    samp_idx = tr_pool[rng.choice(len(tr_pool), min(20000, len(tr_pool)), replace=False)]
    Xs_samp = np.stack([emg_list[fidx][s:s + qk.WIN].T for fidx, s in samp_idx]).astype(np.float32)
    Ms_samp = np.stack([mask_list[fidx] for fidx, s in samp_idx]).astype(np.float32)
    Ys_samp = np.stack([glove_list[fidx][s + qk.WIN - 1] for fidx, s in samp_idx]).astype(np.float32)
    Fs_samp = qk.handcrafted_features(Xs_samp, Ms_samp)
    Ss_samp = qk.channel_cosine_similarity(Xs_samp, Ms_samp)
    xm, xs = Xs_samp.mean((0, 2), keepdims=True), Xs_samp.std((0, 2), keepdims=True) + 1e-8
    ym, ys = Ys_samp.mean(0, keepdims=True), Ys_samp.std(0, keepdims=True) + 1e-8
    fm, fs = Fs_samp.mean(0, keepdims=True), Fs_samp.std(0, keepdims=True) + 1e-8
    sm, ss = Ss_samp.mean(0, keepdims=True), Ss_samp.std(0, keepdims=True) + 1e-8
    del Xs_samp, Ms_samp, Fs_samp, Ss_samp

    net = qk.Net().to(qk.dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=qk.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()

    def make_loader(pool, n_per_epoch, shuffle):
        idx = pool if len(pool) <= n_per_epoch else pool[rng.choice(len(pool), n_per_epoch, replace=False)]
        ds = LazyWindowDataset(emg_list, glove_list, mask_list, idx)
        return DataLoader(ds, batch_size=qk.BATCH, shuffle=shuffle,
                          collate_fn=lambda b: collate_features(b, xm, xs, fm, fs, sm, ss))

    # Both loaders built ONCE, outside the epoch loop -- FIXED sets reused every epoch, matching
    # every other training script in this project (train(), benchmark_repetition_split.py,
    # dense_train_eval.py). Two earlier versions of this script got this wrong in two different
    # ways: first a fresh-every-epoch VAL set caused a spurious early stop from validation noise
    # (fixed above in a prior commit); then, less obviously, a fresh-every-epoch TRAIN set turned
    # out to matter much more -- training loss plateaued around 0.39 (vs. ~0.15-0.22 in every
    # other successful run) because the model was seeing ~720k total distinct windows across 12
    # epochs with almost no repeats, a fundamentally slower-converging regime than "fit the same
    # fixed sample repeatedly," which is what every other script does and what made them converge
    # in the first place. Fixed by drawing ONE random sample from the huge hop=1 candidate pool
    # ONCE, sized to match every other script's MAX_WINDOWS-scale cap, then reusing it every
    # epoch -- isolating whether POOL DENSITY (hop=1 vs hop=20) helps, holding the training
    # regimen itself identical to what's already proven to work.
    tr_idx_fixed = tr_pool[rng.choice(len(tr_pool), min(FIXED_TRAIN_SAMPLE, len(tr_pool)), replace=False)]
    tr_ds_fixed = LazyWindowDataset(emg_list, glove_list, mask_list, tr_idx_fixed)
    val_dl = make_loader(val_pool, RANDOM_PER_EPOCH_VAL, False)
    print(f"fixed training sample: {len(tr_idx_fixed)} windows (drawn once from the hop={DENSE_HOP} "
          f"candidate pool, reused every epoch)")

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for ep in range(qk.EPOCHS):
        net.train()
        tr_dl = DataLoader(tr_ds_fixed, batch_size=qk.BATCH, shuffle=True,
                           collate_fn=lambda b: collate_features(b, xm, xs, fm, fs, sm, ss))
        tot = 0.0
        for xb, yb, mb, fb, sb in tr_dl:
            xb, yb, mb, fb, sb = xb.to(qk.dev), yb.to(qk.dev), mb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev)
            ybn = (yb - torch.from_numpy(ym).to(qk.dev)) / torch.from_numpy(ys).to(qk.dev)
            if qk.NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * qk.NOISE_STD
            out = net(xb, fb, sb, mb)
            loss = lf(out, ybn.float())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / len(tr_dl)

        net.eval()
        with torch.no_grad():
            val_tot = 0.0
            for xb, yb, mb, fb, sb in val_dl:
                xb, yb, mb, fb, sb = xb.to(qk.dev), yb.to(qk.dev), mb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev)
                ybn = (yb - torch.from_numpy(ym).to(qk.dev)) / torch.from_numpy(ys).to(qk.dev)
                val_tot += lf(net(xb, fb, sb, mb), ybn.float()).item()
            val_loss = val_tot / len(val_dl)
        sched.step(val_loss)

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1
        print(f"epoch {ep+1}/{qk.EPOCHS}  train {tr_loss:.4f}  val {val_loss:.4f}{marker}  "
              f"mem={proc.memory_info().rss/1e9:.2f}GB")
        if bad_epochs >= qk.PATIENCE:
            print(f"no val improvement for {qk.PATIENCE} epochs, stopping early")
            break

    net.load_state_dict(best_state)
    net.eval()

    eval_idx = test_idx if len(test_idx) <= EVAL_TARGET_TOTAL else \
        test_idx[rng.choice(len(test_idx), EVAL_TARGET_TOTAL, replace=False)]
    eval_ds = LazyWindowDataset(emg_list, glove_list, mask_list, eval_idx)
    eval_dl = DataLoader(eval_ds, batch_size=qk.BATCH, shuffle=False,
                         collate_fn=lambda b: collate_features(b, xm, xs, fm, fs, sm, ss))
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, mb, fb, sb in eval_dl:
            xb, mb, fb, sb = xb.to(qk.dev), mb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev)
            out = net(xb, fb, sb, mb).cpu().numpy()
            preds.append(out * ys + ym)
            trues.append(yb.numpy())
    pred, Y = np.concatenate(preds), np.concatenate(trues)

    print(f"\n=== streaming dense-trained (hop={DENSE_HOP}, {DENSE_HOP/2}ms step -- sMAPEN's exact "
          f"protocol), {len(eval_idx)} held-out-repetition eval windows ===")
    mean_r2 = qk.report_metrics(pred, Y)
    print(f"\nfor comparison: sMAPEN reports R^2=0.8163 at this exact window density. "
          f"hop_sweep.py's EVAL-ONLY hop=1 point (coarse hop=200-trained model, just evaluated "
          f"densely) got fair R^2=0.840. dense_train_eval.py's hop=20 TRAINED comparison showed "
          f"training density adds a real gain beyond eval-only density (0.706->0.784).")
    return mean_r2


if __name__ == "__main__":
    main(*sys.argv[1:])
