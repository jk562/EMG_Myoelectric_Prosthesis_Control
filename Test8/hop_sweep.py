"""
Step-size (window-overlap) sweep -- tests whether R^2 measured on the SAME
trained, frozen model changes just from evaluating on more densely
overlapping test windows, holding everything else (model weights, held-
out repetitions, files, subjects) fixed.

Motivated by a real, verified difference found in sMAPEN and RoFormer's
own methods text: both use a 100ms window / 0.5ms (1-sample @ 2000Hz)
step for their evaluation -- ~99.5% overlap -- vs. this project's 200ms
window / 100ms (200-sample) step (50% overlap), a ~200x difference in
window density. Neither paper's own text mentions any preprocessing
filter or data augmentation this project lacks, so window density is the
most concrete, verified difference found so far (see
benchmark_repetition_split.py and quick_emg_to_kinematics.py's Encoder
docstring for the three prior architecture-side attempts, all reverted).

This script isolates the EVALUATION-side effect of window density from
any architecture/training difference: loads an ALREADY-TRAINED, frozen
checkpoint (no retraining happens here) and re-evaluates it on the SAME
held-out repetitions (5,6), re-windowed at several hop sizes from the
project's own 200-sample baseline down toward the papers' 1-sample
extreme. At every hop level, the TOTAL evaluated window count is
subsampled down to a fixed target (matching the original 200-hop test
set size) -- so only how DENSELY OVERLAPPING the evaluated windows are
changes across the sweep, not how much test data is looked at. If R^2
climbs toward sMAPEN's 0.8163 purely from denser (near-duplicate)
evaluation windows, that's evidence the published gap is substantially
an evaluation-density artifact (autocorrelated errors between near-
identical adjacent windows), not a real difference in the model's
information-theoretic accuracy.

A literal hop=1 replication of their exact protocol is NOT attempted for
every file at full scale (a ~15-minute DB2 recording has ~1.8M possible
hop=1 starting positions) -- PER_FILE_CAND_CAP bounds how many candidate
windows are drawn per file at each hop level before final subsampling,
so this stays memory-safe (candidates are capped BEFORE building any
window array, and handcrafted_features()/channel_cosine_similarity() --
the two functions whose FFT/einsum intermediates caused the original
benchmark script's OOM, see its own history -- are computed only on the
final subsampled set, never on the full candidate pool).

  python hop_sweep.py CHECKPOINT.pt [data_dir]
"""
import os
import sys
import glob
import numpy as np
import torch
from scipy.io import loadmat

import quick_emg_to_kinematics as qk
from benchmark_repetition_split import TEST_REPS

HOP_LEVELS = [200, 100, 50, 20, 10, 4, 1]   # samples @ 2000Hz -- 200 is this project's baseline
TARGET_TOTAL = 21786    # fixed evaluated-window count at every hop level (matches the original
                        # benchmark's test-set size) -- isolates density as the only variable
PER_FILE_CAND_CAP = 4000


def test_windows_at_hop(path, hop, rng):
    """Windows + labels for just the TEST repetitions (5,6) of one file, at
    a given hop -- built directly (qk.windows_from is hardcoded to
    qk.HOP), capped per file BEFORE any window array is materialized.
    Uses the label at each window's START sample (not a majority vote
    like elsewhere in this project) -- a cheap approximation that's fine
    for selecting which windows fall in test reps, and applies equally at
    every hop level so it doesn't bias the comparison between them."""
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
    keep = np.where(np.isin(rep_at_start, list(TEST_REPS)))[0]
    if len(keep) == 0:
        return None
    if len(keep) > PER_FILE_CAND_CAP:
        keep = rng.choice(keep, PER_FILE_CAND_CAP, replace=False)
    keep_starts = starts[keep]

    X = np.asarray([emg[s:s + qk.WIN].T for s in keep_starts], np.float32)
    Y = np.asarray([glove[s + qk.WIN - 1] for s in keep_starts], np.float32)
    M = np.tile(channel_mask, (len(X), 1))
    return X, Y, M


def main(checkpoint, data_dir="data"):
    ck = torch.load(checkpoint, map_location=qk.dev, weights_only=False)
    net = qk.Net().to(qk.dev)
    net.load_state_dict(ck["sd"])
    net.eval()
    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    fm, fs, sm, ss = ck["fm"], ck["fs"], ck["sm"], ck["ss"]

    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:qk.MAX_FILES]
    rng = np.random.default_rng(0)

    print(f"loaded frozen checkpoint from {checkpoint} -- no retraining in this script, "
          f"only re-windowing the same held-out repetitions {sorted(TEST_REPS)} at different hop sizes")
    print(f"{'hop (ms)':>10} {'hop (samp)':>11} {'n windows':>10} {'pooled R^2':>11} {'fair R^2':>9}")

    results = []
    for hop in HOP_LEVELS:
        Xs, Ys, Ms = [], [], []
        for f in files:
            res = test_windows_at_hop(f, hop, rng)
            if res is None:
                continue
            Xf, Yf, Mf = res
            Xs.append(Xf); Ys.append(Yf); Ms.append(Mf)
        if not Xs:
            print(f"hop={hop}: no usable windows, skipping")
            continue
        X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)

        if len(X) > TARGET_TOTAL:
            idx = rng.choice(len(X), TARGET_TOTAL, replace=False)
            X, Y, M = X[idx], Y[idx], M[idx]

        Feat = qk.handcrafted_features(X, M)
        Sim = qk.channel_cosine_similarity(X, M)
        Xn = ((X - xm) / xs).astype(np.float32)
        Fn = ((Feat - fm) / fs).astype(np.float32)
        Sn = ((Sim - sm) / ss).astype(np.float32)

        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), qk.BATCH):
                xb = torch.from_numpy(Xn[i:i + qk.BATCH]).to(qk.dev)
                mb = torch.from_numpy(M[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
                fb = torch.from_numpy(Fn[i:i + qk.BATCH]).to(qk.dev)
                sb = torch.from_numpy(Sn[i:i + qk.BATCH]).to(qk.dev)
                out.append(net(xb, fb, sb, mb).cpu().numpy())
        pred = np.concatenate(out) * ys + ym

        joint_ss_res = np.sum((pred - Y) ** 2, axis=0)
        joint_ss_tot = np.sum((Y - Y.mean(0, keepdims=True)) ** 2, axis=0)
        joint_r2 = 1 - joint_ss_res / joint_ss_tot
        pooled_r2 = float(1 - joint_ss_res.sum() / joint_ss_tot.sum())
        fair_r2 = float(joint_r2.mean())

        print(f"{hop/2:>10.2f} {hop:>11d} {len(X):>10d} {pooled_r2:>11.3f} {fair_r2:>9.3f}")
        results.append((hop, len(X), pooled_r2, fair_r2))

    print(f"\nfor comparison: sMAPEN (100ms window / 0.5ms step, i.e. hop=1) reports R^2 = 0.8163 +/- 0.0398 "
          f"on this same dataset -- if fair R^2 above trends toward that as hop shrinks, the published gap "
          f"is substantially an evaluation-density effect, not a real accuracy difference.")
    return results


if __name__ == "__main__":
    main(*sys.argv[1:])
