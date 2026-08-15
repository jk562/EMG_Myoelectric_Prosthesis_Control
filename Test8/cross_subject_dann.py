"""
Leave-subject-out (LOSO) evaluation, with an optional Domain-Adversarial
Neural Network (DANN, Ganin & Lempitsky 2015 -- gradient-reversal layer)
head that pushes the shared embedding fed to the regression head to be
subject-invariant, attempting to improve zero-shot accuracy on subjects
the model NEVER sees during training. This is the harder, more
realistic "brand new person puts on the device with zero calibration"
scenario -- the exact one Test6's own CNN got catastrophic negative R^2
on (R^2=-1.445 to -1.762, held out subjects 13 & 15, an older/simpler
architecture without this project's hand-crafted features or cosine-
similarity branch). This project's real, working deliverable remains
per-subject calibration (quick_emg_to_kinematics.py's finetune(),
R^2=0.790) -- that is NOT being replaced or questioned here. This script
is a genuine, honestly-reported ATTEMPT at the harder, largely unsolved
zero-shot problem, run with the same keep/revert discipline as every
other experiment in this project -- not a claim that it's been solved.

HELD_OUT_SUBJECTS matches Test6's own pair (13, 15) for a direct,
comparable number against that established negative result.

DANN mechanism: the embedding that normally feeds straight into the
regression head is ALSO passed (through a gradient-reversal layer) into
a small subject classifier trained to predict which of the TRAINING
subjects a window came from. Because the reversal layer negates the
gradient flowing back from that classification loss, the shared encoder
is pushed to make subject identity harder to recover -- i.e. to discard
subject-specific idiosyncrasies -- while the classifier itself still
tries its best. In principle this should leave more genuinely subject-
invariant movement information in the embedding the regression head
actually uses.

  python cross_subject_dann.py data            # baseline, no DANN
  python cross_subject_dann.py data --dann      # with the DANN branch
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import quick_emg_to_kinematics as qk

HELD_OUT_SUBJECTS = {13, 15}
DANN_WEIGHT = 0.2
DANN_WARMUP_EPOCHS = 10   # linearly ramp lambda 0->1 over this many epochs -- standard DANN
                          # practice (Ganin & Lempitsky), avoids the adversarial signal
                          # dominating before the encoder has learned anything useful yet
PER_FILE_CAP = 2500


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class SubjectHead(nn.Module):
    """Predicts which TRAINING subject a window came from, from the SAME
    shared embedding (CNN + hand-crafted + cosine-sim, concatenated) that
    feeds Net's regression head -- see get_embedding()."""
    def __init__(self, in_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, n_subjects))

    def forward(self, x, lambd):
        return self.net(grad_reverse(x, lambd))


def get_embedding(net, x, feat, sim, mask=None):
    """Reproduces Net.forward's internal embedding computation WITHOUT
    touching quick_emg_to_kinematics.py's Net class -- so the production
    architecture stays untouched by this experimental script."""
    emb = net.encoder.pooled(x, mask)
    femb = net.feat_proj(feat)
    semb = net.sim_proj(sim)
    return torch.cat([emb, femb, semb], dim=-1)


def main(data_dir="data", use_dann=False):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))[:qk.MAX_FILES]
    if not files:
        sys.exit(f"No .mat files in '{data_dir}'.")

    cap_rng = np.random.default_rng(0)
    train_subs, held_out_subs = [], []
    Xs, Ys, Ms, Fs, Ss, Subs = [], [], [], [], [], []
    Xh, Yh, Mh = [], [], []   # held-out subjects, never touched during training
    n_used = n_skipped = 0
    for f in files:
        sub = qk.subject_of(f)
        Xf, Yf, Mf, warns = qk.windows_from(f)
        if Yf is None or len(Xf) == 0:
            n_skipped += 1
            continue
        if sub in HELD_OUT_SUBJECTS:
            held_out_subs.append(sub)
            Xh.append(Xf); Yh.append(Yf); Mh.append(Mf)
        else:
            train_subs.append(sub)
            idx = np.arange(len(Xf))
            if len(idx) > PER_FILE_CAP:
                idx = cap_rng.choice(idx, PER_FILE_CAP, replace=False)
            Xf, Yf, Mf = Xf[idx], Yf[idx], Mf[idx]
            Xs.append(Xf); Ys.append(Yf); Ms.append(Mf)
            Fs.append(qk.handcrafted_features(Xf, Mf)); Ss.append(qk.channel_cosine_similarity(Xf, Mf))
            Subs.append(np.full(len(Xf), sub))
        n_used += 1
    if not Xs or not Xh:
        sys.exit("Missing train or held-out data -- check HELD_OUT_SUBJECTS against available files.")

    X, Y, M = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)
    Feat, Sim = np.concatenate(Fs), np.concatenate(Ss)
    sub_ids = np.concatenate(Subs)
    Xh_all, Yh_all, Mh_all = np.concatenate(Xh), np.concatenate(Yh), np.concatenate(Mh)

    train_subs = sorted(set(train_subs))
    sub_to_idx = {s: i for i, s in enumerate(train_subs)}
    sub_labels = np.array([sub_to_idx[s] for s in sub_ids])

    print(f"files: {len(files)} total -- {n_used} used, {n_skipped} skipped")
    print(f"TRAIN subjects ({len(train_subs)}): {train_subs}")
    print(f"HELD-OUT subjects ({sorted(set(held_out_subs))}), NEVER seen in training, "
          f"{len(Xh_all)} zero-shot eval windows")
    print(f"{len(X)} total train-pool windows, DANN={'ON' if use_dann else 'OFF'}, device={qk.dev}")

    rng = np.random.default_rng(0)
    tr_idx, val_idx = qk.leakage_safe_split(len(X), 1 - qk.VAL_FRAC, rng, gap=1)
    max_val = max(1, int(qk.MAX_WINDOWS * qk.VAL_FRAC))
    max_tr = max(1, qk.MAX_WINDOWS - max_val)
    if len(tr_idx) > max_tr:
        tr_idx = rng.choice(tr_idx, max_tr, replace=False)
    if len(val_idx) > max_val:
        val_idx = rng.choice(val_idx, max_val, replace=False)
    print(f"{len(tr_idx)} train / {len(val_idx)} val windows")

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
                                        torch.from_numpy(Simn[idx].astype(np.float32)),
                                        torch.from_numpy(sub_labels[idx].astype(np.int64))),
                          batch_size=qk.BATCH, shuffle=shuffle)

    tr_dl, val_dl = make_dl(tr_idx, True), make_dl(val_idx, False)

    net = qk.Net().to(qk.dev)
    emb_dim = qk.FEAT_DIM + qk.FEAT_HC_DIM + qk.FEAT_SIM_DIM
    subj_head = SubjectHead(emb_dim, len(train_subs)).to(qk.dev)
    params = list(net.parameters()) + (list(subj_head.parameters()) if use_dann else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=qk.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for ep in range(qk.EPOCHS):
        net.train(); subj_head.train()
        lambd = min(1.0, ep / DANN_WARMUP_EPOCHS) if use_dann else 0.0
        tot, tot_kin, tot_dann = 0.0, 0.0, 0.0
        for xb, yb, mb, fb, sb, subb in tr_dl:
            xb, yb, mb, fb, sb, subb = (xb.to(qk.dev), yb.to(qk.dev), mb.to(qk.dev),
                                        fb.to(qk.dev), sb.to(qk.dev), subb.to(qk.dev))
            if qk.NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * qk.NOISE_STD
            emb = get_embedding(net, xb, fb, sb, mb)
            out = net.head(emb)
            kin_loss = lf(out, yb)
            loss = kin_loss
            dann_loss = torch.tensor(0.0)
            if use_dann:
                logits = subj_head(emb, lambd)
                dann_loss = ce(logits, subb)
                loss = kin_loss + DANN_WEIGHT * dann_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tot_kin += kin_loss.item(); tot_dann += dann_loss.item()
        tr_loss, tr_kin, tr_dann = tot / len(tr_dl), tot_kin / len(tr_dl), tot_dann / len(tr_dl)

        net.eval(); subj_head.eval()
        with torch.no_grad():
            val_loss = sum(lf(net.head(get_embedding(net, xb.to(qk.dev), fb.to(qk.dev), sb.to(qk.dev),
                                                      mb.to(qk.dev))), yb.to(qk.dev)).item()
                           for xb, yb, mb, fb, sb, subb in val_dl) / len(val_dl)
        sched.step(val_loss)

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            marker = " *"
        else:
            bad_epochs += 1
        dann_str = f"  dann-ce {tr_dann:.3f} (lambda={lambd:.2f})" if use_dann else ""
        print(f"epoch {ep+1}/{qk.EPOCHS}  train {tr_kin:.4f}  val {val_loss:.4f}{dann_str}{marker}")
        if bad_epochs >= qk.PATIENCE:
            print(f"no val improvement for {qk.PATIENCE} epochs, stopping early")
            break

    net.load_state_dict(best_state)
    net.eval()
    Xhe = (Xh_all - xm) / xs
    Fhe = (qk.handcrafted_features(Xh_all, Mh_all) - fm) / fs
    She = (qk.channel_cosine_similarity(Xh_all, Mh_all) - sm) / ss
    out = []
    with torch.no_grad():
        for i in range(0, len(Xhe), qk.BATCH):
            xb = torch.from_numpy(Xhe[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            mb = torch.from_numpy(Mh_all[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            fb = torch.from_numpy(Fhe[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            sb = torch.from_numpy(She[i:i + qk.BATCH].astype(np.float32)).to(qk.dev)
            out.append(net(xb, fb, sb, mb).cpu().numpy())
    pred = np.concatenate(out) * ys + ym

    print(f"\n=== ZERO-SHOT on held-out subjects {sorted(set(held_out_subs))} "
          f"(DANN={'ON' if use_dann else 'OFF'}), never seen in training ===")
    mean_r2 = qk.report_metrics(pred, Yh_all)
    print(f"\nfor comparison: Test6's own CNN (older architecture, no hand-crafted features, "
          f"no cosine-similarity branch) got R^2=-1.445 (from-scratch) / -1.762 (masked-SSL) "
          f"held out on this exact subject pair.")
    return mean_r2


if __name__ == "__main__":
    args = sys.argv[1:]
    use_dann = "--dann" in args
    args = [a for a in args if a != "--dann"]
    main(*args, use_dann=use_dann)
