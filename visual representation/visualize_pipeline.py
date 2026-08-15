"""
Builds ONE explanatory figure showing how raw forearm EMG becomes predicted
hand kinematics in this project's Test9 pipeline, end to end, on a real
recording and a real trained/calibrated checkpoint:

  (A) raw multi-channel EMG signal for one real movement
  (B) a schematic of the model pipeline that turns one 200ms window of that
      signal into a predicted kinematics vector
  (C) predicted vs. true joint-angle traces over that movement, for a few
      representative joints
  (D) a simple 2D hand-skeleton rendering comparing the PREDICTED hand pose
      against the TRUE (recorded) hand pose at the end of the movement

Uses Test9/backend.py directly (same model, same predict path the frontend
uses) rather than reimplementing anything, so this figure can't silently
drift from what the actual system does.

  python "visualize_pipeline.py" [subject_file] [movement_restimulus_id]

Defaults to Test9's S13_E1_A1.mat calibrated checkpoint and the first
non-rest movement long enough to look clean in the plot.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
TEST9 = os.path.join(HERE, "..", "Test9")
sys.path.insert(0, TEST9)
import backend as b   # Test9's consolidated backend -- same Net/predict path the frontend uses

# ---- best-effort, illustrative-only hand geometry, same caveat as Test9/frontend.py's own
# reach-trajectory plot: NOT measured for any specific subject or device, only used to give
# this figure a physically-plausible hand SHAPE. Joint mapping is the same best-effort
# CyberGlove channel assignment used throughout this project, not independently verified.
FINGER_JOINTS = {
    "Thumb": [1, 2],
    "Index": [4, 5, 6],
    "Middle": [8, 9, 10],
    "Ring": [12, 13, 14],
    "Little": [16, 17, 18],
}
FINGER_ABDUCTION_CH = {"Thumb": 3, "Index": 7, "Middle": 11, "Ring": 15, "Little": None}
FINGER_SEGMENT_MM = {
    "Thumb": (32, 22), "Index": (40, 25, 20), "Middle": (45, 28, 22),
    "Ring": (42, 26, 20), "Little": (35, 20, 18),
}
FINGER_BASE_X_MM = {"Thumb": -38, "Index": -16, "Middle": 0, "Ring": 16, "Little": 30}
FINGER_BASE_ANGLE_DEG = {"Thumb": -55, "Index": 0, "Middle": 0, "Ring": 0, "Little": 0}
MAX_FLEX_DEG, MAX_ABD_DEG = 45, 20   # per-JOINT cap, deliberately lower than frontend.py's 90 --
                                     # fingers here have up to 3 joints, so 90/joint could curl a
                                     # finger up to 270 degrees total (folding back past itself into
                                     # a tangle); 45/joint caps a 3-joint finger at a realistic-looking
                                     # ~135 degrees of total curl for this illustrative rendering


def normalize_channels(values, ref):
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    return np.clip((values - lo) / span, 0, 1)


def finger_chain_xy(norm, finger):
    """Returns the (x, y) positions of every joint along one finger's kinematic
    chain -- base, then after each segment -- a simple 2D front-view schematic
    (not true 3D), for drawing a recognisable hand shape. Each finger starts
    pointing away from the palm (straight up, or the thumb's own lean) and
    curls TOWARD the hand's centreline as its joints flex -- fingers left of
    centre curl right, fingers right of centre curl left, converging into a
    fist-like shape as flexion increases, which is what makes this read as a
    hand closing rather than an arbitrary tangle of line segments."""
    joints, lens = FINGER_JOINTS[finger], FINGER_SEGMENT_MM[finger]
    abd_ch = FINGER_ABDUCTION_CH.get(finger)
    base_x = FINGER_BASE_X_MM[finger]
    base_angle = np.radians(FINGER_BASE_ANGLE_DEG[finger])
    if abd_ch is not None:
        base_angle += np.radians((float(norm[abd_ch]) - 0.5) * 2 * MAX_ABD_DEG)
    side_sign = 1.0 if base_x < -2 else (-1.0 if base_x > 2 else 0.0)

    x, y = base_x, 0.0
    pts = [(x, y)]
    cum_flex = 0.0
    for j, length in zip(joints, lens):
        cum_flex += float(norm[j]) * np.radians(MAX_FLEX_DEG)
        dir_angle = base_angle + cum_flex * side_sign
        x += length * np.sin(dir_angle)
        y += length * np.cos(dir_angle)
        pts.append((x, y))
    return np.array(pts)


def draw_hand(ax, norm, color, label, alpha=1.0):
    for finger in FINGER_JOINTS:
        pts = finger_chain_xy(norm, finger)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, lw=2.5, ms=4, alpha=alpha,
                label=label if finger == "Middle" else None)
    # simple palm baseline connecting finger bases (no wrist taper -- a flat line reads more
    # clearly as "back of the hand" than a pointed wrist shape did)
    order = ["Little", "Ring", "Middle", "Index", "Thumb"]
    palm_x = [FINGER_BASE_X_MM[f] for f in order]
    ax.plot(palm_x, [0] * len(order), "-", color=color, lw=2.2, alpha=alpha * 0.5)


def pipeline_schematic(ax):
    """Boxes + arrows: raw EMG window -> CNN encoder / hand-crafted features /
    cosine-similarity -> concatenate -> regression head -> kinematics."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf1fb", fontsize=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.2, edgecolor="#2F5496", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color="#2F5496", lw=1.3,
                                     shrinkA=6, shrinkB=6))

    ax.text(5.0, 4.7, "one window -> one predicted kinematics vector, independently per window",
           ha="center", fontsize=8.5, style="italic", color="#444")

    box(0.05, 1.55, 1.75, 1.1, "200ms EMG\nwindow\n(12 channels)", fc="#fff3e0", fontsize=7.5)
    arrow(1.8, 2.1, 2.45, 3.35)
    arrow(1.8, 2.1, 2.45, 2.1)
    arrow(1.8, 2.1, 2.45, 0.85)
    box(2.45, 2.9, 2.35, 0.9, "CNN encoder\n(multi-channel, avg-pool)", fontsize=6.8)
    box(2.45, 1.65, 2.35, 0.9, "hand-crafted\nfeatures (16/ch)", fontsize=6.8)
    box(2.45, 0.4, 2.35, 0.9, "cross-channel\ncosine similarity", fontsize=6.8)
    arrow(4.8, 3.35, 5.5, 2.2)
    arrow(4.8, 2.1, 5.5, 2.1)
    arrow(4.8, 0.85, 5.5, 2.0)
    box(5.5, 1.65, 1.45, 0.9, "concat\nembedding", fc="#e8f5e9", fontsize=7)
    arrow(6.95, 2.1, 7.6, 2.1)
    box(7.6, 1.55, 1.75, 1.1, "regression\nhead (MLP)", fc="#fff3e0", fontsize=7.5)
    arrow(9.35, 2.1, 9.75, 2.1)
    box(9.4, 1.55, 0.55, 1.1, "22\njoint\nangles", fc="#e8f5e9", fontsize=6.5)
    ax.set_xlim(0, 10.2)


def main(subject_file="S13_E1_A1.mat", target_restim=None):
    data_path = os.path.join(TEST9, "data", subject_file)
    ckpt_path = os.path.join(TEST9, f"EMG-KinNet_finetuned_{os.path.splitext(subject_file)[0]}.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(TEST9, "EMG-KinNet.pt")
        print(f"note: no calibrated checkpoint for {subject_file}, using the base model instead")

    ds = b.DATASETS["db2"]
    ck = b.torch.load(ckpt_path, map_location=b.dev, weights_only=False)
    net = b.new_net(ds)
    net.load_state_dict(ck["sd"]); net.eval()

    m = loadmat(data_path)
    restim = m["restimulus"].astype(int).ravel()
    X, Y, M, warns = b.windows_from(ds, data_path)
    pred = b.predict_batches(net, ds, ck, X, M)

    win_restim = np.asarray([np.bincount(restim[s:s + b.WIN]).argmax()
                             for s in range(0, len(restim) - b.WIN, b.HOP)])[:len(X)]

    if target_restim is None:
        for mv in sorted(set(win_restim.tolist()) - {0}):
            idx = np.where(win_restim == mv)[0]
            if len(idx) >= 15:
                target_restim = mv
                break
    idx = np.where(win_restim == target_restim)[0]
    s0, s1 = idx[0], idx[-1]
    print(f"visualising restimulus={target_restim}, windows {s0}-{s1} ({s1 - s0 + 1} windows)")

    emg, glove, mask, _ = b.load_emg_file(ds, data_path)
    n_real = int(mask.sum())
    emg_t0 = s0 * b.HOP
    emg_t1 = s1 * b.HOP + b.WIN
    emg_seg = emg[emg_t0:emg_t1, :n_real]
    tt = np.arange(len(emg_seg)) / ds["fs"]

    # ---- figure layout ----
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.0, 1.4, 2.2, 3.0], hspace=0.55, wspace=0.3)

    # (A) raw EMG
    axA = fig.add_subplot(gs[0, :])
    offset = np.max(np.abs(emg_seg)) * 2.0 + 1e-6
    for c in range(emg_seg.shape[1]):
        axA.plot(tt, emg_seg[:, c] + c * offset, lw=0.7)
    axA.set_yticks([c * offset for c in range(emg_seg.shape[1])])
    axA.set_yticklabels([f"Ch{c+1}" for c in range(emg_seg.shape[1])], fontsize=7)
    axA.set_xlabel("time (s)")
    axA.set_title(f"(A) Raw forearm EMG -- {n_real} channels, subject {subject_file}, "
                 f"movement restimulus={target_restim}", fontsize=11, loc="left")
    axA.margins(x=0)

    # (B) pipeline schematic
    axB = fig.add_subplot(gs[1, :])
    pipeline_schematic(axB)
    axB.set_title("(B) How ONE window becomes ONE prediction (Test9/backend.py's Net)",
                 fontsize=11, loc="left")

    # (C) predicted vs true kinematics over this movement
    axC = fig.add_subplot(gs[2, :])
    show_joints = [4, 8, 12, 16]   # Index/Middle/Ring/Little MCP -- clean, comparable-scale flexion joints
    seg_idx = np.arange(s0, s1 + 1)
    tt_w = seg_idx * b.HOP / ds["fs"]
    for j in show_joints:
        axC.plot(tt_w, pred[seg_idx, j], lw=1.6, label=f"J{j} predicted")
        if Y is not None:
            axC.plot(tt_w, Y[seg_idx, j], lw=1.2, ls="--", alpha=0.7, label=f"J{j} true")
    axC.set_xlabel("time (s)"); axC.set_ylabel("joint angle (raw units)")
    axC.legend(fontsize=7, ncol=4, loc="upper right")
    axC.set_title("(C) Predicted vs. true kinematics across the movement", fontsize=11, loc="left")

    # (D) hand skeleton at the movement's peak/end frame
    axD = fig.add_subplot(gs[3, :])
    axD.set_aspect("equal")
    ref = Y if Y is not None else pred
    peak_local = np.argmax(np.abs(pred[seg_idx, 8] - pred[seg_idx[0], 8]))  # biggest middle-MCP excursion
    peak = seg_idx[peak_local]
    norm_pred = normalize_channels(pred[peak], ref)
    draw_hand(axD, norm_pred, "red", "Predicted")
    if Y is not None:
        norm_true = normalize_channels(Y[peak], ref)
        draw_hand(axD, norm_true, "goldenrod", "True", alpha=0.85)
    axD.set_xlim(-70, 70); axD.set_ylim(-60, 140)
    axD.set_xlabel("x (mm, illustrative)"); axD.set_ylabel("y (mm, illustrative)")
    axD.legend(fontsize=9, loc="upper right")
    axD.set_title(f"(D) Resulting hand pose at window {peak} (predicted vs. true) -- "
                 "best-effort illustrative geometry, not a measured hand", fontsize=11, loc="left")

    fig.suptitle("From raw EMG to predicted hand kinematics -- Test9 pipeline, real data, real trained model",
               fontsize=14, fontweight="bold", y=0.995)

    out_path = os.path.join(HERE, "emg_to_kinematics_pipeline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    subject_file = args[0] if len(args) > 0 else "S13_E1_A1.mat"
    target_restim = int(args[1]) if len(args) > 1 else None
    main(subject_file, target_restim)
