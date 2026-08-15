"""
Test9: unified Streamlit frontend for the consolidated backend.py -- covers
BOTH datasets in one app, switchable from a sidebar selector:

  .mat/.csv/.npy  -- Ninapro DB2, 22-channel CyberGlove kinematics,
                     best-effort anatomical joint names, cosine-similarity
                     feature branch, 3D reach-trajectory visual.
  .fif            -- RAW TEST recordings, 15 generic "Angle" channels (no
                     verified anatomical mapping for this device, so no
                     bone names/reach-trajectory visual).

Shows, for an uploaded/selected EMG recording:
  1. the input EMG signals (real channels only, padding excluded)
  2. the predicted kinematics, over time + current frame
  3. the action being performed, if the file has a label -- .mat via the
     Ninapro `restimulus` field, .fif via MNE annotations
  4. a button to calibrate the model to this specific subject for a
     realistic, honest deployment-accuracy number

Run:
    streamlit run frontend.py

Requires EMG-KinNet.pt and/or RawTest-KinNet.pt (produced by
`python backend.py train db2` / `train rawtest`) depending on which mode
you use. Imports backend.py directly rather than keeping a copy, so
anything the CLI does works here too and stays in sync.
"""

import contextlib
import glob
import io
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import streamlit as st
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import backend

# Each mode bundles everything the shared UI code needs: which dataset key
# (into backend.DATASETS) to use, which file types/folder to browse, which
# checkpoint files to list, and whether the CyberGlove-specific anatomy
# sections (action label, bone names, reach-trajectory plot) apply -- they
# don't for the RAW TEST device, which has no verified joint-name mapping
# for its 15 "Angle" channels.
MODES = {
    ".mat / .csv / .npy  (Ninapro DB2, 22-joint CyberGlove)": {
        "dataset": "db2",
        "ext": ["mat", "csv", "npy"],
        "has_anatomy": True,
    },
    ".fif  (RAW TEST, 15-channel angle device)": {
        "dataset": "rawtest",
        "ext": ["fif"],
        "has_anatomy": False,
    },
}

# Ninapro DB2 movement labels -> readable names (0 = rest). Only used in
# .mat mode. This project's E1/E2/E3 files use a single CONTINUOUS restimulus
# numbering across exercises (verified empirically against the real data:
# E1 files -> values 0-17, E2 -> 18-40, E3 -> 41-49), which lines up exactly
# with Ninapro DB2's official Exercise B (17 movements: 8 isometric/isotonic
# hand configurations + 9 basic wrist movements), Exercise C (23 grasping and
# functional movements), and Exercise D (9 force patterns) -- confirmed via
# ninapro.hevs.ch and the Atzori et al. 2014/2015 papers.
#
# ACTION_NAMES (below) covers Exercise B (1-17) with specific movement names,
# BEST-EFFORT/UNVERIFIED for 16-17 -- not confirmed against official
# documentation, don't cite without checking.
#
# For Exercise C (18-40) and Exercise D (41-49): despite checking the
# official Ninapro site, the original 2014 dataset paper, and the 2015
# benchmark-characterization paper, no machine-readable source with the
# SPECIFIC per-movement names (e.g. which exact grasp #3 is) could be found
# -- only the category-level descriptions above, which are what's used
# below.
ACTION_NAMES = {
    0: "Rest",
    1: "Thumb up",
    2: "Ext. index & middle, flex others",
    3: "Flex ring & little, ext others",
    4: "Thumb opposing little finger",
    5: "Abduct all fingers",
    6: "Fingers flexed together (fist)",
    7: "Pointing index",
    8: "Adduct extended fingers",
    9: "Wrist supination",
    10: "Wrist pronation",
    11: "Wrist flexion",
    12: "Wrist extension",
    13: "Wrist radial deviation",
    14: "Wrist ulnar deviation",
    15: "Wrist extension with closed hand",
    16: "Wrist flexion with closed hand",
    17: "Wrist ulnar deviation with closed hand",
}
UNVERIFIED_ACTIONS = {16, 17}

# Exercise C (18-40) and Exercise D (41-49) -- category-level only, see note
# above for why these aren't specific grasp/force-pattern names.
EXERCISE_C_RANGE = range(18, 41)   # 23 grasping/functional movements
EXERCISE_D_RANGE = range(41, 50)   # 9 force patterns

# 22 CyberGlove channel -> anatomical joint. BEST-EFFORT / UNVERIFIED: this is
# the commonly-documented generic 22-sensor CyberGlove layout (per-finger
# MCP/PIP/DIP + abductions + wrist pitch/yaw), NOT confirmed against Ninapro
# DB2's actual channel order -- verify before citing in a report. Channel 10
# in particular has a raw-value range (roughly -600 to 800) wildly different
# from every other channel (roughly -80 to 150), which is a sign it may not
# be a plain joint angle in the same units as the rest. Only used in .mat mode.
JOINT_NAMES = [
    "Thumb rotation", "Thumb MCP", "Thumb IP", "Thumb-index abduction",
    "Index MCP", "Index PIP", "Index DIP", "Index-middle abduction",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle-ring abduction",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring-little abduction",
    "Little MCP", "Little PIP", "Little DIP",
    "Palm arch", "Wrist flexion/extension", "Wrist abduction/adduction",
]

# per-finger flex/curl channels (MCP/PIP/DIP only, no abduction/wrist/palm) --
# same best-effort caveat as JOINT_NAMES above.
FINGER_JOINTS = {
    "Thumb": [1, 2],
    "Index": [4, 5, 6],
    "Middle": [8, 9, 10],
    "Ring": [12, 13, 14],
    "Little": [16, 17, 18],
}
FINGER_ABDUCTION_CH = {"Thumb": 3, "Index": 7, "Middle": 11, "Ring": 15, "Little": None}

# ILLUSTRATIVE average adult finger segment lengths in mm (proximal, middle,
# distal phalanx) -- generic anthropometric ballpark figures, NOT measured
# for any specific subject or prosthetic device. Only used to give the reach-
# trajectory plot a physically-plausible scale; treat the mm axes there as
# approximate/illustrative, not calibrated.
FINGER_SEGMENT_MM = {
    "Thumb": (32, 22),
    "Index": (40, 25, 20),
    "Middle": (45, 28, 22),
    "Ring": (42, 26, 20),
    "Little": (35, 20, 18),
}
MAX_FLEX_DEG = 90
MAX_ABD_DEG = 20


def fingertip_xyz(norm, finger):
    """Simplified 3D forward kinematics for one finger's tip position from a
    0-1-normalised 22-channel vector at one timestep. Returns (x, y, z) in mm."""
    joints, lens = FINGER_JOINTS[finger], FINGER_SEGMENT_MM[finger]
    abd_ch = FINGER_ABDUCTION_CH.get(finger)
    yaw = np.radians((float(norm[abd_ch]) - 0.5) * 2 * MAX_ABD_DEG) if abd_ch is not None else 0.0

    fwd, up, angle = 0.0, 0.0, 0.0
    for j, length in zip(joints, lens):
        angle += float(norm[j]) * np.radians(MAX_FLEX_DEG)
        fwd += length * np.cos(angle)
        up -= length * np.sin(angle)
    return np.array([fwd * np.sin(yaw), fwd * np.cos(yaw), up])


@st.cache_resource
def load_kinematics_model(path, _ds):
    """_ds is the dataset config dict -- underscore prefix tells
    st.cache_resource not to try hashing it, the path string is what
    actually keys the cache."""
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location=backend.dev, weights_only=False)
    net = backend.new_net(_ds)
    net.load_state_dict(ck["sd"])
    net.eval()
    return net, ck


def fif_action_labels(path, n_windows, win, hop, fs):
    """Per-window action label from MNE .fif annotations, if any are
    present. Returns None if the file has no annotations covering any
    window."""
    import mne
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    ann = raw.annotations
    if len(ann) == 0:
        return None
    labels = np.full(n_windows, "Unlabeled", dtype=object)
    centers = (np.arange(n_windows) * hop + win / 2) / fs
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        labels[(centers >= onset) & (centers < onset + duration)] = desc
    return None if set(labels.tolist()) == {"Unlabeled"} else labels


def windows_and_labels(path, ds):
    """Same loader+windowing backend.predict() uses, plus the recorded
    movement label if this file has one (.mat via `restimulus`, .fif via
    MNE annotations) -- that part is UI-only, not part of the prediction."""
    emg, glove, channel_mask, warns = backend.load_emg_file(ds, path)
    n_real = int(channel_mask.sum())
    X = np.asarray([emg[s:s + backend.WIN].T for s in range(0, len(emg) - backend.WIN, backend.HOP)], np.float32)
    M = np.tile(channel_mask, (len(X), 1)) if len(X) else np.zeros((0, len(channel_mask)), np.float32)
    Y = None
    if glove is not None:
        Y = np.asarray([glove[s + backend.WIN - 1] for s in range(0, len(emg) - backend.WIN, backend.HOP)],
                       np.float32)

    labels = None
    if path.lower().endswith(".mat"):
        m = loadmat(path)
        if "restimulus" in m:
            lab = m["restimulus"].astype(int).ravel()
            labels = np.asarray([np.bincount(lab[s:s + backend.WIN]).argmax()
                                 for s in range(0, len(lab) - backend.WIN, backend.HOP)])
    elif path.lower().endswith(".fif") and len(X):
        labels = fif_action_labels(path, len(X), backend.WIN, backend.HOP, ds["fs"])

    return X, Y, M, labels, emg[:, :n_real], warns


def show_metrics(pred, Y_true):
    """Streamlit version of report_metrics -- same math, same dominant-joint
    warning, so the UI can't tell a different story than the CLI does."""
    joint_ss_res = np.sum((pred - Y_true) ** 2, axis=0)
    joint_ss_tot = np.sum((Y_true - Y_true.mean(0, keepdims=True)) ** 2, axis=0)
    joint_r2 = 1 - joint_ss_res / joint_ss_tot
    pooled_rmse = float(np.sqrt(np.mean((pred - Y_true) ** 2)))
    pooled_r2 = float(1 - joint_ss_res.sum() / joint_ss_tot.sum())
    mean_r2 = float(joint_r2.mean())

    dominant = int(np.argmax(joint_ss_tot))
    dominant_share = joint_ss_tot[dominant] / joint_ss_tot.sum()
    best, worst = int(np.argmax(joint_r2)), int(np.argmin(joint_r2))

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (pooled, raw units)", f"{pooled_rmse:.2f}")
    c2.metric("R² (pooled, raw units)", f"{pooled_r2:.3f}")
    c3.metric("R² averaged across joints (fair)", f"{mean_r2:.3f}")
    if dominant_share > 0.5:
        st.caption(f"note: joint {dominant} alone accounts for {dominant_share*100:.0f}% of the pooled "
                  f"variance -- the pooled numbers mostly measure that one joint, the averaged R² is fairer")
    st.caption(f"best joint: {best} (R² {joint_r2[best]:.3f})  |  worst joint: {worst} (R² {joint_r2[worst]:.3f})")


def normalize_channels(values, ref):
    """values: (22,) at one timestep. ref: (N, 22) whole-recording data used
    to normalise each channel to roughly 0-1. Uses the 1st-99th percentile
    rather than raw min/max, so a handful of noisy outlier samples can't
    single-handedly blow up the scale."""
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    return np.clip((values - lo) / span, 0, 1)


def movement_blocks(labels, min_len=3):
    """One contiguous window-run per unique non-rest action label -- its
    FIRST long-enough occurrence in time order -- so each becomes one
    labeled 'trial' in the trajectory plot below."""
    blocks, seen = [], set()
    i, n = 0, len(labels)
    while i < n:
        lab = int(labels[i])
        j = i
        while j < n and int(labels[j]) == lab:
            j += 1
        if lab != 0 and lab not in seen and (j - i) >= min_len:
            blocks.append((lab, i, j))
            seen.add(lab)
        i = j
    return blocks


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="EMG → Hand Kinematics", layout="wide")
st.title("EMG → Hand Kinematics")

# ---- dataset / mode selection ----
st.sidebar.header("Dataset")
mode_label = st.sidebar.radio("File type", list(MODES.keys()))
mode = MODES[mode_label]
ds = backend.DATASETS[mode["dataset"]]
has_anatomy = mode["has_anatomy"]

st.caption(f"Predicts continuous kinematics from forearm EMG -- {mode_label}, "
          f"1-{ds['max_channels']} channels, with or without ground truth")

# ---- model selection ----
st.sidebar.header("Model")
model_glob = f"{ds['ckpt_prefix']}*.pt"
model_files = sorted(f for f in glob.glob(model_glob) if "-SSL" not in os.path.basename(f))
if not model_files:
    st.error(f"No `{model_glob}` found. Train one first: `python backend.py train {mode['dataset']}`.")
    st.stop()
model_path = st.sidebar.selectbox("Checkpoint", model_files, key=f"ckpt_{mode_label}",
                                  help="Fine-tuned checkpoints (from the Calibrate button below) show up "
                                       "here once saved -- pick one to see calibrated predictions.")
kin_net, kin_ck = load_kinematics_model(model_path, ds)

if __name__ == "__main__":
    import sys, streamlit.web.cli as stcli
    from streamlit import runtime
    if not runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

# ---- input selection ----
st.sidebar.header("Input EMG")
data_dir = ds["data_dir"]
in_mode = st.sidebar.radio("Source", ["Upload file", f"Pick from {data_dir}/"], key=f"src_{mode_label}")

path, tmp_path = None, None
if in_mode == "Upload file":
    up = st.sidebar.file_uploader("EMG file", type=mode["ext"], key=f"upload_{mode_label}")
    if up is not None:
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(up.read())
            tmp_path = path = tmp.name
else:
    files = sorted([f for ext in mode["ext"] for f in glob.glob(os.path.join(data_dir, f"*.{ext}"))])
    if not files:
        st.sidebar.warning(f"No {'/'.join(mode['ext'])} files in {data_dir}/")
    else:
        path = st.sidebar.selectbox("File", files, format_func=os.path.basename, key=f"file_{mode_label}")

if path is None:
    st.info("Select or upload an EMG recording to begin.")
    st.stop()

X, Y_true, M, labels, emg_raw, warns = windows_and_labels(path, ds)

for w in warns:
    st.sidebar.caption(f"note: {w}")

if len(X) == 0:
    st.error("Recording too short to form a single analysis window.")
    if tmp_path is not None:
        os.unlink(tmp_path)
    st.stop()

pred = backend.predict_batches(kin_net, ds, kin_ck, X, M)

# ---- accuracy caveat ----
st.info("This model trains on every available subject (no held-out set), so cold accuracy reflects fit "
       "to data it may have already seen, not generalisation to a new person. Calibrate below first for "
       "an honest, subject-specific number -- that's the intended workflow, not an afterthought.")

# ---- calibrate ----
st.subheader("Calibrate to this subject")
st.caption("Fine-tunes the current checkpoint on 70% of a labeled recording, evaluates on the "
          "other 30% (a leakage-free block split, not just re-testing on what it trained on). "
          "By default this uses the file loaded above -- but if THAT file has no ground truth, "
          "pick a different labeled recording from the same subject instead, then come back and "
          "select the resulting checkpoint to predict on the original file.")

calib_options = sorted([f for ext in mode["ext"] for f in glob.glob(os.path.join(data_dir, f"*.{ext}"))])
calib_source = st.radio("Calibrate using", ["This file (currently loaded)", f"A different file from {data_dir}/"],
                        horizontal=True, disabled=not calib_options, key=f"calibsrc_{mode_label}")
calib_path = path
if calib_source.startswith("A different file"):
    calib_path = st.selectbox("File to calibrate on", calib_options, format_func=os.path.basename,
                              key=f"calibfile_{mode_label}",
                              help="Pick any recording that HAS ground-truth kinematics.")
elif Y_true is None:
    st.caption(f"This file has no ground-truth kinematics -- pick \"A different file from {data_dir}/\" "
              "above, or calibrate is unavailable.")

if calib_path == path and Y_true is None:
    pass   # no usable file selected yet, button below is skipped
elif st.button("Calibrate now"):
    log = io.StringIO()
    with st.spinner("Calibrating..."):
        with contextlib.redirect_stdout(log):
            try:
                backend.finetune(mode["dataset"], calib_path, model_path)
            except SystemExit as e:
                st.error(str(e))
                st.stop()
    st.code(log.getvalue())
    st.success("Saved. Pick the new checkpoint from the **Model** dropdown in the sidebar to see "
              "calibrated predictions.")

# ---- training history ----
history = kin_ck.get("history")
if history:
    with st.expander(f"Training history for {os.path.basename(model_path)}"):
        cols = ["epoch", "train", "val"] if len(history[0]) == 3 else ["epoch", "train"]
        df = pd.DataFrame(history, columns=cols).set_index("epoch")
        st.line_chart(df)
        st.caption("Loss per epoch recorded when this checkpoint was produced (pretrain/train/finetune). "
                  "Older checkpoints saved before this feature was added won't have this.")


def _action_name(a):
    """.mat labels are ints (Ninapro restimulus codes) -- .fif labels are
    the annotation description strings directly, shown as-is."""
    if not isinstance(a, (int, np.integer)):
        return str(a)
    a = int(a)
    if a in ACTION_NAMES:
        return ACTION_NAMES[a]
    if a in EXERCISE_C_RANGE:
        return f"Grasp/functional movement {a - 17} of 23 (Exercise C)"
    if a in EXERCISE_D_RANGE:
        return f"Force pattern {a - 40} of 9 (Exercise D)"
    return f"Movement {a}"


st.sidebar.markdown("---")
if labels is not None:
    unique_actions = sorted(set(labels.tolist()))
    action_names = [_action_name(a) for a in unique_actions]
    chosen = st.sidebar.selectbox("Action", action_names)
    chosen_id = unique_actions[action_names.index(chosen)]
    t = int(np.where(labels == chosen_id)[0][0])   # jump to that action's first occurrence
    st.sidebar.caption(f"window {t}")
else:
    t = st.sidebar.slider("Timestep (window index)", 0, len(pred) - 1, 0, key=f"t_{mode_label}")
    st.sidebar.caption("No action labels found for this file -- showing a plain window-index slider. "
                       "(.mat: needs a `restimulus` field. .fif: needs MNE annotations -- add them with "
                       "`raw.set_annotations(...)` + `raw.save(...)` and this becomes an action picker.)")

window_s = (t * backend.HOP) / ds["fs"]
st.sidebar.caption(f"≈ {window_s:.2f} s into the recording")

# ---- action ----
st.subheader("Action")
if labels is not None:
    lab = labels[t]
    name = _action_name(lab)
    st.metric("Action (from recording label)", name)
    unverified = (isinstance(lab, (int, np.integer))
                 and (int(lab) in UNVERIFIED_ACTIONS or int(lab) in EXERCISE_C_RANGE or int(lab) in EXERCISE_D_RANGE))
    st.caption("This is the recorded ground-truth label, not a model prediction."
              + (" Name not verified against official Ninapro docs." if unverified else ""))
else:
    st.info("No movement label available for this file.")

# ---- EMG input signals ----
st.subheader("Input EMG signals")
show_full = st.checkbox("Show whole recording (instead of current window)", value=False)

fig, ax = plt.subplots(figsize=(11, 5))
if show_full:
    seg = emg_raw
    tt = np.arange(len(seg)) / ds["fs"]
    ax.set_xlabel("time (s)")
else:
    seg = X[t, :emg_raw.shape[1]].T          # real channels only, current window
    tt = np.arange(len(seg)) / ds["fs"]
    ax.set_xlabel("time within window (s)")

offset = np.max(np.abs(seg)) * 2.2 + 1e-6
for c in range(seg.shape[1]):
    ax.plot(tt, seg[:, c] + c * offset, lw=0.6)
ax.set_yticks([c * offset for c in range(seg.shape[1])])
ax.set_yticklabels([f"Ch{c+1}" for c in range(seg.shape[1])])
ax.set_title("EMG channels" + ("" if show_full else f" — window {t}"))
if not show_full:
    ax.margins(x=0)
st.pyplot(fig)
plt.close(fig)

# ---- kinematics ----
st.subheader("Predicted kinematics")
n_kin = ds["n_kin"]
if has_anatomy:
    st.caption("Bone/joint names are a best-effort generic 22-sensor CyberGlove mapping, **not verified** "
              "against Ninapro DB2's actual channel order -- double-check before citing. Joint 10 has a "
              "raw-value range wildly different from every other joint, which may mean it isn't a plain "
              "angle in the same units as the rest.")
    hand_diagram_path = os.path.join(os.path.dirname(__file__), "Hand_Joint_Motion_Diagram.png")
    if os.path.exists(hand_diagram_path):
        with st.expander("Hand Joint Motion Diagram", expanded=False):
            st.image(hand_diagram_path, caption="Hand Joint Motion Diagram", width="stretch")
    ch_label = lambda j: f"J{j}: {JOINT_NAMES[j]}"
else:
    st.caption("Channels are generic (\"Angle 0..14\") -- there's no documented mapping from this device's "
              "angle channels to named anatomical joints, unlike the CyberGlove pipeline's best-effort names.")
    ch_label = lambda j: f"Angle {j}"

c1, c2 = st.columns([1, 2])

with c1:
    st.markdown(f"**Current frame ({n_kin} channels)**")
    rows = {"Channel": [ch_label(j) for j in range(n_kin)],
            "Predicted": [f"{v:.2f}" for v in pred[t]]}
    if Y_true is not None:
        rows["True"] = [f"{v:.2f}" for v in Y_true[t]]
    st.dataframe(rows, height=430, width="stretch")

with c2:
    st.markdown("**Over time**")
    sel = st.multiselect("Channels to plot", list(range(n_kin)),
                         default=list(range(min(4, n_kin))), format_func=ch_label, key=f"sel_{mode_label}")
    if sel:
        fig2, axes = plt.subplots(len(sel), 1, figsize=(9, 1.6 * len(sel)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax2, j in zip(axes, sel):
            ax2.plot(pred[:, j], lw=1.1, label="predicted")
            if Y_true is not None:
                ax2.plot(Y_true[:, j], lw=1.1, alpha=0.7, label="true")
            ax2.axvline(t, color="k", ls="--", lw=0.8)
            ax2.set_ylabel(ch_label(j), fontsize=7)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("window index")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ---- accuracy ----
st.subheader("Accuracy")
if Y_true is not None:
    # if this checkpoint was calibrated ON the currently-loaded file, a
    # whole-file accuracy number is partly inflated -- detect that
    # (checkpoint name matches "<prefix>_finetuned_<this file's stem>") and
    # recompute the SAME leakage-safe held-out split finetune() used (same
    # seed, same params -> identical split) so the number shown here
    # matches what Calibrate actually reported, not an optimistic blend.
    file_stem = os.path.splitext(os.path.basename(path))[0]
    ckpt_prefix = ds["ckpt_prefix"]
    ckpt_stem = os.path.splitext(os.path.basename(model_path))[0]
    calib_marker = f"{ckpt_prefix}_finetuned_"
    if ckpt_stem == f"{calib_marker}{file_stem}":
        eval_rng = np.random.default_rng(0)
        _, eval_idx = backend.leakage_safe_split(len(X), 0.7, eval_rng, gap=1)
        st.warning(f"This checkpoint was calibrated **on this exact file** -- showing accuracy on only the "
                  f"{len(eval_idx)} held-out windows from that calibration run (not the whole file, which "
                  f"would include the windows it was directly trained on and look more accurate than it "
                  f"really generalizes).")
        show_metrics(pred[eval_idx], Y_true[eval_idx])
    else:
        show_metrics(pred, Y_true)
else:
    st.info("No ground-truth kinematics in this file -- predictions only, no accuracy metrics.")

# ---- reach trajectories (CyberGlove/.mat only -- no verified joint mapping for the .fif device) ----
if has_anatomy:
    st.subheader("Reach trajectories: predicted vs. true, per movement")
    st.caption("3D fingertip path for a handful of labeled movements in this file -- drag to rotate. Each "
              "numbered/colored trial below is ONE real movement repetition: ◆ = start (rest, just before "
              "the movement begins), ● (yellow) = target (where the true hand ended up), ● (red) = "
              "endpoint (where the PREDICTED path ended up -- its distance from the target is the model's "
              "positional error for that movement). **Built on best-effort assumptions**: the MCP/PIP/DIP "
              "channel mapping (unverified, see note above) *and* generic illustrative adult finger-segment "
              "lengths (not measured for this subject or any specific prosthetic design) -- treat the mm "
              "axes as approximate scale, the *shape and endpoint-error* comparison is the useful part.")
    if Y_true is not None and labels is not None:
        finger_sel = st.selectbox("Finger", list(FINGER_JOINTS.keys()), index=1, key="ws_finger")
        blocks = movement_blocks(labels)
        if not blocks:
            st.info("No labeled movement segments long enough to plot in this file.")
        else:
            chosen_blocks = blocks[:5]
            ref = Y_true
            norm_true_all = normalize_channels(Y_true, ref)
            norm_pred_all = normalize_channels(pred, ref)
            TRUE_COLOR, PRED_COLOR = "yellow", "red"
            traces, caption_bits = [], []
            for k, (lab, s, e) in enumerate(chosen_blocks):
                pick = np.linspace(s, e - 1, 200).astype(int) if (e - s) > 200 else np.arange(s, e)
                name = _action_name(lab)
                true_pts = np.array([fingertip_xyz(norm_true_all[i], finger_sel) for i in pick])
                pred_pts = np.array([fingertip_xyz(norm_pred_all[i], finger_sel) for i in pick])
                traces.append(go.Scatter3d(x=[true_pts[0, 0]], y=[true_pts[0, 1]], z=[true_pts[0, 2]],
                                           mode="markers+text", marker=dict(color=TRUE_COLOR, size=9, symbol="diamond"),
                                           text=[str(k + 1)], textposition="top center",
                                           name="Start", legendgroup="start", showlegend=(k == 0),
                                           hovertext=f"{k + 1}. {name} -- start", hoverinfo="text"))
                traces.append(go.Scatter3d(x=[true_pts[-1, 0]], y=[true_pts[-1, 1]], z=[true_pts[-1, 2]],
                                           mode="markers", marker=dict(color=TRUE_COLOR, size=7, symbol="circle"),
                                           name="Target (true endpoint)", legendgroup="target", showlegend=(k == 0),
                                           hovertext=f"{k + 1}. {name} -- target", hoverinfo="text"))
                traces.append(go.Scatter3d(x=[pred_pts[-1, 0]], y=[pred_pts[-1, 1]], z=[pred_pts[-1, 2]],
                                           mode="markers", marker=dict(color=PRED_COLOR, size=7, symbol="circle"),
                                           name="Endpoint (predicted)", legendgroup="endpoint", showlegend=(k == 0),
                                           hovertext=f"{k + 1}. {name} -- endpoint", hoverinfo="text"))
                err = float(np.linalg.norm(true_pts[-1] - pred_pts[-1]))
                caption_bits.append(f"**{k + 1}**: {name} (endpoint error ≈ {err:.0f} mm)")

            fig4 = go.Figure(data=traces)
            fig4.update_layout(
                scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                          aspectmode="data"),
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, t=20, b=0),
                height=550,
            )
            st.plotly_chart(fig4, width="stretch")
            st.caption("&nbsp;&nbsp;·&nbsp;&nbsp;".join(caption_bits), unsafe_allow_html=True)
    else:
        st.info("No ground truth and/or movement labels in this file -- nothing to plot trajectories for.")

# ---- compare checkpoints ----
st.subheader("Compare model checkpoints")
st.caption("Runs every saved checkpoint on this file and compares R² averaged across joints -- useful "
          "for seeing whether SSL pretraining or fine-tuning actually helped on this specific recording.")
if Y_true is None:
    st.caption("Unavailable -- this file has no ground-truth kinematics to score against.")
elif st.button("Compare all checkpoints", icon=":material/bar_chart:"):
    results = {}
    with st.spinner("Running all checkpoints..."):
        for mf in model_files:
            net_i, ck_i = load_kinematics_model(mf, ds)
            pred_i = backend.predict_batches(net_i, ds, ck_i, X, M)
            ss_res = np.sum((pred_i - Y_true) ** 2, axis=0)
            ss_tot = np.sum((Y_true - Y_true.mean(0, keepdims=True)) ** 2, axis=0)
            results[mf] = float((1 - ss_res / ss_tot).mean())
    labels_cmp = [os.path.splitext(os.path.basename(m))[0] for m in results]
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    bars = ax5.bar(labels_cmp, list(results.values()))
    ax5.bar_label(bars, fmt="%.3f")
    ax5.set_ylabel("R² averaged across joints")
    ax5.set_title(f"Model comparison on {os.path.basename(path)}")
    plt.xticks(rotation=30, ha="right")
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

if tmp_path is not None:
    os.unlink(tmp_path)

csv = "\n".join(",".join(f"{v:.4f}" for v in row) for row in pred)
st.download_button("Download predicted kinematics (CSV)", csv, "predicted_kinematics.csv", "text/csv")

if __name__ == "__main__":
    import sys, streamlit.web.cli as stcli
    from streamlit import runtime
    if not runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
