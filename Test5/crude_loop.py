"""
Step 3, crudest form: EMG window -> CNN -> MyoHand joints, in a bare MuJoCo viewer.

No PyQt, no multi-panel display, no threading yet -- the point of this script is only to
prove the one thing everything else depends on: does a real prediction from the trained
Masked-SSL model actually move a real musculoskeletal hand model. Direct-angle (kinematic)
control: qpos is set straight from the predicted joint angles and mj_forward() recomputes
derived state -- the 39 muscle actuators are not being driven yet (that's the tracking-
controller step, for later, once this loop is confirmed working).

Run with:  python3 crude_loop.py --task "Holding a cup"
Close the viewer window (or Ctrl+C) to stop.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import gymnasium as gym
import myosuite   # noqa: F401 -- import registers the Myo* environments with gymnasium
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import myohand_mapping as mm

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, 'masked_ssl_model.pt')
MOVEMENT_CATALOG_PATH = os.path.join(HERE, 'movement_catalog.json')
SHARED_STATE_PATH = os.path.join(HERE, 'shared_state.npz')
DATA_DIR = '/Users/kailashjram/Desktop/MSC FINAL PROJECT/Subject Data'
BANDPASS_LOW, BANDPASS_HIGH = 20, 450
SMOOTHING_ALPHA = 0.15   # same reasoning as the web app: raw per-window predictions are
                         # noisier than the true signal -- see every predicted-vs-actual
                         # plot throughout this project. Trades a little lag for a lot less
                         # visible jitter, same fix real myoelectric prostheses apply.
                         # Lowered from 0.25 -- physical joint clamping (see
                         # myohand_mapping.apply_glove_row_to_qpos) turns noisy overshoot
                         # past a joint's real range into visible snapping at the limit,
                         # which reads as extra jerkiness on top of the raw prediction noise.


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
    def forward(self, x):
        return self.conv(x)


class EMGRegressor(nn.Module):
    def __init__(self, encoder, n_joints):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_joints),
        )
    def forward(self, x):
        feat = self.pool(self.encoder(x)).squeeze(-1)
        return self.head(feat)


def bandpass_filter(emg, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, emg, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Holding a cup')
    args = parser.parse_args()

    print('Loading checkpoint...')
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = EMGRegressor(EMGEncoder(ckpt['n_channels']), ckpt['n_joints'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    fs, window_ms, stride_ms = ckpt['fs'], ckpt['window_ms'], ckpt['stride_ms']
    window_size, stride = int(fs * window_ms / 1000), int(fs * stride_ms / 1000)
    xm, xs, ym, ys = ckpt['xm'], ckpt['xs'], ckpt['ym'], ckpt['ys']
    subject = ckpt['fine_tune_subject']

    import json
    with open(MOVEMENT_CATALOG_PATH) as fh:
        catalog = json.load(fh)
    movement_id = catalog[args.task]['movement_id']

    print(f'Loading S{subject} E2 EMG data...')
    mat_path = os.path.join(DATA_DIR, f'S{subject}_E2_A1.mat')
    data = loadmat(mat_path)
    emg = bandpass_filter(data['emg'], fs).astype(np.float32)
    stim = data['restimulus'].flatten()
    starts = [s for s in range(0, len(emg) - window_size + 1, stride)
              if np.bincount(stim[s:s + window_size].astype(int)).argmax() == movement_id]
    print(f'Task "{args.task}" -> movement {movement_id} -> {len(starts)} windows found')

    print('Loading MyoHand model...')
    env = gym.make('myoHandPoseFixed-v0')
    mj_model = env.unwrapped.mj_model
    mj_data = env.unwrapped.mj_data
    joint_index_map = mm.build_joint_index_map(mj_model)
    print(f'Mapped {len(joint_index_map)}/{len(mm.GLOVE_TO_MYOHAND_JOINT)} glove channels to MyoHand joints.')

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    smoothed = None
    frame_index = 0
    print('Opening MuJoCo viewer -- close the window or Ctrl+C to stop.')
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        try:
            while viewer.is_running():
                t0 = time.time()
                s = starts[frame_index % len(starts)]
                window = emg[s:s + window_size]
                window_norm = ((window - xm) / xs).astype(np.float32)
                x = torch.from_numpy(window_norm.T[None]).float()
                with torch.no_grad():
                    pred_norm = model(x).numpy()[0]
                pred_real = pred_norm * ys + ym

                smoothed = pred_real.copy() if smoothed is None else \
                    SMOOTHING_ALPHA * pred_real + (1 - SMOOTHING_ALPHA) * smoothed

                mm.apply_glove_row_to_qpos(smoothed, mj_data.qpos, joint_index_map)
                mujoco.mj_forward(mj_model, mj_data)   # kinematics only, no muscle dynamics yet
                viewer.sync()

                # Shared state for emg_panel.py (a separate process/window -- mjpython's
                # viewer and PyQt both want to own the "main thread" on macOS, so rather than
                # fight that, they're two windows reading/writing a small file between them,
                # same bridge pattern already used for the Test4 web app. Atomic write (temp
                # file + rename) so the reader never sees a half-written file.
                tmp_path = SHARED_STATE_PATH + '.tmp'
                with open(tmp_path, 'wb') as fh:
                    np.savez(fh, emg_window=window, joint_angles=smoothed, frame_index=frame_index)
                os.replace(tmp_path, SHARED_STATE_PATH)

                frame_index += 1
                elapsed = time.time() - t0
                time.sleep(max(0.0, stride_ms / 1000 - elapsed))
        except KeyboardInterrupt:
            pass
    env.close()
    print('Stopped.')


if __name__ == '__main__':
    main()
