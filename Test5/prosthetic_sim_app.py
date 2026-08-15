"""
Single-window PyQt6 app: MuJoCo hand rendered offscreen (no mjpython needed -- separate
from, and does not depend on, crude_loop.py), a 5-finger slider panel, a Manual/EMG mode
toggle, and a live 12-channel EMG panel -- three docked panels around one central 3D
viewport, closer to the "Isaac Sim" layout than crude_loop.py's bare native viewer.

This is a distinct file from crude_loop.py on purpose (per explicit instruction) -- they are
two independent ways to look at the same underlying model/data, not one replacing the other.
crude_loop.py: mjpython + MuJoCo's native interactive viewer, CNN-only, no custom UI.
This file: plain python3 + PyQt6, offscreen-rendered viewport + manual slider control +
live EMG, all in one window.

Manual mode: drag any finger's curl slider (0=fully extended, 100=fully curled, mapped
linearly across that finger's real joint range) to pose the hand directly. Useful for
testing specific poses -- e.g. the "closing all fingers" case from Test3, which the model
was measurably weaker on -- without needing real EMG for that exact case.

EMG mode: real EMG windows for the chosen task stream through the trained Masked-SSL CNN,
smoothed (EMA) and sub-frame interpolated for visual smoothness, driving all 21 mapped
glove-channel joints together. The live EMG panel shows the exact window currently driving
the prediction, updated once per new CNN inference (not per render sub-frame).

Run with:  python3 prosthetic_sim_app.py --task "Holding a cup"
(plain python3 -- no mjpython needed.)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import mujoco
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import myohand_mapping as mm

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, 'masked_ssl_model.pt')
MOVEMENT_CATALOG_PATH = os.path.join(HERE, 'movement_catalog.json')
DATA_DIR = '/Users/kailashjram/Desktop/MSC FINAL PROJECT/Subject Data'
BANDPASS_LOW, BANDPASS_HIGH = 20, 450
N_EMG_CHANNELS = 12

# The standalone MyoSuite hand+forearm model (shoulder-to-fingertip skeleton, no full body --
# no skull/ribs/legs/feet). Same 23 joints / 39 muscles as myoHandPoseFixed-v0's gym-wrapped
# env, but without the extra body parts that env's scene includes for its own task purposes.
MYOHAND_XML_PATH = ('/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/'
                     'site-packages/myosuite/simhive/myo_sim/hand/myohand.xml')

SMOOTHING_ALPHA = 0.2   # raw per-window predictions are noisier than the true signal -- see
                        # every predicted-vs-actual plot throughout this project. Trades a
                        # little lag for a lot less visible jitter, same fix real myoelectric
                        # prostheses apply. Also reduces snapping when noisy predictions
                        # briefly overshoot a joint's real physical range (see
                        # myohand_mapping.apply_glove_row_to_qpos's clipping).
SUB_STEPS = 5   # interpolated render frames between each new CNN prediction. The CNN only
                # updates ~20 times/sec (one per 50ms window); rendering only at that rate
                # looks visibly steppy on top of the prediction noise. Linearly interpolating
                # qpos between the previous and new target over several sub-frames gives a
                # much higher visual update rate without fabricating any motion the model
                # didn't predict -- it's smoothing *between* two real predictions.

# One "curl" slider per finger, driving all of that finger's flexion joints together,
# linearly across each joint's own real range. Thumb abduction is deliberately left out of
# the curl concept (a single 0-1 curl doesn't map naturally onto an abduction axis) --
# add a second slider for it later if you want independent control.
FINGERS = {
    'Thumb':  ['cmc_flexion', 'mp_flexion', 'ip_flexion'],
    'Index':  ['mcp2_flexion', 'pm2_flexion', 'md2_flexion'],
    'Middle': ['mcp3_flexion', 'pm3_flexion', 'md3_flexion'],
    'Ring':   ['mcp4_flexion', 'pm4_flexion', 'md4_flexion'],
    'Little': ['mcp5_flexion', 'pm5_flexion', 'md5_flexion'],
}

CHANNEL_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
]


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


def resolve_joint_indices(mj_model, joint_names):
    """name -> qpos index, resolved against the actual loaded model (not assumed fixed)."""
    name_to_idx = {mj_model.joint(i).name: i for i in range(mj_model.nq)}
    resolved = {}
    for name in joint_names:
        if name not in name_to_idx:
            print(f'WARNING: joint "{name}" not found in this model -- skipping')
            continue
        resolved[name] = name_to_idx[name]
    return resolved


class ProstheticSimApp(QtWidgets.QMainWindow):
    def __init__(self, task):
        super().__init__()
        self.setWindowTitle('Prosthetic Arm Simulation')
        self.mode = 'manual'   # start here so opening the window doesn't depend on EMG data

        self.mj_model = mujoco.MjModel.from_xml_path(MYOHAND_XML_PATH)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        self.glove_joint_index_map = mm.build_joint_index_map(self.mj_model)   # EMG mode
        self.finger_joint_indices = {
            finger: resolve_joint_indices(self.mj_model, joints)
            for finger, joints in FINGERS.items()
        }   # manual mode

        self._load_model_and_data(task)

        self.frame_index = 0
        self.sub_step = 0
        self.smoothed = None
        self.prev_qpos = self.mj_data.qpos.copy()
        self.target_qpos = self.prev_qpos.copy()

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(1, self.stride_ms // SUB_STEPS))

        self._update_render()

    def _load_model_and_data(self, task):
        print('Loading checkpoint...')
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        self.cnn = EMGRegressor(EMGEncoder(ckpt['n_channels']), ckpt['n_joints'])
        self.cnn.load_state_dict(ckpt['model_state_dict'])
        self.cnn.eval()
        self.fs, self.window_ms, self.stride_ms = ckpt['fs'], ckpt['window_ms'], ckpt['stride_ms']
        self.window_size = int(self.fs * self.window_ms / 1000)
        self.stride = int(self.fs * self.stride_ms / 1000)
        self.xm, self.xs, self.ym, self.ys = ckpt['xm'], ckpt['xs'], ckpt['ym'], ckpt['ys']
        subject = ckpt['fine_tune_subject']

        with open(MOVEMENT_CATALOG_PATH) as fh:
            catalog = json.load(fh)
        movement_id = catalog[task]['movement_id']

        print(f'Loading S{subject} E2 EMG data...')
        mat_path = os.path.join(DATA_DIR, f'S{subject}_E2_A1.mat')
        data = loadmat(mat_path)
        self.emg = bandpass_filter(data['emg'], self.fs).astype(np.float32)
        stim = data['restimulus'].flatten()
        self.starts = [s for s in range(0, len(self.emg) - self.window_size + 1, self.stride)
                       if np.bincount(stim[s:s + self.window_size].astype(int)).argmax() == movement_id]
        print(f'Task "{task}" -> movement {movement_id} -> {len(self.starts)} windows found')

    def _build_ui(self):
        self.viewport_label = QtWidgets.QLabel()
        self.viewport_label.setMinimumSize(640, 480)
        self.setCentralWidget(self.viewport_label)

        controls_dock = QtWidgets.QDockWidget('Controls', self)
        controls_dock.setWidget(self._build_controls_panel())
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, controls_dock)

        emg_dock = QtWidgets.QDockWidget('Live EMG (12 channels)', self)
        emg_dock.setWidget(self._build_emg_panel())
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, emg_dock)

    def _build_controls_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        mode_box = QtWidgets.QGroupBox('Control Mode')
        mode_layout = QtWidgets.QHBoxLayout()
        self.manual_radio = QtWidgets.QRadioButton('Manual')
        self.emg_radio = QtWidgets.QRadioButton('EMG (CNN)')
        self.manual_radio.setChecked(True)
        self.manual_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.manual_radio)
        mode_layout.addWidget(self.emg_radio)
        mode_box.setLayout(mode_layout)
        layout.addWidget(mode_box)

        self.sliders = {}
        for finger in FINGERS:
            layout.addWidget(self._make_finger_slider(finger))

        self.status_label = QtWidgets.QLabel('Manual mode -- drag sliders to pose the hand.')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        return panel

    def _build_emg_panel(self):
        pg.setConfigOptions(antialias=True, background='k', foreground='w')
        layout_widget = pg.GraphicsLayoutWidget()
        self.emg_curves = []
        for ch in range(N_EMG_CHANNELS):
            plot = layout_widget.addPlot(row=ch, col=0)
            plot.setLabel('left', f'Ch{ch + 1}')
            plot.showGrid(x=True, y=False, alpha=0.2)
            # Real bandpass-filtered EMG amplitude is tiny and channel-dependent (measured
            # std ~2e-5 to ~5e-4 across channels, peaks up to ~0.01) -- a fixed guessed range
            # made every channel look flat in an earlier version. Auto-range instead.
            plot.enableAutoRange('y', True)
            plot.getAxis('bottom').setStyle(showValues=(ch == N_EMG_CHANNELS - 1))
            curve = plot.plot(pen=pg.mkPen(color=CHANNEL_COLORS[ch], width=1))
            self.emg_curves.append(curve)
        return layout_widget

    def _make_finger_slider(self, name):
        box = QtWidgets.QGroupBox(name)
        layout = QtWidgets.QVBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.valueChanged.connect(lambda v, f=name: self._on_slider_changed(f, v / 100.0))
        self.sliders[name] = slider
        layout.addWidget(QtWidgets.QLabel('Curl'))
        layout.addWidget(slider)
        box.setLayout(layout)
        return box

    def _on_mode_changed(self):
        self.mode = 'manual' if self.manual_radio.isChecked() else 'emg'
        if self.mode == 'manual':
            self.status_label.setText('Manual mode -- drag sliders to pose the hand.')
        else:
            self.status_label.setText('EMG mode -- CNN is driving the hand from real EMG.')
            # Re-anchor the interpolation state to wherever manual mode left the pose, so
            # switching modes doesn't snap the hand instantly to the CNN's current target.
            self.prev_qpos = self.mj_data.qpos.copy()
            self.target_qpos = self.prev_qpos.copy()
            self.sub_step = 0

    def _on_slider_changed(self, finger, curl):
        if self.mode != 'manual':
            return
        for qpos_idx in self.finger_joint_indices[finger].values():
            lo, hi = self.mj_model.jnt_range[qpos_idx]
            self.mj_data.qpos[qpos_idx] = lo + curl * (hi - lo)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_render()

    def _tick(self):
        if self.mode == 'emg':
            self._emg_step()

    def _emg_step(self):
        if self.sub_step == 0:
            s = self.starts[self.frame_index % len(self.starts)]
            window = self.emg[s:s + self.window_size]
            window_norm = ((window - self.xm) / self.xs).astype(np.float32)
            x = torch.from_numpy(window_norm.T[None]).float()
            with torch.no_grad():
                pred_norm = self.cnn(x).numpy()[0]
            pred_real = pred_norm * self.ys + self.ym
            self.smoothed = pred_real.copy() if self.smoothed is None else \
                SMOOTHING_ALPHA * pred_real + (1 - SMOOTHING_ALPHA) * self.smoothed

            self.prev_qpos = self.target_qpos.copy()
            new_target = self.prev_qpos.copy()
            mm.apply_glove_row_to_qpos(self.smoothed, new_target, self.glove_joint_index_map)
            self.target_qpos = new_target
            self.frame_index += 1

            # Update the live EMG panel once per new prediction (not per render sub-frame) --
            # this is the exact window that just drove the CNN, not a resampled/lagged view.
            t = np.arange(window.shape[0])
            for ch in range(N_EMG_CHANNELS):
                self.emg_curves[ch].setData(t, window[:, ch])

        self.sub_step += 1
        blend = self.sub_step / SUB_STEPS
        self.mj_data.qpos[:] = self.prev_qpos + blend * (self.target_qpos - self.prev_qpos)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_render()

        if self.sub_step >= SUB_STEPS:
            self.sub_step = 0

    def _update_render(self):
        self.renderer.update_scene(self.mj_data)
        img = self.renderer.render()   # (H, W, 3) uint8, contiguous
        h, w, _ = img.shape
        qimg = QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.viewport_label.setPixmap(QtGui.QPixmap.fromImage(qimg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Holding a cup')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = ProstheticSimApp(args.task)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
