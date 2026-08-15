"""
Standalone FastAPI web app -- replaces the old Blender-driven pipeline entirely. No Blender
dependency at runtime: this process serves a webpage (Three.js + the converted right_hand.glb)
and streams live model predictions to it over a WebSocket.

Run with:  python3 server.py
Then open: http://127.0.0.1:8000

Model/data logic ported directly from the old inference_server.py -- same checkpoint, same
movement lookup, same EMA smoothing, same honest limitation: the model's regression head was
only fine-tuned on FINE_TUNE_SUBJECT (see checkpoint), so this always drives the simulation
from that one calibrated subject's real recorded EMG for the selected movement. Patient intake
fields (age/hand/amputation) affect logging and which EMG channel is shown, not which
subject's data drives the motion.
"""
import asyncio
import json
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

HERE = os.path.dirname(os.path.abspath(__file__))
TEST4_DIR = os.path.dirname(HERE)
CHECKPOINT_PATH = os.path.join(TEST4_DIR, 'masked_ssl_model.pt')
MOVEMENT_CATALOG_PATH = os.path.join(TEST4_DIR, 'movement_catalog.json')
DATA_DIR = '/Users/kailashjram/Desktop/MSC FINAL PROJECT/Subject Data'
STATIC_DIR = os.path.join(HERE, 'static')

BANDPASS_LOW, BANDPASS_HIGH = 20, 450
SMOOTHING_ALPHA = 0.25   # see inference_server.py's original note: trades a little lag for
                         # much less visible prediction jitter -- same fix real myoelectric
                         # prostheses apply to raw regressor/classifier output


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


print('Loading checkpoint...')
_ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
_model = EMGRegressor(EMGEncoder(_ckpt['n_channels']), _ckpt['n_joints'])
_model.load_state_dict(_ckpt['model_state_dict'])
_model.eval()
_FS, _WINDOW_MS, _STRIDE_MS = _ckpt['fs'], _ckpt['window_ms'], _ckpt['stride_ms']
_WINDOW_SIZE, _STRIDE = int(_FS * _WINDOW_MS / 1000), int(_FS * _STRIDE_MS / 1000)
_XM, _XS, _YM, _YS = _ckpt['xm'], _ckpt['xs'], _ckpt['ym'], _ckpt['ys']
_SUBJECT = _ckpt['fine_tune_subject']
print(f'Model loaded. Fine-tuned on subject S{_SUBJECT}. Device: cpu (real-time inference on this small model is fast enough without a GPU).')

with open(MOVEMENT_CATALOG_PATH) as fh:
    _catalog = json.load(fh)

print(f'Loading S{_SUBJECT} E2 EMG data...')
_mat_path = os.path.join(DATA_DIR, f'S{_SUBJECT}_E2_A1.mat')
_data = loadmat(_mat_path)
_emg = bandpass_filter(_data['emg'], _FS).astype(np.float32)
_stim = _data['restimulus'].flatten()
print('EMG data loaded.')


def windows_for_movement(movement_id):
    return [s for s in range(0, len(_emg) - _WINDOW_SIZE + 1, _STRIDE)
             if np.bincount(_stim[s:s + _WINDOW_SIZE].astype(int)).argmax() == movement_id]


def simulate_bicep_emg(effort_fraction, n_samples, rng):
    # NinaPro DB2 has no bicep channel -- nothing real to show here. Same approach as the
    # original inference_server.py: a plausible burst scaled by predicted exertion, always
    # clearly labelled SIMULATED wherever it's surfaced (see index.html).
    t = np.arange(n_samples)
    burst = np.abs(np.sin(2 * np.pi * 8 * t / n_samples)) * effort_fraction
    return (burst + rng.normal(0, 0.02, n_samples)).tolist()


app = FastAPI()


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get('/api/tasks')
def get_tasks():
    return {'tasks': list(_catalog.keys())}


@app.websocket('/ws/session')
async def session_ws(websocket: WebSocket):
    await websocket.accept()
    config = json.loads(await websocket.receive_text())
    task, age = config['task'], int(config['age'])
    hand, amputation = config['hand'], config['amputation']

    movement_id = _catalog[task]['movement_id']
    starts = windows_for_movement(movement_id)
    if not starts:
        await websocket.send_text(json.dumps({'error': f'No windows found for movement {movement_id}'}))
        await websocket.close()
        return

    session_id = db.start_session(age, hand, amputation, task, movement_id, _SUBJECT)
    print(f'Session {session_id}: task="{task}" movement={movement_id} hand={hand} amputation={amputation}')

    rng = np.random.default_rng(0)
    smoothed = None
    frame_index = 0
    t0 = asyncio.get_event_loop().time()

    try:
        while True:
            for s in starts:
                window = _emg[s:s + _WINDOW_SIZE]
                window_norm = ((window - _XM) / _XS).astype(np.float32)
                x = torch.from_numpy(window_norm.T[None]).float()
                with torch.no_grad():
                    pred_norm = _model(x).numpy()[0]
                pred_real = pred_norm * _YS + _YM

                smoothed = pred_real.copy() if smoothed is None else \
                    SMOOTHING_ALPHA * pred_real + (1 - SMOOTHING_ALPHA) * smoothed
                pred_real = smoothed

                forearm_emg = window[:, :2].mean(axis=1).tolist() if amputation == 'wrist' else None
                effort = float(np.clip(np.mean(pred_real) / 60.0, 0, 1))
                bicep_emg = simulate_bicep_emg(effort, _WINDOW_SIZE, rng)

                elapsed_ms = int((asyncio.get_event_loop().time() - t0) * 1000)
                payload = {
                    'frame_index': frame_index, 'elapsed_ms': elapsed_ms,
                    'joint_angles': pred_real.tolist(),
                    'forearm_emg': forearm_emg, 'bicep_emg_simulated': bicep_emg,
                    'amputation_level': amputation,
                }
                await websocket.send_text(json.dumps(payload))
                db.log_frame(session_id, frame_index, elapsed_ms, pred_real,
                             forearm_emg=forearm_emg, bicep_emg=bicep_emg)

                frame_index += 1
                await asyncio.sleep(_STRIDE_MS / 1000)
    except WebSocketDisconnect:
        print(f'Session {session_id}: client disconnected after {frame_index} frames.')


app.mount('/', StaticFiles(directory=STATIC_DIR), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
