"""
Separate window: live scrolling 12-channel EMG display, reading the shared state file that
crude_loop.py writes each frame.

Why a separate process/window instead of one combined window: mjpython's MuJoCo viewer
needs to own the "main thread" on macOS (that's the whole reason mjpython exists), and PyQt
also wants the main thread for its event loop -- rather than fight that conflict, this is
deliberately the "prove it with separate windows first" step your plan called for. Folding
this into one PyQt window with docked panels (replacing the native MuJoCo viewer with an
embedded render, or restructuring the threading) is the step after this one, once both
pieces are independently confirmed working.

Run alongside crude_loop.py, in a second terminal:
    python3 emg_panel.py
(plain python3 is fine here -- no mjpython needed, this window has no MuJoCo viewer in it.)
"""
import os
import sys
import zipfile

import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_STATE_PATH = os.path.join(HERE, 'shared_state.npz')
N_CHANNELS = 12
POLL_MS = 50   # matches crude_loop.py's ~50ms window stride

CHANNEL_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
]


class EmgPanel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live EMG (12 channels)')

        pg.setConfigOptions(antialias=True, background='k', foreground='w')
        layout_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout_widget)

        self.curves = []
        for ch in range(N_CHANNELS):
            plot = layout_widget.addPlot(row=ch, col=0)
            plot.setLabel('left', f'Ch{ch + 1}')
            plot.showGrid(x=True, y=False, alpha=0.2)
            # Real bandpass-filtered EMG amplitude turned out to be tiny and channel-
            # dependent (measured std ranged ~2e-5 to ~5e-4 across channels, peaks up to
            # ~0.01 at most) -- a fixed guessed range (originally +/-0.5) made every
            # channel look flat. Auto-ranging on Y adapts to whatever each channel's
            # real signal actually is, rather than guessing a single number for all 12.
            plot.enableAutoRange('y', True)
            plot.getAxis('bottom').setStyle(showValues=(ch == N_CHANNELS - 1))
            curve = plot.plot(pen=pg.mkPen(color=CHANNEL_COLORS[ch], width=1))
            self.curves.append(curve)

        self.status_label = QtWidgets.QLabel('Waiting for crude_loop.py to start writing...')
        self.statusBar().addWidget(self.status_label)

        self._last_frame_index = -1
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)

        # Position on the right half of the screen, per the request that started this file.
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w, h = screen.width() // 2, screen.height()
        self.setGeometry(screen.x() + screen.width() - w, screen.y(), w, h)

    def poll(self):
        if not os.path.exists(SHARED_STATE_PATH):
            return
        try:
            with np.load(SHARED_STATE_PATH) as data:
                frame_index = int(data['frame_index'])
                if frame_index == self._last_frame_index:
                    return
                self._last_frame_index = frame_index
                emg_window = data['emg_window']       # (400, 12)
                joint_angles = data['joint_angles']    # (22,)
        except (OSError, ValueError, zipfile.BadZipFile):
            return   # crude_loop.py may be mid-write; just try again next tick

        t = np.arange(emg_window.shape[0])
        for ch in range(N_CHANNELS):
            self.curves[ch].setData(t, emg_window[:, ch])

        thumb_deg = joint_angles[1]
        index_deg = joint_angles[4]
        self.status_label.setText(
            f'Frame {frame_index}  |  Thumb MCP {thumb_deg:.0f} deg  |  Index MCP {index_deg:.0f} deg')


def main():
    app = QtWidgets.QApplication(sys.argv)
    panel = EmgPanel()
    panel.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
