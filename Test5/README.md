# Test 5 — MuJoCo Physical Hand Simulation

Moves the simulation from `Test4`'s web demo to a real musculoskeletal hand model (MyoSuite's MyoHand, in MuJoCo), to prove that a real prediction from the trained masked-SSL model can actually move a real physically-simulated hand.

## What's here

- `myohand_mapping.py` — maps the CNN's 22 CyberGlove-channel angle outputs (degrees) to MyoHand's MuJoCo joint positions (radians, `qpos`). Verified directly against a loaded `myoHandPoseFixed-v0` environment: 23 `qpos` joints with real anatomical names (`mcp2_flexion`, `pm3_flexion`, `cmc_abduction`, etc.), 39 muscle actuators. Targets `qpos` directly (kinematic/position control), not the 39 muscle actuators — deliberately the crude first version, proving the CNN→hand loop before adding a muscle-activation tracking controller.
- `crude_loop.py` — the bare-bones version: EMG window → CNN → MyoHand joints, in MuJoCo's native interactive viewer (`mjpython`). No custom UI. Run with `python3 crude_loop.py --task "Holding a cup"`.
- `emg_panel.py` — a **separate** window showing a live scrolling 12-channel EMG display, reading a shared-state file `crude_loop.py` writes each frame. Separate process because `mjpython`'s viewer and PyQt both want to own the main thread on macOS — run this in a second terminal alongside `crude_loop.py` (plain `python3`, no `mjpython` needed).
- `prosthetic_sim_app.py` — a combined single-window PyQt6 app: offscreen-rendered MuJoCo viewport + a 5-finger slider panel + a Manual/EMG mode toggle + the live EMG panel, all docked together. A distinct, independent alternative to `crude_loop.py` (not a replacement) — useful for posing the hand manually (e.g. reproducing `Test3`'s "closing all fingers" test case) as well as driving it from EMG.
- `masked_ssl_model.pt`, `movement_catalog.json` — carried over from `Test4`.
- `shared_state.npz` — the live frame-exchange file between `crude_loop.py` and `emg_panel.py`.

## Run order

```bash
mjpython crude_loop.py --task "Holding a cup"    # terminal 1
python3 emg_panel.py                             # terminal 2
```

or, for the combined single-window version:

```bash
python3 prosthetic_sim_app.py
```
