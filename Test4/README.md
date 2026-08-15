# Test 4 — Real-Time Prosthetic Arm Simulation (Web)

A browser-based, real-time simulation front end driven by the SSL-pretrained model from `Test2`/`Test3` — the first attempt at an interactive demo, before `Test5` moved to a MuJoCo physics simulation and `Test7`–`Test9` moved to the Streamlit-based analysis frontend.

## What's here

- `web/` — `server.py` + `static/` assets: the browser-based simulation app.
- `db.py` — SQLite storage for the simulation (`sessions`, `movement_catalog`, `session_frames` tables). Session frames are logged for later review only — **not** used to train the CNN, which is trained entirely offline via the Ninapro pipeline in `Test2`/`Test3`.
- `prosthetic_sim.db` — the SQLite database file itself (large; session-log data only, not a project deliverable).
- `train_masked_checkpoint.py` — standalone reproduction of just the masked-reconstruction SSL slice of `Test2`'s pipeline (pretrain → fine-tune), because `Test2`'s own trained model only ever lived in that notebook's kernel process and was never saved. Same hyperparameters as the original run (`MASKED_EPOCHS=100`, `FT_EPOCHS=80`) — an exact reproduction, not a fresh tune. Produces `masked_ssl_model.pt`.
- `movement_catalog.json` / `movement_mapping.ipynb` — the task/movement label set (carried over from `Test3`) and how it maps to specific EMG recordings.
- `fig_emg_holding_a_cup.png`, `fig_emg_holding_a_pen.png`, `fig_emg_picking_up_a_box.png`, `fig_emg_turning_on_a_switch.png`, `fig_emg_unlocking_a_door_key_lateral_grip.png` — per-task EMG figures used by the simulation.

## Where this leads

`masked_ssl_model.pt` (the standalone checkpoint reproduced here) is reused directly by `Test5`'s MuJoCo simulation.
