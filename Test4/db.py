"""
SQLite storage for the real-time prosthetic arm simulation.

Three tables, deliberately simple:
  sessions          -- one row per app run: patient intake fields + which task was chosen
  movement_catalog  -- mirrors Test4/movement_catalog.json so the app can query it by SQL
                       instead of re-parsing the JSON every time
  session_frames    -- one row per animation frame: predicted joint angles + EMG snippets,
                       for later review/debugging -- NOT used to train the CNN, which is
                       trained entirely from the offline NinaPro pipeline in Test2/Test3.
"""
import sqlite3
import json
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), 'prosthetic_sim.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    age              INTEGER,
    hand_side        TEXT CHECK(hand_side IN ('left', 'right')),
    amputation_level TEXT CHECK(amputation_level IN ('wrist', 'forearm')),
    task_name        TEXT NOT NULL,
    movement_id      INTEGER NOT NULL,
    source_subject   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS movement_catalog (
    task_name        TEXT PRIMARY KEY,
    movement_id      INTEGER NOT NULL,
    evidence_json    TEXT NOT NULL,
    confidence_note  TEXT
);

CREATE TABLE IF NOT EXISTS session_frames (
    frame_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(session_id),
    frame_index      INTEGER NOT NULL,
    elapsed_ms       INTEGER NOT NULL,
    joint_angles_json TEXT NOT NULL,      -- 22 predicted glove-channel angles, degrees
    forearm_emg_json  TEXT,               -- real recorded EMG snippet, or NULL if not shown
    bicep_emg_json    TEXT,               -- SIMULATED snippet (NinaPro has no bicep channel)
    bicep_emg_is_simulated INTEGER DEFAULT 1
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def load_movement_catalog_into_db(catalog_json_path=None):
    catalog_json_path = catalog_json_path or os.path.join(os.path.dirname(__file__), 'movement_catalog.json')
    with open(catalog_json_path) as fh:
        catalog = json.load(fh)
    conn = get_connection()
    for task_name, info in catalog.items():
        conn.execute(
            "INSERT OR REPLACE INTO movement_catalog (task_name, movement_id, evidence_json, confidence_note) "
            "VALUES (?, ?, ?, ?)",
            (task_name, info['movement_id'], json.dumps(info['evidence_mcp_degrees']), info['confidence']),
        )
    conn.commit()
    conn.close()
    print(f'Loaded {len(catalog)} movements into {DB_PATH}')


def start_session(age, hand_side, amputation_level, task_name, movement_id, source_subject):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO sessions (created_at, age, hand_side, amputation_level, task_name, movement_id, source_subject) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.strftime('%Y-%m-%d %H:%M:%S'), age, hand_side, amputation_level, task_name, movement_id, source_subject),
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def log_frame(session_id, frame_index, elapsed_ms, joint_angles, forearm_emg=None, bicep_emg=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO session_frames (session_id, frame_index, elapsed_ms, joint_angles_json, "
        "forearm_emg_json, bicep_emg_json) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, frame_index, elapsed_ms, json.dumps(list(map(float, joint_angles))),
         json.dumps(list(map(float, forearm_emg))) if forearm_emg is not None else None,
         json.dumps(list(map(float, bicep_emg))) if bicep_emg is not None else None),
    )
    conn.commit()
    conn.close()


if __name__ == '__main__':
    load_movement_catalog_into_db()
    print(f'Database ready at: {DB_PATH}')
