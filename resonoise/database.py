"""SQLite persistence: original audio, extracted features, analysis results and processed audio."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .errors import NotFoundError

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL, original_filename TEXT,
    created_at REAL NOT NULL, sample_rate INTEGER NOT NULL, original_sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL, duration REAL NOT NULL, size_bytes INTEGER NOT NULL,
    audio_path TEXT NOT NULL, reference_path TEXT, container TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    recording_id TEXT PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    created_at REAL NOT NULL, features_json TEXT NOT NULL, payload_json TEXT NOT NULL,
    source_class TEXT, confidence REAL
);
CREATE TABLE IF NOT EXISTS processed (
    id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    created_at REAL NOT NULL, profile TEXT, params_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
    payload_json TEXT NOT NULL, audio_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    label TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'overall', features_json TEXT NOT NULL, created_at REAL NOT NULL,
    UNIQUE(recording_id, scope)
);
CREATE INDEX IF NOT EXISTS idx_processed_rec ON processed(recording_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "resonoise.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # ------------------------------------------------------------------ recordings
    def add_recording(self, *, name, source, original_filename, sample_rate, original_sample_rate, channels, duration,
                      size_bytes, audio_bytes: bytes, container, reference_bytes: bytes | None = None) -> dict:
        rid = new_id()
        audio_path = self.audio_dir / f"{rid}.wav"
        audio_path.write_bytes(audio_bytes)
        ref_path = None
        if reference_bytes is not None:
            ref_path = self.audio_dir / f"{rid}_reference.wav"
            ref_path.write_bytes(reference_bytes)
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, name, source, original_filename, time.time(), sample_rate, original_sample_rate, channels,
                 duration, size_bytes, str(audio_path), str(ref_path) if ref_path else None, container))
        return self.get_recording(rid)

    def get_recording(self, rid: str) -> dict:
        with self._connect() as con:
            row = con.execute("SELECT * FROM recordings WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise NotFoundError("That recording does not exist (it may have been deleted).")
        return dict(row)

    def list_recordings(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """SELECT r.*, a.source_class, a.confidence,
                          (SELECT COUNT(*) FROM processed p WHERE p.recording_id = r.id) AS n_processed
                   FROM recordings r LEFT JOIN analyses a ON a.recording_id = r.id
                   ORDER BY r.created_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def delete_recording(self, rid: str) -> None:
        rec = self.get_recording(rid)
        with self._lock, self._connect() as con:
            paths = [rec["audio_path"], rec["reference_path"]]
            paths += [r["audio_path"] for r in con.execute("SELECT audio_path FROM processed WHERE recording_id=?", (rid,))]
            con.execute("DELETE FROM recordings WHERE id=?", (rid,))
        for p in paths:
            if p:
                Path(p).unlink(missing_ok=True)

    # ------------------------------------------------------------------ analyses
    def save_analysis(self, rid: str, features: dict, payload: dict, source_class: str | None, confidence: float | None):
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?)",
                (rid, time.time(), json.dumps(features), json.dumps(payload), source_class, confidence))

    def get_analysis(self, rid: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM analyses WHERE recording_id=?", (rid,)).fetchone()
        if row is None:
            return None
        return {"created_at": row["created_at"], "features": json.loads(row["features_json"]),
                "payload": json.loads(row["payload_json"])}

    # ------------------------------------------------------------------ processed audio
    def add_processed(self, rid: str, *, profile, params: dict, metrics: dict, payload: dict, audio_bytes: bytes) -> str:
        pid = new_id()
        path = self.audio_dir / f"{pid}_processed.wav"
        path.write_bytes(audio_bytes)
        payload = dict(payload, processed_id=pid)
        with self._lock, self._connect() as con:
            con.execute("INSERT INTO processed VALUES (?,?,?,?,?,?,?,?)",
                        (pid, rid, time.time(), profile, json.dumps(params), json.dumps(metrics),
                         json.dumps(payload), str(path)))
        return pid

    def get_processed(self, pid: str) -> dict:
        with self._connect() as con:
            row = con.execute("SELECT * FROM processed WHERE id=?", (pid,)).fetchone()
        if row is None:
            raise NotFoundError("That processed result does not exist.")
        d = dict(row)
        d["params"], d["metrics"] = json.loads(d.pop("params_json")), json.loads(d.pop("metrics_json"))
        d["payload"] = json.loads(d.pop("payload_json"))
        return d

    def list_processed(self, rid: str) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT id, created_at, profile, params_json, metrics_json FROM processed "
                               "WHERE recording_id=? ORDER BY created_at DESC", (rid,)).fetchall()
        return [{"id": r["id"], "created_at": r["created_at"], "profile": r["profile"],
                 "params": json.loads(r["params_json"]), "metrics": json.loads(r["metrics_json"])} for r in rows]

    def delete_processed(self, pid: str) -> None:
        rec = self.get_processed(pid)
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM processed WHERE id=?", (pid,))
        Path(rec["audio_path"]).unlink(missing_ok=True)

    # ------------------------------------------------------------------ labels used for retraining
    def set_label(self, rid: str, label: str, features: dict, scope: str = "overall") -> None:
        with self._lock, self._connect() as con:
            con.execute("INSERT OR REPLACE INTO labels (recording_id, label, scope, features_json, created_at) "
                        "VALUES (?,?,?,?,?)", (rid, label, scope, json.dumps(features), time.time()))

    def list_labels(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM labels ORDER BY id").fetchall()
        return [{"recording_id": r["recording_id"], "label": r["label"], "scope": r["scope"],
                 "features": json.loads(r["features_json"])} for r in rows]

    def get_label(self, rid: str, scope: str = "overall") -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT label FROM labels WHERE recording_id=? AND scope=?", (rid, scope)).fetchone()
        return row["label"] if row else None

    def stats(self) -> dict:
        with self._connect() as con:
            def count(t):
                return con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            return {"recordings": count("recordings"), "analyses": count("analyses"),
                    "processed": count("processed"), "labels": count("labels")}
