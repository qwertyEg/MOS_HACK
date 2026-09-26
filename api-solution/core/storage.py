"""SQLite: объекты, календарный план, кадры, кэш разборов и учёт расходов.

Разбор кадра кэшируется по ключу (хэш фото, модель, режим, версия промптов и
чек-листа) — одно и то же фото, загруженное повторно или в другой объект,
не тратит токены второй раз.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    object_type TEXT NOT NULL,
    floors_total INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plan (
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    stage_id INTEGER NOT NULL,
    start TEXT,
    end TEXT,
    PRIMARY KEY (object_id, stage_id)
);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    date_source TEXT,
    image_path TEXT NOT NULL,
    UNIQUE (object_id, sha256)
);
CREATE TABLE IF NOT EXISTS analyses (
    cache_key TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    thinking INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    cache_key TEXT,
    step TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    cached_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);
"""


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _now():
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, path=config.DB_PATH):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- объекты и план ---

    def list_objects(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM objects ORDER BY name")]

    def create_object(self, name, object_type, floors_total=None):
        with self._conn() as c:
            cur = c.execute("INSERT INTO objects (name, object_type, floors_total, created_at) VALUES (?, ?, ?, ?)",
                            (name, object_type, floors_total, _now()))
            return cur.lastrowid

    def update_object(self, object_id, object_type, floors_total):
        with self._conn() as c:
            c.execute("UPDATE objects SET object_type = ?, floors_total = ? WHERE id = ?",
                      (object_type, floors_total, object_id))

    def get_plan(self, object_id):
        with self._conn() as c:
            rows = c.execute("SELECT stage_id, start, end FROM plan WHERE object_id = ? ORDER BY stage_id",
                             (object_id,))
            return {r["stage_id"]: (r["start"], r["end"]) for r in rows}

    def save_plan(self, object_id, plan):
        """plan: {stage_id: (начало, окончание)} — date или ISO-строка, либо None."""
        with self._conn() as c:
            c.execute("DELETE FROM plan WHERE object_id = ?", (object_id,))
            c.executemany("INSERT INTO plan (object_id, stage_id, start, end) VALUES (?, ?, ?, ?)",
                          [(object_id, sid, _iso(s), _iso(e)) for sid, (s, e) in plan.items()])

    # --- кадры ---

    def add_frame(self, object_id, sha, filename, taken_at, date_source, jpeg):
        path = config.IMAGES_DIR / f"{sha}.jpg"
        if not path.exists():
            path.write_bytes(jpeg)
        with self._conn() as c:
            c.execute("""INSERT INTO frames (object_id, sha256, filename, taken_at, date_source, image_path)
                         VALUES (?, ?, ?, ?, ?, ?)
                         ON CONFLICT (object_id, sha256) DO UPDATE
                         SET taken_at = excluded.taken_at, date_source = excluded.date_source,
                             filename = excluded.filename""",
                      (object_id, sha, filename, taken_at, date_source, str(path)))

    def list_frames(self, object_id):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM frames WHERE object_id = ? ORDER BY taken_at, id", (object_id,))
            return [dict(r) for r in rows]

    def delete_frame(self, frame_id):
        with self._conn() as c:
            c.execute("DELETE FROM frames WHERE id = ?", (frame_id,))

    # --- кэш разборов и расходы ---

    def get_analysis(self, cache_key):
        with self._conn() as c:
            row = c.execute("SELECT result FROM analyses WHERE cache_key = ?", (cache_key,)).fetchone()
            return json.loads(row["result"]) if row else None

    def latest_analysis(self, sha):
        """Последний разбор фото при любых настройках — чтобы смена модели в UI не обнуляла сводку."""
        with self._conn() as c:
            row = c.execute("SELECT result FROM analyses WHERE sha256 = ? ORDER BY created_at DESC LIMIT 1",
                            (sha,)).fetchone()
            return json.loads(row["result"]) if row else None

    def save_analysis(self, cache_key, sha, model, thinking, strategy, result):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO analyses
                         (cache_key, sha256, model, thinking, strategy, result, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (cache_key, sha, model, int(thinking), strategy,
                       json.dumps(result, ensure_ascii=False), _now()))

    def log_call(self, cache_key, step, model, usage):
        with self._conn() as c:
            c.execute("""INSERT INTO calls (cache_key, step, model, prompt_tokens, cached_tokens,
                         completion_tokens, cost_usd, latency_ms, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (cache_key, step, model, usage.prompt_tokens, usage.cached_tokens,
                       usage.completion_tokens, usage.cost_usd, usage.latency_ms, _now()))

    def spend(self):
        with self._conn() as c:
            row = c.execute("""SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd), 0) AS cost,
                               COALESCE(SUM(prompt_tokens), 0) AS prompt,
                               COALESCE(SUM(cached_tokens), 0) AS cached,
                               COALESCE(SUM(completion_tokens), 0) AS completion FROM calls""").fetchone()
            return dict(row)
