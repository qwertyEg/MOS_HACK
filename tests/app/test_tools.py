"""Консольные утилиты: засев демо и загрузка папки — отдельными процессами, как у пользователя.

Модули ядра в дочернем процессе могут отсутствовать (или не иметь весов) —
утилиты всё равно должны сохранить кадры и честно отчитаться.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from tests.app.conftest import series

ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], tmp: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp}/cli.db", "LOCAL_STORAGE_DIR": str(tmp / "storage"),
           "TMP_DIR": str(tmp / "tmp"), "DEMO_DIR": str(tmp / "demo"), "STORAGE_BACKEND": "local"}
    return subprocess.run([sys.executable, *args], cwd=tmp, env=env, capture_output=True, text=True, timeout=180)


def test_seed_demo_and_ingest_folder_cli(tmp_path):
    site_dir = tmp_path / "demo" / "sites" / "one"
    (site_dir / "cam1").mkdir(parents=True)
    for name, data in series(2, base=dt.datetime(2025, 5, 5, 9)):
        (site_dir / "cam1" / name).write_bytes(data)
    (site_dir / "site.json").write_text(json.dumps({"name": "CLI-объект", "timezone": "UTC"}), encoding="utf-8")

    r = _run([str(ROOT / "tools" / "seed_demo.py"), "--no-process"], tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "камера 1: 2 файл(ов)" in r.stdout

    extra = tmp_path / "more"
    extra.mkdir()
    for name, data in series(3, base=dt.datetime(2025, 5, 6, 9), seed0=50):
        (extra / name).write_bytes(data)
    r = _run([str(ROOT / "tools" / "ingest_folder.py"), "--camera", "1", "--folder", str(extra), "--timeout", "60"],
             tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "кадров 3" in r.stdout

    with sqlite3.connect(tmp_path / "cli.db") as conn:
        assert conn.execute("select count(*) from frames").fetchone()[0] == 5
        statuses = {row[0] for row in conn.execute("select status from frames where job_id is not null")}
    # без модулей/весов моделей — отложено с причиной, с ними — обработано; но не потеряно и не упало
    assert statuses <= {"pending", "done", "postponed"}

    # повторный запуск по той же папке ничего не дублирует
    r = _run([str(ROOT / "tools" / "ingest_folder.py"), "--camera", "1", "--folder", str(extra), "--no-process"],
             tmp_path)
    assert r.returncode == 0 and "дубликатов 3" in r.stdout
