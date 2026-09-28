"""Общая часть утилит tools/seed_demo.py, tools/ingest_folder.py, tools/ingest_video.py.

Утилиты работают в процессе, без HTTP: кадры пишутся в ту же БД и хранилище,
что у сервиса. С `--no-process` кадры остаются `pending` — запущенный сервис
подберёт их сам (сканирование каждые RECOVER_SCAN_S секунд). Без флага кадры
обрабатываются здесь же, с прогрессом в консоли; от двойной обработки при
работающем сервисе защищает атомарный «захват» кадра (pipeline.claim).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from app import auth, db
from app.config import settings
from app.models import Camera, Job
from app.services import ingest, pipeline, views
from app.services.queue import frame_queue

FINAL = ("done", "postponed", "failed")


def bootstrap() -> None:
    settings.export_env()
    db.init_db()
    with db.session() as s:
        auth.ensure_admin(s)


def ingest_files(camera_id: int, paths: list[Path], params: ingest.UploadParams, kind: str = "cli") -> str:
    """Сохранить файлы в камеру синхронно. → id задания."""
    with db.session() as s:
        cam = s.get(Camera, camera_id)
        if cam is None:
            raise SystemExit(f"камера {camera_id} не найдена")
        job = ingest.new_job(s, kind, cam, message=f"{len(paths)} файл(ов) из консоли")
    ingest.run_job(job.id, camera_id, paths, params, submit=False)
    return job.id


def jobs_state(job_ids: list[str]) -> list[dict]:
    with db.session() as s:
        return [views.job_json(s, s.get(Job, j)) for j in job_ids if s.get(Job, j) is not None]


def process(job_ids: list[str], timeout: float | None = None, echo: Callable[[str], None] = print) -> list[dict]:
    """Обработать кадры заданий в этом процессе и дождаться конца (или таймаута)."""
    frame_queue.start(supervise=False)
    started = time.monotonic()
    last_line = ""
    try:
        while True:
            jobs = jobs_state(job_ids)
            total = sum(j["total"] for j in jobs)
            done = sum(j["done"] + j["failed"] + j["postponed"] for j in jobs)
            line = f"обработано {done}/{total}"
            if line != last_line:
                echo(line)
                last_line = line
            if all(j["state"] in FINAL for j in jobs) and frame_queue.wait_idle(1.0):
                break
            if timeout and time.monotonic() - started > timeout:
                echo("таймаут: остальное дообработает сервис")
                break
            time.sleep(0.5)
    finally:
        frame_queue.shutdown()
    sites = {j["site_id"] for j in jobs_state(job_ids) if j["site_id"]}
    for site_id in sites:
        pipeline.recompute_site(site_id)
    return jobs_state(job_ids)


def print_jobs(jobs: list[dict], echo: Callable[[str], None] = print) -> None:
    for j in jobs:
        echo(f"задание {j['id'][:8]}: {j['state']}, кадров {j['total']}, готово {j['done']}, "
             f"отложено {j['postponed']}, ошибок {j['failed']}, дубликатов {j['duplicates']}, "
             f"пропущено {j['skipped']}")
        if j.get("postponed_reason"):
            echo(f"  отложено: {j['postponed_reason']}")
        for err in j["errors"][:10]:
            echo(f"  ! {err}")
