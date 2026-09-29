"""Окружение тестов бэкенда: временная SQLite, временное хранилище, фейковые модули ядра.

Очередь — настоящая (потоки на камеру), пересчёт площадки — без дебаунса,
чтобы тесты были детерминированы: после `env.wait()` всё посчитано.
"""
from __future__ import annotations

import datetime as dt
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.app.fakes import Fakes


@dataclass
class Env:
    client: TestClient
    fakes: Fakes
    tmp: Path

    def wait(self, timeout: float = 20.0) -> None:
        from app.services.queue import frame_queue
        assert frame_queue.wait_idle(timeout), "очередь не затихла"

    def wait_job(self, job_id: str, timeout: float = 20.0, until=("done", "postponed", "failed")) -> dict:
        deadline = time.monotonic() + timeout
        job = {}
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["state"] in until:
                self.wait()
                return self.client.get(f"/api/jobs/{job_id}").json()
            time.sleep(0.05)
        raise AssertionError(f"задание не завершилось: {job}")

    def site(self, **kw) -> dict:
        r = self.client.post("/api/sites", json={"name": kw.pop("name", "Тестовый объект"), **kw})
        assert r.status_code == 201, r.text
        return r.json()

    def camera(self, site_id: int, **kw) -> dict:
        r = self.client.post(f"/api/sites/{site_id}/cameras", json={"name": kw.pop("name", "Камера 1"), **kw})
        assert r.status_code == 201, r.text
        return r.json()

    def upload(self, camera_id: int, files: list[tuple[str, bytes]], **form) -> dict:
        r = self.client.post(f"/api/cameras/{camera_id}/upload",
                             files=[("files", (name, data)) for name, data in files],
                             data={k: str(v) for k, v in form.items()})
        assert r.status_code == 202, r.text
        return self.wait_job(r.json()["job_id"])

    def frames(self, camera_id: int) -> list[dict]:
        return self.client.get(f"/api/cameras/{camera_id}/frames", params={"order": "asc", "limit": 1000}).json()


@pytest.fixture
def env(tmp_path, monkeypatch):
    from app import db, storage
    from app.config import settings
    from app.main import create_app
    from app.services import pipeline, providers, views
    from app.services import settings as settings_svc

    monkeypatch.setattr(settings, "recompute_debounce_s", 0.0)
    monkeypatch.setattr(settings, "provider_status_ttl_s", 0.0)
    monkeypatch.setattr(settings, "postponed_retry_s", 3600.0)
    monkeypatch.setattr(settings, "recover_scan_s", 3600.0)
    monkeypatch.setattr(settings, "tmp_dir", str(tmp_path / "tmp"))
    monkeypatch.setattr(settings, "demo_dir", str(tmp_path / "demo"))
    monkeypatch.setattr(settings, "default_mode", "local")
    monkeypatch.setattr(settings, "warm_models", False)   # фоновый прогрев сбил бы счётчики вызовов фейков
    db.configure(f"sqlite:///{tmp_path}/app.db")
    storage.configure("local", tmp_path / "storage")

    from app import auth
    auth.limiter.reset()                  # лимит неверных паролей — в памяти процесса, тесты не делят счёт

    fakes = Fakes()
    fakes.install(providers)
    providers.registry.reset()
    settings_svc.invalidate()
    pipeline.reset_caches()
    views._catalog_names.cache_clear()

    with TestClient(create_app()) as client:
        r = client.post("/api/login", json={"login": "admin", "password": "admin"})
        assert r.status_code == 200, r.text
        yield Env(client, fakes, tmp_path)

    providers.clear_overrides()
    providers.registry.reset()
    settings_svc.invalidate()
    pipeline.reset_caches()
    views._catalog_names.cache_clear()


# --------------------------------------------------------------------------
# синтетические кадры
# --------------------------------------------------------------------------

def scene(excavator_x: int | None = 60, truck_x: int | None = None, bg: int = 120, seed: int = 0,
          size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """Кадр «площадки»: шумный серый фон, синий прямоугольник — экскаватор, красный — самосвал."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.clip(rng.normal(bg, 12 if bg > 60 else 4, (h, w, 3)), 0, 255).astype(np.uint8)
    if excavator_x is not None:
        cv2.rectangle(img, (excavator_x, 100), (excavator_x + 40, 130), (255, 0, 0), -1)
    if truck_x is not None:
        cv2.rectangle(img, (truck_x, 160), (truck_x + 50, 190), (0, 0, 255), -1)
    return img


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def stamp(base: dt.datetime, minutes: int) -> str:
    return (base + dt.timedelta(minutes=minutes)).strftime("%Y_%m_%d_%H_%M_%S")


def series(n: int, base: dt.datetime = dt.datetime(2025, 5, 12, 8, 0), step_min: int = 20,
           move: int = 15, truck: bool = False, prefix: str = "cam", seed0: int = 0) -> list[tuple[str, bytes]]:
    """n кадров раз в step_min минут; экскаватор сдвигается на move пикселей каждый кадр.
    seed0 — другой шум фона: иначе одинаковые кадры разных серий отсеются как дубликаты."""
    return [(f"{prefix}_{stamp(base, i * step_min)}.jpg",
             jpeg(scene(40 + i * move, truck_x=200 if truck else None, seed=seed0 + i))) for i in range(n)]


def make_zip(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def make_video(path: Path, frames: int = 30, fps: float = 10.0) -> Path | None:
    """Видео с едущим экскаватором. mp4 — avc1/mp4v (что умеет сборка OpenCV), avi — MJPG."""
    codecs = ("MJPG",) if path.suffix == ".avi" else ("avc1", "mp4v")
    for codec in codecs:
        if path.exists():
            path.unlink()
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (320, 240))
        if not writer.isOpened():
            continue
        for i in range(frames):
            writer.write(scene(20 + i * 8, seed=i))
        writer.release()
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return path
    return None
