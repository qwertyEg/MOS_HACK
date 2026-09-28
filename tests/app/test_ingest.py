"""Приём кадров: метки времени, дедуп, zip, видео, поток камеры (/api/ingest)."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from app.services import ingest
from tests.app.conftest import jpeg, make_video, make_zip, scene, series


@pytest.mark.parametrize("name, expected, has_time", [
    ("doric_2006_03_08_12_30_22.jpg", dt.datetime(2006, 3, 8, 12, 30, 22), True),
    ("IMG_20240314_103000.jpg", dt.datetime(2024, 3, 14, 10, 30, 0), True),
    ("cam1-2024-03-14T10-30-00.png", dt.datetime(2024, 3, 14, 10, 30, 0), True),
    ("2024-03-14_10-30.jpg", dt.datetime(2024, 3, 14, 10, 30), True),
    ("report_2024-03-14.jpg", dt.datetime(2024, 3, 14, 12, 0), False),
    ("14.03.2024.jpg", dt.datetime(2024, 3, 14, 12, 0), False),
    ("photo_2024_13_45_10_00_00.jpg", None, None),   # месяц 13 — не дата
    ("IMG_0042.jpg", None, None),
])
def test_parse_timestamp(name, expected, has_time):
    got = ingest.parse_timestamp(name)
    if expected is None:
        assert got is None
    else:
        assert got == (expected, has_time)


def test_upload_images_timestamps_from_names_in_site_tz(env):
    site = env.site(timezone="Europe/Moscow")
    cam = env.camera(site["id"])
    files = series(3, base=dt.datetime(2025, 5, 12, 8, 0))
    job = env.upload(cam["id"], list(reversed(files)))       # порядок загрузки не важен
    assert job["state"] == "done" and job["total"] == 3 and job["done"] == 3, job
    frames = env.frames(cam["id"])
    # 08:00 по Москве = 05:00 UTC; кадры идут по времени съёмки
    assert [f["captured_at"] for f in frames] == [
        "2025-05-12T05:00:00+00:00", "2025-05-12T05:20:00+00:00", "2025-05-12T05:40:00+00:00"]
    assert all(f["ts_source"] == "имя файла" for f in frames)
    assert env.client.get(f"/api/cameras/{cam['id']}").json()["image_w"] == 320

    # повторная загрузка тех же файлов — дубликаты, новых кадров нет
    job2 = env.upload(cam["id"], files)
    assert job2["total"] == 0 and job2["duplicates"] == 3
    assert len(env.frames(cam["id"])) == 3


def test_upload_without_dates_uses_start_at_and_interval(env):
    site = env.site(timezone="UTC")
    cam = env.camera(site["id"])
    files = [(f"photo_{i}.jpg", jpeg(scene(30 + 10 * i, seed=i))) for i in range(3)]
    env.upload(cam["id"], files, start_at="2025-06-01T10:00:00", interval_min=30)
    frames = env.frames(cam["id"])
    assert [f["captured_at"] for f in frames] == [
        "2025-06-01T10:00:00+00:00", "2025-06-01T10:30:00+00:00", "2025-06-01T11:00:00+00:00"]
    assert frames[0]["ts_source"] == "start_at + i·interval"


def test_upload_without_dates_and_start_at_uses_upload_time(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    before = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=5)
    env.upload(cam["id"], [("a.jpg", jpeg(scene(10, seed=1))), ("b.jpg", jpeg(scene(90, seed=2)))], interval_min=20)
    frames = env.frames(cam["id"])
    last = dt.datetime.fromisoformat(frames[-1]["captured_at"])
    first = dt.datetime.fromisoformat(frames[0]["captured_at"])
    assert last >= before and last - first == dt.timedelta(minutes=20)
    assert "время загрузки" in frames[0]["ts_source"]


def test_same_date_only_names_get_distinct_times(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], [("day_2025-05-01.jpg", jpeg(scene(10, seed=1))),
                           ("again_2025-05-01.jpg", jpeg(scene(90, seed=2)))])
    frames = env.frames(cam["id"])
    assert len(frames) == 2
    assert frames[0]["captured_at"] != frames[1]["captured_at"]
    assert all(f["captured_at"].startswith("2025-05-01T12:00") for f in frames)


def test_upload_zip(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    files = series(4, base=dt.datetime(2025, 5, 1, 9, 0), prefix="site/cam")
    archive = make_zip(files + [("notes.txt", b"hello"), ("__MACOSX/._x.jpg", b"junk")])
    job = env.upload(cam["id"], [("batch.zip", archive)])
    assert job["total"] == 4 and job["skipped"] == 1, job
    assert any("notes.txt" in e for e in job["errors"])
    assert [f["captured_at"][11:16] for f in env.frames(cam["id"])] == ["09:00", "09:20", "09:40", "10:00"]


@pytest.mark.parametrize("suffix", [".avi", ".mp4"])
def test_upload_video_realtime(env, suffix):
    path = make_video(env.tmp / f"clip{suffix}", frames=30, fps=10)
    if path is None:
        pytest.skip(f"сборка OpenCV не пишет {suffix}")
    cam = env.camera(env.site(timezone="UTC")["id"])
    # 3 с ролика, шаг 1 с → кадры 0, 10, 20; метки = start_at + позиция в ролике
    job = env.upload(cam["id"], [(path.name, path.read_bytes())], start_at="2025-05-01T12:00:00", step_s=1)
    assert job["total"] == 3, job
    frames = env.frames(cam["id"])
    assert [f["captured_at"] for f in frames] == [
        "2025-05-01T12:00:00+00:00", "2025-05-01T12:00:01+00:00", "2025-05-01T12:00:02+00:00"]
    detail = env.client.get(f"/api/frames/{frames[1]['id']}").json()
    assert detail["meta"]["video_frame"] == 10


def test_upload_video_timelapse(env):
    path = make_video(env.tmp / "timelapse.avi", frames=12, fps=15)
    assert path is not None
    cam = env.camera(env.site(timezone="UTC")["id"])
    job = env.upload(cam["id"], [(path.name, path.read_bytes())], start_at="2025-05-01T08:00:00",
                     video_mode="timelapse", every_n=3, interval_min=30)
    assert job["total"] == 4, job
    assert [f["captured_at"][11:16] for f in env.frames(cam["id"])] == ["08:00", "08:30", "09:00", "09:30"]


def test_upload_invalid_input(env):
    c = env.client
    cam = env.camera(env.site()["id"])
    url = f"/api/cameras/{cam['id']}/upload"
    assert c.post(url, data={"interval_min": "20"}).status_code == 400
    assert c.post(url, files=[("files", ("a.txt", b"x"))]).status_code == 400
    assert c.post(url, files=[("files", ("a.jpg", b"x"))], data={"start_at": "вчера"}).status_code == 400
    assert c.post(url, files=[("files", ("a.jpg", b"x"))], data={"interval_min": "0"}).status_code == 400
    assert c.post(url, files=[("files", ("a.jpg", b"x"))], data={"video_mode": "slow"}).status_code == 400
    assert c.post("/api/cameras/999/upload", files=[("files", ("a.jpg", b"x"))]).status_code == 404
    # битый jpg принимается заданием, но не становится кадром
    r = c.post(url, files=[("files", ("broken_2025_01_01_10_00_00.jpg", b"not an image"))])
    job = env.wait_job(r.json()["job_id"])
    assert job["total"] == 0 and job["skipped"] == 1 and "прочитать" in job["errors"][0]
    assert c.get("/api/jobs/nope").status_code == 404


def _ingest(env, cam, key, data, captured_at="2025-05-01T10:00:00+03:00", meta=None):
    return env.client.post("/api/ingest", headers={"X-Camera-Key": key},
                           files={"file": ("f.jpg", data, "image/jpeg")},
                           data={"camera_id": str(cam["id"]), "captured_at": captured_at,
                                 "meta": json.dumps(meta or {"sequence": 1})})


def test_ingest_stream_contract(env):
    cam = env.camera(env.site()["id"], kind="stream")
    data = jpeg(scene(50))
    env.client.post("/api/logout")                 # камера ходит без сессии, по ключу
    assert _ingest(env, cam, "wrong", data).status_code == 403
    assert _ingest(env, {"id": 999}, cam["ingest_key"], data).status_code == 403
    assert _ingest(env, cam, cam["ingest_key"], b"").status_code == 400
    assert _ingest(env, cam, cam["ingest_key"], data, captured_at="позавчера").status_code == 400
    assert _ingest(env, cam, cam["ingest_key"], b"garbage bytes").status_code == 400

    r = _ingest(env, cam, cam["ingest_key"], data, meta={"sequence": 7, "camera": "sim"})
    assert r.status_code == 202 and r.json()["queued"] is True
    frame_id = r.json()["frame_id"]
    # повтор той же метки (камера переслала после сбоя) — не новый кадр
    r2 = _ingest(env, cam, cam["ingest_key"], jpeg(scene(80)))
    assert r2.status_code == 202 and r2.json()["duplicate"] is True

    env.wait()
    env.client.post("/api/login", json={"login": "admin", "password": "admin"})
    detail = env.client.get(f"/api/frames/{frame_id}").json()
    assert detail["captured_at"] == "2025-05-01T07:00:00+00:00"
    assert detail["meta"]["sequence"] == 7 and detail["meta"]["ts_source"] == "камера"
    assert detail["status"] == "done" and detail["processed_a"] is True


def test_ingest_queue_full_returns_503(env, monkeypatch):
    from app.config import settings
    cam = env.camera(env.site()["id"])
    monkeypatch.setattr(settings, "queue_limit", 0)
    r = _ingest(env, cam, cam["ingest_key"], jpeg(scene(50)))
    assert r.status_code == 503 and "очередь" in r.json()["error"]
