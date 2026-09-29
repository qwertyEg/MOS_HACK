"""Надёжность и доступ из сети (ревью «надёжность» 28.09): российские имена кадров,
«бомбы» изображений, рестарт посреди работы, неверные часы камеры, NaN/переполнения,
вход, сессии, CSRF, заголовки, лимиты тела, SSRF, сшивка провайдеров модели Б."""
from __future__ import annotations

import datetime as dt
import json
import struct
import zlib

import pytest
from sqlalchemy import select

from app import db
from app.models import Frame, Job
from app.services import ingest
from tests.app.conftest import jpeg, make_zip, scene, series


# --------------------------------------------------------------------------
# имена кадров российских регистраторов
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("Камера 1_14.10.2020_07-06-00.jpg", dt.datetime(2020, 10, 14, 7, 6, 0)),      # Trassir / NVR
    ("14.03.2024 10.30.00.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),
    ("snapshot_14-03-2024_103000.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),
    ("14032024_103000.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),
    ("Фото 14.03.2024 в 10.30.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),           # мессенджер
    ("ch01_20240314103000.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),               # Hikvision
    ("2024-03-14 10.30.00[M][0@0][0].jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),    # Dahua
    ("photo_2024-03-14_10-30-00.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),         # Telegram
    ("IMG_20240314_103000.jpg", dt.datetime(2024, 3, 14, 10, 30, 0)),
    ("doric_2006_03_08_12_30_22.jpg", dt.datetime(2006, 3, 8, 12, 30, 22)),
])
def test_russian_recorder_names_keep_date_and_time(name, expected):
    assert ingest.parse_timestamp(name) == (expected, True)


def test_one_day_of_russian_recorder_frames_stays_one_day(env):
    """Раньше 4 кадра 14.10.2020 получали даты 06.07, 30.09, 14.10 и 10.12, все в 12:00."""
    cam = env.camera(env.site(timezone="Europe/Moscow")["id"])
    names = [f"Камера 1_14.10.2020_{t}.jpg" for t in ("07-06-00", "09-30-00", "12-10-00", "15-45-00")]
    job = env.upload(cam["id"], [(n, jpeg(scene(30 + 20 * i, seed=i))) for i, n in enumerate(names)])
    assert job["total"] == 4, job
    got = [f["captured_at"] for f in env.frames(cam["id"])]
    assert got == ["2020-10-14T04:06:00+00:00", "2020-10-14T06:30:00+00:00",
                   "2020-10-14T09:10:00+00:00", "2020-10-14T12:45:00+00:00"]


# --------------------------------------------------------------------------
# «бомбы» изображений, zip, рестарт
# --------------------------------------------------------------------------

def png_header(width: int, height: int) -> bytes:
    """PNG из одного заголовка: 57 байт вместо 900 МБ пикселей — декодировать нельзя, открыть — можно."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 16)) + chunk(b"IEND", b"")


def test_png_bomb_is_rejected_before_decoding(env):
    bomb = png_header(20000, 15000)
    with pytest.raises(ingest.ImageTooLarge):
        ingest.decode_image(bomb)
    r = env.client.post("/api/analyze", files={"file": ("big.png", bomb, "image/png")})
    assert r.status_code == 413 and "Мп" in r.json()["detail"]
    cam = env.camera(env.site(timezone="UTC")["id"])
    job = env.upload(cam["id"], [("big_2025_05_01_10_00_00.png", bomb)])
    assert job["total"] == 0 and job["skipped"] == 1 and "Мп" in job["errors"][0], job
    env.client.post("/api/logout")
    r = env.client.post("/api/ingest", headers={"X-Camera-Key": cam["ingest_key"]},
                        files={"file": ("f.png", bomb, "image/png")},
                        data={"camera_id": str(cam["id"]), "captured_at": "2025-05-01T10:00:00Z"})
    assert r.status_code == 400 and "Мп" in r.json()["detail"]


def test_large_jpeg_is_reduced_on_read_and_stored_reduced(env, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_image_mp", 0.05)          # 50 тыс. пикселей; кадр 320×240 = 77 тыс.
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(1))
    (fr,) = env.frames(cam["id"])
    detail = env.client.get(f"/api/frames/{fr['id']}").json()
    assert (detail["width"], detail["height"]) == (160, 120)
    assert detail["meta"]["reduced_x"] == 2


def test_frame_that_crashed_the_process_twice_is_not_retried_forever(env):
    from app.services.queue import frame_queue
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(1))
    fid = env.frames(cam["id"])[0]["id"]

    def crash() -> tuple:
        """Процесс умер посреди кадра: кадр остался «processing»; при старте — reset_stale."""
        with db.session() as s:
            s.get(Frame, fid).status = "processing"
            s.commit()
        frame_queue.reset_stale()
        with db.session() as s:
            fr = s.get(Frame, fid)
            return fr.status, fr.meta.get("crashes"), fr.note

    assert crash()[:2] == ("pending", 1)               # первый раз — ещё одна попытка
    status, crashes, note = crash()
    assert (status, crashes) == ("error", 2) and "обрывалась" in note
    # оператор разобрался — «повторить кадры с ошибкой»; дошёл до конца — счётчик падений сброшен
    site_id = env.client.get(f"/api/cameras/{cam['id']}").json()["site_id"]
    assert env.client.post(f"/api/sites/{site_id}/retry-errors").json()["frames"] == 1
    env.wait()
    assert env.frames(cam["id"])[0]["status"] == "done"
    with db.session() as s:
        assert "crashes" not in s.get(Frame, fid).meta


def test_upload_jobs_cut_by_restart_are_failed_and_tmp_cleaned(env):
    import os
    import time

    from app.config import settings
    from app.models import Camera
    cam = env.camera(env.site()["id"])
    with db.session() as s:
        job_id = ingest.new_job(s, "upload", s.get(Camera, cam["id"])).id
    stale = settings.path(settings.tmp_dir) / "uploads" / "deadbeef"
    stale.mkdir(parents=True)
    (stale / "x.jpg").write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    ingest.recover_after_restart()
    with db.session() as s:
        job = s.get(Job, job_id)
        assert job.state == "failed" and "перезапустился" in job.message
    assert not stale.exists()


def test_zip_bomb_is_skipped(env, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "zip_max_unpacked_mb", 0)
    cam = env.camera(env.site(timezone="UTC")["id"])
    job = env.upload(cam["id"], [("batch.zip", make_zip(series(2)))])
    assert job["total"] == 0 and any("распакованном виде" in e for e in job["errors"]), job


# --------------------------------------------------------------------------
# неверные часы камеры, догрузка задним числом, кадры без даты на архиве
# --------------------------------------------------------------------------

def test_frame_from_the_future_is_rejected(env):
    cam = env.camera(env.site(timezone="UTC")["id"], kind="stream")
    r = env.client.post("/api/ingest", headers={"X-Camera-Key": cam["ingest_key"], "X-Camera-Id": str(cam["id"])},
                        files={"file": ("f.jpg", jpeg(scene(50)), "image/jpeg")},
                        data={"captured_at": "2036-01-01T00:00:00Z"})
    assert r.status_code == 400 and "в будущем" in r.json()["detail"]
    assert env.frames(cam["id"]) == []
    # ключ проверяется до чтения тела, если камера назвалась заголовком
    r = env.client.post("/api/ingest", headers={"X-Camera-Key": "wrong", "X-Camera-Id": str(cam["id"])},
                        files={"file": ("f.jpg", b"x" * 1000, "image/jpeg")})
    assert r.status_code == 403


def test_late_upload_warns_that_hours_need_reprocessing(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    files = series(6)
    env.upload(cam["id"], files[3:])
    job = env.upload(cam["id"], files[:3])
    assert job["total"] == 3
    assert any("старше уже разобранных" in e and "Переанализировать" in e for e in job["errors"]), job


def test_undated_files_on_archive_site_need_start_at(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(2, base=dt.datetime(2020, 10, 14, 8, 0)))
    job = env.upload(cam["id"], [("photo.jpg", jpeg(scene(200, seed=9)))])
    assert job["total"] == 0 and job["skipped"] == 1
    assert any("start_at" in e for e in job["errors"]), job
    job = env.upload(cam["id"], [("photo.jpg", jpeg(scene(200, seed=9)))], start_at="2020-10-14T12:00:00")
    assert job["total"] == 1


# --------------------------------------------------------------------------
# NaN, Infinity, переполнения — 400/404, а не 500
# --------------------------------------------------------------------------

def test_non_finite_and_huge_numbers_are_400_not_500(env):
    c = env.client
    site = env.site()
    sid = site["id"]
    raw = lambda url, text, method="post": getattr(c, method)(   # noqa: E731
        url, content=text.encode(), headers={"content-type": "application/json"})
    assert raw("/api/settings", '{"thresholds":{"pipeline":{"stage_every_h":NaN}}}', "put").status_code == 400
    assert c.put("/api/settings", json={"thresholds": {"pipeline": {"recent_window_h": 1e300}}}).status_code == 400
    assert raw(f"/api/sites/{sid}", '{"floors_total": Infinity}', "patch").status_code == 400
    assert raw(f"/api/sites/{sid}/fleet", '[{"cls":"excavator","count":1e400}]', "put").status_code == 400
    assert raw(f"/api/sites/{sid}/hours", '{"cls":"excavator","hours":NaN}').status_code == 400
    assert c.post(f"/api/sites/{sid}/hours", json={"cls": "excavator", "hours": 1, "at": "0001-01-01"}).status_code == 400
    assert raw("/api/detect", '{"frame_id": 1e400}').status_code == 400
    assert raw("/api/detect", '{"frame_id": 99999999999999999999}').status_code == 400
    for url in ("/api/sites/99999999999999999999", "/api/frames/99999999999999999999",
                "/api/units/99999999999999999999"):
        assert c.get(url).status_code == 404, url
    assert c.delete(f"/api/sites/{sid}/hours/99999999999999999999").status_code in (400, 404)
    cam = env.camera(sid)
    assert c.get(f"/api/cameras/{cam['id']}/frames", params={"before": "0001-01-01"}).status_code == 400
    assert c.get(f"/api/cameras/{cam['id']}/frames", params={"before": "9999-12-31T23:59:59-23:59"}).status_code == 400
    plan = [{"stage_id": 3, "planned_start": "2026-01-01", "planned_end": "9999-12-31"}]
    assert c.put(f"/api/sites/{sid}/plan", json=plan).status_code == 400
    # сервис после всего этого жив и считает
    assert c.post(f"/api/sites/{sid}/recompute").status_code == 200


def test_unhandled_error_does_not_leak_details(env, monkeypatch):
    from fastapi.testclient import TestClient

    from app.services import views

    env.site()

    def boom(*a, **kw):
        raise RuntimeError("SELECT password_hash FROM users")
    monkeypatch.setattr(views, "site_card", boom)
    c = TestClient(env.client.app, raise_server_exceptions=False)      # без lifespan: очередь та же
    c.cookies.update(dict(env.client.cookies))
    r = c.get("/api/sites")
    assert r.status_code == 500
    assert "SELECT" not in r.text and "номер" in r.json()["detail"]


# --------------------------------------------------------------------------
# вход, сессии, CSRF, заголовки, лимиты, служебные страницы
# --------------------------------------------------------------------------

def test_login_is_rate_limited(env):
    c = env.client
    c.post("/api/logout")
    for _ in range(5):
        assert c.post("/api/login", json={"login": "admin", "password": "nope"}).status_code == 401
    r = c.post("/api/login", json={"login": "admin", "password": "nope"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    # во время паузы и верный пароль не проверяется
    assert c.post("/api/login", json={"login": "admin", "password": "admin"}).status_code == 429
    r = c.post("/login", data={"login": "admin", "password": "admin"}, follow_redirects=False)
    assert r.status_code == 429
    from app import auth
    auth.limiter.reset()
    assert c.post("/api/login", json={"login": "admin", "password": "admin"}).status_code == 200


def test_logout_revokes_copied_cookie(env):
    c = env.client
    stolen = dict(c.cookies)
    assert c.get("/api/me").status_code == 200
    c.post("/api/logout")
    from fastapi.testclient import TestClient
    other = TestClient(c.app)
    other.cookies.update(stolen)
    assert other.get("/api/me").status_code == 401


def test_security_headers_and_docs_behind_login(env):
    c = env.client
    for path in ("/", "/api/sites", "/static/js/ui.js"):
        h = c.get(path).headers
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff", path
        assert "frame-ancestors 'none'" in h["content-security-policy"], path
    assert c.get("/api/openapi.json").status_code == 200
    c.post("/api/logout")
    assert c.get("/api/openapi.json").status_code == 401
    assert c.get("/api/docs", follow_redirects=False).status_code == 303
    assert set(c.get("/api/health").json()) == {"ok", "version"}      # без входа — без адресов и провайдеров


def test_cross_site_writes_are_rejected(env):
    c = env.client
    assert c.post("/api/sites", json={"name": "свой"}, headers={"Origin": "http://testserver"}).status_code == 201
    assert c.post("/api/sites", json={"name": "чужой"}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/sites", json={"name": "чужой"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # простая форма text/plain с чужой страницы на JSON-API
    r = c.post("/api/sites", content=b'{"name":"csrf-test"}', headers={"content-type": "text/plain"})
    assert r.status_code == 415
    # ссылка «выйти» с чужой страницы не разлогинивает
    c.get("/logout", headers={"Sec-Fetch-Site": "cross-site"}, follow_redirects=False)
    assert c.get("/api/me").status_code == 200


def test_body_limits(env):
    c = env.client
    big = b'{"name": "' + b"x" * (9 * 1024 * 1024) + b'"}'
    assert c.post("/api/sites", content=big, headers={"content-type": "application/json"}).status_code == 413
    r = c.post("/login", content=b"login=admin&password=" + b"x" * 100_000,
               headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 413


def test_multi_range_to_static_is_served_whole(env):
    r = env.client.get("/static/js/ui.js", headers={"Range": "bytes=" + ",".join(f"{i}-{i}" for i in range(2000))})
    assert r.status_code == 200 and len(r.content) > 1000


def test_camera_address_cannot_point_to_service_networks(env):
    cam = env.camera(env.site()["id"], kind="stream")
    for uri in ("http://169.254.169.254/latest", "ftp://10.0.0.5/", "file:///etc/passwd", "http://0.0.0.0:9101"):
        r = env.client.post(f"/api/cameras/{cam['id']}/connect", json={"source_uri": uri})
        assert r.status_code == 400, (uri, r.text)
    from app.config import settings
    r = env.client.post(f"/api/cameras/{cam['id']}/connect", json={"source_uri": f"http://127.0.0.1:{settings.port}"})
    assert r.status_code == 400


def test_media_needs_login_and_stays_inside_storage(env):
    c = env.client
    for key in ("..%2F..%2Fetc%2Fpasswd", "%2Fetc%2Fpasswd", "a%5C..%5Cb", "x%00y"):
        assert c.get(f"/media/{key}").status_code == 404, key
    c.post("/api/logout")
    assert c.get("/media/frames/1/x.jpg").status_code == 401


# --------------------------------------------------------------------------
# смена провайдера модели Б не выбрасывает историю этапов
# --------------------------------------------------------------------------

def test_stage_history_is_stitched_when_provider_switches(env):
    from app.models import StageObservation
    from app.services import adapters
    site_id = env.site(timezone="UTC")["id"]
    cam = env.camera(site_id)
    env.upload(cam["id"], series(4, step_min=90))
    frames = env.frames(cam["id"])
    with db.session() as s:
        before = len(adapters.observations(s, site_id, "siglip"))
        assert before >= 2
        s.add(StageObservation(frame_id=frames[-1]["id"], provider="glm", model="glm-4.6v", answers={},
                               scores={}, stage_likelihood={}, unsure_ratio=0.0, latency_ms=1.0, cost_usd=0.0,
                               raw={"provider_kind": "external"}))
        s.commit()
        sources: dict = {}
        got = adapters.observations(s, site_id, "glm", sources)
    assert len(got) >= before, "первый ответ GLM не заменяет всю историю SigLIP"
    assert sources["glm"] == 1 and sources["siglip"] >= before - 1
