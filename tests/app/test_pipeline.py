"""Очередь и конвейер кадра: модели А/Б, отложенный анализ, устойчивость к ошибкам,
восстановление после рестарта, отбор кадров для модели Б, ручные отметки, отклонения."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select

from app import db
from app.models import ActivityInterval, Deviation, Detection, EquipmentUnit, Frame, StageObservation
from app.services import pipeline
from app.services.providers import registry
from app.services.queue import frame_queue
from tests.app.conftest import jpeg, scene, series, stamp


def _count(model, *where) -> int:
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where)) or 0


def test_queue_processes_frames_with_fake_detector(env):
    site = env.site(timezone="UTC")
    cam = env.camera(site["id"])
    job = env.upload(cam["id"], series(4, step_min=20, move=15))
    assert job["state"] == "done" and job["done"] == 4, job

    frames = env.frames(cam["id"])
    assert all(f["status"] == "done" and f["processed_a"] and f["processed_b"] for f in frames)
    assert _count(Detection, Detection.provider == "yolo") == 4
    # одна машина на четырёх кадрах = одна единица техники, три засчитанных интервала по 20 мин
    with db.session() as s:
        units = s.scalars(select(EquipmentUnit)).all()
        assert [(u.uid, u.cls, u.status) for u in units] == [("excavator-1", "excavator", "active")]
        hours = [iv.hours for iv in s.scalars(select(ActivityInterval).order_by(ActivityInterval.start))]
    assert [round(h, 3) for h in hours] == [0.333, 0.333, 0.333]

    detail = env.client.get(f"/api/frames/{frames[1]['id']}").json()
    det = detail["detections"][0]
    assert det["class"] == "excavator" and det["moved_since_prev"] is True and det["activity"] == "working"
    assert det["unit_id"] == "excavator-1" and det["name"] == "Экскаватор"

    eq = env.client.get(f"/api/sites/{site['id']}/equipment").json()
    assert eq["units"][0]["worked_hours"] == 1.0


def test_hours_expected_counts_from_first_frame(env):
    """Требование 5: этап идёт с 1 мая, камера начала снимать 12 мая в 08:00 и сняла час.
    Ожидаемое к «сейчас» — доля смены с первого кадра (0.1 смены), а не 9 смен с начала этапа."""
    import importlib
    env.fakes.equipment.hours = importlib.import_module("core.equipment.hours")
    site = env.site(timezone="Europe/Moscow")
    cam = env.camera(site["id"])
    r = env.client.put(f"/api/sites/{site['id']}/plan", json=[{
        "stage_id": 3, "planned_start": "2025-05-01", "planned_end": "2025-05-31",
        "planned_hours": {"excavator": 140.0}, "hours_manual": True}])
    assert r.status_code == 200, r.text
    env.upload(cam["id"], series(4, step_min=20, move=15))
    env.client.post(f"/api/sites/{site['id']}/recompute")
    ov = env.client.get(f"/api/sites/{site['id']}/overview").json()
    assert ov["report"]["observed_from"].startswith("2025-05-12T05:00")      # 08:00 МСК
    ex = next(r for r in ov["equipment"] if r["cls"] == "excavator")
    per_workday = 140.0 / 27                                                 # май 2025: 27 рабочих дней пн–сб
    assert abs(ex["expected_hours"] - round(per_workday * 0.1, 2)) < 0.02, ex
    assert ex["expected_from"].startswith("2025-05-12T05:00")
    assert abs(ex["planned_observed_hours"] - per_workday * 18) < 0.05      # 12–31 мая: 18 рабочих дней
    assert ex["worked_hours"] > ex["expected_hours"] and ex["detectable"] is True
    bal = env.client.get(f"/api/sites/{site['id']}/equipment").json()["balances"]
    assert bal[0]["expected_hours"] == ex["expected_hours"]


def test_provider_not_ready_postpones_then_resumes(env):
    det = env.fakes.equipment.get_detector("yolo")
    det.is_ready, det.reason = False, "нет весов models/equipment.pt"
    registry.reset()
    cam = env.camera(env.site()["id"])
    job = env.upload(cam["id"], series(2))
    assert job["state"] == "postponed" and job["total"] == 2, job
    assert "нет весов" in job["postponed_reason"]
    frames = env.frames(cam["id"])
    # кадры сохранены, модель Б отработала, модель А ждёт
    assert all(f["status"] == "postponed" and not f["processed_a"] and f["processed_b"] for f in frames)
    assert "модель А (yolo) отложена" in frames[0]["note"]

    det.is_ready, det.reason = True, ""
    registry.reset()
    assert frame_queue.recover(include_postponed=True) == 2
    env.wait()
    assert all(f["status"] == "done" and f["processed_a"] for f in env.frames(cam["id"]))
    assert env.client.get(f"/api/jobs/{job['id']}").json()["state"] == "done"


def test_error_on_one_frame_does_not_stop_queue(env):
    bad_time = dt.datetime(2025, 5, 12, 8, 20, tzinfo=dt.UTC)
    env.fakes.equipment.get_detector("yolo").fail_if = lambda img, frame: frame.captured_at == bad_time
    cam = env.camera(env.site(timezone="UTC")["id"])
    job = env.upload(cam["id"], series(3))
    assert job["failed"] == 1 and job["done"] == 2, job
    assert any("детектор упал" in e for e in job["errors"])
    statuses = [(f["captured_at"][11:16], f["status"]) for f in env.frames(cam["id"])]
    assert statuses == [("08:00", "done"), ("08:20", "error"), ("08:40", "done")]
    # модель Б на сбойном кадре всё равно отработала (шаги независимы)
    broken = env.frames(cam["id"])[1]
    assert broken["processed_b"] is True and broken["processed_a"] is False


def test_night_frame_goes_to_model_a_but_not_b(env):
    cam = env.camera(env.site(timezone="UTC")["id"])
    night = jpeg(scene(60, bg=15, seed=3))
    env.upload(cam["id"], [(f"cam_{stamp(dt.datetime(2025, 5, 12, 2, 0), 0)}.jpg", night)])
    f = env.frames(cam["id"])[0]
    assert f["is_night"] is True and f["processed_a"] and f["processed_b"] and not f["stage_used"]
    assert f["detections_count"] == 1
    assert "модель Б пропустила кадр" in f["note"]
    assert env.fakes.classifiers.get("siglip") is None or not env.fakes.classifiers["siglip"].calls


def test_model_b_not_more_often_than_every_hour(env):
    # внеочередной вызов по маске выключаем — проверяем чистое правило «раз в час»
    r = env.client.put("/api/settings", json={"thresholds": {"pipeline": {"stage_mask_change": 1.0}}})
    assert r.status_code == 200
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(5, step_min=20))          # 08:00 … 09:20
    used = [f["captured_at"][11:16] for f in env.frames(cam["id"]) if f["stage_used"]]
    assert used == ["08:00", "09:00"]
    assert _count(StageObservation) == 2
    # контекст стройки передаётся классификатору
    _fid, ctx = env.fakes.classifiers["siglip"].calls[-1]
    assert ctx["object_type"] == "Жильё"


def test_strong_mask_change_triggers_extra_stage_call(env):
    """Фейковая маска прирастает на 10 % за кадр — больше порога 5 %: каждый кадр внеочередной."""
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(3, step_min=20))
    assert [f["stage_used"] for f in env.frames(cam["id"])] == [True, True, True]
    info = env.client.get(f"/api/cameras/{cam['id']}").json()
    assert info["mask"]["masked_ratio"] > 0 and info["mask"]["windows"] == 3
    png = env.client.get(f"/api/cameras/{cam['id']}/mask.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_recovery_after_restart(env):
    """Кадры, записанные без работающей очереди (рестарт, утилита), подбираются при старте."""
    frame_queue.shutdown()
    cam = env.camera(env.site()["id"])
    r = env.client.post(f"/api/cameras/{cam['id']}/upload", files=[("files", (n, d)) for n, d in series(3)])
    job = env.wait_job(r.json()["job_id"], until=("processing",))
    assert job["pending"] == 3
    # «упавший» процесс оставил один кадр в processing
    with db.session() as s:
        fr = s.scalars(select(Frame)).first()
        fr.status = "processing"
        s.commit()
    frame_queue.start(supervise=False)
    env.wait()
    assert all(f["status"] == "done" for f in env.frames(cam["id"]))


def test_manual_stage_mark_survives_recompute(env):
    site = env.site()
    cam = env.camera(site["id"])
    r = env.client.patch(f"/api/sites/{site['id']}/stages/3",
                         json={"status": "done", "actual_end": "2025-05-10", "note": "принято по акту"})
    assert r.status_code == 200 and r.json()["manual"] is True and r.json()["progress"] == 1.0
    env.upload(cam["id"], series(3))                       # фейковая хронология ставит этап 3 «идёт»
    pipeline.recompute_site(site["id"])
    stages = {s["id"]: s for s in env.client.get(f"/api/sites/{site['id']}/overview").json()["stages"]}
    assert stages[3]["status"] == "done" and stages[3]["manual"] is True and stages[3]["actual_end"] == "2025-05-10"
    assert stages[1]["status"] == "done" and stages[1]["manual"] is False    # модельные — обновились
    # вернуть этап модели
    env.client.patch(f"/api/sites/{site['id']}/stages/3", json={"manual": False})
    stages = {s["id"]: s for s in env.client.get(f"/api/sites/{site['id']}/overview").json()["stages"]}
    assert stages[3]["status"] == "active" and stages[3]["manual"] is False
    # невалидная отметка
    assert env.client.patch(f"/api/sites/{site['id']}/stages/3", json={"status": "почти"}).status_code == 400
    assert env.client.patch(f"/api/sites/{site['id']}/stages/3", json={"progress": 2}).status_code == 400
    assert env.client.patch(f"/api/sites/{site['id']}/stages/42", json={"status": "done"}).status_code == 404


def test_stage_is_inferred_with_equipment_of_model_a(env):
    """Пересчёт отдаёт хронологии этапов технику модели А (журнал моточасов + рамки), а в отчёт
    пишет основание этапа. Настоящие core.stage.sequence/fusion поверх фейковых моделей."""
    from app.services import providers
    from core.stage import fusion, sequence

    providers.override_module("core.stage.sequence", sequence)
    providers.override_module("core.stage.fusion", fusion)
    site = env.site(timezone="UTC")
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(4, step_min=20, move=15))   # экскаватор копает, чек-лист видит котлован
    pipeline.recompute_site(site["id"])
    report = env.client.get(f"/api/sites/{site['id']}/overview").json()["report"]
    basis = report["stage_basis"]
    assert report["current_stage"] == 3 and basis["stage"] == 3
    assert basis["equipment"][0]["cls"] == "excavator" and basis["equipment"][0]["hours"] == 1.0
    assert basis["equipment_relation"] == "agree" and "экскаватор (1,0 ч)" in basis["text"]

    # вес техники 0 — этап только по чек-листу, техника в основание не попадает
    r = env.client.put("/api/settings", json={"thresholds": {"stage": {"equipment_weight": 0}}})
    assert r.status_code == 200, r.text
    pipeline.recompute_site(site["id"])
    basis = env.client.get(f"/api/sites/{site['id']}/overview").json()["report"]["stage_basis"]
    assert basis["stage"] == 3 and basis["equipment"] == [] and basis["decided_by"] == "checklist"


def test_deviations_upsert_by_key_ack_and_auto_resolve(env):
    site = env.site(timezone="UTC")
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3))                       # экскаватор без самосвалов
    pipeline.recompute_site(site["id"])
    pipeline.recompute_site(site["id"])
    assert _count(Deviation) == 1
    devs = env.client.get(f"/api/sites/{site['id']}/deviations").json()
    assert devs[0]["type"] == "pair_broken" and devs[0]["frames"] and devs[0]["camera_name"] == "Камера 1"
    assert devs[0]["frames"][0]["annotated_url"].endswith("/annotated.jpg")

    r = env.client.patch(f"/api/deviations/{devs[0]['id']}", json={"status": "ack", "note": "подрядчик уведомлён"})
    assert r.json()["status"] == "ack"
    pipeline.recompute_site(site["id"])
    assert env.client.get(f"/api/sites/{site['id']}/deviations").json()[0]["status"] == "ack"
    assert env.client.patch(f"/api/deviations/{devs[0]['id']}", json={"status": "забыть"}).status_code == 400
    assert env.client.get(f"/api/sites/{site['id']}/deviations?status=lost").status_code == 400

    # приехали самосвалы — отклонение закрывается само
    env.upload(cam["id"], series(1, base=dt.datetime(2025, 5, 12, 9, 0), truck=True, prefix="late"))
    pipeline.recompute_site(site["id"])
    assert env.client.get(f"/api/sites/{site['id']}/deviations").json() == []
    resolved = env.client.get(f"/api/sites/{site['id']}/deviations?status=resolved").json()
    assert len(resolved) == 1 and resolved[0]["data"]["auto_resolved"] is True
    assert _count(Deviation) == 1
