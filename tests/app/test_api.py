"""JSON API: контракт /api/detect, /api/analyze, настройки, план, сводка объекта,
кадры, аннотированный кадр, переанализ, демо, работа без модулей ядра."""
from __future__ import annotations

import datetime as dt
import json

import cv2
import numpy as np
import pytest

from app import storage
from app.services import providers, views
from app.services.providers import registry
from tests.app.conftest import jpeg, scene, series

CONTRACT_KEYS = {"class", "bbox", "conf", "zone_id", "moved_since_prev", "displacement_px",
                 "bbox_shape_delta", "track_id", "unit_id", "activity"}


def test_detect_contract_with_file(env):
    r = env.client.post("/api/detect", files={"file": ("x.jpg", jpeg(scene(60, truck_x=180)))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frame_id"] is None and body["provider"] == "yolo"
    assert {d["class"] for d in body["detections"]} == {"excavator", "dump_truck"}
    for d in body["detections"]:
        assert set(d) == CONTRACT_KEYS                    # ровно контракт PLAN §4.3
        assert len(d["bbox"]) == 4 and d["activity"] == "unknown"


def test_detect_by_frame_id_returns_tracked_fields(env):
    cam = env.camera(env.site()["id"])
    env.upload(cam["id"], series(2))
    second = env.frames(cam["id"])[1]
    body = env.client.post("/api/detect", json={"frame_id": second["id"]}).json()
    assert body["frame_id"] == second["id"] and body["source"] == "stored"
    d = body["detections"][0]
    assert set(d) == CONTRACT_KEYS and d["moved_since_prev"] is True and d["unit_id"] == "excavator-1"


def test_detect_invalid_input(env):
    c = env.client
    assert c.post("/api/detect", json={}).status_code == 400
    assert c.post("/api/detect", json={"frame_id": 12345}).status_code == 404
    assert c.post("/api/detect", json={"frame_id": "abc"}).status_code == 400
    assert c.post("/api/detect", files={"file": ("x.jpg", b"not image")}).status_code == 400
    assert c.post("/api/detect", content=b"raw", headers={"content-type": "text/plain"}).status_code == 400
    assert c.post("/api/detect", json={"frame_id": 1, "provider": "resnet"}).status_code == 400
    det = env.fakes.equipment.get_detector("yolo")
    det.is_ready, det.reason = False, "нет весов"
    registry.reset()
    r = c.post("/api/detect", files={"file": ("x.jpg", jpeg(scene(60)))})
    assert r.status_code == 503 and "нет весов" in r.json()["detail"]


def test_analyze_returns_boxes_checklist_stage_and_timing(env):
    r = env.client.post("/api/analyze", files={"file": ("x.jpg", jpeg(scene(60)))}, data={"provider": "external"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_a"] == "glm" and body["model_b"] == "glm" and body["mode"] == "external"
    assert body["counts"] == {"excavator": 1}
    assert body["checklist"]["answers"]["pit"] == "yes" and body["checklist"]["unsure_ratio"] == 0.25
    assert body["stage"]["front"] == 3 and body["stage"]["name"] == "Земляные работы, котлован"
    assert body["annotated"].startswith("data:image/jpeg;base64,")
    assert {"quality_ms", "detect_ms", "stage_ms", "total_ms"} <= body["timings"].keys()
    assert body["errors"] == []
    # ночной снимок: модель Б не спрашивается, но объясняется почему
    night = env.client.post("/api/analyze", files={"file": ("n.jpg", jpeg(scene(60, bg=10)))}).json()
    assert night["checklist"] is None and "ночной" in night["errors"][0]
    assert env.client.post("/api/analyze", files={"file": ("x.jpg", b"zzz")}).status_code == 400
    assert env.client.post("/api/analyze", files={"file": ("x.jpg", jpeg(scene(60)))},
                           data={"provider": "cloud"}).status_code == 400


def test_settings_modes_thresholds_and_classes(env):
    c = env.client
    s = c.get("/api/settings").json()
    assert (s["mode"], s["model_a"], s["model_b"]) == ("local", "yolo", "siglip")
    assert {"pipeline", "stage", "equipment", "analytics"} <= s["thresholds"].keys()
    assert s["thresholds"]["equipment"]["merge_radius_m"] == 5.0       # из EquipmentConfig модели А
    classes = {x["key"]: x for x in s["classes"]}
    assert len(classes) == 21 and classes["excavator"]["tz"] is True and classes["excavator"]["supported"]
    assert classes["tower_crane"]["supported"] is False                # YOLO — только классы ТЗ

    s = c.put("/api/settings", json={"mode": "external"}).json()
    assert (s["mode"], s["model_a"], s["model_b"]) == ("external", "glm", "glm")
    assert all(x["supported"] for x in s["classes"])
    s = c.put("/api/settings", json={"model_a": "yolo"}).json()
    assert (s["mode"], s["model_b"]) == ("hybrid", "glm")
    s = c.put("/api/settings", json={"model_b": "local_vlm"}).json()
    assert s["mode"] == "local"
    s = c.put("/api/settings", json={"thresholds": {"stage": {"yes_thr": 0.7}, "equipment": {"parked_after_h": 24}}}).json()
    assert s["thresholds"]["stage"]["yes_thr"] == 0.7 and s["thresholds"]["stage"]["no_thr"] == 0.38
    assert s["thresholds"]["equipment"]["parked_after_h"] == 24
    assert c.get("/api/settings").json()["thresholds"]["stage"]["yes_thr"] == 0.7

    for bad in ({"mode": "turbo"}, {"model_a": "sam"}, {"thresholds": {"stage": {"yes_thr": "high"}}},
                {"thresholds": {"stage": {"no_thr": 0.9}}}, {"thresholds": {"magic": {}}},
                {"thresholds": {"pipeline": {"clock": "moon"}}}, ["mode"]):
        assert c.put("/api/settings", json=bad).status_code == 400, bad


def test_health_shape(env):
    body = env.client.get("/api/health").json()
    assert body["ok"] is True and body["version"]
    assert set(body["providers"]) == {"yolo", "siglip", "glm", "local_vlm"}
    assert all({"ready", "reason"} <= p.keys() for p in body["providers"].values())


def test_plan_roundtrip_hours_and_validation(env):
    c = env.client
    site = env.site()
    plan = [
        {"stage_id": 3, "planned_start": "2025-05-05", "planned_end": "2025-05-10", "work_codes": ["12.3.1."]},
        {"stage_id": 4, "planned_start": "2025-05-11", "planned_end": "2025-06-10",
         "equipment": {"concrete_mixer": 2}, "planned_hours": {"concrete_mixer": 99}, "hours_manual": True},
    ]
    r = c.put(f"/api/sites/{site['id']}/plan", json=plan)
    assert r.status_code == 200, r.text
    got = {p["stage_id"]: p for p in r.json()}
    assert set(got[3]) >= {"stage_id", "name", "work_codes", "planned_start", "planned_end", "equipment",
                           "planned_hours", "hours_manual"}
    # пустой парк строки → нормы этапа; часы = единицы × 6 раб. дней × 10 ч × 0.7
    assert got[3]["equipment"] == {"excavator": 1, "dump_truck": 2}
    assert got[3]["planned_hours"] == {"excavator": 42.0, "dump_truck": 84.0}
    assert got[4]["planned_hours"] == {"concrete_mixer": 99.0}          # ручные часы не пересчитываются
    assert c.get(f"/api/sites/{site['id']}/plan").json() == r.json()

    # парк влияет на часы строк без своего парка
    r = c.put(f"/api/sites/{site['id']}/fleet", json=[{"cls": "bulldozer", "count": 2}])
    assert r.json() == [{"cls": "bulldozer", "count": 2}]
    assert c.put(f"/api/sites/{site['id']}/fleet", json=[{"cls": "ufo", "count": 1}]).status_code == 400
    assert c.put(f"/api/sites/{site['id']}/fleet", json=[{"cls": "bulldozer", "count": -1}]).status_code == 400

    for bad in ([{"stage_id": 9}], [{"stage_id": 3, "planned_start": "2025-05-10", "planned_end": "2025-05-01"}],
                [{"stage_id": 3, "planned_start": "10 мая"}], [{"stage_id": 3, "equipment": {"ufo": 1}}],
                [{"stage_id": 3}, {"stage_id": 3}], {"stage_id": 3}, [{"stage_id": 3, "equipment": {"excavator": 1.5}}]):
        assert c.put(f"/api/sites/{site['id']}/plan", json=bad).status_code == 400, bad
    assert c.get("/api/sites/999/plan").status_code == 404


def test_plan_import_and_demo(env):
    c = env.client
    site = env.site()
    csv = "stage_id,start,end,codes\n3,2025-05-01,2025-05-20,12.3.1.\n5,2025-06-01,2025-08-01\nмусор\n"
    r = c.post(f"/api/sites/{site['id']}/plan/import", files={"file": ("plan.csv", csv.encode())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is True and [p["stage_id"] for p in body["plan"]] == [3, 5]
    assert any("строка 4" in w for w in body["warnings"])
    assert c.get(f"/api/sites/{site['id']}/plan").json()[0]["work_codes"] == ["12.3.1."]
    assert c.post(f"/api/sites/{site['id']}/plan/import", files={"file": ("plan.pdf", b"%PDF")}).status_code == 400
    assert c.post(f"/api/sites/{site['id']}/plan/import", files={"file": ("plan.csv", b"")}).status_code == 400
    preview = c.post(f"/api/sites/{site['id']}/plan/import", files={"file": ("p.csv", b"4,2025-01-01,2025-02-01")},
                     data={"apply": "0"}).json()
    assert preview["applied"] is False and c.get(f"/api/sites/{site['id']}/plan").json()[0]["stage_id"] == 3

    cam = env.camera(site["id"])
    env.upload(cam["id"], series(2, base=dt.datetime(2025, 3, 1, 9)) +
               series(1, base=dt.datetime(2025, 9, 1, 9), prefix="late", seed0=100))
    body = c.post(f"/api/sites/{site['id']}/plan/demo", json={"stage_ids": [1, 2, 3]}).json()
    assert [p["stage_id"] for p in body["plan"]] == [1, 2, 3]
    assert body["plan"][0]["planned_start"] == "2025-03-01" and body["plan"][-1]["planned_end"] == "2025-09-01"
    assert all(p["source"] == "demo" for p in body["plan"]) and "Демо-план" in body["warnings"][0]
    assert c.post(f"/api/sites/{site['id']}/plan/demo", json={"start": "2025-05-01", "end": "2025-01-01"}
                  ).status_code == 400


OVERVIEW_REPORT = {"verdict", "lag_days", "expected_progress", "actual_progress", "forecast_finish", "explanation"}
OVERVIEW_STAGE = {"id", "name", "status", "progress", "planned_start", "planned_end", "actual_start", "actual_end",
                  "manual", "works"}
OVERVIEW_EQUIPMENT = {"cls", "name", "units", "active", "idle", "parked", "planned_hours", "worked_hours",
                      "remaining_hours"}


def test_overview_has_all_contract_fields(env):
    c = env.client
    site = env.site(timezone="UTC")
    c.put(f"/api/sites/{site['id']}/plan", json=[
        {"stage_id": 3, "planned_start": "2025-05-01", "planned_end": "2025-05-31", "work_codes": ["12.3.1."]}])
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(4))
    ov = c.get(f"/api/sites/{site['id']}/overview").json()
    assert {"site", "report", "stages", "equipment", "deviations", "cameras", "series"} <= ov.keys()
    assert OVERVIEW_REPORT <= ov["report"].keys()
    assert ov["report"]["verdict"] == "behind" and ov["report"]["lag_days"] == 4.0
    assert len(ov["stages"]) == 8 and all(OVERVIEW_STAGE <= s.keys() for s in ov["stages"])
    st3 = ov["stages"][2]
    assert st3["planned_start"] == "2025-05-01" and st3["status"] == "active"
    assert st3["works"] == [{"code": "12.3.1.", "name": "Устройство котлована", "key": "12.3.1."}]
    assert ov["stages"][0]["works"] == [{"code": "10.2.", "name": "Вынос инженерных систем",
                                        "key": "10.2."}]   # из каталога
    exc = next(e for e in ov["equipment"] if e["cls"] == "excavator")
    assert OVERVIEW_EQUIPMENT <= exc.keys()
    assert exc["units"] == 1 and exc["active"] == 1 and exc["worked_hours"] == 1.0
    assert exc["planned_hours"] == pytest.approx(1 * 27 * 10 * 0.7)  # май 2025: 27 рабочих дней пн–сб and exc["remaining_hours"] == exc["planned_hours"] - 1.0
    cam_ov = ov["cameras"][0]
    assert {"id", "name", "last_frame", "units_now"} <= cam_ov.keys() and cam_ov["units_now"] == 1
    assert {"id", "url", "captured_at"} <= cam_ov["last_frame"].keys()
    assert {"days", "expected", "actual"} <= ov["series"].keys() and ov["series"]["days"]
    assert ov["deviations"][0]["type"] == "pair_broken"
    card = c.get("/api/sites").json()[0]
    assert card["verdict"] == "behind" and card["current_stage"] == 3 and card["active_units"] == 1
    assert card["open_deviations"] == 1 and card["thumb"].startswith("/media/")


def test_frames_list_detail_and_annotated(env):
    c = env.client
    cam = env.camera(env.site(timezone="UTC")["id"])
    env.upload(cam["id"], series(5))
    newest = c.get(f"/api/cameras/{cam['id']}/frames", params={"limit": 2}).json()
    assert [f["captured_at"][11:16] for f in newest] == ["09:20", "09:00"]
    older = c.get(f"/api/cameras/{cam['id']}/frames", params={"limit": 2, "before": newest[-1]["captured_at"]}).json()
    assert [f["captured_at"][11:16] for f in older] == ["08:40", "08:20"]
    assert c.get(f"/api/cameras/{cam['id']}/frames", params={"before": "когда-то"}).status_code == 400

    d = c.get(f"/api/frames/{older[0]['id']}").json()
    assert d["prev_id"] == older[1]["id"] and d["next_id"] == newest[1]["id"]
    assert d["quality"]["quality_ok"] is True and d["checklist"] is None or isinstance(d["checklist"], dict)
    first = c.get(f"/api/frames/{env.frames(cam['id'])[0]['id']}").json()
    assert first["checklist"]["answers"]["pit"] == "yes" and "pit" in first["checklist"]["questions"]

    img = c.get(d["annotated_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (240, 320)
    # без модуля отрисовки модели А — простые рамки OpenCV
    providers.override_module("core.equipment", providers.MISSING)
    try:
        assert c.get(d["annotated_url"]).status_code == 200
    finally:
        env.fakes.install(providers)
    assert c.get("/api/frames/999/annotated.jpg").status_code == 404
    assert c.get("/media/../../etc/passwd").status_code == 404


def test_reprocess_with_other_provider_keeps_old_results(env):
    c = env.client
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3))
    c.put("/api/settings", json={"mode": "external"})
    r = c.post(f"/api/sites/{site['id']}/reprocess")
    assert r.status_code == 202 and r.json()["frames"] == 3 and r.json()["model_a"] == "glm"
    job = env.wait_job(r.json()["job_id"])
    assert job["state"] == "done" and job["done"] == 3
    from sqlalchemy import func, select
    from app import db
    from app.models import Detection
    with db.session() as s:
        by_provider = dict(s.execute(select(Detection.provider, func.count()).group_by(Detection.provider)).all())
    assert by_provider == {"yolo": 3, "glm": 3}
    eq = c.get(f"/api/sites/{site['id']}/equipment").json()
    assert len(eq["units"]) == 1 and eq["units"][0]["worked_hours"] == pytest.approx(2 / 3, abs=0.01)
    unit = c.get(f"/api/units/{eq['units'][0]['id']}").json()
    assert len(unit["intervals"]) == 2 and len(unit["detections"]) == 3
    assert c.get("/api/units/999").status_code == 404


def test_demo_seed_from_dataset_dir(env):
    site_dir = env.tmp / "demo" / "sites" / "pit"
    (site_dir / "north").mkdir(parents=True)
    for name, data in series(3, base=dt.datetime(2025, 5, 5, 9)):
        (site_dir / "north" / name).write_bytes(data)
    (site_dir / "site.json").write_text(json.dumps({
        "name": "Демо: котлован", "timezone": "UTC", "plan": {"stage_ids": [3]},
        "cameras": [{"name": "Север", "dir": "north", "interval_min": 20,
                     "calibration": {"image_points": [[0, 0], [320, 0], [320, 240], [0, 240]],
                                     "site_points": [[0, 0], [32, 0], [32, 24], [0, 24]]},
                     "zones": [{"name": "Котлован", "kind": "work", "polygon": [[0, 0], [320, 0], [320, 240], [0, 240]]}]}],
    }), encoding="utf-8")
    r = env.client.post("/api/demo/seed", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    site_id = body["sites"][0]["id"]
    for job_id in body["jobs"]:
        assert env.wait_job(job_id)["done"] == 3
    plan = env.client.get(f"/api/sites/{site_id}/plan").json()
    assert plan[0]["stage_id"] == 3 and plan[0]["planned_start"] == "2025-05-05" and plan[0]["planned_hours"]
    assert env.client.get(f"/api/sites/{site_id}/fleet").json() == [{"cls": "dump_truck", "count": 2},
                                                                     {"cls": "excavator", "count": 1}]
    cam = env.client.get(f"/api/sites/{site_id}/cameras").json()[0]
    assert cam["calibrated"] is True
    frame = env.client.get(f"/api/frames/{env.frames(cam['id'])[0]['id']}").json()
    assert frame["detections"][0]["zone_id"] is not None
    # повторный засев не плодит дубли
    again = env.client.post("/api/demo/seed", json={}).json()
    assert again["sites"] == [] and "уже есть" in again["warnings"][0]


def test_demo_seed_without_data_is_404(env):
    r = env.client.post("/api/demo/seed")
    assert r.status_code == 404
    assert "sites/<объект>" in r.json()["detail"]


def test_works_without_core_modules(env):
    """Модулей ядра нет вовсе: кадры сохраняются, качество — запасное, анализ отложен с причиной."""
    providers.clear_overrides()
    for name in ("core.equipment", "core.stage", "core.plan", "core.analytics"):
        providers.override_module(name, providers.MISSING)
    registry.reset()
    views._catalog_names.cache_clear()
    site = env.site()
    cam = env.camera(site["id"])
    job = env.upload(cam["id"], series(2))
    assert job["state"] == "postponed" and job["total"] == 2
    f = env.frames(cam["id"])[0]
    assert f["quality_ok"] is True and "не установлен" in f["note"]
    ov = env.client.get(f"/api/sites/{site['id']}/overview").json()
    assert ov["report"]["verdict"] == "no_plan" and ov["report"]["errors"]
    health = env.client.get("/api/health").json()
    assert health["ok"] is True and health["providers"]["yolo"]["ready"] is False


def test_local_storage_rejects_path_traversal(tmp_path):
    st = storage.LocalStorage(tmp_path / "root")
    (tmp_path / "root-evil").mkdir()
    for key in ("../root-evil/x.jpg", "/etc/passwd", "a/../../x", "a\\b", ""):
        with pytest.raises(ValueError):
            st.put(key, b"x")
    st.put("frames/1/a.jpg", b"data")
    assert st.get("frames/1/a.jpg") == b"data" and st.exists("frames/1/a.jpg")
    assert not st.exists("../root-evil/x.jpg")


def test_manual_hours_correction_moves_the_bar_and_survives_reprocess(env):
    c = env.client
    site = env.site(timezone="UTC")
    c.put(f"/api/sites/{site['id']}/plan", json=[
        {"stage_id": 3, "planned_start": "2025-05-01", "planned_end": "2025-05-31"}])
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(4))                       # модель насчитала 1 ч экскаватора

    def excavator():
        ov = c.get(f"/api/sites/{site['id']}/overview").json()
        return next(e for e in ov["equipment"] if e["cls"] == "excavator")

    before = excavator()
    assert before["worked_hours"] == 1.0
    r = c.post(f"/api/sites/{site['id']}/hours",
               json={"cls": "excavator", "hours": 3, "stage_id": 3, "at": "2025-05-12T18:00", "note": "ночная смена"})
    assert r.status_code == 201 and r.json()["manual"] is True
    corr_id = r.json()["id"]
    assert excavator()["worked_hours"] == 4.0
    assert excavator()["remaining_hours"] == pytest.approx(before["remaining_hours"] - 3)
    assert [h["note"] for h in c.get(f"/api/sites/{site['id']}/hours").json()] == ["ночная смена"]

    job = env.wait_job(c.post(f"/api/sites/{site['id']}/reprocess").json()["job_id"])
    assert job["done"] == 4 and excavator()["worked_hours"] == 4.0     # поправка пережила переанализ

    for bad in ({"cls": "ufo", "hours": 1}, {"cls": "excavator", "hours": 0}, {"cls": "excavator", "hours": "3"},
                {"cls": "excavator", "hours": 1, "stage_id": 11}, {"cls": "excavator", "hours": 1, "at": "вчера"}):
        assert c.post(f"/api/sites/{site['id']}/hours", json=bad).status_code == 400, bad
    assert c.delete(f"/api/sites/{site['id']}/hours/{corr_id}").status_code == 200
    assert excavator()["worked_hours"] == 1.0
    assert c.delete(f"/api/sites/{site['id']}/hours/{corr_id}").status_code == 404


def test_catalog_queue_and_jobs_list(env):
    c = env.client
    cat = c.get("/api/catalog").json()
    assert len(cat["stages"]) == 8 and len(cat["equipment"]) == 21 and len(cat["signs"]) == 60
    pit = cat["stages"][2]
    assert pit["name"] == "Земляные работы, котлован" and pit["substages"][0]["xlsx"]
    assert pit["rule"]["default_equipment"] == {"excavator": 1, "dump_truck": 2}
    assert [w["code"] for w in pit["works"]] == ["12.3.1.", "12.3.9."]
    cam = env.camera(env.site()["id"])
    env.upload(cam["id"], series(2))
    q = c.get("/api/queue").json()
    assert q["running"] is True and q["pending"] == 0
    jobs = c.get("/api/jobs").json()
    assert jobs[0]["camera_id"] == cam["id"] and jobs[0]["state"] == "done"
