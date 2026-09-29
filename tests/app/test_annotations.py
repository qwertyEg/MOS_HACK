"""Ручная разметка техники (требование 3): правки рамок и машин сразу меняют аналитику,
переживают переанализ, отменяются целиком и выгружаются датасетом YOLO."""
from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pytest

from app.config import settings
from app.services.queue import replayer
from tests.app.conftest import series

BASE = dt.datetime(2025, 5, 12, 8, 0)


def frame(env, fid: int) -> dict:
    r = env.client.get(f"/api/frames/{fid}")
    assert r.status_code == 200, r.text
    return r.json()


def classes(env, fid: int) -> list[str]:
    return sorted(d["class"] for d in frame(env, fid)["detections"])


def units(env, site_id: int) -> list[dict]:
    return env.client.get(f"/api/sites/{site_id}/equipment").json()["units"]


def reprocess(env, site_id: int) -> None:
    r = env.client.post(f"/api/sites/{site_id}/reprocess")
    assert r.status_code == 202, r.text
    env.wait_job(r.json()["job_id"])


def box_of(env, fid: int, cls: str) -> dict:
    return next(d for d in frame(env, fid)["detections"] if d["class"] == cls)


def test_relabel_one_box_survives_reprocess_and_undo(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    f = [x["id"] for x in env.frames(cam["id"])]
    truck = box_of(env, f[1], "dump_truck")

    r = env.client.patch(f"/api/detections/{truck['id']}", json={"cls": "concrete_mixer", "scope": "box"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["kind"] == "box_relabel" and out["batch"] and "Самосвал" in out["note"]
    fixed = next(d for d in out["frame"]["detections"] if d["bbox"] == truck["bbox"])
    assert fixed["class"] == "concrete_mixer"
    assert fixed["manual"]["orig_cls"] == "dump_truck" and fixed["manual"]["cls"] == "concrete_mixer"
    assert out["frame"]["annotations"]["reviewed"] is True
    env.wait()
    assert classes(env, f[0]) == ["dump_truck", "excavator"], "правилась только одна рамка"

    reprocess(env, site["id"])                             # детектор снова скажет «самосвал» — правка ляжет сверху
    assert classes(env, f[1]) == ["concrete_mixer", "excavator"]
    assert classes(env, f[2]) == ["dump_truck", "excavator"]

    r = env.client.delete(f"/api/annotations/batches/{out['batch']}")
    assert r.status_code == 200, r.text
    assert sorted(d["class"] for d in r.json()["frame"]["detections"]) == ["dump_truck", "excavator"]
    env.wait()
    back = box_of(env, f[1], "dump_truck")
    assert "manual" not in back
    assert env.client.delete(f"/api/annotations/batches/{out['batch']}").status_code == 404


def test_box_edit_is_visible_at_once_and_replay_follows(env, monkeypatch):
    """Большой архив: кадр обновляется сразу, перепрогон техники — следом в фоне."""
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    fid = env.frames(cam["id"])[2]["id"]
    monkeypatch.setattr(settings, "recompute_debounce_s", 30.0)
    truck = box_of(env, fid, "dump_truck")
    r = env.client.patch(f"/api/detections/{truck['id']}", json={"cls": "truck"})
    assert r.status_code == 200
    assert "truck" in [d["class"] for d in r.json()["frame"]["detections"]]
    assert r.json()["replay"]["state"] == "queued"
    monkeypatch.setattr(settings, "recompute_debounce_s", 0.0)   # пересчёт аналитики после перепрогона — сразу
    env.wait(timeout=30)                                   # фоновый перепрогон отработал
    assert replayer.state(site["id"])["last"]["error"] is None
    assert "truck" in classes(env, fid)


def test_delete_place_hides_static_false_machine_everywhere(env):
    """«Это место камеры — не техника»: контейнер/мачта не заводит машину ни на прошлых, ни на новых кадрах."""
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    f = [x["id"] for x in env.frames(cam["id"])]
    assert any(u["cls"] == "dump_truck" for u in units(env, site["id"]))
    truck = box_of(env, f[0], "dump_truck")

    r = env.client.delete(f"/api/detections/{truck['id']}", params={"scope": "camera"})
    assert r.status_code == 200, r.text
    assert [d["class"] for d in r.json()["frame"]["detections"]] == ["excavator"]
    env.wait()
    for fid in f:
        assert classes(env, fid) == ["excavator"]
    assert [u["cls"] for u in units(env, site["id"])] == ["excavator"]

    env.upload(cam["id"], series(2, base=BASE + dt.timedelta(hours=2), truck=True, seed0=50))
    for x in env.frames(cam["id"]):
        assert "dump_truck" not in classes(env, x["id"])
    journal = env.client.get(f"/api/sites/{site['id']}/annotations").json()
    assert journal["batches"][0]["kind"] == "box_delete" and journal["stats"]["reviewed_frames"] == 1


def test_delete_one_box_only_on_this_frame(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    f = [x["id"] for x in env.frames(cam["id"])]
    truck = box_of(env, f[1], "dump_truck")
    assert env.client.patch(f"/api/detections/{truck['id']}", json={"deleted": True}).status_code == 200
    env.wait()
    assert classes(env, f[1]) == ["excavator"]
    assert classes(env, f[0]) == ["dump_truck", "excavator"] == classes(env, f[2])
    reprocess(env, site["id"])
    assert classes(env, f[1]) == ["excavator"]


def test_add_missed_box_creates_machine_and_is_validated(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3))
    fid = env.frames(cam["id"])[1]["id"]
    r = env.client.post(f"/api/frames/{fid}/detections", json={"cls": "roller", "bbox": [250, 20, 40, 30]})
    assert r.status_code == 201, r.text
    added = next(d for d in r.json()["frame"]["detections"] if d["class"] == "roller")
    assert added["manual"]["added"] is True and added["bbox"] == [250.0, 20.0, 40.0, 30.0]
    env.wait()
    assert any(u["cls"] == "roller" for u in units(env, site["id"]))
    reprocess(env, site["id"])
    assert "roller" in classes(env, fid)

    c = env.client
    assert c.post(f"/api/frames/{fid}/detections", json={"cls": "ufo", "bbox": [1, 1, 40, 40]}).status_code == 400
    assert c.post(f"/api/frames/{fid}/detections", json={"cls": "roller", "bbox": [1, 1, 3, 3]}).status_code == 400
    assert c.post(f"/api/frames/{fid}/detections", json={"cls": "roller", "bbox": [1, 1]}).status_code == 400
    assert c.post(f"/api/frames/{fid}/detections", json={"cls": "roller", "bbox": [900, 900, 40, 40]}).status_code == 400
    assert c.post("/api/frames/99999/detections", json={"cls": "roller", "bbox": [1, 1, 40, 40]}).status_code == 404
    assert c.patch("/api/detections/99999", json={"cls": "roller"}).status_code == 404
    det = frame(env, fid)["detections"][0]
    assert c.patch(f"/api/detections/{det['id']}", json={"cls": "roller", "scope": "all"}).status_code == 400
    assert c.patch(f"/api/detections/{det['id']}", json={}).status_code == 400
    assert c.patch(f"/api/detections/{det['id']}", json={"deleted": True, "scope": "site"}).status_code == 400


def test_merge_two_cameras_into_one_machine_survives_reprocess_and_undo(env):
    """Некалиброванные камеры видят один кран двумя машинами — оператор склеивает."""
    site = env.site()
    cam1, cam2 = env.camera(site["id"], name="Север"), env.camera(site["id"], name="Юг")
    env.upload(cam1["id"], series(3))
    env.upload(cam2["id"], series(3, seed0=100, prefix="south"))
    before = units(env, site["id"])
    assert len(before) == 2 and {u["cls"] for u in before} == {"excavator"}
    hours_before = sum(u["worked_hours"] for u in before)

    r = env.client.post("/api/units/merge", json={"unit_ids": [u["id"] for u in before], "cls": "mobile_crane"})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "unit_merge" and r.json()["replay"]["state"] == "idle"
    (one,) = units(env, site["id"])
    assert one["manual"] is True and one["uid"].startswith("m") and one["cls"] == "mobile_crane"
    assert sorted(one["cameras"]) == sorted([str(cam1["id"]), str(cam2["id"])])
    assert one["worked_hours"] == pytest.approx(hours_before, abs=0.02)
    ivs = env.client.get(f"/api/sites/{site['id']}/hours", params={"manual": "false"}).json()
    assert ivs and {iv["cls"] for iv in ivs} == {"mobile_crane"}, "моточасы ушли к новому классу"
    for x in env.frames(cam2["id"]):
        assert classes(env, x["id"]) == ["mobile_crane"]

    reprocess(env, site["id"])
    (again,) = units(env, site["id"])
    assert again["uid"] == one["uid"] and again["cls"] == "mobile_crane"

    r = env.client.delete(f"/api/annotations/batches/{r.json()['batch']}")
    assert r.status_code == 200
    env.wait()
    assert sorted(u["cls"] for u in units(env, site["id"])) == ["excavator", "excavator"]

    assert env.client.post("/api/units/merge", json={"unit_ids": [before[0]["id"]]}).status_code == 400
    assert env.client.post("/api/units/merge", json={"unit_ids": [1, "2"]}).status_code == 400
    after = units(env, site["id"])
    assert env.client.post("/api/units/merge", json={"unit_ids": [after[0]["id"], 99999]}).status_code == 404


def test_merge_drops_second_box_of_the_same_machine_on_one_frame(env):
    """Кран разрезан детектором на две рамки одного кадра: склейка оставляет одну."""
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(2, truck=True))
    fid = env.frames(cam["id"])[1]["id"]
    ids = [u["id"] for u in units(env, site["id"])]
    assert len(ids) == 2
    r = env.client.post("/api/units/merge", json={"unit_ids": ids, "cls": "tower_crane"})
    assert r.status_code == 200, r.text
    assert "лишних рамок" in r.json()["note"]
    (one,) = units(env, site["id"])
    assert one["cls"] == "tower_crane"
    assert classes(env, fid) == ["tower_crane"]


def test_split_machine_from_a_frame(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(6))
    (unit,) = units(env, site["id"])
    tl = env.client.get(f"/api/units/{unit['id']}/timeline").json()
    assert [i["frame_id"] for i in tl["items"]] == [x["id"] for x in env.frames(cam["id"])]
    r = env.client.post(f"/api/units/{unit['id']}/split", json={"frame_id": tl["items"][3]["frame_id"]})
    assert r.status_code == 200, r.text
    got = units(env, site["id"])
    assert len(got) == 2 and all(u["manual"] for u in got)
    sizes = sorted(len(env.client.get(f"/api/units/{u['id']}/timeline").json()["items"]) for u in got)
    assert sizes == [3, 3]
    reprocess(env, site["id"])
    got = units(env, site["id"])
    assert len(got) == 2

    first = min(got, key=lambda u: u["first_seen"])
    tl = env.client.get(f"/api/units/{first['id']}/timeline").json()
    assert env.client.post(f"/api/units/{first['id']}/split",
                           json={"frame_id": tl["items"][0]["frame_id"]}).status_code == 400
    other = env.frames(cam["id"])[-1]["id"]
    assert env.client.post(f"/api/units/{first['id']}/split", json={"frame_id": other}).status_code == 400


def test_relabel_whole_machine_moves_boxes_and_hours(env):
    site = env.site(timezone="UTC")
    env.client.put(f"/api/sites/{site['id']}/plan", json=[
        {"stage_id": 3, "planned_start": "2025-05-01", "planned_end": "2025-05-31"}])
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(4))
    (unit,) = units(env, site["id"])
    fid = env.frames(cam["id"])[2]["id"]
    det = frame(env, fid)["detections"][0]
    r = env.client.patch(f"/api/detections/{det['id']}", json={"cls": "bulldozer", "scope": "unit"})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "unit_relabel"
    (u,) = units(env, site["id"])
    assert u["cls"] == "bulldozer" and u["worked_hours"] == pytest.approx(unit["worked_hours"])
    for x in env.frames(cam["id"]):
        assert classes(env, x["id"]) == ["bulldozer"]
    ivs = env.client.get(f"/api/sites/{site['id']}/hours", params={"manual": "false"}).json()
    assert {iv["cls"] for iv in ivs} == {"bulldozer"} and sum(iv["hours"] for iv in ivs) == pytest.approx(1.0, abs=0.01)
    ov = env.client.get(f"/api/sites/{site['id']}/overview").json()
    assert next(e for e in ov["equipment"] if e["cls"] == "excavator")["worked_hours"] == 0.0, \
        "полоска экскаватора больше не считает часы бульдозера"
    r = env.client.patch(f"/api/units/{u['id']}", json={"cls": "roller"})
    assert r.status_code == 200
    (u,) = units(env, site["id"])
    assert u["cls"] == "roller"
    assert env.client.patch(f"/api/units/{u['id']}", json={"cls": "ufo"}).status_code == 400
    assert env.client.patch(f"/api/units/{u['id']}", json={}).status_code == 400


def test_unit_is_not_equipment(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    truck = next(u for u in units(env, site["id"]) if u["cls"] == "dump_truck")
    r = env.client.delete(f"/api/units/{truck['id']}")
    assert r.status_code == 200, r.text
    assert "не техника" in r.json()["note"] and "место" in r.json()["note"]   # стояла — скрыто и место
    assert [u["cls"] for u in units(env, site["id"])] == ["excavator"]
    env.upload(cam["id"], series(2, base=BASE + dt.timedelta(hours=3), truck=True, seed0=70))
    assert [u["cls"] for u in units(env, site["id"])] == ["excavator"]


def test_verify_frame_and_export_yolo_dataset(env):
    site = env.site(name="Сборный каркас")
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    f = [x["id"] for x in env.frames(cam["id"])]
    r = env.client.post(f"/api/frames/{f[0]}/verify", json={})
    assert r.status_code == 200 and r.json()["frame"]["annotations"]["verified"] is True
    assert env.client.post(f"/api/frames/{f[0]}/verify", json={"verified": "да"}).status_code == 400
    truck = box_of(env, f[1], "dump_truck")
    env.client.patch(f"/api/detections/{truck['id']}", json={"cls": "concrete_mixer"})
    env.wait()

    r = env.client.get(f"/api/sites/{site['id']}/dataset.zip")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip" and r.headers["x-dataset-frames"] == "2"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert {"data.yaml", "classes.txt", "manifest.json", "README.txt"} <= set(names)
    yaml = zf.read("data.yaml").decode()
    assert "nc: 21" in yaml and "0: excavator" in yaml
    classes_list = zf.read("classes.txt").decode().split()
    images = [n for n in names if n.startswith("images/")]
    labels = [n for n in names if n.startswith("labels/")]
    assert len(images) == 2 and len(labels) == 2
    manifest = json.loads(zf.read("manifest.json"))
    assert {fr["frame_id"] for fr in manifest["frames"]} == {f[0], f[1]}
    assert all(fr["reviewed"] for fr in manifest["frames"])
    by_frame = {fr["frame_id"]: fr for fr in manifest["frames"]}
    mixer = next(b for b in by_frame[f[1]]["boxes"] if b["cls"] == "concrete_mixer")
    assert mixer["source"] == "manual" and mixer["orig_cls"] == "dump_truck"
    lines = zf.read(by_frame[f[1]]["image"].replace("images/", "labels/").rsplit(".", 1)[0] + ".txt").decode().split("\n")
    idx = sorted(int(line.split()[0]) for line in lines if line)
    assert idx == sorted([classes_list.index("excavator"), classes_list.index("concrete_mixer")])
    for line in lines:
        if line:
            assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])
    assert {fr["split"] for fr in manifest["frames"]} == {"train", "val"}

    full = zipfile.ZipFile(io.BytesIO(env.client.get(f"/api/sites/{site['id']}/dataset.zip",
                                                     params={"scope": "all"}).content))
    assert len([n for n in full.namelist() if n.startswith("images/")]) == 3
    assert env.client.get(f"/api/sites/{site['id']}/dataset.zip", params={"scope": "x"}).status_code == 400
    assert env.client.get("/api/sites/999/dataset.zip").status_code == 404
    multi = env.client.get("/api/dataset.zip", params={"sites": str(site["id"]), "scope": "all"})
    assert multi.status_code == 200 and multi.headers["x-dataset-frames"] == "3"

    # снять отметку «проверен»
    r = env.client.post(f"/api/frames/{f[0]}/verify", json={"verified": False})
    assert r.status_code == 200 and r.json()["frame"]["annotations"]["verified"] is False


def test_export_dataset_cli(env, tmp_path):
    from tools import export_dataset
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(2, truck=True))
    out = tmp_path / "ds.zip"
    assert export_dataset.main(["--site", str(site["id"]), "--scope", "all", "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as zf:
        assert len([n for n in zf.namelist() if n.startswith("labels/")]) == 2
    target = tmp_path / "ds"
    assert export_dataset.main(["--site", str(site["id"]), "--scope", "all", "--dir", str(target)]) == 0
    assert (target / "data.yaml").exists() and len(list((target / "images").rglob("*.jpg"))) == 2


def test_manual_stage_plan_and_hours_survive_reprocess_and_replay(env):
    """Ручная отметка этапа, правка сроков плана и моточасов — не затираются ни переанализом,
    ни перепрогоном техники после правки разметки."""
    c = env.client
    site = env.site(timezone="UTC")
    plan = [{"stage_id": 3, "planned_start": "2025-05-01", "planned_end": "2025-05-20", "equipment": {"excavator": 1}},
            {"stage_id": 4, "planned_start": "2025-05-21", "planned_end": "2025-06-30"}]
    assert c.put(f"/api/sites/{site['id']}/plan", json=plan).status_code == 200
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(4))
    assert c.patch(f"/api/sites/{site['id']}/stages/4", json={
        "status": "active", "progress": 0.3, "actual_start": "2025-05-12", "note": "по акту"}).status_code == 200
    assert c.post(f"/api/sites/{site['id']}/hours", json={
        "cls": "excavator", "hours": 5, "stage_id": 3, "at": "2025-05-12T18:00", "note": "ночная смена"}).status_code == 201

    def state():
        ov = c.get(f"/api/sites/{site['id']}/overview").json()
        st = {s["id"]: s for s in ov["stages"]}
        exc = next(e for e in ov["equipment"] if e["cls"] == "excavator")
        return st[4], st[3]["planned_start"], st[3]["planned_end"], exc["worked_hours"]

    s4, a, b, worked = state()
    assert s4["manual"] and s4["progress"] == 0.3 and (a, b) == ("2025-05-01", "2025-05-20")
    reprocess(env, site["id"])
    fid = env.frames(cam["id"])[1]["id"]
    det = frame(env, fid)["detections"][0]
    c.post(f"/api/frames/{fid}/detections", json={"cls": "roller", "bbox": [250, 20, 40, 30]})
    c.patch(f"/api/detections/{det['id']}", json={"cls": "excavator"})
    env.wait()
    s4b, a2, b2, worked2 = state()
    assert s4b["manual"] and s4b["progress"] == 0.3 and s4b["actual_start"] == "2025-05-12"
    assert (a2, b2) == (a, b) and worked2 == pytest.approx(worked)
    assert [h["note"] for h in c.get(f"/api/sites/{site['id']}/hours").json()] == ["ночная смена"]


def test_journal_lists_actions_newest_first(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(2, truck=True))
    f = [x["id"] for x in env.frames(cam["id"])]
    env.client.post(f"/api/frames/{f[0]}/verify", json={})
    env.client.post(f"/api/frames/{f[1]}/detections", json={"cls": "grader", "bbox": [10, 10, 30, 30]})
    env.wait()
    j = env.client.get(f"/api/sites/{site['id']}/annotations").json()
    assert [b["kind"] for b in j["batches"]] == ["box_add", "frame_verify"]
    assert j["batches"][0]["author"] == "admin" and j["batches"][0]["frame_ids"] == [f[1]]
    assert j["stats"]["reviewed_frames"] == 2 and j["replay"]["state"] == "idle"
    assert env.client.get(f"/api/frames/{f[1]}/annotations").json()["count"] == 1
    assert env.client.post(f"/api/sites/{site['id']}/annotations/replay").status_code == 202
    env.wait()
    assert env.client.get(f"/api/sites/{site['id']}/annotations/status").json()["last"]["frames"] == 2
    assert env.client.get("/api/sites/999/annotations").status_code == 404


def test_stale_ids_after_replay_are_resolved_or_refused(env):
    """Перепрогон переписывает строки рамок и машин, а открытая страница помнит старые id:
    рамку находим по кадру и её рамке, машину — по uid; иначе 409, а не правка чужой машины."""
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(3, truck=True))
    fid = env.frames(cam["id"])[1]["id"]
    old = frame(env, fid)["detections"]
    exc, truck = (next(d for d in old if d["class"] == c) for c in ("excavator", "dump_truck"))
    assert env.client.patch(f"/api/detections/{exc['id']}", json={"cls": "excavator"}).status_code == 200
    env.wait()
    assert all(d["id"] != truck["id"] for d in frame(env, fid)["detections"]), "перепрогон пересоздал строки рамок"
    r = env.client.patch(f"/api/detections/{truck['id']}", json={"deleted": True, "frame_id": fid, "bbox": truck["bbox"]})
    assert r.status_code == 200, r.text
    env.wait()
    assert classes(env, fid) == ["excavator"]
    assert env.client.patch(f"/api/detections/{truck['id']}", json={"deleted": True}).status_code == 404

    unit = units(env, site["id"])[0]
    r = env.client.patch(f"/api/units/{unit['id']}", json={"cls": "roller", "uid": "u9999", "site_id": site["id"]})
    assert r.status_code == 409 and "устарел" in r.json()["detail"]
    r = env.client.patch("/api/units/99999", json={"cls": "roller", "uid": unit["uid"], "site_id": site["id"]})
    assert r.status_code == 200, "машина найдена по uid, хотя её id уже другой"


def test_annotation_api_requires_login(env):
    env.client.post("/api/logout")
    c = env.client
    for method, path, body in (
            ("PATCH", "/api/detections/1", {"cls": "roller"}), ("DELETE", "/api/detections/1", None),
            ("POST", "/api/frames/1/detections", {"cls": "roller", "bbox": [1, 1, 20, 20]}),
            ("POST", "/api/frames/1/verify", {}), ("GET", "/api/frames/1/annotations", None),
            ("POST", "/api/units/merge", {"unit_ids": [1, 2]}), ("POST", "/api/units/1/split", {"frame_id": 1}),
            ("PATCH", "/api/units/1", {"cls": "roller"}), ("DELETE", "/api/units/1", None),
            ("GET", "/api/units/1/timeline", None), ("GET", "/api/sites/1/annotations", None),
            ("GET", "/api/sites/1/annotations/status", None), ("POST", "/api/sites/1/annotations/replay", None),
            ("DELETE", "/api/annotations/batches/x", None), ("GET", "/api/sites/1/dataset.zip", None),
            ("GET", "/api/dataset.zip", None)):
        r = c.request(method, path, json=body) if body is not None else c.request(method, path)
        assert r.status_code == 401, (method, path, r.status_code)
