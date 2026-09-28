"""CRUD объектов и камер, зоны, калибровка, невалидный ввод → 4xx."""
from __future__ import annotations

from tests.app.conftest import series

CARD_FIELDS = {"id", "name", "verdict", "lag_days", "current_stage", "progress", "active_units",
               "open_deviations", "last_frame_at", "thumb"}


def test_site_crud(env):
    c = env.client
    site = env.site(name="ЖК Ромашка", address="Москва", floors_total=17)
    assert CARD_FIELDS <= site.keys()
    assert site["verdict"] == "no_plan" and site["floors_total"] == 17

    assert [s["name"] for s in c.get("/api/sites").json()] == ["ЖК Ромашка"]
    r = c.patch(f"/api/sites/{site['id']}", json={"name": "ЖК Ромашка-2", "shift_hours": 12})
    assert r.status_code == 200 and r.json()["name"] == "ЖК Ромашка-2" and r.json()["shift_hours"] == 12
    assert c.get(f"/api/sites/{site['id']}").json()["name"] == "ЖК Ромашка-2"
    assert c.delete(f"/api/sites/{site['id']}").json()["ok"] is True
    assert c.get(f"/api/sites/{site['id']}").status_code == 404
    assert c.get("/api/sites").json() == []


def test_site_invalid_input(env):
    c = env.client
    assert c.post("/api/sites", json={}).status_code == 400
    assert c.post("/api/sites", json={"name": "   "}).status_code == 400
    assert c.post("/api/sites", json={"name": "x", "timezone": "Mars/Base"}).status_code == 400
    assert c.post("/api/sites", json={"name": "x", "floors_total": -1}).status_code == 400
    assert c.post("/api/sites", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
    r = c.post("/api/sites", json={"name": "x", "shift_hours": "десять"})
    assert r.status_code == 400 and "shift_hours" in r.json()["detail"]
    assert c.get("/api/sites/abc").status_code == 400
    assert c.patch("/api/sites/999", json={"name": "y"}).status_code == 404


def test_camera_crud_and_delete_cascades(env):
    c = env.client
    site = env.site()
    cam = env.camera(site["id"], name="Север", interval_min=30, kind="folder")
    assert cam["ingest_key"] and cam["interval_min"] == 30 and cam["kind"] == "folder"
    assert c.post("/api/sites/999/cameras", json={"name": "x"}).status_code == 404
    assert c.post(f"/api/sites/{site['id']}/cameras", json={"name": "x", "kind": "rtsp"}).status_code == 400
    assert c.post(f"/api/sites/{site['id']}/cameras", json={"name": "x", "interval_min": 0}).status_code == 400

    old_key = cam["ingest_key"]
    r = c.patch(f"/api/cameras/{cam['id']}", json={"name": "Юг", "regenerate_key": True})
    assert r.json()["name"] == "Юг" and r.json()["ingest_key"] != old_key
    assert len(c.get(f"/api/sites/{site['id']}/cameras").json()) == 1

    env.upload(cam["id"], series(2))
    assert c.get(f"/api/cameras/{cam['id']}").json()["frames_total"] == 2
    assert c.get(f"/api/sites/{site['id']}/equipment").json()["units"]
    assert c.delete(f"/api/cameras/{cam['id']}").status_code == 200
    assert c.get(f"/api/cameras/{cam['id']}").status_code == 404
    assert c.get(f"/api/sites/{site['id']}/cameras").json() == []
    # техника, которую видела только удалённая камера, не остаётся «фантомом» (отчёт UI)
    assert c.get(f"/api/sites/{site['id']}/equipment").json()["units"] == []


def test_zones_create_list_delete(env):
    c = env.client
    cam = env.camera(env.site()["id"])
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    r = c.post(f"/api/cameras/{cam['id']}/zones", json={"name": "Котлован", "kind": "work", "polygon": poly})
    assert r.status_code == 201
    zone = r.json()
    assert zone["polygon"] == [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    assert c.post(f"/api/cameras/{cam['id']}/zones", json={"name": "z", "polygon": [[0, 0], [1, 1]]}).status_code == 400
    assert c.post(f"/api/cameras/{cam['id']}/zones",
                  json={"name": "z", "kind": "лес", "polygon": poly}).status_code == 400
    assert c.post(f"/api/cameras/{cam['id']}/zones", json={"name": "z", "polygon": [[0, "a"], [1, 1], [2, 2]]}
                  ).status_code == 400
    assert [z["name"] for z in c.get(f"/api/cameras/{cam['id']}/zones").json()] == ["Котлован"]
    assert c.delete(f"/api/cameras/{cam['id']}/zones/{zone['id']}").status_code == 200
    assert c.delete(f"/api/cameras/{cam['id']}/zones/{zone['id']}").status_code == 404
    assert c.get(f"/api/cameras/{cam['id']}/zones").json() == []


def test_calibration(env):
    c = env.client
    cam = env.camera(env.site()["id"])
    img = [[0, 0], [320, 0], [320, 240], [0, 240]]
    site_pts = [[0, 0], [32, 0], [32, 24], [0, 24]]
    r = c.post(f"/api/cameras/{cam['id']}/calibration", json={"image_points": img, "site_points": site_pts})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["homography"]) == 3 and body["reproj_error"] < 1e-6 and body["points"] == 4
    assert c.get(f"/api/cameras/{cam['id']}").json()["calibrated"] is True
    # мало точек, разное число, мусор
    assert c.post(f"/api/cameras/{cam['id']}/calibration",
                  json={"image_points": img[:3], "site_points": site_pts[:3]}).status_code == 400
    assert c.post(f"/api/cameras/{cam['id']}/calibration",
                  json={"image_points": img, "site_points": site_pts + [[1, 1]]}).status_code == 400
    assert c.post(f"/api/cameras/{cam['id']}/calibration", json={"image_points": "x"}).status_code == 400
    # сброс калибровки
    r = c.patch(f"/api/cameras/{cam['id']}", json={"homography": None})
    assert r.json()["calibrated"] is False and r.json()["calib_points"] is None
