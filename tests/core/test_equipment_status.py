"""Статусы единиц: работает / стоит / на стоянке / уехала — на сценариях длиной в дни."""
from __future__ import annotations

import datetime as dt

from core.contracts import FrameInfo, UnitState, UnitStatus, Zone
from core.equipment import EquipmentConfig, EquipmentEngine, synthetic as S
from core.equipment.status import compute_status

T0 = dt.datetime(2026, 9, 28, 6, 0, tzinfo=dt.timezone.utc)
BOX = (500.0, 300.0, 160.0, 110.0)


def fi(cam, minutes, k):
    return FrameInfo(frame_id=f"{cam}-{k}", camera_id=cam, site_id=1,
                     captured_at=T0 + dt.timedelta(minutes=minutes), width=1280, height=720)


def feed(eng, cam, schedule, zones=()):
    """schedule: [(минуты, [Detection])] → последний EquipmentUpdate."""
    upd = None
    for k, (m, dets) in enumerate(schedule):
        upd = eng.process(fi(cam, m, k), None, dets, None, list(zones), [])
    return upd


def test_dump_truck_that_left_becomes_departed():
    eng = EquipmentEngine()
    present = [(25 * k, [S.det("dump_truck", (500 + 30 * k, 300, 160, 110))]) for k in range(4)]
    gone = [(25 * k, []) for k in range(4, 12)]                      # камера снимает, самосвала нет
    upd = feed(eng, "c1", present + gone[:5])                          # 2 ч без самосвала
    assert upd.units[0].status != UnitStatus.DEPARTED
    upd = feed(eng, "c1", present[:0] + [(25 * k, []) for k in range(9, 13)])
    assert upd.units[0].status == UnitStatus.DEPARTED


def test_silent_camera_does_not_make_machine_depart():
    """Камера c1 замолчала на 5 ч (нет кадров) — это не доказательство, что самосвал уехал.
    Кадры другой камеры, которая его никогда не видела, тоже не доказательство."""
    eng = EquipmentEngine()
    feed(eng, "c1", [(25 * k, [S.det("dump_truck", BOX)]) for k in range(3)])
    upd = feed(eng, "c2", [(60 + 25 * k, []) for k in range(14)])
    assert upd.units[0].status == UnitStatus.IDLE


def test_machine_standing_three_days_is_parked():
    eng = EquipmentEngine()
    sched = [(30 * k, [S.det("excavator", BOX)]) for k in range(0, 72 * 2 + 1)]   # 3 суток раз в 30 мин
    statuses = {}
    for k, (m, dets) in enumerate(sched):
        upd = eng.process(fi("c1", m, k), None, dets, None, [], [])
        statuses[m / 60] = upd.units[0].status
    assert statuses[24.0] == UnitStatus.IDLE
    assert statuses[47.5] == UnitStatus.IDLE
    assert statuses[48.0] == UnitStatus.PARKED
    assert statuses[72.0] == UnitStatus.PARKED


def test_working_machine_is_active_then_idle_after_it_stops():
    eng = EquipmentEngine()
    moving = [(25 * k, [S.det("excavator", (400 + 40 * k, 300, 160, 110))]) for k in range(4)]
    upd = feed(eng, "c1", moving)
    assert upd.units[0].status == UnitStatus.ACTIVE
    upd = feed(eng, "c1", [(100 + 25, [S.det("excavator", (520, 300, 160, 110))])])
    assert upd.units[0].status == UnitStatus.IDLE


def test_parking_zone_parks_idle_machine_immediately():
    zone = Zone(id=7, name="Отстой техники", kind="parking", camera_id="c1",
                polygon=[(450, 250), (750, 250), (750, 500), (450, 500)])
    eng = EquipmentEngine()
    upd = feed(eng, "c1", [(25 * k, [S.det("bulldozer", BOX)]) for k in range(3)], zones=[zone])
    assert upd.detections[0].zone_id == 7
    assert upd.units[0].status == UnitStatus.PARKED


def test_returning_truck_gets_its_old_unit_id_back():
    """Челночный самосвал уехал и вернулся — единица та же, а не «Самосвал №2»."""
    eng = EquipmentEngine()
    first = feed(eng, "c1", [(25 * k, [S.det("dump_truck", BOX)]) for k in range(3)])
    uid = first.detections[0].unit_id
    feed(eng, "c1", [(25 * k, []) for k in range(3, 12)])           # уехал больше чем на 3 ч
    assert eng.units()[0].status == UnitStatus.DEPARTED
    back = feed(eng, "c1", [(25 * 13, [S.det("dump_truck", (200, 320, 160, 110))])])
    assert back.detections[0].unit_id == uid
    assert len(back.units) == 1 and back.units[0].status == UnitStatus.IDLE


def test_status_rules_directly():
    cfg = EquipmentConfig()
    u = UnitState("u1", "excavator", UnitStatus.IDLE, first_seen=T0, last_seen=T0 + dt.timedelta(hours=1),
                  last_moved=T0 + dt.timedelta(minutes=55), cameras={"c1", "c2"})
    last = {"c1": u.last_seen, "c2": u.last_seen}
    assert compute_status(u, u.last_seen, last, cfg) == UnitStatus.ACTIVE
    # одна из двух камер продолжает снимать 4 ч и не видит машину — уехала
    assert compute_status(u, u.last_seen, {**last, "c2": u.last_seen + dt.timedelta(hours=4)}, cfg) \
        == UnitStatus.DEPARTED
