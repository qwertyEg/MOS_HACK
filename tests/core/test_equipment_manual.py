"""Ручная разметка в модели А: машина, названная оператором, класс, закреплённый им,
рамка оператора сильнее фильтров детектора (app/services/annotations.py → Detection.extra)."""
from __future__ import annotations

import datetime as dt

import pytest

from core.contracts import FrameInfo, PlanItem
from core.equipment import EquipmentEngine, postprocess, synthetic as S
from core.equipment.engine import MANUAL_CLS, MANUAL_UNIT, MANUAL_UNIT_CLS, is_manual

T0 = dt.datetime(2026, 10, 14, 5, 0, tzinfo=dt.timezone.utc)      # 08:00 по Москве
PLAN = [PlanItem(5, dt.date(2026, 10, 1), dt.date(2026, 11, 30), name="Монолит",
                 planned_hours={"mobile_crane": 50.0, "tower_crane": 100.0})]


def fi(cam, minutes, k=0):
    return FrameInfo(frame_id=k, camera_id=cam, site_id=1, captured_at=T0 + dt.timedelta(minutes=minutes),
                     width=1280, height=720)


def keyed(cls, box, key, lock=None, conf=0.6):
    extra = {MANUAL_UNIT: key, "manual": {"unit": key}}
    if lock:
        extra[MANUAL_UNIT_CLS] = lock
    return S.det(cls, box, conf=conf, **extra)


def test_operator_merge_joins_uncalibrated_cameras_into_one_machine():
    """Две некалиброванные камеры — независимы (две единицы), пока оператор не сказал «это одна машина»."""
    eng = EquipmentEngine()
    for k in range(3):
        eng.process(fi("A", 25 * k, k), None, [S.det("mobile_crane", (300 + 40 * k, 300, 160, 110))], None, [], PLAN)
        eng.process(fi("B", 25 * k + 5, 10 + k), None, [S.det("mobile_crane", (500, 200 + 40 * k, 160, 110))],
                    None, [], PLAN)
    assert len(eng.units()) == 2

    eng = EquipmentEngine()
    ivs = []
    for k in range(3):
        ivs += eng.process(fi("A", 25 * k, k), None, [keyed("mobile_crane", (300 + 40 * k, 300, 160, 110), "m1", "mobile_crane")],
                           None, [], PLAN).intervals
        ivs += eng.process(fi("B", 25 * k + 5, 10 + k), None,
                           [keyed("mobile_crane", (500, 200 + 40 * k, 160, 110), "m1", "mobile_crane")], None, [], PLAN).intervals
    (unit,) = eng.units()
    assert unit.unit_id == "m1" and is_manual(unit.unit_id)
    assert unit.cameras == {"A", "B"}
    # обе камеры видели одну работу — часы не удвоены (объединение интервалов 0→55 мин)
    assert sum(iv.hours for iv in ivs) == pytest.approx(55 / 60)


def test_locked_class_beats_detector_votes_and_goes_to_hours():
    """Оператор: «это башенный кран», детектор упорно пишет «автокран» — класс и моточасы у крана."""
    eng = EquipmentEngine()
    ivs = []

    def boom(k):          # кран стоит, стрела ходит: рамка то выше, то ниже — это работа
        return (300, 100, 160, 400 if k % 2 == 0 else 480)

    for k in range(4):
        d = keyed("mobile_crane", boom(k), "m7", "tower_crane", conf=0.9)
        d.cls = "tower_crane"          # так рамку отдаёт annotations.apply
        up = eng.process(fi("A", 25 * k, k), None, [d], None, [], PLAN)
        ivs += up.intervals
        assert up.detections[0].cls == "tower_crane"
    # новые кадры без правки (живая камера): трек продолжает ручную машину, класс держится
    for k in range(4, 12):
        up = eng.process(fi("A", 25 * k, k), None, [S.det("mobile_crane", boom(k), conf=0.95)], None, [], PLAN)
        ivs += up.intervals
        assert up.detections[0].unit_id == "m7"
        assert up.detections[0].cls == "tower_crane"
    assert ivs and {iv.cls for iv in ivs} == {"tower_crane"}
    assert eng.units()[0].label.startswith("Башенный кран")


def test_split_moves_the_rest_of_the_track_to_a_new_machine():
    """Разделение с кадра: у трека новый ключ — дальше это другая машина, прошлое остаётся за старой."""
    eng = EquipmentEngine()
    uids = []
    for k in range(6):
        box = (300 + 30 * k, 300, 160, 110)
        d = keyed("excavator", box, "m_a", "excavator") if k < 3 else keyed("excavator", box, "m_b", "excavator")
        uids.append(eng.process(fi("A", 25 * k, k), None, [d], None, [], PLAN).detections[0].unit_id)
    assert uids == ["m_a"] * 3 + ["m_b"] * 3
    # и после разделения на кадрах без правок трек остаётся за новой машиной
    up = eng.process(fi("A", 150, 6), None, [S.det("excavator", (480, 300, 160, 110))], None, [], PLAN)
    assert up.detections[0].unit_id == "m_b"
    assert len(eng.units()) == 2


def test_manual_unit_is_never_merged_away_by_camera_fusion():
    eng = EquipmentEngine()
    eng.process(fi("A", 0, 0), None, [keyed("excavator", (300, 300, 160, 110), "m9", "excavator")], None, [], PLAN)
    assert not eng._young_single("m9", T0)
    assert is_manual("m9") and not is_manual("u0001") and not is_manual(None)


def test_two_boxes_of_one_frame_with_the_same_key_stay_two_machines():
    eng = EquipmentEngine()
    up = eng.process(fi("A", 0, 0), None, [keyed("excavator", (100, 300, 160, 110), "m1", "excavator"),
                                           keyed("excavator", (600, 300, 160, 110), "m1", "excavator")],
                     None, [], PLAN)
    assert len({d.unit_id for d in up.detections}) == 2


def test_box_class_set_by_operator_is_not_overwritten_by_unit_class():
    """«Только эта рамка»: у рамки ручной класс, у машины — свой по большинству."""
    eng = EquipmentEngine()
    for k in range(5):
        eng.process(fi("A", 25 * k, k), None, [S.det("excavator", (300 + 5 * k, 300, 160, 110))], None, [], PLAN)
    d = S.det("mobile_crane", (325, 300, 160, 110), conf=1.0, **{MANUAL_CLS: "mobile_crane", "manual": {"cls": "mobile_crane"}})
    up = eng.process(fi("A", 125, 5), None, [d], None, [], PLAN)
    assert up.detections[0].cls == "mobile_crane"
    assert eng.units()[0].cls == "excavator"


def test_set_manual_classes_after_restore_and_counter_ignores_manual_ids():
    eng = EquipmentEngine()
    for k in range(3):
        eng.process(fi("A", 25 * k, k), None, [keyed("excavator", (300 + 40 * k, 300, 160, 110), "m123", "excavator")],
                    None, [], PLAN)
    units = eng.units()
    last = {"A": (T0 + dt.timedelta(minutes=50), [])}
    eng2 = EquipmentEngine()
    eng2.restore(units, last)
    eng2.set_manual_classes({"m123": "bulldozer"})
    assert eng2.units()[0].cls == "bulldozer"
    up = eng2.process(fi("A", 75, 3), None, [S.det("excavator", (900, 500, 100, 80))], None, [], PLAN)
    # новая машина получает u0001, а не u0124 от хвоста ручного ключа
    assert up.detections[0].unit_id == "u0001"


def test_operator_box_survives_filters_and_wins_nms():
    """Рамка оператора: маленькая, у края, с любой «уверенностью» — не отсеивается и побеждает NMS."""
    small = S.det("excavator", (0, 700, 12, 12), conf=1.0, manual={"added": True}, **{MANUAL_CLS: "excavator"})
    weak = S.det("truck", (400, 300, 200, 120), conf=0.5, manual={"cls": "truck"})
    strong = S.det("dump_truck", (402, 301, 200, 120), conf=0.95)
    out = postprocess.clean([small, weak, strong], 1280, 720)
    assert {d.cls for d in out} == {"excavator", "truck"}
    kept = next(d for d in out if d.cls == "truck")
    assert "alt" not in kept.extra, "голос погашенной рамки модели не спорит с оператором"
    # две рамки оператора друг друга не гасят
    a = S.det("truck", (400, 300, 200, 120), conf=1.0, manual={"added": True})
    b = S.det("dump_truck", (405, 305, 200, 120), conf=1.0, manual={"added": True})
    assert len(postprocess.clean([a, b], 1280, 720)) == 2
