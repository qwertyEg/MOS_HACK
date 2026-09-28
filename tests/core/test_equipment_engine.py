"""EquipmentEngine целиком: сценарии площадки, моточасы, рестарт, потоки, формат вывода."""
from __future__ import annotations

import datetime as dt
import json
import threading

import numpy as np
import pytest

from core.contracts import Activity, CameraGeometry, Detection, FrameInfo, PlanItem, UnitStatus
from core.equipment import EquipmentConfig, EquipmentEngine, EquipmentUpdate, fusion, hours, synthetic as S
from core.equipment.status import count_by_class

T0 = dt.datetime(2026, 9, 28, 6, 0, tzinfo=dt.timezone.utc)      # 09:00 по Москве, понедельник
PLAN = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 10, 10), name="Котлован",
                 planned_hours={"excavator": 16.0, "dump_truck": 28.0})]


def fi(cam, minutes, k=0, night=False):
    return FrameInfo(frame_id=k, camera_id=cam, site_id=1, captured_at=T0 + dt.timedelta(minutes=minutes),
                     width=1280, height=720, is_night=night)


def moving(k, cls="excavator", y=300):
    """Машина, которая на каждом кадре сдвигается на 40 px — работает."""
    return S.det(cls, (300 + 40 * k, y, 160, 110))


def total(intervals):
    return sum(iv.hours for iv in intervals)


# --------------------------------------------------------------------------
# моточасы
# --------------------------------------------------------------------------


def test_three_working_intervals_of_25_minutes_give_1_25_hours():
    eng = EquipmentEngine()
    ivs = []
    for k in range(4):
        ivs += eng.process(fi("c1", 25 * k, k), None, [moving(k)], None, [], PLAN).intervals
    assert total(ivs) == pytest.approx(1.25)
    assert {iv.stage_id for iv in ivs} == {3}
    assert all(len(iv.frame_ids) == 2 for iv in ivs), "у каждого интервала — два кадра-доказательства"
    assert eng.units()[0].worked_hours == pytest.approx(1.25)
    (bal,) = [b for b in hours.balances(PLAN, ivs) if b.cls == "excavator"]
    assert bal.remaining_hours == pytest.approx(16 - 1.25)


def test_single_isolated_move_is_not_work_but_consecutive_moves_are():
    """confirm_moves=2: один сдвиг (машину переставили, перед ней прошёл человек) — не работа."""
    eng = EquipmentEngine()
    boxes = [(300, 300), (340, 300), (340, 300), (340, 300)]           # сдвиг только один раз
    ivs = []
    for k, (x, y) in enumerate(boxes):
        ivs += eng.process(fi("c1", 25 * k, k), None, [S.det("excavator", (x, y, 160, 110))], None, [], PLAN).intervals
    assert ivs == []
    # при confirm_moves=1 тот же единичный сдвиг засчитывается (как в формуле методики)
    eng1 = EquipmentEngine(EquipmentConfig(confirm_moves=1))
    ivs1 = []
    for k, (x, y) in enumerate(boxes):
        ivs1 += eng1.process(fi("c1", 25 * k, k), None, [S.det("excavator", (x, y, 160, 110))], None, [], PLAN).intervals
    assert total(ivs1) == pytest.approx(25 / 60)


def test_gap_in_frames_caps_credit_at_45_minutes():
    """Камера молчала 100 мин (меньше max_gap) — машина сдвинулась, но засчитываем не больше 45 мин."""
    eng = EquipmentEngine()
    times = [0, 25, 125]
    ivs = []
    for k, m in enumerate(times):
        ivs += eng.process(fi("c1", m, k), None, [moving(k)], None, [], PLAN).intervals
    assert total(ivs) == pytest.approx((25 + 45) / 60)


def test_five_hour_gap_credits_at_most_45_minutes():
    eng = EquipmentEngine()
    ivs = []
    for k, m in enumerate([0, 25, 50, 50 + 300]):
        ivs += eng.process(fi("c1", m, k), None, [moving(k)], None, [], PLAN).intervals
    gap_part = [iv for iv in ivs if iv.end == T0 + dt.timedelta(minutes=350)]
    assert sum(iv.hours for iv in gap_part) <= 0.75
    assert total(ivs) == pytest.approx(50 / 60)


def test_night_frames_count_too():
    eng = EquipmentEngine()
    night0 = 20 * 60            # 02:00 МСК следующего дня
    ivs = []
    for k in range(3):
        ivs += eng.process(fi("c1", night0 + 25 * k, k, night=True), None, [moving(k)], None, [], PLAN).intervals
    assert total(ivs) == pytest.approx(50 / 60)


def test_two_cameras_seeing_one_working_machine_do_not_double_count_hours():
    img_a, site = [[100, 650], [1180, 650], [900, 250], [380, 250]], [[0, 0], [40, 0], [40, 30], [0, 30]]
    Ha, _ = fusion.homography_from_points(img_a, site)
    g = CameraGeometry("A", Ha, (1280, 720))
    gb = CameraGeometry("B", Ha, (1280, 720))       # вторая камера с тем же видом — проще для проверки часов
    eng = EquipmentEngine()
    ivs = []
    for k in range(4):
        box = (300 + 40 * k, 400, 160, 110)
        ivs += eng.process(fi("A", 25 * k, k), None, [S.det("excavator", box)], g, [], PLAN).intervals
        box_b = (300 + 40 * k + 20, 400, 160, 110)
        ivs += eng.process(fi("B", 25 * k + 10, 100 + k), None, [S.det("excavator", box_b)], gb, [], PLAN).intervals
    assert len(eng.units()) == 1
    # A видит движение 0→75, B — 10→85: объединение — 85 минут, а не 150
    assert total(ivs) == pytest.approx(85 / 60)
    assert eng.units()[0].worked_hours == pytest.approx(85 / 60)


def test_hours_outside_plan_have_no_stage():
    eng = EquipmentEngine()
    ivs = []
    for k in range(3):
        ivs += eng.process(fi("c1", 25 * k, k), None, [moving(k, "bulldozer")], None, [], []).intervals
    assert ivs and all(iv.stage_id is None for iv in ivs)


# --------------------------------------------------------------------------
# сценарий площадки на синтетических кадрах
# --------------------------------------------------------------------------


def test_excavation_scene_excavator_digs_truck_shuttles():
    """Экскаватор копает на месте (шасси стоит, стрела ходит), самосвал подъезжает и уезжает."""
    eng = EquipmentEngine()
    exc = (700.0, 250.0, 240.0, 180.0)
    poses = [0, 25, 5, 30, 10, 28, 2]
    truck_at = {1: (380.0, 330.0, 220.0, 140.0), 2: (380.0, 330.0, 220.0, 140.0), 3: (200.0, 360.0, 220.0, 140.0)}
    ivs, upd = [], None
    for k, pose in enumerate(poses):
        img = S.ground(1280, 720, seed=5)
        S.draw_machine(img, exc, pose=pose)
        dets = [S.det("excavator", exc)]
        if k in truck_at:
            S.draw_machine(img, truck_at[k], color=(40, 40, 200), kind="dump_truck")
            dets.append(S.det("dump_truck", truck_at[k], conf=0.7))
        img = S.relight(img, 1.0 + 0.05 * (k % 3), seed=k)
        upd = eng.process(fi("c1", 25 * k, k), img, dets, None, [], PLAN)
        ivs += upd.intervals
        if k >= 1:
            assert upd.detections[0].activity == Activity.WORKING, (k, upd.detections[0].appearance_delta)
    exc_hours = sum(iv.hours for iv in ivs if iv.cls == "excavator")
    assert exc_hours == pytest.approx(6 * 25 / 60)
    assert count_by_class(upd.units) == {"excavator": 1, "dump_truck": 1}
    assert {u.cls: u.status for u in upd.units}["excavator"] == UnitStatus.ACTIVE


def test_output_is_serializable_and_matches_contract():
    eng = EquipmentEngine()
    upd = None
    for k in range(3):
        upd = eng.process(fi("c1", 25 * k, k), None,
                          [moving(k), S.det("truck", (900, 300, 200, 120), 0.6, alt=[["dump_truck", 0.55]])],
                          None, [], PLAN)
    assert isinstance(upd, EquipmentUpdate)
    for d in upd.detections:
        c = d.to_contract()
        assert set(c) >= {"class", "bbox", "conf", "zone_id", "moved_since_prev", "displacement_px",
                          "bbox_shape_delta", "track_id", "unit_id", "activity"}
        json.dumps(c)
        json.dumps(d.extra)          # extra пишется в БД как JSON — никаких numpy внутри
        assert d.unit_id and d.track_id
        assert d.extra["unit_label"].endswith("№1")
    assert upd.detections[0].activity == Activity.WORKING and upd.detections[1].activity == Activity.IDLE


def test_label_majority_over_frames_gives_stable_class():
    eng = EquipmentEngine()
    labels = ["truck", "dump_truck", "dump_truck", "truck", "dump_truck", "dump_truck"]
    upd = None
    for k, c in enumerate(labels):
        upd = eng.process(fi("c1", 25 * k, k), None, [S.det(c, (500, 300, 200, 120), 0.7)], None, [], [])
    assert len(upd.units) == 1 and upd.units[0].cls == "dump_truck"
    assert upd.detections[0].cls == "dump_truck"
    assert upd.units[0].label == "Самосвал №1"


def test_zone_assignment_uses_smallest_containing_zone():
    from core.contracts import Zone
    big = Zone(1, "Площадка", "work", "c1", [(0, 0), (1280, 0), (1280, 720), (0, 720)])
    pit = Zone(2, "Котлован", "work", "c1", [(400, 300), (900, 300), (900, 600), (400, 600)])
    other_cam = Zone(3, "Чужая", "work", "c9", [(0, 0), (1280, 0), (1280, 720), (0, 720)])
    eng = EquipmentEngine()
    upd = eng.process(fi("c1", 0), None, [S.det("excavator", (500, 350, 160, 110))], None, [big, pit, other_cam], [])
    assert upd.detections[0].zone_id == 2


# --------------------------------------------------------------------------
# рестарт, порядок кадров, потоки, настройки
# --------------------------------------------------------------------------


def test_restore_continues_ids_tracks_and_does_not_recredit():
    eng = EquipmentEngine()
    last_dets, ivs = None, []
    for k in range(3):
        upd = eng.process(fi("c1", 25 * k, k), None, [moving(k)], None, [], PLAN)
        last_dets, ivs = upd.detections, ivs + upd.intervals
    units = eng.units()

    fresh = EquipmentEngine()
    fresh.restore(units, {"c1": (T0 + dt.timedelta(minutes=50), last_dets)})
    upd = fresh.process(fi("c1", 75, 3), None, [moving(3)], None, [], PLAN)
    assert upd.detections[0].track_id == last_dets[0].track_id
    assert upd.detections[0].unit_id == units[0].unit_id
    assert upd.detections[0].activity == Activity.WORKING
    # серия движения до рестарта не сохранилась: один интервал — ещё не подтверждение
    upd2 = fresh.process(fi("c1", 100, 4), None, [moving(4)], None, [], PLAN)
    assert total(upd2.intervals) == pytest.approx(50 / 60)
    assert fresh.units()[0].worked_hours == pytest.approx(units[0].worked_hours + 50 / 60)
    new = fresh.process(fi("c1", 125, 5), None, [moving(5), S.det("bulldozer", (50, 500, 150, 100))],
                        None, [], PLAN)
    assert new.detections[1].unit_id == "u0002", "счётчик единиц продолжился после рестарта"


def test_out_of_order_frame_changes_nothing():
    eng = EquipmentEngine()
    for k in range(3):
        eng.process(fi("c1", 25 * k, k), None, [moving(k)], None, [], PLAN)
    before = eng.units()
    upd = eng.process(fi("c1", 10, 99), None, [S.det("excavator", (100, 100, 160, 110))], None, [], PLAN)
    assert upd.intervals == [] and upd.notes
    assert upd.detections[0].unit_id is None and upd.detections[0].activity == Activity.UNKNOWN
    assert eng.units() == before


def test_parallel_cameras_under_lock():
    eng = EquipmentEngine()
    errors = []

    def camera(cam, y):
        try:
            for k in range(30):
                # машина ходит туда-обратно на 40 px — каждый кадр движение, трек не рвётся
                eng.process(fi(cam, 25 * k, k), None, [moving(k % 2, y=y)], None, [], PLAN)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=camera, args=(f"c{i}", 100 + 50 * i)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(eng.units()) == 4, "некалиброванные камеры: по единице на камеру"


def test_daily_camera_does_not_breed_units_when_machines_are_moved_overnight():
    """Камера снимает раз в сутки (архив Эдинбурга): экскаватор каждый день в новом месте
    кадра, трекер его по рамке не узнаёт — но единица та же. Со второй машиной одновременно —
    две единицы, не больше. Моточасов по суточным снимкам нет: движение не оценивается."""
    eng = EquipmentEngine()
    xs = [100, 700, 300, 900, 200, 600, 1000, 150]
    ivs, upd = [], None
    for k, x in enumerate(xs):
        dets = [S.det("excavator", (x, 300, 160, 110))]
        if k >= 4:
            dets.append(S.det("excavator", ((x + 500) % 1100, 500, 160, 110)))
        upd = eng.process(fi("c1", 24 * 60 * k, k), None, dets, None, [], PLAN)
        ivs += upd.intervals
        assert len({d.unit_id for d in upd.detections}) == len(dets), "две рамки кадра — две машины"
    assert len(eng.units()) == 2, [u.label for u in eng.units()]
    assert ivs == []
    # неделю в кадре, но движение по суточным снимкам не оценивается: «стоит», а не «на стоянке,
    # ждёт вывоза» — иначе аналитика пишет «работы встали» там, где работу просто не видно
    assert {u.status for u in eng.units()} == {UnitStatus.IDLE}


def test_machine_seen_standing_for_two_days_by_a_regular_camera_is_parked():
    """Обычная камера (25 мин) видит, что экскаватор не двигается больше parked_after_h, — PARKED."""
    eng = EquipmentEngine()
    for k in range(0, 50 * 60 // 25 + 1):
        eng.process(fi("c1", 25 * k, k), None, [S.det("excavator", (300, 300, 160, 110))], None, [], [])
    assert eng.units()[0].status == UnitStatus.PARKED


def test_daily_camera_machine_back_after_many_snapshots_is_the_same_unit():
    """Без номера машины одного типа не различить: экскаватор, которого суточная камера
    не видела десять снимков, — та же единица, а не «Экскаватор №2»."""
    eng = EquipmentEngine()
    for day in range(11):
        dets = [S.det("excavator", (100 + 600 * (day == 10), 300, 160, 110))] if day in (0, 10) else []
        eng.process(fi("c1", 24 * 60 * day, day), None, dets, None, [], [])
    assert len(eng.units()) == 1


def test_two_departed_machines_coming_back_together_keep_their_units():
    """Два самосвала уехали (камера снимала без них больше departed_after_h) и вернулись
    вместе: каждый получает одну из прежних единиц, третьей не появляется."""
    eng = EquipmentEngine()
    trucks = lambda x0: [S.det("dump_truck", (x0, 300, 200, 130)), S.det("dump_truck", (x0 + 500, 300, 200, 130))]  # noqa: E731
    eng.process(fi("c1", 0, 0), None, trucks(100), None, [], [])
    for k in range(1, 9):
        eng.process(fi("c1", 25 * k, k), None, [], None, [], [])
    assert {u.status for u in eng.units()} == {UnitStatus.DEPARTED}
    upd = eng.process(fi("c1", 25 * 9, 9), None, trucks(300), None, [], [])
    assert len({d.unit_id for d in upd.detections}) == 2
    assert len(eng.units()) == 2, [u.label for u in eng.units()]


def test_regular_camera_keeps_separate_units_for_a_second_machine():
    """Без разрыва съёмки поведение прежнее: вторая машина в другом месте — новая единица."""
    eng = EquipmentEngine()
    eng.process(fi("c1", 0, 0), None, [S.det("excavator", (100, 300, 160, 110))], None, [], [])
    upd = eng.process(fi("c1", 25, 1), None, [S.det("excavator", (100, 300, 160, 110)),
                                              S.det("excavator", (900, 300, 160, 110))], None, [], [])
    assert len({d.unit_id for d in upd.detections}) == 2 and len(eng.units()) == 2


def test_config_roundtrip_and_unknown_keys():
    cfg = EquipmentConfig.from_dict({"merge_radius_m": "7.5", "workdays": [0, 1, 2, 3, 4], "confirm_moves": 1,
                                     "no_such_threshold": 1, "shape_needs_appearance": "false"})
    assert cfg.merge_radius_m == 7.5 and cfg.workdays == (0, 1, 2, 3, 4) and cfg.confirm_moves == 1
    assert cfg.shape_needs_appearance is False
    again = EquipmentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    with pytest.raises(ValueError):
        EquipmentConfig.from_dict({"parked_after_h": -1})


def test_engine_accepts_frames_without_image_or_detections():
    eng = EquipmentEngine()
    upd = eng.process(fi("c1", 0), None, [], None, [], [])
    assert upd.detections == [] and upd.units == [] and upd.intervals == []
    img = np.zeros((720, 1280, 3), np.uint8)
    upd = eng.process(fi("c1", 25, 1), img, [Detection("excavator", 0.9, (10, 10, 5, 5))], None, [], [])
    assert upd.detections == [], "мелочь отфильтрована"
