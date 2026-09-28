"""Регрессии по итогам ревью модели А: каждый тест — конкретный найденный сценарий поломки."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from core.contracts import ActivityInterval, CameraGeometry, FrameInfo, UnitStatus, Zone
from core.equipment import EquipmentConfig, EquipmentEngine, fusion, hours, synthetic as S
from core.equipment.detect_vlm import VlmDetector
from core.equipment.detect_yolo import YoloDetector
from core.equipment.status import count_by_class

T0 = dt.datetime(2026, 9, 28, 6, 0, tzinfo=dt.timezone.utc)
SITE = ([[100, 650], [1180, 650], [900, 250], [380, 250]], [[0, 0], [40, 0], [40, 30], [0, 30]])
H, _ = fusion.homography_from_points(*SITE)


def fi(cam, minutes, k=0, w=1280, h=720):
    return FrameInfo(frame_id=f"{cam}{k}", camera_id=cam, site_id=1,
                     captured_at=T0 + dt.timedelta(minutes=minutes), width=w, height=h)


def test_junk_plate_readings_do_not_glue_different_machines():
    """VLM ответила «нет»/«UNKNOWN» вместо номера у разных машин — это не один номер."""
    for junk in ("нет", "н/д", "UNKNOWN", "0000000"):
        assert fusion.normalize_plate(junk) is None, junk
    eng = EquipmentEngine()
    g = CameraGeometry("A", H, (1280, 720))
    gb = CameraGeometry("B", H, (1280, 720))
    eng.process(fi("A", 0), None, [S.det("dump_truck", (200, 500, 120, 80), plate="нет")], g, [], [])
    upd = eng.process(fi("B", 5), None, [S.det("dump_truck", (900, 300, 120, 80), plate="нет")], gb, [], [])
    assert count_by_class(upd.units) == {"dump_truck": 2}
    assert upd.detections[0].extra == {"plate_raw": "нет", "unit_label": "Самосвал №2", "unit_status": "idle"}


def test_balances_do_not_double_count_overlapping_rows_of_one_unit():
    """После склейки дубля веб-слой переписывает unit_id старых строк: 10:00–10:25 и 10:10–10:35 — это 35 минут."""
    m = lambda k: T0 + dt.timedelta(minutes=k)   # noqa: E731
    ivs = [ActivityInterval("u1", "excavator", m(0), m(25), 25 / 60, None),
           ActivityInterval("u1", "excavator", m(10), m(35), 25 / 60, None),
           ActivityInterval("u2", "excavator", m(10), m(35), 25 / 60, None)]      # другая машина — своё время
    (b,) = hours.balances([], ivs)
    assert b.worked_hours == pytest.approx((35 + 25) / 60)


def test_projection_without_frame_size_does_not_crash():
    """Переразбор по сохранённым рамкам: размеров кадра нет, картинки нет — не делить на ноль."""
    assert fusion.project(H, (640, 500), image_size=(1280, 720), frame_size=(0, 0)) == fusion.project(H, (640, 500))
    eng = EquipmentEngine()
    upd = eng.process(fi("A", 0, w=0, h=0), None, [S.det("excavator", (500, 400, 120, 80))],
                      CameraGeometry("A", H, (1280, 720)), [], [])
    assert upd.detections[0].site_xy is not None


def test_vlm_pixel_answer_is_relative_to_the_downscaled_copy_it_saw():
    """Кадр 1920×1080 уходит в модель копией 1280×720; пиксели ответа — пиксели копии."""
    class Client:
        model_id = "local-vlm"

        def __init__(self, bbox):
            self.bbox = bbox

        def ask_json(self, *a):
            return SimpleNamespace(data={"objects": [{"class": "excavator", "bbox": self.bbox, "conf": 0.9}]},
                                   text="", latency_ms=1)

    img = np.zeros((1080, 1920, 3), np.uint8)
    (d,) = VlmDetector(client=Client([640, 360, 1280, 720])).detect(img)
    assert d.bbox == pytest.approx((960.0, 540.0, 960.0, 540.0))
    (d,) = VlmDetector(client=Client([100, 100, 400, 300]), coords="pixels").detect(img)
    assert d.bbox == pytest.approx((150.0, 150.0, 450.0, 300.0))


def test_camera_parking_zone_beats_big_site_work_zone():
    parking = Zone(1, "Отстой", "parking", "A", [(150, 380), (450, 380), (450, 650), (150, 650)])
    pit = Zone(2, "Котлован", "work", None, [(-100, -100), (100, -100), (100, 100), (-100, 100)])   # метры
    eng = EquipmentEngine()
    g = CameraGeometry("A", H, (1280, 720))
    upd = None
    for k in range(3):
        upd = eng.process(fi("A", 25 * k, k), None, [S.det("bulldozer", (240, 450, 120, 80))], g, [parking, pit], [])
    assert upd.detections[0].zone_id == 1 and upd.units[0].status == UnitStatus.PARKED


def test_camera_without_zones_does_not_release_machine_from_parking():
    parking = Zone(1, "Отстой", "parking", "A", [(150, 380), (450, 380), (450, 650), (150, 650)])
    g, gb = CameraGeometry("A", H, (1280, 720)), CameraGeometry("B", H, (1280, 720))
    eng = EquipmentEngine()
    statuses = []
    for k in range(4):
        statuses.append(eng.process(fi("A", 25 * k, k), None, [S.det("crane_manipulator", (240, 450, 120, 80))],
                                    g, [parking], []).units[0].status)
        statuses.append(eng.process(fi("B", 25 * k + 5, 10 + k), None, [S.det("crane_manipulator", (245, 452, 120, 80))],
                                    gb, [parking], []).units[0].status)
    assert len(eng.units()) == 1
    assert set(statuses) == {UnitStatus.PARKED}


def test_parked_in_zone_survives_restart():
    parking = Zone(1, "Отстой", "parking", "A", [(150, 380), (450, 380), (450, 650), (150, 650)])
    eng = EquipmentEngine()
    for k in range(3):
        upd = eng.process(fi("A", 25 * k, k), None, [S.det("bulldozer", (240, 450, 120, 80))], None, [parking], [])
    assert upd.units[0].status == UnitStatus.PARKED
    fresh = EquipmentEngine()
    fresh.restore(eng.units(), {"A": (T0 + dt.timedelta(minutes=50), upd.detections)})
    # первый кадр после рестарта — от другой камеры, машину не видит
    out = fresh.process(fi("C", 60, 99), None, [], None, [], [])
    assert out.units[0].status == UnitStatus.PARKED


@pytest.mark.parametrize("key", ["merge_radius_m", "split_after", "vote_window", "max_credit_gap_min"])
def test_zero_thresholds_are_rejected(key):
    with pytest.raises(ValueError):
        EquipmentConfig.from_dict({key: 0})


def test_single_move_does_not_make_unit_active():
    """Человек прошёл перед машиной — один «сдвиг»: полоска не уменьшилась, значит и статус не «работает»."""
    eng = EquipmentEngine()
    xs = [300, 340, 340, 340]
    seen = []
    for k, x in enumerate(xs):
        upd = eng.process(fi("A", 25 * k, k), None, [S.det("excavator", (x, 300, 160, 110))], None, [], [])
        seen.append((upd.detections[0].moved_since_prev, upd.units[0].status))
    assert seen[1] == (True, UnitStatus.IDLE)
    assert upd.units[0].worked_hours == 0.0


def test_parked_timer_is_not_reset_by_a_single_spurious_move():
    eng = EquipmentEngine()
    upd = None
    for k in range(0, 97):                                     # 48 ч раз в 30 мин
        x = 340 if k == 40 else 300                            # один ложный «сдвиг» посередине
        upd = eng.process(fi("A", 30 * k, k), None, [S.det("excavator", (x, 300, 160, 110))], None, [], [])
    assert upd.units[0].status == UnitStatus.PARKED


def test_yolo_per_class_threshold_below_global_conf_reaches_the_network(tmp_path):
    import json
    cj = tmp_path / "equipment_classes.json"
    cj.write_text(json.dumps({"names": {"0": "tower_crane", "1": "excavator"}}), encoding="utf-8")

    class Model:
        names: dict = {}

        def predict(self, img, **kw):
            self.conf = kw["conf"]
            rows = [(r, c, k) for r, c, k in [((100, 50, 300, 600), 0.2, 0), ((600, 300, 800, 450), 0.2, 1)]
                    if c >= kw["conf"]]
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=np.array([r for r, _, _ in rows], np.float32).reshape(-1, 4),
                conf=np.array([c for _, c, _ in rows]), cls=np.array([k for _, _, k in rows])))]

    model = Model()
    cfg = EquipmentConfig(conf_by_class={"tower_crane": 0.15})
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model, config=cfg)
    out = det.detect(S.ground(1280, 720), fi("A", 0))
    assert model.conf == 0.15
    assert [d.cls for d in out] == ["tower_crane"], "экскаватору 0.2 по-прежнему мало (общий порог 0.25)"


def test_interval_set_floor_is_not_an_interval():
    s = hours.IntervalSet(floor=T0)
    s.add(T0 - dt.timedelta(minutes=10), T0 + dt.timedelta(minutes=20))
    assert list(s) == [(T0, T0 + dt.timedelta(minutes=20))], "новое время не прилипает к границе рестарта"


def test_one_frame_box_jump_over_static_machine_is_not_work_with_images():
    """Детектор на одном кадре «сдвинул» рамку неподвижной машины и вернул: старое место
    не изменилось — ни один из двух сдвигов не работа (с картинкой это видно по содержимому)."""
    base = S.ground(1280, 720, seed=3)
    S.draw_machine(base, (300, 300, 160, 110))
    eng = EquipmentEngine()
    moved = []
    upd = None
    for k in range(6):
        x = 340 if k == 3 else 300
        upd = eng.process(fi("A", 25 * k, k), S.relight(base, seed=k), [S.det("excavator", (x, 300, 160, 110))],
                          None, [], [])
        moved.append(upd.detections[0].moved_since_prev)
    assert moved == [False] * 6
    assert upd.units[0].worked_hours == 0.0 and upd.units[0].status == UnitStatus.IDLE


def test_box_flicker_without_images_is_marked_and_not_credited():
    eng = EquipmentEngine()
    extras = []
    for k, x in enumerate([300, 300, 340, 300, 300]):
        upd = eng.process(fi("A", 25 * k, k), None, [S.det("excavator", (x, 300, 160, 110))], None, [], [])
        extras.append(upd.detections[0].extra.get("box_flicker", False))
    assert extras == [False, False, False, True, False]
    assert upd.units[0].worked_hours == 0.0
    # а настоящая езда (A → B → C) — работа
    eng = EquipmentEngine()
    for k, x in enumerate([300, 340, 380, 420]):
        upd = eng.process(fi("A", 25 * k, k), None, [S.det("excavator", (x, 300, 160, 110))], None, [], [])
    assert upd.units[0].worked_hours == pytest.approx(75 / 60)


def test_vlm_detector_reports_supported_classes():
    det = VlmDetector(client=object(), classes=["excavator", "truck", "бульдозер-невидимка"])
    assert det.supported_classes == ["excavator", "truck"]
