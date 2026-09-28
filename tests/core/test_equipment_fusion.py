"""Слияние камер: одна машина с разных ракурсов — одна единица; калибровка."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from core.contracts import CameraGeometry, FrameInfo, UnitStatus
from core.equipment import EquipmentConfig, EquipmentEngine, fusion, synthetic as S
from core.equipment.status import count_by_class

T0 = dt.datetime(2026, 9, 28, 8, 0, tzinfo=dt.timezone.utc)

# Две камеры смотрят на площадку 40×30 м с разных сторон: точки кадра ↔ точки плана.
CAM_A = ([[100, 650], [1180, 650], [900, 250], [380, 250]], [[0, 0], [40, 0], [40, 30], [0, 30]])
CAM_B = ([[120, 680], [1150, 640], [870, 230], [350, 260]], [[40, 30], [0, 30], [0, 0], [40, 0]])


def geom(cam, pts):
    H, _ = fusion.homography_from_points(*pts)
    return CameraGeometry(cam, H, (1280, 720))


GA, GB = geom("A", CAM_A), geom("B", CAM_B)


def to_image(g: CameraGeometry, site_xy, w=120, h=80):
    """Рамка, у которой середина нижней кромки проецируется в точку плана site_xy."""
    Hinv = np.linalg.inv(np.asarray(g.homography))
    v = Hinv @ np.array([site_xy[0], site_xy[1], 1.0])
    fx, fy = v[0] / v[2], v[1] / v[2]
    return (fx - w / 2, fy - h, float(w), float(h))


def frame(cam, minutes, k=0):
    return FrameInfo(frame_id=f"{cam}{k}", camera_id=cam, site_id=1, captured_at=T0 + dt.timedelta(minutes=minutes),
                     width=1280, height=720)


def see(engine, cam, g, minutes, machines, k=0, image=None):
    """machines: [(cls, (X, Y), extra)] → EquipmentUpdate."""
    dets = [S.det(cls, to_image(g, xy) if g else xy, **(extra or {})) for cls, xy, extra in machines]
    return engine.process(frame(cam, minutes, k), image, dets, g, [], [])


# --------------------------------------------------------------------------


def test_homography_recovers_exact_points():
    img, site = CAM_A
    H, err = fusion.homography_from_points(img, site)
    assert err < 1e-6
    for p, q in zip(img, site):
        assert fusion.project(H, p) == pytest.approx(tuple(q), abs=1e-6)


def test_homography_ransac_survives_one_bad_click_but_reports_it():
    rng = np.random.default_rng(0)
    site = rng.uniform(0, 40, (8, 2))
    Hinv = np.linalg.inv(np.asarray(GA.homography))
    img = [(lambda v: [v[0] / v[2], v[1] / v[2]])(Hinv @ np.array([x, y, 1.0])) for x, y in site]
    img[3] = [img[3][0] + 150, img[3][1] - 90]          # одна точка кликнута мимо
    H, err = fusion.homography_from_points(img, site.tolist())
    assert fusion.project(H, img[0]) == pytest.approx(tuple(site[0]), abs=0.05)
    assert err > 1.0, "ошибка репроекции должна показать пользователю плохую точку"


@pytest.mark.parametrize("img,site,msg", [
    ([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]], "минимум 4"),
    ([[0, 0], [10, 10], [20, 20], [30, 30]], [[0, 0], [1, 0], [1, 1], [0, 1]], "одной прямой"),
    ([[0, 0], [10, 0], [10, 10], [0, 10]], [[0, 0], [1, 0], [1, 1]], "поровну"),
])
def test_homography_rejects_bad_input(img, site, msg):
    with pytest.raises(ValueError, match=msg):
        fusion.homography_from_points(img, site)


def test_projection_rescales_when_frame_resolution_differs():
    H = GA.homography
    full = fusion.project(H, (640, 500))
    half = fusion.project(H, (320, 250), image_size=(1280, 720), frame_size=(640, 360))
    assert half == pytest.approx(full)


def test_point_above_horizon_is_not_projected():
    # у горизонта знаменатель гомографии уходит в ноль и за него
    Hm = np.asarray(GA.homography)
    y_h = -Hm[2, 2] / Hm[2, 1] if abs(Hm[2, 1]) > 1e-12 else -1e9
    assert fusion.project(GA.homography, (640, y_h - 50)) is None


def test_plate_normalization():
    assert fusion.normalize_plate("а 123 вс 77") == fusion.normalize_plate("A123BC77") == "A123BC77"
    assert fusion.normalize_plate("") is None and fusion.normalize_plate("7") is None


# --------------------------------------------------------------------------


def test_two_calibrated_cameras_one_machine_one_unit():
    eng = EquipmentEngine()
    ua = see(eng, "A", GA, 0, [("excavator", (20, 10), None)])
    ub = see(eng, "B", GB, 5, [("excavator", (20.6, 10.4), None)])
    assert ua.detections[0].unit_id == ub.detections[0].unit_id
    assert count_by_class(ub.units) == {"excavator": 1}
    assert ub.units[0].cameras == {"A", "B"}
    assert ub.detections[0].site_xy == pytest.approx((20.6, 10.4), abs=0.05)


def test_three_close_machines_in_overlap_stay_three_units():
    """Три экскаватора в 4 м друг от друга (ближе радиуса склейки) — всё равно три машины."""
    eng = EquipmentEngine()
    xs = [(10, 10), (14, 10), (18, 10)]
    ua = see(eng, "A", GA, 0, [("excavator", p, None) for p in xs])
    ub = see(eng, "B", GB, 5, [("excavator", (x + 0.4, y - 0.3), None) for x, y in reversed(xs)])
    by_pos_a = {round(d.site_xy[0]): d.unit_id for d in ua.detections}
    by_pos_b = {round(d.site_xy[0]): d.unit_id for d in ub.detections}
    assert len(set(by_pos_a.values())) == 3
    assert by_pos_b == by_pos_a, "каждая машина камеры B склеена со своей машиной камеры A"
    assert count_by_class(ub.units) == {"excavator": 3}


def test_uncalibrated_cameras_are_independent():
    eng = EquipmentEngine()
    see(eng, "A", None, 0, [("excavator", (500, 300, 120, 80), None)])
    ub = see(eng, "B", None, 5, [("excavator", (500, 300, 120, 80), None)])
    assert count_by_class(ub.units) == {"excavator": 2}
    assert any("не откалибрована" in n for n in ub.notes)


def test_same_plate_merges_unconditionally():
    eng = EquipmentEngine()
    ua = see(eng, "A", GA, 0, [("dump_truck", (5, 5), {"plate": "А123ВС77"})])
    ub = see(eng, "B", GB, 5, [("truck", (35, 25), {"plate": "a123bc77"})])
    assert ua.detections[0].unit_id == ub.detections[0].unit_id


def test_different_plates_never_merge_even_when_close():
    eng = EquipmentEngine()
    see(eng, "A", GA, 0, [("dump_truck", (10, 10), {"plate": "A123BC77"})])
    ub = see(eng, "B", GB, 5, [("dump_truck", (10.3, 10.2), {"plate": "K777KK99"})])
    assert count_by_class(ub.units) == {"dump_truck": 2}


def test_gray_zone_is_decided_by_colour():
    """6.5 м — дальше радиуса склейки: жёлтую с жёлтой склеиваем, жёлтую с синей — нет."""
    def yellow_img(g, xy, color):
        img = S.ground(1280, 720, seed=3)
        S.draw_machine(img, to_image(g, xy), color=color)
        return img

    for color_b, expected in (((0, 200, 240), 1), ((200, 60, 20), 2)):
        eng = EquipmentEngine()
        see(eng, "A", GA, 0, [("excavator", (20, 10), None)], image=yellow_img(GA, (20, 10), (0, 200, 240)))
        ub = see(eng, "B", GB, 5, [("excavator", (26.5, 10), None)], image=yellow_img(GB, (26.5, 10), color_b))
        assert count_by_class(ub.units) == {"excavator": expected}, color_b


def test_young_duplicate_from_boundary_is_merged_and_reported():
    """Камера B сначала спроецировала машину с ошибкой (7 м) — родился дубль; следующий кадр его склеивает."""
    eng = EquipmentEngine()
    ua = see(eng, "A", GA, 0, [("excavator", (10, 10), None)])
    ub1 = see(eng, "B", GB, 5, [("excavator", (17, 10), None)], k=1)
    assert ub1.detections[0].unit_id != ua.detections[0].unit_id
    ub2 = see(eng, "B", GB, 30, [("excavator", (17.2, 10.1), None)], k=2)
    see(eng, "A", GA, 32, [("excavator", (10.1, 10), None)], k=3)
    ub3 = see(eng, "B", GB, 35, [("excavator", (10.6, 10.2), None)], k=4)
    dup = ub1.detections[0].unit_id
    merged = {**ub2.merged, **ub3.merged}
    assert merged.get(dup) == ua.detections[0].unit_id
    assert ub3.detections[0].unit_id == ua.detections[0].unit_id
    assert dup not in {u.unit_id for u in eng.units()}


def test_track_that_drifts_away_from_its_unit_is_split_off():
    """Склеили две разные машины (стояли рядом), потом та, что видна камере B, уехала:
    трек B не прерывается, но на плане устойчиво расходится с машиной камеры A — отделяем."""
    eng = EquipmentEngine()
    see(eng, "A", GA, 0, [("excavator", (10, 10), None)])
    first = see(eng, "B", GB, 5, [("excavator", (10.5, 10), None)])
    uid = first.detections[0].unit_id
    ids = []
    for k, x in enumerate((16, 22, 28, 34)):
        m = 25 * (k + 1)
        see(eng, "A", GA, m, [("excavator", (10, 10), None)], k=10 + k)
        ub = see(eng, "B", GB, m + 3, [("excavator", (x, 10), None)], k=20 + k)
        ids.append((ub.detections[0].track_id, ub.detections[0].unit_id))
    assert {t for t, _ in ids} == {first.detections[0].track_id}, "трек камеры B непрерывен"
    assert [u for _, u in ids[:3]] == [uid] * 3, "одно расхождение — ещё не повод разделять"
    assert ids[3][1] != uid
    assert count_by_class(ub.units) == {"excavator": 2}


def test_cluster_never_puts_two_boxes_of_one_camera_together():
    cfg = EquipmentConfig()
    o = lambda cam, xy: fusion.Observation(frozenset({cam}), T0, "excavator", xy)   # noqa: E731
    obs = [o("A", (0, 0)), o("A", (1, 0)), o("B", (0.5, 0))]
    groups = fusion.cluster(obs, cfg)
    assert len(groups) == 2
    assert fusion.count_units(obs) == 2


def test_departed_and_parked_units_count_separately():
    eng = EquipmentEngine()
    see(eng, "A", GA, 0, [("excavator", (20, 10), None)])
    units = eng.units()
    assert units[0].status == UnitStatus.IDLE
    assert count_by_class(units, statuses=(UnitStatus.ACTIVE,)) == {}
