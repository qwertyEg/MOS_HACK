"""Трекер одной камеры: «работает / стоит» по кадрам раз в 20–30 минут.

Сцены синтетические (core/equipment/synthetic.py): детектор здесь идеальный
или с дрожанием рамки, проверяется то, что делает сама модель А.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from core.contracts import Activity, FrameInfo
from core.equipment import synthetic as S
from core.equipment.config import EquipmentConfig
from core.equipment.tracker import CameraTracker

T0 = dt.datetime(2026, 9, 28, 8, 0, tzinfo=dt.timezone.utc)
BOX = (250.0, 150.0, 120.0, 90.0)


def fi(k: int, minutes: float = 25.0, night: bool = False, t0=T0) -> FrameInfo:
    return FrameInfo(frame_id=k, camera_id="cam1", site_id=1, captured_at=t0 + dt.timedelta(minutes=minutes * k),
                     width=640, height=360, is_night=night)


def scene(pose: float = 0.0, box=BOX, seed: int = 0, gain: float = 1.0, offset: float = 0.0,
          night: bool = False, shift=(0, 0), color=(0, 170, 240), kind="excavator") -> np.ndarray:
    img = S.ground(seed=1)
    S.draw_machine(img, box, color=color, pose=pose, kind=kind)
    img = S.darken(img, seed=seed) if night else S.relight(img, gain, offset, seed=seed)
    return S.shift(img, *shift) if shift != (0, 0) else img


def jitter(box, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dw, box[3] + dh)


def run(tracker, frames):
    """frames: [(FrameInfo, image, [Detection])] → список шагов по кадрам."""
    out = []
    for f, img, dets in frames:
        steps, _ = tracker.update(f, img, dets)
        out.append(steps)
    return out


def test_first_frame_of_track_is_unknown():
    tr = CameraTracker("cam1", EquipmentConfig())
    steps, _ = tr.update(fi(0), scene(), [S.det("excavator", BOX)])
    assert steps[0].detection.activity == Activity.UNKNOWN
    assert steps[0].detection.track_id == "cam1-t1"
    assert steps[0].prev_seen is None and not steps[0].judged


def test_excavator_digging_in_place_is_working():
    """Центр рамки стоит, меняется поза стрелы/ковша — это работа."""
    tr = CameraTracker("cam1", EquipmentConfig())
    poses = [0, 20, 5, 30, 12]
    res = run(tr, [(fi(k), scene(p, seed=k), [S.det("excavator", jitter(BOX, dx=(-1) ** k))])
                   for k, p in enumerate(poses)])
    later = [s[0].detection for s in res[1:]]
    assert all(d.activity == Activity.WORKING for d in later), [(d.appearance_delta, d.displacement_px) for d in later]
    assert all(d.displacement_px < 8 for d in later), "шасси не двигалось — сработать должна внешность"
    assert len({s[0].detection.track_id for s in res}) == 1


def test_static_machine_idle_despite_noise_light_and_box_jitter():
    tr = CameraTracker("cam1", EquipmentConfig())
    lights = [(1.0, 0), (0.7, 10), (1.3, -25), (0.85, 30), (1.15, 0), (0.6, 5)]
    res = run(tr, [(fi(k), scene(0, seed=k, gain=g, offset=o),
                    [S.det("excavator", jitter(BOX, dx=2 * (-1) ** k, dy=1, dw=3 * (-1) ** k, dh=-2))])
                   for k, (g, o) in enumerate(lights)])
    later = [s[0].detection for s in res[1:]]
    assert all(d.activity == Activity.IDLE for d in later), [d.appearance_delta for d in later]
    assert max(d.appearance_delta for d in later) < EquipmentConfig().appearance_thr


def test_static_machine_idle_at_night():
    tr = CameraTracker("cam1", EquipmentConfig())
    res = run(tr, [(fi(k, night=True), scene(0, seed=k, night=True), [S.det("excavator", BOX)])
                   for k in range(5)])
    assert all(s[0].detection.activity == Activity.IDLE for s in res[1:])


def test_camera_shake_is_compensated():
    """Камера на ветру сместилась на несколько пикселей — вся картинка и рамки поехали, машина стоит."""
    tr = CameraTracker("cam1", EquipmentConfig())
    shifts = [(0, 0), (6, -4), (-3, 5), (7, 2)]
    res = run(tr, [(fi(k), scene(0, seed=k, shift=sh), [S.det("excavator", jitter(BOX, *sh))])
                   for k, sh in enumerate(shifts)])
    later = [s[0].detection for s in res[1:]]
    assert all(d.activity == Activity.IDLE for d in later), [(d.displacement_px, d.appearance_delta) for d in later]
    assert all(d.displacement_px < 3 for d in later)


def test_large_camera_move_resets_motion():
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    steps, notes = tr.update(fi(1), scene(0, shift=(60, 0)), [S.det("excavator", jitter(BOX, 60))])
    assert steps[0].detection.activity == Activity.UNKNOWN
    assert any("камера сдвинулась" in n for n in notes)


def test_driving_machine_is_moved():
    tr = CameraTracker("cam1", EquipmentConfig())
    b1 = jitter(BOX, dx=45)
    tr.update(fi(0), scene(0, kind="dump_truck"), [S.det("dump_truck", BOX)])
    steps, _ = tr.update(fi(1), scene(0, box=b1, kind="dump_truck", seed=1), [S.det("dump_truck", b1)])
    d = steps[0].detection
    assert d.moved_since_prev and d.activity == Activity.WORKING
    assert d.displacement_px == pytest.approx(45, abs=1)


def test_label_flip_truck_dump_truck_keeps_one_track_with_majority_label():
    tr = CameraTracker("cam1", EquipmentConfig())
    labels = ["dump_truck", "truck", "dump_truck", "dump_truck", "truck", "dump_truck"]
    res = run(tr, [(fi(k), scene(0, kind="dump_truck", seed=k), [S.det(c, BOX, conf=0.7)])
                   for k, c in enumerate(labels)])
    ids = {s[0].detection.track_id for s in res}
    assert len(ids) == 1
    assert res[-1][0].track.label == "dump_truck"


def test_alt_votes_from_nms_count_towards_label():
    tr = CameraTracker("cam1", EquipmentConfig())
    for k in range(4):
        d = S.det("truck", BOX, conf=0.55, alt=[["dump_truck", 0.54]])
        steps, _ = tr.update(fi(k), None, [d])
    # голос проигравшей рамки — вполсилы, поэтому метка остаётся truck,
    # но оба класса видны в истории голосов
    assert steps[0].track.label == "truck"
    assert {c for c, _ in steps[0].track.votes} == {"truck", "dump_truck"}


def test_gap_longer_than_max_gap_restarts_motion_judgement():
    cfg = EquipmentConfig(max_gap_min=120)
    tr = CameraTracker("cam1", cfg)
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    far = jitter(BOX, dx=60)
    f = FrameInfo(1, "cam1", 1, T0 + dt.timedelta(hours=5), 640, 360)
    steps, _ = tr.update(f, scene(0, box=far, seed=2), [S.det("excavator", far)])
    d = steps[0].detection
    assert d.activity == Activity.UNKNOWN and not d.moved_since_prev
    assert d.track_id == "cam1-t1", "машину узнали (трек тот же), но смещение за 5 ч работой не считаем"
    assert d.extra["gap_min"] == 300


def test_far_move_within_frame_rematched_by_colour_when_old_spot_vacated():
    tr = CameraTracker("cam1", EquipmentConfig())
    far = (40.0, 230.0, 120.0, 90.0)
    tr.update(fi(0), scene(0, kind="dump_truck", color=(30, 30, 200)), [S.det("dump_truck", BOX)])
    steps, _ = tr.update(fi(1), scene(0, box=far, kind="dump_truck", color=(30, 30, 200), seed=1),
                         [S.det("dump_truck", far)])
    d = steps[0].detection
    assert d.track_id == "cam1-t1" and d.moved_since_prev


def test_detector_miss_does_not_steal_track_for_other_machine():
    """Экскаватор на месте, детектор его пропустил, а в другом углу появился другой — это новый трек."""
    tr = CameraTracker("cam1", EquipmentConfig())
    other = (40.0, 230.0, 120.0, 90.0)
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    img = scene(0, seed=1)
    S.draw_machine(img, other, pose=10)
    steps, _ = tr.update(fi(1), img, [S.det("excavator", other)])
    assert steps[0].detection.track_id == "cam1-t2"
    assert steps[0].detection.activity == Activity.UNKNOWN


def test_two_neighbours_keep_their_tracks():
    """Две машины рядом, обе чуть сдвинулись — венгерский алгоритм не перепутает их местами."""
    tr = CameraTracker("cam1", EquipmentConfig())
    a, b = (100.0, 150.0, 100.0, 80.0), (215.0, 150.0, 100.0, 80.0)
    tr.update(fi(0), None, [S.det("excavator", a), S.det("excavator", b)])
    a2, b2 = jitter(a, dx=30), jitter(b, dx=30)
    steps, _ = tr.update(fi(1), None, [S.det("excavator", b2), S.det("excavator", a2)])
    assert steps[0].detection.track_id == "cam1-t2"
    assert steps[1].detection.track_id == "cam1-t1"


def test_shape_change_without_image_counts_as_moved():
    """Без картинки (переразбор по сохранённым рамкам) форма рамки работает как в методике."""
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(0), None, [S.det("excavator", BOX)])
    steps, _ = tr.update(fi(1), None, [S.det("excavator", jitter(BOX, dw=25, dh=5))])
    assert steps[0].detection.bbox_shape_delta > 0.15 and steps[0].detection.moved_since_prev


def test_shape_jitter_without_content_change_is_not_work():
    """Детектор «вздохнул» рамкой на стоящей машине: форма поменялась, содержимое — нет."""
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    steps, _ = tr.update(fi(1), scene(0, seed=1), [S.det("excavator", jitter(BOX, dw=18, dh=6))])
    d = steps[0].detection
    assert d.bbox_shape_delta > 0.15
    assert not d.moved_since_prev and d.activity == Activity.IDLE


def test_day_night_switch_ignores_appearance():
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    steps, notes = tr.update(fi(1, night=True), scene(0, night=True, seed=1), [S.det("excavator", BOX)])
    assert steps[0].detection.activity == Activity.IDLE
    assert steps[0].detection.appearance_delta == 0.0
    assert any("день/ночь" in n for n in notes)


def test_rain_on_lens_frame_uses_geometry_only():
    """Капли на объективе: содержимое рамки «изменилось», но это не работа — и такой кроп
    не должен стать образцом для следующего кадра."""
    from core.contracts import Weather
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(0), scene(0), [S.det("excavator", BOX)])
    rainy = scene(0, seed=1)
    rng = np.random.default_rng(3)
    for _ in range(40):                                   # размытые пятна-капли
        x, y = int(rng.integers(200, 420)), int(rng.integers(100, 280))
        rainy[y - 6:y + 6, x - 6:x + 6] = rainy[y - 6:y + 6, x - 6:x + 6] // 2 + 100
    f = FrameInfo(1, "cam1", 1, T0 + dt.timedelta(minutes=25), 640, 360, weather=Weather.RAIN, quality_ok=False)
    steps, notes = tr.update(f, rainy, [S.det("excavator", jitter(BOX, dw=20))])
    assert steps[0].detection.activity == Activity.IDLE and steps[0].detection.appearance_delta == 0.0
    assert any("капли" in n for n in notes)
    steps, _ = tr.update(fi(2), scene(0, seed=2), [S.det("excavator", BOX)])
    assert steps[0].detection.activity == Activity.IDLE


def test_assignment_never_returns_forbidden_pairs():
    """Регрессия: венгерский алгоритм назначал строку и по «запрещённой» (∞) стоимости."""
    from core.equipment.tracker import _assign, _greedy
    inf = float("inf")
    cost = np.array([[inf, 0.5], [inf, inf]])        # строка 1 может получить столбец 0 только по ∞
    assert _assign(cost, inf) == [(0, 1)]
    assert _greedy(cost, inf) == [(0, 1)]
    assert _assign(np.array([[inf, 2.5]]), 1.8) == []


def test_out_of_order_frame_does_not_touch_state():
    tr = CameraTracker("cam1", EquipmentConfig())
    tr.update(fi(2), scene(0), [S.det("excavator", BOX)])
    steps, notes = tr.update(fi(1), scene(0), [S.det("excavator", BOX)])
    assert steps[0].detection.activity == Activity.UNKNOWN and steps[0].detection.track_id is None
    assert tr.last_frame_at == fi(2).captured_at and len(tr.tracks) == 1
    assert notes
