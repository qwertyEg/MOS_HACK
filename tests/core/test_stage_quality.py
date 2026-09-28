"""Качество кадра на синтетике и реальных снимках стройки (Edinburgh, test_photos Никиты).

Главные сценарии из требований: ночь отсеивается, пасмурный серый день — НЕТ
(баг Дениса: sat < 12 → «ночь»), капли на объективе исключают кадр из модели Б,
снег не исключает, размытый кадр — брак.
"""
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.contracts import Weather
from core.stage import quality as Q

PHOTOS = Path(__file__).resolve().parent.parent / "fixtures" / "stage_photos"
MSK = dt.timezone(dt.timedelta(hours=3))
NOON_JUNE = dt.datetime(2026, 6, 15, 12, 0, tzinfo=MSK)
NOON_DEC = dt.datetime(2026, 12, 10, 12, 30, tzinfo=MSK)
NIGHT_DEC = dt.datetime(2026, 12, 10, 2, 0, tzinfo=MSK)


def scene(seed: int = 0, h: int = 480, w: int = 640) -> np.ndarray:
    """Небо (гладкий градиент), текстурная земля, дома с окнами, мелкая техника."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for y in range(h // 3):
        img[y] = (200 - y * 0.2, 190 - y * 0.2, 170 - y * 0.2)
    ground = cv2.GaussianBlur(rng.normal(110, 25, (h - h // 3, w)).astype(np.float32), (0, 0), 1.2)
    img[h // 3:] = np.stack([ground * 0.9, ground, ground * 1.1], axis=2)
    for bx in range(20, w - 100, 150):
        bh = int(rng.integers(120, 220))
        y0 = max(0, h // 3 + 60 - bh)
        col = rng.integers(60, 200, 3).astype(np.float32)
        img[y0:h // 3 + 60, bx:bx + 110] = col
        for wy in range(y0 + 8, h // 3 + 50, 18):
            for wx in range(bx + 8, bx + 100, 16):
                img[wy:wy + 9, wx:wx + 8] = col * 0.4
    for _ in range(25):
        x, y = int(rng.integers(0, w - 30)), int(rng.integers(h // 2, h - 20))
        img[y:y + 12, x:x + 25] = rng.integers(0, 255, 3)
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def overcast(img: np.ndarray) -> np.ndarray:
    """Пасмурный день: цвет почти выцвел, свет приглушён, но контраст сцены на месте."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.15
    hsv[..., 2] *= 0.85
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def with_drops(img: np.ndarray, n: int = 7, seed: int = 1) -> np.ndarray:
    """Капли на стекле: круглые размытые пятна чуть светлее фона, с бликом внутри."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (0, 0), w / 70)
    for _ in range(n):
        r = int(rng.integers(w // 40, w // 22))
        cx, cy = int(rng.integers(r + 10, w - r - 10)), int(rng.integers(h // 3 + r, h - r - 10))
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (cx, cy), r, 255, -1)
        m = m > 0
        out[m] = np.clip(blurred[m].astype(np.int16) + 25, 0, 255).astype(np.uint8)
        cv2.circle(out, (cx - r // 3, cy - r // 3), max(2, r // 6), (255, 255, 255), -1)
    return out


def night_scene() -> np.ndarray:
    img = (scene() * 0.22).astype(np.uint8)
    for x, y in [(100, 200), (300, 250), (500, 180), (200, 400)]:
        cv2.circle(img, (x, y), 6, (200, 220, 255), -1)      # фонари
    return img


def test_clean_day_is_usable():
    r = Q.assess(scene(), NOON_JUNE)
    assert r.quality_ok and not r.is_night and r.usable_for_stage
    assert r.weather is Weather.CLEAR and r.reject_reason == ""
    assert r.blur > 100 and 100 < r.brightness < 150


def test_dark_frame_is_night_and_not_for_stage():
    for when in (None, NIGHT_DEC):
        r = Q.assess(night_scene(), when)
        assert r.is_night and r.quality_ok and not r.usable_for_stage   # модель А его ещё обработает
        assert "ночь" in r.reject_reason


@pytest.mark.parametrize("when", [None, NOON_DEC])
def test_overcast_gray_day_is_not_night(when):
    """Баг Дениса: saturation < 12 считалось ночью, и пасмурные дни выпадали из анализа."""
    img = overcast(scene())
    mt = Q.measure(img)
    assert mt.saturation < 12                     # у Дениса это была бы «ночь»
    r = Q.assess(img, when)
    assert not r.is_night and r.usable_for_stage


def test_real_photos_are_clear_day_even_desaturated():
    for p in sorted(PHOTOS.glob("*.jpg")):
        img = cv2.imread(str(p))
        when = dt.datetime.strptime(p.stem[6:25], "%Y_%m_%d_%H_%M_%S").replace(tzinfo=dt.timezone.utc)
        for variant in (img, overcast(img)):
            r = Q.assess(variant, when)
            assert r.usable_for_stage and not r.is_night and r.weather is Weather.CLEAR, (p.name, r)


def test_ir_monochrome_is_night_without_clock_but_bw_camera_by_day_is_not():
    ir = cv2.cvtColor(cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert Q.assess(ir).is_night                      # ИК-режим: каналы совпадают идеально
    assert not Q.assess(ir, NOON_JUNE).is_night       # светлый монохром днём — ч/б камера


def test_clock_night_is_overridden_only_by_clearly_daylit_frame():
    assert Q.assess(night_scene(), NIGHT_DEC).is_night
    # по часам ночь, а кадр яркий и цветной — вероятнее, что врут часы или часовой пояс
    assert not Q.assess(scene(), NIGHT_DEC).is_night


def test_blurred_frame_is_rejected_as_defect():
    r = Q.assess(cv2.GaussianBlur(scene(), (0, 0), 6), NOON_JUNE)
    assert not r.quality_ok and not r.usable_for_stage
    assert "размыт" in r.reject_reason and not r.is_night


def test_synthetic_drops_on_lens_mean_rain():
    r = Q.assess(with_drops(scene()), NOON_JUNE)
    assert r.weather is Weather.RAIN and r.quality_ok and not r.usable_for_stage
    assert "капли" in r.reject_reason


def test_drops_on_real_photos_are_detected_and_clean_ones_are_not():
    hits = 0
    for p in sorted(PHOTOS.glob("*.jpg")):
        img = cv2.imread(str(p))
        assert Q.assess(img).weather is not Weather.RAIN, p.name
        hits += Q.assess(with_drops(img)).weather is Weather.RAIN
    assert hits >= 8          # на 320×256 капля может слиться с размытым кадром — это ограничение


def test_overexposed_and_no_contrast_are_defects():
    assert Q.assess(np.full((480, 640, 3), 252, np.uint8)).reject_reason == "засвет"
    flat = np.full((480, 640, 3), 120, np.uint8)
    r = Q.assess(flat, NOON_JUNE)
    assert not r.quality_ok and "контраст" in r.reject_reason


def test_snow_is_recognised_but_still_usable():
    img = scene()
    rng = np.random.default_rng(3)
    img[240:] = np.clip(rng.normal(225, 12, img[240:].shape), 0, 255).astype(np.uint8)
    img[300:330, 100:400] = 40                         # тёмный забор на снегу
    r = Q.assess(img, NOON_DEC)
    assert r.weather is Weather.SNOW and r.usable_for_stage   # снег уйдёт в фон маски
    assert Q.looks_snowy(img) and not Q.looks_snowy(scene())


def test_fog_is_recognised_and_dense_fog_excluded():
    base = scene()
    fog = cv2.addWeighted(base, 0.3, np.full_like(base, 215), 0.7, 0)
    r = Q.assess(fog, NOON_JUNE)
    assert r.weather is Weather.FOG and not r.usable_for_stage and "туман" in r.reject_reason
    light = cv2.addWeighted(base, 0.75, np.full_like(base, 215), 0.25, 0)
    assert Q.assess(light, NOON_JUNE).usable_for_stage


def test_sun_elevation_for_moscow():
    assert Q.sun_elevation_deg(NOON_JUNE) > 50
    assert Q.sun_elevation_deg(NIGHT_DEC) < -45
    assert Q.sun_elevation_deg(dt.datetime(2026, 12, 10, 16, 30, tzinfo=MSK)) < 0   # декабрьский вечер
    naive_utc = dt.datetime(2026, 6, 15, 9, 0)          # наивное время — UTC по контракту
    assert Q.sun_elevation_deg(naive_utc) == pytest.approx(Q.sun_elevation_deg(NOON_JUNE))


def test_resolution_does_not_change_verdict():
    img = scene()
    big = cv2.resize(img, (1920, 1440), interpolation=cv2.INTER_CUBIC)
    a, b = Q.assess(img, NOON_JUNE), Q.assess(big, NOON_JUNE)
    assert (a.quality_ok, a.is_night, a.weather) == (b.quality_ok, b.is_night, b.weather)


def test_empty_frame_is_defect_not_crash():
    r = Q.assess(np.zeros((0, 0, 3), np.uint8))
    assert not r.quality_ok and not r.usable_for_stage
