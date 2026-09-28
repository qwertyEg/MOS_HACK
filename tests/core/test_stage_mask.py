"""Динамическая маска: сосед скрыт, стройка открывается, маска не растёт, окно по суткам.

Сцена — как у Дениса в tests/test_mask.py, но кадрами в течение суток:

    NEIGHBOUR — статичный соседний дом у левого края: маска обязана уцелеть,
                что бы ни делали погода, освещение и проезжающий грузовик;
    OURS      — наше здание растёт справа на этаж каждые трое суток: маска
                обязана отступить.
"""
import datetime as dt

import numpy as np
import pytest

from core.contracts import Weather
from core.stage.mask import DynamicMask, MaskConfig, apply_background, masked_for_model

H, W = 384, 480
GROUND_Y = 300
NEIGHBOUR = (20, 150, 120, GROUND_Y)          # x0, x1, y0, y1
OURS_X = (300, 440)
T0 = dt.datetime(2026, 4, 1, 6, 0, tzinfo=dt.timezone.utc)   # 09:00 по Москве


def frame(day: int, k: int, weather: float = 0.0, truck: bool = False, snow: bool = False) -> np.ndarray:
    rng = np.random.default_rng(day * 100 + k)
    img = np.zeros((H, W), np.float32)
    img[:GROUND_Y] = 195 + rng.normal(0, 6, (GROUND_Y, W)) + weather
    img[GROUND_Y:] = (225 if snow else 95) + rng.normal(0, 4, (H - GROUND_Y, W)) + weather
    x0, x1, y0, y1 = NEIGHBOUR
    img[y0:y1, x0:x1] = 135 + weather
    if snow:
        img[y0:y0 + 40, x0:x1] = 235                 # снег на крыше соседа
    floors = day // 3
    if floors:
        top = max(0, GROUND_Y - floors * 18)
        img[top:GROUND_Y, OURS_X[0]:OURS_X[1]] = 60 + weather
    if truck:                                        # грузовик перед соседом в одном кадре
        img[220:290, 40:130] = 20
    g = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def when(day: int, k: int) -> dt.datetime:
    return T0 + dt.timedelta(days=day, hours=3 * k)


def operator_mask() -> np.ndarray:
    """Оператор закрасил всё выше земли: небо и соседа (стройки там ещё нет)."""
    bmp = np.zeros((H, W), np.uint8)
    bmp[:GROUND_Y] = 255
    return bmp


def region(mask_hw: np.ndarray, x0, x1, y0, y1) -> float:
    return float(mask_hw[y0:y1, x0:x1].mean())


def run_days(m: DynamicMask, days, per_day=4, snow_from=None, weather_flag=True):
    history = []
    for d in days:
        for k in range(per_day):
            snowy = snow_from is not None and d >= snow_from
            wx = Weather.SNOW if snowy else Weather.CLEAR
            m.update(frame(d, k, 30 * np.sin(d / 7.0), truck=(k == 1 and d % 2 == 0), snow=snowy), when(d, k),
                     weather=wx if weather_flag else Weather.CLEAR)
        history.append(m.background.copy())
    return history


@pytest.fixture(scope="module")
def grown():
    m = DynamicMask.new((H, W))
    m.set_background(operator_mask())
    history = run_days(m, range(61))
    return m, history


def test_mask_never_grows(grown):
    _, history = grown
    for before, after in zip(history, history[1:]):
        assert not (after & ~before).any()


def test_static_neighbour_stays_hidden_despite_weather_and_trucks(grown):
    m, _ = grown
    x0, x1, y0, y1 = NEIGHBOUR
    assert region(~m.visible(), x0, x1, y0, y1) > 0.9


def test_growing_building_is_reopened(grown):
    m, _ = grown
    top = max(0, GROUND_Y - (50 // 3) * 18)          # высота здания ~10 суток назад — окно её уже увидело
    assert region(m.visible(), OURS_X[0] + 16, OURS_X[1] - 16, top + 16, GROUND_Y - 16) > 0.8
    assert m.retained < 1.0 and m.windows > 0


def test_auto_init_hides_static_edges_and_keeps_site_visible():
    """Никто не рисовал маску: после init_days суток статичное у края кадра — фон;
    центр-низ (площадка) и статичный островок посреди изменчивой области — нет."""
    m = DynamicMask.new((H, W), MaskConfig(init_days=5))
    rng = np.random.default_rng(7)
    for d in range(6):
        strip = int(rng.integers(40, 200))                        # деревья/дорога справа: меняются день ото дня
        for k in range(4):
            img = frame(0, k, weather=20 * np.sin(d))             # сосед, небо, земля неподвижны
            img[100:300, 360:480] = strip
            img[170:230, 400:440] = 90                            # статичный островок, края кадра не касается
            img[GROUND_Y - 120 + d * 10:GROUND_Y + 40, 180:330] = 40 + 30 * d   # стройка идёт
            y = int(rng.integers(200, 330))
            img[y:y + 30, 200 + 20 * k:260 + 20 * k] = 10         # техника ездит
            upd = m.update(img, when(d, k), weather=Weather.CLEAR)
        if d == 4:
            assert not m.initialized
    assert m.initialized and m.source == "auto" and upd.skipped == ""
    hidden = ~m.visible()
    _, _, y0, y1 = NEIGHBOUR
    assert region(hidden, 20, 90, y0, y1) > 0.9                   # сосед у левого края скрыт…
    assert region(hidden, 110, 150, 200, y1) < 0.1                # …кроме края, зашедшего в эллипс площадки
    assert region(hidden, 190, 320, 250, 370) < 0.05              # площадка видна
    assert region(hidden, 404, 436, 174, 226) < 0.2               # островок не касается края — не фон
    assert region(hidden, 360, 480, 100, 300) < 0.2               # изменчивое — не фон
    assert 0.03 < m.masked_ratio <= MaskConfig().init_max_ratio


def test_auto_init_never_covers_the_site_even_if_everything_is_static():
    """Выходные/простой: за init_days ничего не менялось. Центр-низ всё равно открыт,
    и доля маски не выше предела."""
    m = DynamicMask.new((H, W), MaskConfig(init_days=3, init_max_ratio=0.5))
    for d in range(4):
        for k in range(3):
            m.update(frame(0, k), when(d, k), weather=Weather.CLEAR)
    assert m.initialized
    assert region(~m.visible(), 200, 280, 250, 360) == 0.0
    assert m.masked_ratio <= 0.5 + 1e-6


def test_window_is_counted_in_days_not_frames():
    """Баг D9: окно из 10 «дней» по кадрам покрывало 3 часа съёмки."""
    m = DynamicMask.new((H, W))
    m.set_background(operator_mask())
    for k in range(30):                                   # 30 кадров за одни сутки
        m.update(frame(0, k % 4), T0 + dt.timedelta(minutes=20 * k), weather=Weather.CLEAR)
    assert len(m.ring) == 0 and m.windows == 0
    upd = m.update(frame(1, 0), when(1, 0), weather=Weather.CLEAR)
    assert upd.day_closed and len(m.ring) == 1
    late = m.update(frame(0, 0), when(0, 0), weather=Weather.CLEAR)
    assert late.skipped and m.late_frames == 1            # назад окно не пересчитывается


def test_dumps_loads_roundtrip_continues_identically():
    a = DynamicMask.new((H, W))
    a.set_background(operator_mask(), lock=False)
    run_days(a, range(14))
    a.update(frame(14, 0), when(14, 0), weather=Weather.CLEAR)        # незакрытый буфер суток тоже сохраняется
    blob = a.dumps()
    assert isinstance(blob, bytes) and len(blob) < 2_000_000
    b = DynamicMask.loads(blob)
    assert (b.background == a.background).all() and (b.hot_count == a.hot_count).all()
    assert b.ring_days == a.ring_days and b.buffer_day == a.buffer_day and len(b.buffer) == len(a.buffer)
    assert b.masked_ratio == pytest.approx(a.masked_ratio) and b.source == "manual"
    run_days(a, range(15, 30))
    run_days(b, range(15, 30))
    assert (a.background == b.background).all() and (a.evidence == b.evidence).all()


def test_resolution_change_does_not_crash():
    """Баг D8: один кадр 16:9 среди 4:3 навсегда останавливал разбор потока."""
    m = DynamicMask.new((H, W))
    m.set_background(operator_mask())
    run_days(m, range(3))
    big = np.repeat(np.repeat(frame(3, 0), 2, axis=0), 2, axis=1)          # то же 4:3, вдвое крупнее
    assert m.update(big, when(3, 0), weather=Weather.CLEAR).skipped == ""
    wide = np.zeros((360, 640, 3), np.uint8)                                  # 16:9
    upd = m.update(wide, when(3, 1), weather=Weather.CLEAR)
    assert "соотношение" in upd.skipped and m.initialized
    run_days(m, range(4, 6))                                                  # поток продолжается
    assert len(m.ring) >= 4 and m.mismatch_streak == 0
    for i in range(m.config.reinit_after_mismatch):                          # камеру заменили
        upd = m.update(wide, when(6, 0) + dt.timedelta(minutes=i), weather=Weather.CLEAR)
    assert m.frame_shape == (360, 640) and not m.initialized and "заново" in upd.skipped
    assert m.visible().shape == (360, 640)


def test_snow_is_absorbed_instead_of_erasing_the_neighbour():
    """Снег на крыше соседа — не стройка. Пока окно смешанное, стирание заморожено;
    когда всё окно снежное, снег становится базой. Контроль — без флага снега крышу стирает."""
    x0, x1, y0, _ = NEIGHBOUR
    frozen = DynamicMask.new((H, W))
    frozen.set_background(operator_mask())
    run_days(frozen, range(12), snow_from=None)
    run_days(frozen, range(12, 40), snow_from=12)
    control = DynamicMask.new((H, W))
    control.set_background(operator_mask())
    run_days(control, range(12))
    run_days(control, range(12, 40), snow_from=12, weather_flag=False)
    assert region(~frozen.visible(), x0, x1, y0, y0 + 40) > 0.9
    assert region(~control.visible(), x0, x1, y0, y0 + 40) < 0.5


def test_locked_area_is_never_erased():
    m = DynamicMask.new((H, W))
    bmp = operator_mask()
    m.set_background(bmp, lock=True)
    run_days(m, range(40))
    assert m.masked_ratio == pytest.approx(bmp.astype(bool).mean(), abs=0.02)


def test_apply_modes():
    m = DynamicMask.new((H, W))
    img = frame(0, 0)
    assert (m.apply(img) == img).all()                    # маски ещё нет — кадр как есть
    m.set_background(operator_mask())
    dark = m.apply(img, "darken")
    assert dark[50, 50, 0] == int(img[50, 50, 0] * 0.3) and (dark[350] == img[350]).all()
    assert (m.apply(img, "black")[:GROUND_Y - 16] == 0).all()
    crop = m.apply(img, "crop")
    assert crop.shape[0] < H and crop.shape[1] == W
    assert m.apply(img, "blur").shape == img.shape
    with pytest.raises(ValueError):
        apply_background(img, ~m.visible(), "sepia")
    big = np.zeros((2 * H, 2 * W, 3), np.uint8) + 100     # маска масштабируется под любой кадр
    assert m.apply(big).shape == big.shape
    assert m.visible((2 * H, 2 * W)).shape == (2 * H, 2 * W)


def test_masked_for_model_accepts_mask_object_or_visible_array():
    m = DynamicMask.new((H, W))
    img = frame(0, 0)
    out, applied = masked_for_model(img, {"mask": m})
    assert not applied and out is img                     # неинициализированная маска не трогает кадр
    m.set_background(operator_mask())
    out, applied = masked_for_model(img, {"mask": m, "mask_mode": "black"})
    assert applied and out[10, 10, 0] == 0
    out, applied = masked_for_model(img, {"mask": m.visible()})
    assert applied and out[10, 10, 0] < img[10, 10, 0]
    assert masked_for_model(img, None) == (img, False)
    assert masked_for_model(img, {"mask": m, "mask_mode": "none"})[1] is False
