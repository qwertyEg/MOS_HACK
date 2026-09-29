"""Динамическая маска: сосед скрыт, стройка открывается, маска не растёт, окно по суткам.

Сцена — как у Дениса в tests/test_mask.py, но кадрами в течение суток:

    NEIGHBOUR — статичный соседний дом у левого края: маска обязана уцелеть,
                что бы ни делали погода, освещение и проезжающий грузовик;
    OURS      — наше здание растёт справа на этаж каждые трое суток: маска
                обязана отступить.

Автоматическая начальная маска — своя сцена (`far_scene`): небо, дальний город,
соседняя башня выше горизонта с краном на крыше, котлован с техникой и наш кран.
"""
import datetime as dt
import io
import json

import cv2
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


# --------------------------------------------------------------------------
# автоматическая начальная маска: «дальний план» (небо, горизонт, соседние башни)
# --------------------------------------------------------------------------

HORIZON = 120          # нижняя кромка неба
SKYLINE = 150          # дальний город и основания соседних башен
TOWER = (300, 380, 30)  # соседняя башня: x0, x1, верх (выше горизонта)
OUR_MAST = (96, 108)    # мачта нашего башенного крана — стоит в котловане
LOCAL = dt.timezone(dt.timedelta(hours=3))


def _texture(h: int, w: int, seed: int, base: float, amp: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.uniform(-amp, amp, (max(1, h // 8), max(1, w // 8)))
    return base + cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


FAR_CITY = _texture(SKYLINE - HORIZON, W, 11, 120)
NEIGHBOUR_FACADE = _texture(SKYLINE - TOWER[2], TOWER[1] - TOWER[0], 12, 150, 50)


def far_scene(hour: int, day: int = 0, sky_shift: float = 0.0, busy: bool = True, sky: bool = True) -> np.ndarray:
    """Стройка «как у ЖК»: сверху небо, у горизонта — дальний город и соседняя башня выше
    горизонта; ниже — котлован, где всё меняется от часа к часу (техника, грунт)."""
    rng = np.random.default_rng(day * 100 + hour)
    img = np.zeros((H, W, 3), np.float32)
    light = 25 * np.sin(hour / 3.0) + sky_shift
    if sky:
        img[:HORIZON] = (225 + light, 195 + light, 160 + light)              # голубое небо (BGR)
        img[:HORIZON] += rng.normal(0, 2, (HORIZON, W, 1))
    else:
        img[:HORIZON] = _texture(HORIZON, W, 13, 110)[..., None]            # сверху тоже грунт
    img[HORIZON:SKYLINE] = (FAR_CITY + 0.5 * light)[..., None]
    x0, x1, top = TOWER
    img[top:SKYLINE, x0:x1] = (NEIGHBOUR_FACADE + 0.5 * light)[..., None]
    ground = _texture(H - SKYLINE, W, 1000 + day * 24 + hour if busy else 7, 90, 45)
    img[SKYLINE:] = ground[..., None]
    img[20:330, OUR_MAST[0]:OUR_MAST[1]] = 60                                 # мачта нашего крана
    if busy:
        x = 150 + 25 * (hour % 8)
        img[260:300, x:x + 50] = (20, 140, 230)                               # экскаватор ездит
    return np.clip(img, 0, 255).astype(np.uint8)


def far_boxes(hour: int, busy: bool = True) -> list[tuple]:
    boxes = [(OUR_MAST[0] - 4, 20, 60, 312, "tower_crane", 0.8),              # опора — в котловане
             (TOWER[0] + 20, 0, 30, TOWER[2] + 10, "tower_crane", 0.7)]       # кран на соседней башне
    if busy:
        boxes.append((150 + 25 * (hour % 8), 260, 50, 40, "excavator", 0.9))
    return boxes


def at(day: int, hour: int) -> dt.datetime:
    return dt.datetime(2026, 8, 1 + day, hour, 5, tzinfo=LOCAL)


def feed(m: DynamicMask, hours, day: int = 0, **kw):
    upd = None
    for hr in hours:
        upd = m.update(far_scene(hr, day, **kw), at(day, hr), weather=Weather.CLEAR,
                       boxes=far_boxes(hr, kw.get("busy", True)))
        if upd.initialized_now:
            return upd
    return upd


def test_auto_init_hides_sky_and_neighbour_keeps_pit_and_our_crane():
    """Маска строится в первые часы съёмки (6 срезов по часу), закрывает небо, дальний город
    и соседнюю башню вместе с краном на её крыше; котлован и наш кран видны."""
    m = DynamicMask.new((H, W))
    feed(m, range(7, 12))
    assert not m.initialized and m.bootstrap()["slices"] == 5
    upd = feed(m, range(12, 15))
    assert upd is not None and upd.initialized_now and m.source == "auto"
    hidden = ~m.visible()
    assert region(hidden, 150, 280, 0, HORIZON - 16) > 0.95                   # небо
    assert region(hidden, 0, 60, HORIZON, SKYLINE - 8) > 0.8                  # дальний город у горизонта
    x0, x1, top = TOWER
    assert region(hidden, x0 + 8, x1 - 8, top, SKYLINE - 16) > 0.9           # соседняя башня выше горизонта
    assert region(hidden, 0, W, SKYLINE + 40, H) < 0.01                       # котлован виден
    # наш кран: ниже горизонта (минус запас) рамка его отстаивает, в небе — фон, как всё небо
    assert region(hidden, OUR_MAST[0], OUR_MAST[1], HORIZON, 330) < 0.2
    assert m.init_info["horizon"] == pytest.approx(HORIZON / H, abs=0.05)
    assert 0.2 < m.masked_ratio <= MaskConfig().init_max_ratio


def test_auto_init_needs_hours_of_observation():
    """Шесть кадров за полчаса — это один срез: маски нет, пока съёмка не покроет ≥ 4 ч."""
    m = DynamicMask.new((H, W))
    for k in range(12):
        m.update(far_scene(9, 0), at(0, 9) + dt.timedelta(minutes=2 * k), weather=Weather.CLEAR)
    assert not m.initialized and m.bootstrap()["slices"] == 1


def test_top_down_camera_masks_only_the_top_edge():
    """Нет неба (камера смотрит на площадку сверху, как в Эдинбурге): статичный грунт
    в тихие первые часы — не фон; фоном может быть только верхняя кромка кадра."""
    m = DynamicMask.new((H, W))
    feed(m, range(7, 15), sky=False, busy=False)
    assert m.initialized
    hidden = ~m.visible()
    band = int(round(MaskConfig().top_band * H / 16)) * 16 + 16
    assert region(hidden, 0, W, band, H) == 0.0
    assert m.init_info["horizon"] is None


def test_idle_site_after_holidays_never_hides_the_pit():
    """Стресс «праздники»: площадка стоит, всё статично. Маска — только дальний план,
    котлован и нижняя часть кадра открыты (у первой редакции здесь закрывалось 70 % кадра)."""
    m = DynamicMask.new((H, W))
    feed(m, range(7, 15), busy=False)
    assert m.initialized
    hidden = ~m.visible()
    assert region(hidden, 0, W, int(MaskConfig().far_max * H), H) == 0.0
    assert region(hidden, 0, W, SKYLINE + 32, H) < 0.01
    assert m.masked_ratio <= MaskConfig().far_max


def test_letterbox_is_not_background():
    m = DynamicMask.new((H, W))
    for hr in range(7, 15):
        img = far_scene(hr)
        img[:, :48] = 0
        img[:, -48:] = 0
        m.update(img, at(0, hr), weather=Weather.CLEAR, boxes=far_boxes(hr))
    assert m.initialized
    hidden = ~m.visible()
    assert region(hidden, 0, 32, 0, H) == 0.0 and region(hidden, W - 32, W, 0, H) == 0.0


def test_site_equipment_reopens_background_under_its_box():
    """Кран, поставленный на площадке после построения маски, стрелой заходит в дальний
    план — со второго кадра маска под его рамкой (ниже горизонта) стирается. Кран с опорой
    на фоне (соседняя площадка) маску не трогает. Маска при этом не растёт."""
    m = DynamicMask.new((H, W))
    feed(m, range(7, 15))
    assert m.initialized
    before = m.background.copy()
    new_crane = (400, 100, 40, 120, "mobile_crane", 0.8)          # опора y=220 — на площадке
    foreign = (20, 60, 40, 40, "tower_crane", 0.8)                # опора y=100 — в небе/на фоне
    for k in range(2):
        m.update(far_scene(15 + k), at(0, 15 + k), weather=Weather.CLEAR, boxes=[new_crane, foreign])
    hidden = ~m.visible()
    top = int(np.ceil(m.protect_top)) * 16
    assert region(hidden, 400, 440, top, SKYLINE) < 0.2           # под рамкой нового крана — открыто
    assert region(hidden, 20, 60, 60, 100) > 0.9                  # чужой кран — фон
    assert not (m.background & ~before).any()
    assert [h["event"] for h in m.history] == ["auto", "equipment"]


def test_history_gives_each_frame_the_mask_of_its_time():
    """Кадр, разобранный задним числом, получает маску своего времени: до построения — нет
    маски; после сжатия старый кадр видит старую (большую) маску."""
    m = DynamicMask.new((H, W))
    feed(m, range(7, 15))
    first = m.history[0]
    assert first["event"] == "auto" and first["from"] <= at(0, 8).astimezone(dt.timezone.utc).isoformat()
    assert m.visible_at(at(0, 6)) is None                          # до съёмки маски не было
    assert (m.background_at(at(0, 10)) == m.background).all()      # кадры накопления — с маской
    old = m.background.copy()
    m.update(far_scene(16), at(0, 16), boxes=[(400, 100, 40, 120, "mobile_crane", 0.8)])
    m.update(far_scene(17), at(0, 17), boxes=[(400, 100, 40, 120, "mobile_crane", 0.8)])
    assert m.masked_ratio < float(old.mean())
    assert (m.background_at(at(0, 10)) == old).all()               # старый кадр — старая маска
    assert (m.background_at(at(0, 18)) == m.background).all()
    assert m.visible_at(at(0, 10), (H // 2, W // 2)).shape == (H // 2, W // 2)


def test_manual_mask_resets_history_and_applies_to_all_frames():
    m = DynamicMask.new((H, W))
    feed(m, range(7, 15))
    m.set_background(operator_mask(), lock=True)
    assert m.source == "manual" and [h["event"] for h in m.history] == ["manual"]
    assert (m.background_at(at(0, 6)) == m.background).all()      # и на кадрах до автоматики
    assert m.locked_ratio == pytest.approx(m.masked_ratio)


def test_bootstrap_survives_dumps_loads():
    """Состояние накопления (срезы, рамки техники) хранится: рестарт сервиса посреди первых
    часов не обнуляет их и даёт ту же маску."""
    a = DynamicMask.new((H, W))
    feed(a, range(7, 11))
    b = DynamicMask.loads(a.dumps())
    assert len(b.boot) == len(a.boot) and len(b.slice_buf) == len(a.slice_buf) and b.boot_boxes == a.boot_boxes
    feed(a, range(11, 15))
    feed(b, range(11, 15))
    assert a.initialized and b.initialized and (a.background == b.background).all()
    c = DynamicMask.loads(b.dumps())
    assert c.history == b.history and (c.background_at(at(0, 9)) == b.background_at(at(0, 9))).all()


def test_version1_blob_still_loads():
    """Маски, сохранённые до второй редакции (без истории и накопления), читаются."""
    m = DynamicMask.new((H, W))
    m.set_background(operator_mask())
    with np.load(io.BytesIO(m.dumps())) as z:
        arrays = {k: z[k] for k in ("background", "locked", "evidence", "hot_count", "ring", "buffer")}
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    for k in ("history", "boot_times", "boot_boxes", "slice_key", "slice_t0", "init_info", "protect_top", "last_at"):
        meta.pop(k, None)
    meta["version"] = 1
    for k in ("init_days", "init_static_threshold", "protect_center", "protect_axes", "require_border_contact"):
        meta["config"][k] = 1                                     # старые поля конфига — игнорируются
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
    old = DynamicMask.loads(buf.getvalue())
    assert old.initialized and (old.background == m.background).all()
    assert len(old.history) == 1 and (old.background_at(at(0, 9)) == m.background).all()


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
    assert (m.apply(img, "gray")[:GROUND_Y - 16] == 127).all()
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
