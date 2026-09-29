"""Качество кадра и помехи на камере → QualityReport (брак, ночь, капли, туман, солнце, снег…).

Зачем: модель Б отвечает на чек-лист по дневным чистым кадрам, а модель А работает
всегда (днём и ночью, в дождь и снег) — поэтому «брак» и «непригоден для этапа» —
разные вещи. `quality_ok=False` — кадр испорчен (почти чёрный, засвечен, без контраста,
сильно размыт, объектив залит или перекрыт). `usable_for_stage=False` — ещё и ночь,
капли на объективе, туман, солнце в объективе, сдвиг камеры. `flags` — все найденные
помехи кодами (`FLAGS`), `reject_reason` — по-русски, почему кадр не пошёл в модель Б.

Три уровня проверки, от дешёвого к точному:

1. **Один кадр** (всегда, ~10 мс на ширине 640): яркость, контраст, резкость, солнце
   над городом объекта по часам кадра, «тёмный канал», снег на земле, яркий диск солнца,
   круглые мягкие пятна-капли. Единственный уровень для «Проверить снимок» без нейросети.
2. **Норма камеры** (`core/stage/camera_norm.py`, конвейер): сравнение с собственными
   чистыми дневными кадрами той же камеры. Ловит то, что по одному кадру не отличить:
   подтёки воды на колпаке (окна фасада и белые бытовки есть и в норме — не капли),
   запотевание и пелену (текстура сцены упала относительно обычной), перекрытие части
   объектива, сдвиг камеры.
3. **Погода по SigLIP** (`core/stage/weather_clip.py`, если локальная модель Б
   доступна): zero-shot «туман / сильный дождь / снегопад / ночь» по смыслу кадра —
   для одиночных снимков и перед вызовом модели Б. Результат приходит сюда
   вероятностями (`weather_probs`), решение — здесь же, по порогам `QualityConfig`.

Числа проверки (docs/limitations.md): российские таймлапсы с камер московских строек
(5 объектов, ~990 кадров с настоящими метками времени), testset/conditions (Commons),
дневные фото техники (testset/equipment, московские фото) — для ложных отбраковок.

**Ночь** — баг Дениса: `saturation < 12 or mean < 55` отбрасывал пасмурный серый день
как ночь. Здесь по картинке ночь — темно (< 45), ИК-монохром (каналы совпадают до
уровня) или тёмное небо при многих точечных огнях (московская стройка ночью под
прожекторами светлая: средняя яркость 73–100); по времени съёмки — солнце ниже −6° над
городом часового пояса объекта (`config_for_timezone`, формулы NOAA); яркий цветной
кадр «ночью по часам» — вероятнее врут часы, но не оранжевый (натриевые прожекторы).

**Капли по одному кадру**: круглые «мягкие» пятна (низкая энергия лапласиана) с резким
кольцом вокруг; прямоугольные пятна (окна, панели, бытовки — площадь пятна почти равна
описанному прямоугольнику) каплями не считаются, нужно не меньше `rain_score` признаков.
Настоящие капли на колпаке уличной камеры — чаще размытые подтёки, их ловит норма камеры.

**Снег**: много яркого ненасыщенного на нижней половине кадра при сохранённом контрасте,
и не тёплого оттенка (песок карьера — жёлтый: R − B > `snow_max_warm`). Снег на площадке
кадр не портит — модель Б его видит, снег со временем уходит в фон маски. **Туман** по
одному кадру — высокий «тёмный канал» (He et al., 2009) при низком контрасте; густой
туман (контраст < 18) — не для модели Б. **Солнце в объективе** — компактное пересвеченное
пятно с ореолом; кадр исключается, если пересвечено ≥ 12 % или норма камеры видит, что
текстура сцены упала вдвое (контровой свет, блик).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from core.contracts import QualityReport, Weather

if TYPE_CHECKING:  # pragma: no cover
    from core.stage.camera_norm import CameraNorm

MOSCOW_LAT = 55.7558
MOSCOW_LON = 37.6173
_ZONE_COORD = re.compile(r"([+-])(\d{2})(\d{2})(\d{2})?([+-])(\d{3})(\d{2})(\d{2})?")

# Код помехи → (подпись по-русски, исключает ли кадр из модели Б).
FLAGS: dict[str, tuple[str, bool]] = {
    "night": ("Ночь", True),
    "ir": ("ИК-подсветка", True),
    "twilight": ("Сумерки", False),
    "drops": ("Капли на объективе", True),
    "rain": ("Сильный дождь", True),
    "fog": ("Туман", True),
    "haze": ("Дымка", False),
    "snow_cover": ("Снег на площадке", False),
    "snowfall": ("Снегопад", True),
    "occluded": ("Объектив перекрыт", True),
    "shifted": ("Камера сдвинута", True),
    "low_visibility": ("Плохая видимость", True),
    "blur": ("Размыто", True),
    "dark": ("Слишком темно", True),
    "overexposed": ("Засвечено", True),
    "low_contrast": ("Нет контраста", True),
}
BLOCKING = frozenset(k for k, (_, blocks) in FLAGS.items() if blocks)


@functools.lru_cache(maxsize=256)
def tz_coordinates(tz_name: str | None) -> tuple[float, float] | None:
    """Широта и долгота главного города часового пояса — из zone1970.tab базы tz.

    Высота солнца по умолчанию считается над Москвой; для объекта в другом поясе
    (архив Чикаго, Эдинбурга, Канберры) ночь по часам уезжала на 8–10 часов, и
    ночные кадры с фонарями шли в модель Б как дневные. Координат у объекта в
    карточке нет, а часовой пояс есть всегда — город пояса даёт солнце с
    точностью в пару градусов, этого хватает для «ночь / сумерки / день».
    None — пояс неизвестен или базы нет (тогда остаётся Москва)."""
    if not tz_name:
        return None
    for path in _zone_tables():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 3 and not line.startswith("#") and parts[2] == tz_name:
                m = _ZONE_COORD.fullmatch(parts[1])
                if not m:
                    return None
                la = int(m[2]) + int(m[3]) / 60 + int(m[4] or 0) / 3600
                lo = int(m[6]) + int(m[7]) / 60 + int(m[8] or 0) / 3600
                return (la if m[1] == "+" else -la, lo if m[5] == "+" else -lo)
    return None


def _zone_tables() -> list[Path]:
    out = []
    try:
        import tzdata  # пакет из requirements: есть и в slim-образе без системной базы
        out += [Path(tzdata.__file__).parent / "zoneinfo" / n for n in ("zone1970.tab", "zone.tab")]
    except ImportError:
        pass
    return out + [Path("/usr/share/zoneinfo") / n for n in ("zone1970.tab", "zone.tab")]


def config_for_timezone(tz_name: str | None, base: "QualityConfig | None" = None) -> "QualityConfig":
    """QualityConfig с солнцем над городом часового пояса объекта (см. tz_coordinates)."""
    cfg = base or QualityConfig()
    ll = tz_coordinates(tz_name)
    return dataclasses.replace(cfg, latitude=ll[0], longitude=ll[1]) if ll else cfg


@dataclass
class QualityConfig:
    work_width: int = 640
    border_level: int = 16                 # чёрные поля видео 4:3 в кадре 16:9 (ниже — поле) срезаются
    # брак
    dark_brightness: float = 18.0          # средняя яркость ниже — кадр почти чёрный
    overexposed_brightness: float = 243.0
    overexposed_clip_frac: float = 0.6     # доля пикселей ≥ 250 — засвет
    min_contrast: float = 8.0              # std яркости ниже — «нет контраста» (заслонён, запотел)
    min_sharpness: float = 15.0            # дисперсия лапласиана на ширине 640 ниже — сильная размытость
    # ночь
    night_brightness: float = 45.0
    twilight_brightness: float = 60.0      # в сумерках (по часам) порог мягче
    ir_chroma: float = 1.0                 # средняя межканальная разница ниже — ИК-монохром
    ir_saturation: float = 5.0
    mono_day_brightness: float = 70.0      # монохромный, но светлый днём — ч/б камера, не ночь
    clock_override_brightness: float = 95.0   # «по часам ночь», а кадр ярок и цветной — часы врут
    clock_override_saturation: float = 25.0
    clock_override_max_warm: float = 40.0     # … но не с оранжевым светом прожекторов (R − B выше)
    night_sky: float = 75.0                # без часов: верхняя полоса кадра темнее …
    night_lights: int = 8                  # … и точечных огней на тёмном не меньше — ночь под прожекторами
    night_image_max_brightness: float = 115.0
    sun_night_deg: float = -6.0
    sun_day_deg: float = 0.0
    twilight_deg: float = 3.0              # солнце ниже (но не ночь) — «сумерки», пометка без исключения
    latitude: float = MOSCOW_LAT
    longitude: float = MOSCOW_LON
    # капли на объективе (одиночный кадр)
    drop_softness: float = 0.3             # «мягко» — энергия лапласиана ниже доли от типичной по кадру
    drop_min_area_frac: float = 0.0008
    drop_max_area_frac: float = 0.06
    drop_min_circularity: float = 0.6
    drop_min_fill: float = 0.68            # площадь пятна / площадь описанного круга
    drop_max_rect_fill: float = 0.85       # площадь / описанный прямоугольник выше — окно, панель, не капля
    drop_ring_ratio: float = 2.5           # кольцо вокруг резче пятна во столько раз
    drop_highlight: float = 60.0           # блик внутри капли ярче кольца вокруг на столько уровней
    rain_score: float = 4.0
    low_contrast: float = 35.0
    # снег и туман
    snow_value: int = 170
    snow_saturation: int = 40
    snow_ground_frac: float = 0.35
    snow_min_contrast: float = 25.0
    snow_max_warm: float = 12.0            # R − B «снежных» пикселей выше — песок или тёплый свет, не снег
    fog_dark_channel: float = 110.0
    fog_max_contrast: float = 40.0
    fog_unusable_contrast: float = 18.0
    # норма камеры (core/stage/camera_norm.py)
    norm_min_sun: float = 3.0              # сравнивать с нормой, только когда солнце выше (или часов нет)
    norm_drops: float = 0.008              # доля кадра в провалах текстуры — капли / подтёки
    norm_occluded: float = 0.10            # один провал больше — объектив перекрыт
    norm_broken: float = 0.5               # … больше половины кадра — брак
    norm_low_gain: float = 0.5             # текстура сцены ниже доли от обычной — плохая видимость
    norm_blind_gain: float = 0.2           # … ниже этой — сцены не видно, брак
    norm_fog_veil: float = 12.0            # тёмный канал поднялся на столько уровней — туман / пелена
    norm_shift: float = 0.03               # сдвиг больше доли ширины — камера сдвинута
    norm_shift_response: float = 0.08
    # погода по SigLIP (core/stage/weather_clip.py): порог вероятности zero-shot класса
    clip_fog: float = 0.3
    clip_rain: float = 0.45                # «сильный дождь» + «капли на стекле»
    clip_snowfall: float = 0.4
    clip_night: float = 0.4                # только когда часов нет


@dataclass
class QualityMetrics:
    """Сырые числа — для отладки, журнала кадра и подбора порогов."""
    brightness: float
    contrast: float
    sharpness: float
    saturation: float
    chroma: float
    clip_frac: float
    dark_channel: float
    snow_ground_frac: float
    soft_drops: int
    highlighted_drops: int
    sun_elevation: float | None = None
    warm_cast: float = 0.0          # средняя разность R − B: оранжевый свет натриевых прожекторов
    sky: float = 255.0              # яркость верхней полосы кадра
    lights: int = 0                 # точечные огни на тёмном фоне
    snow_warm: float = 0.0          # R − B «снежных» пикселей
    norm: dict | None = None        # сравнение с нормой камеры (NormCheck.as_dict)
    clip: dict | None = None        # вероятности погоды по SigLIP
    extra: dict = field(default_factory=dict)


def sun_elevation_deg(when: dt.datetime, lat: float = MOSCOW_LAT, lon: float = MOSCOW_LON) -> float:
    """Высота солнца над горизонтом, градусы (упрощённые формулы NOAA, точность ~1°).

    Наивное время считается UTC — так договорено в контрактах.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    t = when.astimezone(dt.timezone.utc)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon          # истинное солнечное время, минуты
    ha = math.radians(tst / 4 - 180)
    phi = math.radians(lat)
    cos_zen = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))


def crop_borders(img: np.ndarray, level: int = 16) -> np.ndarray:
    """Срезать чёрные поля по краям (видео 4:3 в кадре 16:9, «почтовый ящик»): они занижали
    яркость и контраст и давали ложные «мягкие пятна». Поле — не меньше 2 % стороны."""
    if img.ndim != 3 or img.shape[0] < 16 or img.shape[1] < 16:
        return img
    m = img.max(axis=2)
    cols = np.flatnonzero(m.max(axis=0) > level)
    rows = np.flatnonzero(m.max(axis=1) > level)
    if len(cols) == 0 or len(rows) == 0:
        return img
    h, w = m.shape
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0 = x0 if x0 >= 0.02 * w else 0
    x1 = x1 if w - x1 >= 0.02 * w else w
    y0 = y0 if y0 >= 0.02 * h else 0
    y1 = y1 if h - y1 >= 0.02 * h else h
    if (x1 - x0) < 0.5 * w or (y1 - y0) < 0.5 * h:
        return img                     # «полей» больше половины — это не поля, а тёмная сцена
    return img[y0:y1, x0:x1]


def _prepare(image_bgr: np.ndarray, width: int, border_level: int = 16) -> np.ndarray:
    img = image_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = crop_borders(img, border_level)
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA if w > width else cv2.INTER_LINEAR)
    return img


def _drops(gray: np.ndarray, cfg: QualityConfig) -> tuple[int, int]:
    """(круглые мягкие пятна в резкой сцене, из них с ярким бликом внутри).

    Форма берётся по выпуклой оболочке пятна: блик внутри капли выедает в «мягкой»
    области дыру или полумесяц, а оболочка остаётся круглой. Прямоугольник (окно,
    стеклопакет, панель фасада, торец бытовки) почти целиком заполняет описанный
    прямоугольник — это не капля: ложные «капли» на фасадах выбрасывали из модели Б
    дневные кадры московских ЖК.
    """
    h, w = gray.shape
    area = float(h * w)
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    energy = cv2.GaussianBlur(lap, (0, 0), 3)
    ref = float(np.percentile(energy, 60))
    if ref <= 2.0:  # сцена без текстуры: каплю не отличить от гладкого участка — признак молчит
        return 0, 0
    soft = (energy < cfg.drop_softness * ref).astype(np.uint8)
    soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(soft, connectivity=8)
    soft_count = highlighted = 0
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # касается края — небо, земля, край кадра
        pad = max(3, int(0.35 * max(bw, bh) / 2))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        ha, hp = cv2.contourArea(hull), cv2.arcLength(hull, True)
        if hp <= 0 or not (cfg.drop_min_area_frac * area <= ha <= cfg.drop_max_area_frac * area):
            continue
        (_, _), r = cv2.minEnclosingCircle(hull)
        circularity = 4 * math.pi * ha / hp ** 2
        fill = ha / (math.pi * r * r) if r > 0 else 0.0
        if circularity < cfg.drop_min_circularity or fill < cfg.drop_min_fill or a < 0.5 * ha:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
        if rw * rh > 0 and a / (rw * rh) > cfg.drop_max_rect_fill:
            continue  # прямоугольник: окно, панель, бытовка
        inside = np.zeros_like(comp)
        cv2.drawContours(inside, [hull], -1, 1, thickness=-1)
        ring = cv2.dilate(inside, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool) & ~inside.astype(bool)
        inside = inside.astype(bool)
        e = energy[y0:y1, x0:x1]
        # медиана внутри: блик даёт резкое пятнышко, но капля в целом мягкая
        if not ring.any() or e[ring].mean() < cfg.drop_ring_ratio * max(float(np.median(e[inside])), 1e-3):
            continue
        soft_count += 1
        g = gray[y0:y1, x0:x1]
        if float(g[inside].max()) >= float(np.median(g[ring])) + cfg.drop_highlight:
            highlighted += 1
    return soft_count, highlighted


def _lights(gray: np.ndarray, hsv: np.ndarray) -> int:
    """Точечные огни на тёмном фоне: прожекторы, фонари, окна ночью."""
    h, w = gray.shape
    bright = (hsv[..., 2] >= 245).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return 0
    bg = cv2.blur(gray, (31, 31))
    count = 0
    for i in range(1, n):
        if 2 <= stats[i, 4] <= 0.002 * h * w:
            cx, cy = int(cents[i][0]), int(cents[i][1])
            if bg[min(h - 1, cy), min(w - 1, cx)] < 90:
                count += 1
    return count


def looks_snowy(image_bgr: np.ndarray, config: QualityConfig | None = None) -> bool:
    """Дешёвая проверка «земля под снегом» — её зовёт и динамическая маска."""
    cfg = config or QualityConfig()
    small = _prepare(image_bgr, 160, cfg.border_level)
    frac, warm = _snow(small, cfg)
    return frac >= cfg.snow_ground_frac and warm <= cfg.snow_max_warm and float(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std()) >= cfg.snow_min_contrast


def _snow(img: np.ndarray, cfg: QualityConfig) -> tuple[float, float]:
    """(доля «снежных» пикселей нижней половины, их средний R − B). Песок карьера светлый и
    малонасыщенный, как снег, но тёплый: R − B у него 20–40, у снега — около нуля или меньше."""
    low = img[img.shape[0] // 2:]
    hsv = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] >= cfg.snow_value) & (hsv[..., 1] <= cfg.snow_saturation)
    if not m.any():
        return 0.0, 0.0
    warm = float((low[..., 2][m].astype(np.int16) - low[..., 0][m].astype(np.int16)).mean())
    return float(m.mean()), warm


def _snow_frac(img: np.ndarray, cfg: QualityConfig) -> float:
    return _snow(img, cfg)[0]


def measure(image_bgr: np.ndarray, captured_at: dt.datetime | None = None,
            config: QualityConfig | None = None) -> QualityMetrics:
    cfg = config or QualityConfig()
    return _measure(_prepare(image_bgr, cfg.work_width, cfg.border_level), captured_at, cfg)


def _measure(img: np.ndarray, captured_at: dt.datetime | None, cfg: QualityConfig) -> QualityMetrics:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    chroma = float((np.abs(r - g) + np.abs(g - b)).mean() / 2)
    h = gray.shape[0]
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    soft, highlighted = _drops(gray, cfg)
    snow_frac, snow_warm = _snow(img, cfg)
    return QualityMetrics(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        saturation=float(hsv[..., 1].mean()),
        chroma=chroma,
        clip_frac=float((gray >= 250).mean()),
        dark_channel=float(dark[h // 3:].mean()),
        snow_ground_frac=snow_frac,
        soft_drops=soft,
        highlighted_drops=highlighted,
        sun_elevation=(sun_elevation_deg(captured_at, cfg.latitude, cfg.longitude)
                       if captured_at is not None else None),
        warm_cast=float((r - b).mean()),
        sky=float(gray[: max(1, h // 7)].mean()),
        lights=_lights(gray, hsv),
        snow_warm=snow_warm,
    )


def _is_mono(mt: QualityMetrics, cfg: QualityConfig) -> bool:
    return mt.chroma < cfg.ir_chroma and mt.saturation < cfg.ir_saturation


def _is_night(mt: QualityMetrics, cfg: QualityConfig) -> bool:
    mono = _is_mono(mt, cfg)
    dark = mt.brightness < cfg.night_brightness
    sun = mt.sun_elevation
    if sun is None:
        # Без часов: ночная московская стройка под прожекторами светла (средняя яркость
        # 73–100), но небо тёмное и в кадре десятки точечных огней.
        lit_night = (mt.sky < cfg.night_sky and mt.lights >= cfg.night_lights
                     and mt.brightness < cfg.night_image_max_brightness)
        by_clip = bool(mt.clip) and mt.clip.get("night", 0.0) >= cfg.clip_night
        return dark or mono or lit_night or by_clip
    if sun < cfg.sun_night_deg:
        # Яркий цветной кадр ночью «по часам» — вероятнее, врут часы. Но не оранжевый:
        # стройка под натриевыми прожекторами так же ярка (архив Чикаго: яркость ~100,
        # R − B 50–83 ночью против −2…17 днём), и без этой оговорки ночь шла в модель Б.
        clearly_day = (mt.brightness >= cfg.clock_override_brightness
                       and mt.saturation >= cfg.clock_override_saturation
                       and mt.warm_cast < cfg.clock_override_max_warm)
        return not clearly_day
    if sun >= cfg.sun_day_deg:
        return dark or (mono and mt.brightness < cfg.mono_day_brightness)
    return mt.brightness < cfg.twilight_brightness or (mono and mt.brightness < cfg.mono_day_brightness)


def assess(image_bgr: np.ndarray, captured_at: dt.datetime | None = None,
           config: QualityConfig | None = None, norm: "CameraNorm | None" = None,
           weather_probs: dict[str, float] | None = None) -> QualityReport:
    """Кадр → QualityReport. reject_reason заполняется и для «непригоден для этапа»
    (ночь, капли, туман…) при quality_ok=True — чтобы UI объяснял, почему кадр не пошёл в модель Б.

    norm — норма камеры (конвейер): кадр сравнивается с ней и, если он чистый дневной,
    учится в неё (норма меняется — сохраните её). weather_probs — вероятности погоды по
    SigLIP (weather_clip), если их посчитали."""
    cfg = config or QualityConfig()
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return QualityReport(quality_ok=False, is_night=False, weather=Weather.UNKNOWN,
                             reject_reason="пустой кадр", usable_for_stage=False, flags=["dark"])
    img = _prepare(image_bgr, cfg.work_width, cfg.border_level)
    mt = _measure(img, captured_at, cfg)
    mt.clip = dict(weather_probs) if weather_probs else None
    check = None
    daylight = not _is_night(mt, cfg) and (mt.sun_elevation is None or mt.sun_elevation >= cfg.norm_min_sun)
    if norm is not None and daylight:
        check = norm.compare(img, when=captured_at)
        mt.norm = check.as_dict() if check is not None else None
    report = report_from_metrics(mt, cfg)
    if norm is not None:
        clean = (daylight and report.quality_ok and report.usable_for_stage
                 and not (set(report.flags) - {"snow_cover"}))
        mismatch = check is not None and bool({"shifted", "occluded", "drops"} & set(report.flags))
        event = norm.observe(img, clean, captured_at, mismatch=mismatch, check=check)
        if event:
            report.details["norm_event"] = event
        report.details["norm_frames"] = len(norm.frames)
    return report


def add_weather(image_bgr: np.ndarray, captured_at: dt.datetime | None, config: QualityConfig | None,
                report: QualityReport, weather_probs: dict[str, float]) -> QualityReport:
    """Пересчитать вердикт с погодой по SigLIP, когда кадр уже оценён (и, может быть, выучен
    нормой камеры): те же метрики и сравнение с нормой из report.details, плюс вероятности.
    Норму второй раз не трогает."""
    cfg = config or QualityConfig()
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return report
    mt = _measure(_prepare(image_bgr, cfg.work_width, cfg.border_level), captured_at, cfg)
    mt.norm = report.details.get("norm")
    mt.clip = dict(weather_probs)
    out = report_from_metrics(mt, cfg)
    out.details.update({k: v for k, v in report.details.items() if k.startswith("norm_")})
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %" if x >= 0.095 else f"{x * 100:.1f} %".replace(".", ",")


def report_from_metrics(mt: QualityMetrics, cfg: QualityConfig | None = None) -> QualityReport:
    cfg = cfg or QualityConfig()
    night = _is_night(mt, cfg)
    flags: list[str] = []
    reasons: dict[str, str] = {}
    ok = True
    reason = ""
    nm = mt.norm if not night else None
    clip = mt.clip or {}

    if mt.brightness < cfg.dark_brightness:
        ok, reason = False, "слишком темно — кадр почти чёрный"
        flags.append("dark")
    elif mt.brightness > cfg.overexposed_brightness or mt.clip_frac > cfg.overexposed_clip_frac:
        ok, reason = False, "засвет"
        flags.append("overexposed")
    elif mt.contrast < cfg.min_contrast:
        ok, reason = False, "нет контраста (объектив заслонён, запотел или густой туман)"
        flags.append("low_contrast")
    elif nm and nm["gain"] < cfg.norm_blind_gain:
        ok = False
        reason = (f"объектив запотел, залит водой или густой туман — сцены почти не видно "
                  f"(текстура {_pct(nm['gain'])} от обычной для этой камеры)")
        flags.append("low_visibility")
    elif nm and nm["loss_max"] >= cfg.norm_broken:
        ok = False
        reason = f"объектив перекрыт: не видно {_pct(nm['loss_max'])} кадра"
        flags.append("occluded")

    if night:
        flags.append("night")
        if _is_mono(mt, cfg):
            flags.append("ir")
    elif mt.sun_elevation is not None and mt.sun_elevation < cfg.twilight_deg:
        flags.append("twilight")

    if ok and not night:
        # --- норма камеры: помехи относительно обычного вида этой камеры
        if nm:
            if nm["loss_max"] >= cfg.norm_occluded:
                flags.append("occluded")
                reasons["occluded"] = (f"объектив частично перекрыт (грязь, предмет, сетка): не видно "
                                       f"{_pct(nm['loss_max'])} кадра — кадр исключён из определения этапа")
            elif nm["loss_frac"] >= cfg.norm_drops:
                flags.append("drops")
                reasons["drops"] = (f"капли или подтёки на объективе: размыто {_pct(nm['loss_frac'])} кадра "
                                    f"— кадр исключён из определения этапа")
            if nm["shift"] >= cfg.norm_shift and nm["shift_response"] >= cfg.norm_shift_response:
                flags.append("shifted")
                reasons["shifted"] = (f"камера сдвинута на {_pct(nm['shift'])} ширины кадра относительно "
                                      f"обычного вида — проверьте крепление; кадр исключён из определения этапа")
            if nm["gain"] < cfg.norm_low_gain:
                # Туман и вода на колпаке поднимают тёмные места (пелена); солнце в объективе и
                # контровой свет — нет (у камеры «Cityzen» в 17:59 тёмный канал даже падает).
                if nm["veil"] >= cfg.norm_fog_veil:
                    flags.append("fog")
                    reasons["fog"] = (f"туман или дымка: текстура сцены {_pct(nm['gain'])} от обычной "
                                      f"для этой камеры — кадр исключён из определения этапа")
                else:
                    flags.append("low_visibility")
                    reasons["low_visibility"] = (
                        f"засветка или плохая видимость (солнце в объективе, контровой свет, пелена): "
                        f"текстура сцены {_pct(nm['gain'])} от обычной для этой камеры — кадр исключён "
                        f"из определения этапа")
        # --- один кадр
        rain_score = float(mt.soft_drops)
        if mt.highlighted_drops >= 1:
            rain_score += 1
        if mt.soft_drops >= 1 and mt.contrast < cfg.low_contrast:
            rain_score += 1
        if nm is None and rain_score >= cfg.rain_score and "drops" not in flags:
            flags.append("drops")
            reasons["drops"] = "капли на объективе — кадр исключён из определения этапа"
        fog_single = mt.dark_channel >= cfg.fog_dark_channel and mt.contrast < cfg.fog_max_contrast
        if clip.get("fog", 0.0) >= cfg.clip_fog and "fog" not in flags:
            flags.append("fog")
            reasons["fog"] = f"туман (по смыслу кадра, SigLIP {clip['fog']:.2f}) — кадр исключён из определения этапа"
        elif fog_single and "fog" not in flags:
            if mt.contrast < cfg.fog_unusable_contrast:
                flags.append("fog")
                reasons["fog"] = "густой туман — кадр исключён из определения этапа"
            else:
                flags.append("haze")
        rain_p = clip.get("rain", 0.0) + clip.get("drops", 0.0)
        if rain_p >= cfg.clip_rain:
            flags.append("rain")
            reasons["rain"] = f"сильный дождь (SigLIP {rain_p:.2f}) — кадр исключён из определения этапа"
        if clip.get("snowfall", 0.0) >= cfg.clip_snowfall:
            flags.append("snowfall")
            reasons["snowfall"] = (f"снегопад (SigLIP {clip['snowfall']:.2f}) — кадр исключён из "
                                   f"определения этапа")
        if (mt.snow_ground_frac >= cfg.snow_ground_frac and mt.contrast >= cfg.snow_min_contrast
                and mt.snow_warm <= cfg.snow_max_warm):
            flags.append("snow_cover")
        if "fog" not in flags and "haze" not in flags and mt.sharpness < cfg.min_sharpness:
            ok, reason = False, "сильная размытость (не в фокусе или смазано)"
            flags.append("blur")

    if "drops" in flags or "rain" in flags:
        weather = Weather.RAIN
    elif "fog" in flags or "haze" in flags:
        weather = Weather.FOG
    elif "snowfall" in flags or "snow_cover" in flags:
        weather = Weather.SNOW
    else:
        weather = Weather.CLEAR

    blocking = [f for f in flags if f in BLOCKING and f not in ("night", "ir")]
    usable = ok and not night and not blocking
    if ok and not night and blocking:
        reason = next((reasons[f] for f in blocking if f in reasons), "кадр исключён из определения этапа")
    elif ok and night:
        reason = "ночь — кадр не идёт в определение этапа (модель А его обрабатывает)"
    if not ok and mt.brightness < cfg.dark_brightness:
        night = night or mt.sun_elevation is None or mt.sun_elevation < cfg.sun_day_deg
        if night and "night" not in flags:
            flags.append("night")

    details: dict[str, Any] = {
        "brightness": round(mt.brightness, 1), "contrast": round(mt.contrast, 1),
        "sharpness": round(mt.sharpness, 1), "saturation": round(mt.saturation, 1),
        "dark_channel": round(mt.dark_channel, 1), "sky": round(mt.sky, 1), "lights": mt.lights,
        "soft_drops": mt.soft_drops, "clip_frac": round(mt.clip_frac, 4),
        "snow": round(mt.snow_ground_frac, 3),
    }
    if mt.sun_elevation is not None:
        details["sun_elevation"] = round(mt.sun_elevation, 1)
    if nm:
        details["norm"] = nm
    if mt.clip:
        details["weather_clip"] = {k: round(float(v), 3) for k, v in mt.clip.items()}
    return QualityReport(quality_ok=ok, is_night=night, weather=weather, reject_reason=reason,
                         blur=round(mt.sharpness, 2), brightness=round(mt.brightness, 2),
                         usable_for_stage=usable, flags=list(dict.fromkeys(flags)), details=details)
