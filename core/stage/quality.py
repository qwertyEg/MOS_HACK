"""Качество кадра: брак, ночь, дождь (капли на объективе), снег, туман → QualityReport.

Зачем: модель Б отвечает на чек-лист по дневным чистым кадрам, а модель А
работает всегда (днём и ночью, в дождь и снег) — поэтому брак и «непригоден
для этапа» — разные вещи. `quality_ok=False` — кадр испорчен (почти чёрный,
засвечен, без контраста, сильно размыт). `usable_for_stage=False` — ещё и ночь,
капли на объективе, густой туман. Снег модель Б не останавливает: снег со
временем становится статичным и уходит в фон динамической маски.

Все признаки считаются на уменьшенной копии (ширина 640), чтобы пороги не
зависели от разрешения камеры: дисперсия лапласиана растёт с разрешением.

**Ночь** — баг Дениса: `saturation < 12 or mean < 55` отбрасывал пасмурный
серый день как ночь, а в пасмурную погоду кадры нужны не меньше. Здесь:

- по картинке ночь — это темно (средняя яркость < 45) или ИК-режим камеры:
  кадр почти идеально серый (межканальная разница < 1 уровня) — у пасмурного
  дня цвет приглушён, но каналы не совпадают;
- по времени съёмки (если есть) считается высота солнца над Москвой или над
  городом часового пояса объекта (`config_for_timezone`, формулы NOAA):
  солнце ниже −6° (гражданские сумерки) — ночь, если кадр не
  ярок и не цветен (тогда вероятнее, что врут часы или часовой пояс) или
  залит оранжевым светом натриевых прожекторов (R − B ≥ 40);
  солнце выше горизонта — день, если кадр не тёмный; монохромный, но светлый
  кадр днём — ч/б камера, а не ночь; в сумерках решает яркость.

**Капли на объективе** — эвристика из трёх признаков, пороги в QualityConfig:

1. пятнистая локальная размытость: круглые «мягкие» пятна (низкая энергия
   лапласиана) посреди резкой сцены — дефокусированная капля; от неба и
   гладкой стены их отличают форма (близки к кругу, не касаются края кадра)
   и резкое кольцо вокруг;
2. яркие круглые блики — капля преломляет небо и фонари;
3. общий низкий контраст усиливает первые два признака.

Ограничения (честно, для docs/limitations.md): штрихи падающего дождя без капель
на стекле не ловятся; одна крупная капля, закрывающая пол-кадра, выглядит как
размытие (уйдёт в брак по резкости или пройдёт); сцена без текстуры (сплошной
бетон, туман) не даёт отличить каплю от гладкого участка — там признак молчит.
Пороги подобраны на синтетике и здравом смысле, а не на размеченном датасете.

**Снег**: много яркого малонасыщенного на нижней половине кадра (земля) при
сохранённом контрасте. **Туман**: высокий «тёмный канал» (He et al., 2009 — в
чистом уличном кадре минимум по каналам в окрестности почти ноль, дымка его
поднимает) при низком контрасте; густой туман (контраст < 18) — не для модели Б.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.contracts import QualityReport, Weather

MOSCOW_LAT = 55.7558
MOSCOW_LON = 37.6173
_ZONE_COORD = re.compile(r"([+-])(\d{2})(\d{2})(\d{2})?([+-])(\d{3})(\d{2})(\d{2})?")


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
    sun_night_deg: float = -6.0
    sun_day_deg: float = 0.0
    latitude: float = MOSCOW_LAT
    longitude: float = MOSCOW_LON
    # капли на объективе
    drop_softness: float = 0.3             # «мягко» — энергия лапласиана ниже доли от типичной по кадру
    drop_min_area_frac: float = 0.0008
    drop_max_area_frac: float = 0.06
    drop_min_circularity: float = 0.6
    drop_min_fill: float = 0.68            # площадь пятна / площадь описанного круга
    drop_ring_ratio: float = 2.5           # кольцо вокруг резче пятна во столько раз
    drop_highlight: float = 60.0           # блик внутри капли ярче кольца вокруг на столько уровней
    rain_score: float = 3.0
    low_contrast: float = 35.0
    # снег и туман
    snow_value: int = 170
    snow_saturation: int = 40
    snow_ground_frac: float = 0.35
    snow_min_contrast: float = 25.0
    fog_dark_channel: float = 110.0
    fog_max_contrast: float = 40.0
    fog_unusable_contrast: float = 18.0


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


def _prepare(image_bgr: np.ndarray, width: int) -> np.ndarray:
    img = image_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA if w > width else cv2.INTER_LINEAR)
    return img


def _drops(gray: np.ndarray, cfg: QualityConfig) -> tuple[int, int]:
    """(круглые мягкие пятна в резкой сцене, из них с ярким бликом внутри).

    Форма берётся по выпуклой оболочке пятна: блик внутри капли выедает в «мягкой»
    области дыру или полумесяц, а оболочка остаётся круглой.
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


def looks_snowy(image_bgr: np.ndarray, config: QualityConfig | None = None) -> bool:
    """Дешёвая проверка «земля под снегом» — её зовёт и динамическая маска."""
    cfg = config or QualityConfig()
    small = _prepare(image_bgr, 160)
    return _snow_frac(small, cfg) >= cfg.snow_ground_frac and float(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std()) >= cfg.snow_min_contrast


def _snow_frac(img: np.ndarray, cfg: QualityConfig) -> float:
    hsv = cv2.cvtColor(img[img.shape[0] // 2:], cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] >= cfg.snow_value) & (hsv[..., 1] <= cfg.snow_saturation)).mean())


def measure(image_bgr: np.ndarray, captured_at: dt.datetime | None = None,
            config: QualityConfig | None = None) -> QualityMetrics:
    cfg = config or QualityConfig()
    img = _prepare(image_bgr, cfg.work_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    chroma = float((np.abs(r - g) + np.abs(g - b)).mean() / 2)
    h = gray.shape[0]
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    soft, highlighted = _drops(gray, cfg)
    return QualityMetrics(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        saturation=float(hsv[..., 1].mean()),
        chroma=chroma,
        clip_frac=float((gray >= 250).mean()),
        dark_channel=float(dark[h // 3:].mean()),
        snow_ground_frac=_snow_frac(img, cfg),
        soft_drops=soft,
        highlighted_drops=highlighted,
        sun_elevation=(sun_elevation_deg(captured_at, cfg.latitude, cfg.longitude)
                       if captured_at is not None else None),
        warm_cast=float((r - b).mean()),
    )


def _is_night(mt: QualityMetrics, cfg: QualityConfig) -> bool:
    mono = mt.chroma < cfg.ir_chroma and mt.saturation < cfg.ir_saturation
    dark = mt.brightness < cfg.night_brightness
    sun = mt.sun_elevation
    if sun is None:
        return dark or mono
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
           config: QualityConfig | None = None) -> QualityReport:
    """Кадр → QualityReport. reject_reason заполняется и для «непригоден для этапа»
    (ночь, капли, туман) при quality_ok=True — чтобы UI объяснял, почему кадр не пошёл в модель Б."""
    cfg = config or QualityConfig()
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return QualityReport(quality_ok=False, is_night=False, weather=Weather.UNKNOWN,
                             reject_reason="пустой кадр", usable_for_stage=False)
    return report_from_metrics(measure(image_bgr, captured_at, cfg), cfg)


def report_from_metrics(mt: QualityMetrics, cfg: QualityConfig | None = None) -> QualityReport:
    cfg = cfg or QualityConfig()
    night = _is_night(mt, cfg)
    weather = Weather.CLEAR
    reason = ""
    ok = True

    if mt.brightness < cfg.dark_brightness:
        ok, reason = False, "слишком темно — кадр почти чёрный"
    elif mt.brightness > cfg.overexposed_brightness or mt.clip_frac > cfg.overexposed_clip_frac:
        ok, reason = False, "засвет"
    elif mt.contrast < cfg.min_contrast:
        ok, reason = False, "нет контраста (объектив заслонён, запотел или густой туман)"

    if ok and not night:
        fog = mt.dark_channel >= cfg.fog_dark_channel and mt.contrast < cfg.fog_max_contrast
        snow = (mt.snow_ground_frac >= cfg.snow_ground_frac and mt.contrast >= cfg.snow_min_contrast)
        # Мягкие круглые пятна — главный признак; блики и низкий контраст только поддерживают его.
        # Одни блики дождём не считаются: фонари и солнечные зайчики на технике дают то же самое.
        rain_score = float(mt.soft_drops)
        if mt.highlighted_drops >= 1:
            rain_score += 1
        if mt.soft_drops >= 1 and mt.contrast < cfg.low_contrast:
            rain_score += 1
        if rain_score >= cfg.rain_score:
            weather = Weather.RAIN
        elif fog:
            weather = Weather.FOG
        elif snow:
            weather = Weather.SNOW
        if weather is not Weather.FOG and mt.sharpness < cfg.min_sharpness:
            ok, reason = False, "сильная размытость (не в фокусе или смазано)"

    usable = ok and not night
    if usable and weather is Weather.RAIN:
        usable, reason = False, "капли на объективе — кадр исключён из определения этапа"
    elif usable and weather is Weather.FOG and mt.contrast < cfg.fog_unusable_contrast:
        usable, reason = False, "густой туман — кадр исключён из определения этапа"
    elif ok and night:
        reason = "ночь — кадр не идёт в определение этапа (модель А его обрабатывает)"
    if not ok and mt.brightness < cfg.dark_brightness:
        night = night or mt.sun_elevation is None or mt.sun_elevation < cfg.sun_day_deg

    return QualityReport(quality_ok=ok, is_night=night, weather=weather, reject_reason=reason,
                         blur=round(mt.sharpness, 2), brightness=round(mt.brightness, 2),
                         usable_for_stage=usable)
