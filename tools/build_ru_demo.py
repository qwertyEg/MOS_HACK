#!/usr/bin/env python3
"""Российские демо-объекты из таймлапсов настоящих камер московских строек.

Главный источник — помесячные таймлапсы, которые оператор камер videocam.online публикует на
страницах онлайн-трансляций ЖК (кнопка «TimeLapse» у плеера; сами трансляции снимает
tools/capture_camera.py). Камера в таймлапсе снимает раз в ~10 минут, а в углу кадра — экранная
подпись с датой и временем. Её мы и читаем, поэтому метки времени в именах файлов НАСТОЯЩИЕ:

1. первый проход по видео — вырезаем подпись каждого кадра (у плотных роликов — каждого stride-го);
2. распознаём цифры: моноширинный шрифт, известная сетка знакомест, шаблоны цифр — из
   нескольких кадров, прочитанных глазами (`refs`); сравнение — нормированная корреляция;
3. чистим хронологию: остаётся самая длинная согласованная цепочка меток, выпавшие (блик, капля,
   «3» вместо «8») восстанавливаются интерполяцией по номеру кадра и помечены в frames_source.csv;
4. отбираем кадры по сетке (плотно — несколько рабочих дней для моточасов, редко — весь
   месяц для хронологии этапов), подряд идущие одинаковые кадры (камера «зависла») выкидываем;
5. второй проход — сохраняем `<объект>_<камера>_YYYY_MM_DD_HH_MM_SS.jpg` и пишем site.json
   (формат app/services/demo.py) с планом в кодах перечня работ организаторов.

Ещё два вида источников: ролик без подписи (таймлапс с YouTube под CC BY — метки синтетические,
равномерно по датам из названия) и папка снимков с подписью (датасет arh-df — снимки не
упорядочены, ненадёжные метки выбрасываются, а не интерполируются).

    python tools/build_ru_demo.py --raw /root/mos_hack/data/raw/ru --ext /root/mos_hack/data/ext \\
        --out /root/mos_hack/data/demo/sites
    python tools/build_ru_demo.py --raw ... --dump-osd ru_cityzen_pit 16 --dump-to /tmp/osd.png
                                                   # лист подписей кадров — прочитать refs глазами

Видео скачиваются отдельно (публичные HLS-плейлисты videocam.online, см. SOURCE_URL; YouTube):
    yt-dlp --hls-prefer-native --referer https://videocam.online/ -o vc_Tushino2.mp4 \\
        https://rtsp2.videocam.online/vod/_definst_/Tushino2.mp4/playlist.m3u8
    yt-dlp -f "bv*[height<=1080]" -o "yt_%(id)s.%(ext)s" https://www.youtube.com/watch?v=Q3yjW_csgrc
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SOURCE_URL = "https://rtsp2.videocam.online/vod/_definst_/{stream}.mp4/playlist.m3u8"
PAGE_URL = "https://videocam.online/wowza2.php?id={stream}.stream"
LICENSE = ("не указана: публичная онлайн-трансляция и таймлапс, опубликованные оператором камер "
           "на странице ЖК; используется только как демо-данные хакатона, не распространять")

# Позиции знаков в подписи. ymd: «2026-08-01 21:29:58», dmy: «03/05/2026 04:19:57».
FORMATS = {
    "ymd": {"year": (0, 4), "month": (5, 7), "day": (8, 10)},
    "dmy": {"day": (0, 2), "month": (3, 5), "year": (6, 10)},
}
TIME_POS = {"hour": (11, 13), "minute": (14, 16), "second": (17, 19)}
DIGIT_POS = {"ymd": [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18],
             "dmy": [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]}


@dataclasses.dataclass
class OsdSpec:
    """Где на кадре подпись: левый край первого знака даты, шаг знакоместа, строки, формат."""
    x0: float
    pitch: float
    y0: int
    y1: int
    fmt: str = "ymd"
    refs: dict[int, str] = dataclasses.field(default_factory=dict)   # номер кадра → текст подписи

    def cell(self, strip: np.ndarray, k: int) -> np.ndarray:
        """Знакоместо k из полосы подписи (полоса — строки y0:y1, столбцы от 0), нормированное."""
        import cv2

        a = int(round(self.x0 + k * self.pitch))
        b = int(round(self.x0 + (k + 1) * self.pitch))
        c = strip[:, a:b].astype(np.float32)
        c = cv2.resize(c, (int(round(self.pitch)), self.y1 - self.y0), interpolation=cv2.INTER_AREA)
        v = c - c.mean()
        n = float(np.linalg.norm(v))
        return (v / n).ravel() if n > 1e-6 else v.ravel()

    def strip(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Полоса подписи: минимум по каналам — белый текст яркий, цветной фон гаснет."""
        right = int(round(self.x0 + 19 * self.pitch)) + 2
        return frame_bgr[self.y0:self.y1, :right].min(axis=2)


class OsdReader:
    """Распознавание цифр подписи по шаблонам из кадров с известным текстом (1-NN по корреляции)."""

    def __init__(self, spec: OsdSpec):
        self.spec = spec
        self.samples: list[tuple[str, np.ndarray]] = []

    def fit(self, strips: dict[int, np.ndarray]) -> None:
        for idx, text in self.spec.refs.items():
            if idx not in strips:
                raise ValueError(f"кадра {idx} нет в видео — refs не от этого ролика?")
            for k in DIGIT_POS[self.spec.fmt]:
                self.samples.append((text[k], self.spec.cell(strips[idx], k)))
        missing = set("0123456789") - {d for d, _ in self.samples}
        if missing:
            raise ValueError(f"в refs нет цифр {sorted(missing)} — добавьте кадры с ними")
        self._labels = [d for d, _ in self.samples]
        self._mat = np.stack([v for _, v in self.samples])

    def read(self, strip: np.ndarray) -> tuple[str, float, float]:
        """→ (текст подписи, средняя корреляция, худший отрыв лучшей цифры от другой цифры)."""
        chars = list("0000-00-00 00:00:00" if self.spec.fmt == "ymd" else "00/00/0000 00:00:00")
        scores, margins = [], []
        for k in DIGIT_POS[self.spec.fmt]:
            sims = self._mat @ self.spec.cell(strip, k)
            best: dict[str, float] = {}
            for d, s in zip(self._labels, sims):
                best[d] = max(best.get(d, -1.0), float(s))
            ranked = sorted(best.items(), key=lambda kv: -kv[1])
            chars[k] = ranked[0][0]
            scores.append(ranked[0][1])
            margins.append(ranked[0][1] - ranked[1][1])
        return "".join(chars), float(np.mean(scores)), float(min(margins))


def parse_osd(text: str, fmt: str) -> dt.datetime | None:
    pos = {**FORMATS[fmt], **TIME_POS}
    try:
        parts = {k: int(text[a:b]) for k, (a, b) in pos.items()}
        return dt.datetime(parts["year"], parts["month"], parts["day"], parts["hour"], parts["minute"],
                           parts["second"])
    except ValueError:
        return None


def _fits(secs: list[float], a: int, b: int, rate: float) -> bool:
    """Метки кадров a < b совместимы: время не идёт назад и не скачет быстрее rate секунд на кадр."""
    d = secs[b] - secs[a]
    return 0 <= d <= (b - a) * rate


def clean_timeline(stamps: list[dt.datetime | None], max_rate: dt.timedelta = dt.timedelta(hours=3),
                   window: int = 400) -> tuple[list[dt.datetime], list[bool]]:
    """Убрать ошибки распознавания: оставить самую длинную согласованную цепочку меток.

    Цепочка — неубывающие метки, между соседями не больше `max_rate` на кадр (камера снимает
    раз в ~10 минут: «+5 дней через кадр» — это «3» вместо «8», а не перерыв). Жадный проход
    «не назад относительно последней» не годится: одна ошибка вперёд (23 → 28 августа) отбрасывает
    все честные метки после неё. Отвергнутые метки, которые всё же ложатся между соседями
    цепочки, возвращаются; остальные — интерполяция по номеру кадра между надёжными соседями.
    Возвращает метки всех кадров и флаги «интерполировано».
    """
    n = len(stamps)
    rate = max_rate.total_seconds()
    secs = [s.timestamp() if s else 0.0 for s in stamps]
    cand = [i for i in range(n) if stamps[i] is not None]
    length: dict[int, int] = {}
    prev: dict[int, int | None] = {}
    for pos, i in enumerate(cand):
        length[i], prev[i] = 1, None
        for j in cand[max(0, pos - window):pos]:
            if length[j] + 1 > length[i] and _fits(secs, j, i, rate):
                length[i], prev[i] = length[j] + 1, j
    ok = [False] * n
    end = max(cand, key=lambda k: length[k]) if cand else None
    while end is not None:
        ok[end] = True
        end = prev[end]
    good = [i for i in range(n) if ok[i]]
    for i in cand:
        if ok[i]:
            continue
        k = int(np.searchsorted(good, i))
        left = good[k - 1] if k > 0 else None
        right = good[k] if k < len(good) else None
        if (left is None or _fits(secs, left, i, rate)) and (right is None or _fits(secs, i, right, rate)):
            ok[i] = True
    good = [i for i in range(n) if ok[i]]
    if len(good) < 2:
        raise ValueError("подпись не читается: меньше двух надёжных меток")
    out, interp = [], []
    gi = 0
    for i in range(n):
        if ok[i]:
            out.append(stamps[i])
            interp.append(False)
            continue
        while gi + 1 < len(good) and good[gi + 1] < i:
            gi += 1
        a = good[gi] if good[gi] < i else None
        b = next((g for g in good[gi:] if g > i), None)
        if a is None:
            a, b = good[0], good[1]
        elif b is None:
            a, b = good[-2], good[-1]
        t = secs[a] + (secs[b] - secs[a]) * (i - a) / (b - a)
        v = dt.datetime.fromtimestamp(t).replace(microsecond=0)
        v = (v + dt.timedelta(seconds=30)).replace(second=stamps[a].second)
        out.append(v)
        interp.append(True)
    return out, interp


@dataclasses.dataclass
class Rule:
    """Сетка отбора: с `start` по `end` (даты включительно) раз в `step_min`, только часы [h0, h1)."""
    start: str
    end: str
    step_min: int
    h0: int = 0
    h1: int = 24


def select(stamps: list[dt.datetime], rules: list[Rule], signatures: list[np.ndarray] | None = None,
           frozen_diff: float = 0.5) -> list[int]:
    """Номера кадров, ближайших к узлам сеток (не дальше половины шага); без «зависших» повторов."""
    secs = np.array([s.timestamp() for s in stamps])
    chosen: set[int] = set()
    for r in rules:
        t = dt.datetime.fromisoformat(r.start)
        end = dt.datetime.fromisoformat(r.end) + dt.timedelta(days=1)
        while t < end:
            if r.h0 <= t.hour < r.h1:
                i = int(np.argmin(np.abs(secs - t.timestamp())))
                if abs(secs[i] - t.timestamp()) <= r.step_min * 30:
                    chosen.add(i)
            t += dt.timedelta(minutes=r.step_min)
    out: list[int] = []
    for i in sorted(chosen):
        if out and stamps[i] == stamps[out[-1]]:
            continue
        if signatures is not None and out and float(np.abs(signatures[i] - signatures[out[-1]]).mean()) < frozen_diff:
            continue            # картинка не изменилась — камера зависла, в таймлапсе повтор кадра
        out.append(i)
    return out


def signature(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def first_pass(video: Path, spec: OsdSpec | None, stride: int = 1
               ) -> tuple[list[int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Проход по видео: номера кадров (каждый stride-й), полосы подписи и «подписи» картинки.

    Номера — исходные номера кадров в ролике, поэтому refs и повторный проход от stride не зависят.
    Пропущенные кадры только grab()-аются: плотный таймлапс (кадр в полминуты) не держим в памяти.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    idx, strips, sigs = [], {}, {}
    i = 0
    while True:
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        idx.append(i)
        if spec is not None:
            strips[i] = spec.strip(fr)
        sigs[i] = signature(fr)
        i += 1
    cap.release()
    if not idx:
        raise ValueError(f"{video}: кадры не читаются")
    return idx, strips, sigs


def read_stamps(osd: OsdSpec, idx: list[int], strips: dict[int, np.ndarray]
                ) -> tuple[list[dt.datetime | None], list[str], list[float], list[float]]:
    reader = OsdReader(osd)
    reader.fit(strips)
    stamps, texts, scores, margins = [], [], [], []
    for i in idx:
        text, score, margin = reader.read(strips[i])
        texts.append(text)
        scores.append(score)
        margins.append(margin)
        # порог: ниже — цифру легко спутать (3/8 мелким шрифтом на светлом фоне), такая метка
        # не участвует в хронологии и будет восстановлена по соседям
        stamps.append(parse_osd(text, osd.fmt) if score >= 0.55 and margin >= 0.03 else None)
    return stamps, texts, scores, margins


# ---------------------------------------------------------------------------
# Объекты. Подписи (x0, pitch, y0, y1) сняты по профилю ярких столбцов кадра,
# refs прочитаны глазами на листах --dump-osd (номера — исходные номера кадров ролика).

VIDEOCAM_SOURCE = {"title": "Онлайн-трансляция и помесячный таймлапс камеры стройки",
                   "operator": "videocam.online (видеонаблюдение за стройками Москвы, камеры застройщика)",
                   "catalog": "https://hodstroitelstva.ru/spb/index.php/construction-photos/"
                              "268-new-buildings-moscow-with-webcams"}
VIDEOCAM_NOTES = ["Российская стройка, настоящая камера, настоящее московское время кадров.",
                  "План правдоподобный, в кодах «Сводного перечня строительных работ» организаторов "
                  "(work_codes); это не график застройщика.",
                  "Детектор техники на этих кадрах не обучался."]


def _plan(stage_id: int, name: str, start: str, end: str, codes: list[str], equipment: dict) -> dict:
    return {"stage_id": stage_id, "name": name, "planned_start": start, "planned_end": end,
            "work_codes": codes, "equipment": equipment}


SITES: dict[str, dict] = {}


def site_spec(key: str, **kw) -> None:
    SITES[key] = kw


site_spec(
    "ru_cityzen_pit",
    name="ЖК «Cityzen», 8-я очередь — котлован (Москва, Тушино)",
    address="Москва, СЗАО, Покровское-Стрешнево, Волоколамское ш. (ЖК «Cityzen», MR Group)",
    object_type="Жильё",
    description=("Настоящая камера московской стройки: площадка 8-й очереди ЖК «Cityzen» в августе 2026 г. "
                 "Нулевой цикл — буровые установки и гусеничные краны на сваях и ограждении котлована, "
                 "экскаваторы и самосвалы на разработке грунта; на заднем плане — монолитные башни "
                 "соседних очередей того же ЖК с башенными кранами (для анализа площадки 8-й очереди это "
                 "«чужие» здания). Кадры круглосуточно: ночь под прожекторами, дождь, закатный засвет."),
    cams={"cam_8och": {"stream": "Tushino2", "name": "Камера 8-й очереди — общий вид площадки",
                       "osd": OsdSpec(x0=325.5, pitch=18.0, y0=1047, y1=1080, fmt="ymd", refs={
                           493: "2026-08-05 06:39:58", 986: "2026-08-09 12:49:58", 1233: "2026-08-11 15:39:58",
                           1480: "2026-08-13 19:39:58", 2219: "2026-08-20 07:29:58", 2466: "2026-08-22 17:59:58",
                           2713: "2026-08-24 14:49:58", 2959: "2026-08-26 11:29:58"})}},
    rules=[Rule("2026-08-18", "2026-08-20", 20), Rule("2026-08-01", "2026-08-31", 120, 6, 22)],
    plan=[
        _plan(2, "Ограждение котлована, шпунт, сваи", "2026-07-15", "2026-08-20",
              ["12.3.6."], {"drilling_rig": 2, "crawler_crane": 1, "concrete_mixer": 2, "excavator": 1}),
        _plan(3, "Земляные работы, котлован", "2026-08-10", "2026-09-30",
              ["12.3.1.", "12.3.7."], {"excavator": 3, "dump_truck": 6, "bulldozer": 1}),
        _plan(4, "Монолит подземной части", "2026-09-20", "2026-12-20",
              ["12.3.4.", "12.3.9.", "12.3.10."],
              {"tower_crane": 2, "concrete_pump": 1, "concrete_mixer": 4, "truck": 1}),
    ],
    fleet={"excavator": 3, "dump_truck": 6, "drilling_rig": 2, "crawler_crane": 2, "mobile_crane": 1,
           "concrete_mixer": 2, "bulldozer": 1},
    demonstrates=["этапы 2–3 на настоящей московской камере: сваи/ограждение котлована и разработка грунта",
                  "работающая техника → этап (буровые, гусеничные краны, экскаваторы, самосвалы)",
                  "задний план: монолитные башни соседних очередей — не должны «поднимать» этап площадки",
                  "ночь под прожекторами, дождь, засвет от солнца — отбраковка кадров для модели Б",
                  "моточасы: 3 рабочих дня (18–20.08, вт–чт) с шагом 20 мин круглосуточно, "
                  "остальной месяц — раз в 2 ч днём"],
    plan_note=("План правдоподобный, не от застройщика: сваи и ограждение котлована по плану должны "
               "закончиться 20.08, котлован — с 10.08. Буровые на кадрах работают до конца месяца."),
    notes=["Буровые установки (буронабивные сваи) детектор не знает — ждём pile_driver/mobile_crane "
           "(замена буровой по equipment_rules.json); гусеничные краны — mobile_crane/tower_crane."],
)

site_spec(
    "ru_cityzen_frame",
    name="ЖК «Cityzen», 1-я очередь — монолитный каркас (Москва, Тушино)",
    address="Москва, СЗАО, Волоколамское ш. (ЖК «Cityzen», MR Group)",
    object_type="Жильё",
    description=("Настоящая камера: крупный план монолитного каркаса 1-й очереди ЖК «Cityzen», 1–5 мая 2026 г. "
                 "Опалубка стен и перекрытия, армирование, башенный кран, бетононасос; рабочие в касках, "
                 "работы идут и ночью под прожектором. Повторы одного кадра (камера «зависала») выброшены."),
    cams={"cam_1och": {"stream": "Tushino", "name": "Камера 1-й очереди — каркас крупным планом",
                       "osd": OsdSpec(x0=522.0, pitch=20.0, y0=1042, y1=1080, fmt="dmy", refs={
                           0: "01/05/2026 00:39:57", 103: "01/05/2026 22:59:57", 206: "02/05/2026 18:49:57",
                           240: "03/05/2026 00:49:57", 343: "03/05/2026 20:59:57", 377: "04/05/2026 05:09:57",
                           480: "04/05/2026 23:39:57"})}},
    rules=[Rule("2026-05-01", "2026-05-05", 40)],
    plan=[
        _plan(5, "Монолит надземной части", "2026-03-01", "2026-09-30",
              ["12.4.4.", "12.4.9.", "12.4.10.", "12.4.28."],
              {"tower_crane": 1, "concrete_pump": 1, "concrete_mixer": 3, "truck": 1}),
    ],
    fleet={"tower_crane": 2, "concrete_pump": 1, "concrete_mixer": 3, "truck": 1},
    demonstrates=["этап 5 (монолит надземной части) на российском ЖК", "башенный кран, бетононасос, опалубка",
                  "майские праздники 1–5 мая: работы идут и ночью под прожектором"],
    plan_note="Этап 5 по плану; ожидаемый вердикт — «в срок».",
)

site_spec(
    "ru_paveletskaya",
    name="ЖК «Павелецкая Сити», 4-я очередь — фасад башни (Москва)",
    address="Москва, ЮАО, Дубининская ул. (ЖК «Павелецкая Сити»)",
    object_type="Жильё",
    description=("Настоящая камера: башни 4-й очереди «Павелецкой Сити» в августе 2026 г. — монолит закончен, "
                 "идут навесной фасад и остекление снизу вверх, у башни — мачта подъёмника и кран. "
                 "Кадры 4 раза в день за весь месяц: утро, день, вечер, ночная подсветка."),
    cams={"cam_tower": {"stream": "Dubininsky5", "name": "Камера на соседнем доме — вид на башни",
                        "crop": (240, 0, 1680, 1080),
                        "osd": OsdSpec(x0=241.3, pitch=12.6, y0=1058, y1=1080, fmt="ymd", refs={
                            0: "2026-08-01 00:39:58", 237: "2026-08-03 07:39:58", 475: "2026-08-05 10:19:58",
                            712: "2026-08-07 13:49:58", 1186: "2026-08-11 19:29:58", 1424: "2026-08-13 20:59:58",
                            1661: "2026-08-15 23:39:58", 1898: "2026-08-18 05:39:58", 2373: "2026-08-22 18:49:58",
                            3559: "2026-08-31 23:59:58"})}},
    rules=[Rule("2026-08-01", "2026-08-31", 240, 8, 24)],
    plan=[
        _plan(5, "Монолит надземной части", "2025-10-01", "2026-06-30",
              ["12.4.4.", "12.4.28."], {"tower_crane": 1, "concrete_pump": 1, "concrete_mixer": 3, "truck": 1}),
        _plan(7, "Фасад и остекление", "2026-05-15", "2026-11-30",
              ["12.4.11.", "12.4.38.", "12.4.54."], {"facade_hoist": 2, "tower_crane": 1, "crane_manipulator": 1}),
    ],
    fleet={"tower_crane": 1, "facade_hoist": 2, "crane_manipulator": 1},
    demonstrates=["этап 7 (фасад и остекление) на высотке", "рост облицовки за месяц",
                  "ночные кадры (подсветка окон) — не идут в модель Б"],
    plan_note="Фасад по плану с 15.05; ожидаемый этап 7.",
)

site_spec(
    "ru_seliger_city",
    name="ЖК «Селигер Сити», 4-я очередь — благоустройство и школа (Москва)",
    address="Москва, САО, Ильменский пр. (ЖК «Селигер Сити»)",
    object_type="Жильё",
    description=("Настоящая камера: двор 4-й очереди ЖК «Селигер Сити» в августе 2026 г. — жилые корпуса "
                 "построены, во дворе благоустройство (покрытия, спортплощадки, МАФы), рядом достраивают школу; "
                 "на заднем плане — чужие стройки с башенными кранами."),
    cams={"cam_yard": {"stream": "ilmensky2", "name": "Камера на корпусе — вид во двор",
                       "osd": OsdSpec(x0=248.5, pitch=13.0, y0=1056, y1=1080, fmt="ymd", refs={
                           0: "2026-08-01 00:59:58", 685: "2026-08-07 17:59:58", 913: "2026-08-09 18:39:58",
                           1142: "2026-08-11 21:09:58", 1827: "2026-08-18 05:09:58", 2055: "2026-08-20 07:39:58",
                           2283: "2026-08-22 21:29:58", 2512: "2026-08-24 17:49:58", 3197: "2026-08-30 06:59:58"})}},
    rules=[Rule("2026-08-01", "2026-08-31", 240, 8, 24), Rule("2026-08-19", "2026-08-19", 30, 7, 21)],
    plan=[
        _plan(7, "Фасад и остекление", "2026-03-01", "2026-07-31",
              ["12.4.11.", "12.4.38."], {"facade_hoist": 1, "crane_manipulator": 1}),
        _plan(8, "Наружные сети и благоустройство", "2026-07-01", "2026-09-30",
              ["12.7.1.", "12.7.2.", "12.7.3.", "12.7.4."],
              {"excavator": 1, "dump_truck": 2, "roller": 1, "wheel_loader": 1, "crane_manipulator": 1}),
    ],
    fleet={"excavator": 1, "dump_truck": 2, "roller": 1, "wheel_loader": 1, "crane_manipulator": 1},
    demonstrates=["этап 8 (благоустройство) на российском ЖК — готовые корпуса не должны тянуть этап назад",
                  "чужие стройки с кранами на заднем плане", "мелкая техника благоустройства"],
    plan_note="Благоустройство по плану с 01.07; ожидаемый этап 8.",
)

site_spec(
    "ru_slava",
    name="МФК «Слава» — грязный объектив, блики, ночь (Москва)",
    address="Москва, САО, Ленинградский пр-т (МФК «Слава», MR Group)",
    object_type="Офисно-деловой центр",
    description=("Настоящая камера: МФК «Слава» в августе 2026 г. Башни в стекле, внизу — работы по "
                 "благоустройству и сетям, за башнями — чужая высотка с краном. Объектив грязный, в разводах, "
                 "днём блики и засвет, ночью — подсветка окон: образец «плохой камеры», кадры которой нельзя "
                 "пускать в модель Б без проверки качества."),
    cams={"cam_roof": {"stream": "Slava", "name": "Камера на кровле — вид вниз между башнями",
                       "osd": OsdSpec(x0=242.5, pitch=20.0, y0=1036, y1=1070, fmt="ymd", refs={
                           0: "2026-08-01 00:09:58", 261: "2026-08-03 05:39:58", 522: "2026-08-05 06:39:58",
                           783: "2026-08-07 07:29:58", 1305: "2026-08-11 08:49:58", 1566: "2026-08-13 09:09:58",
                           2088: "2026-08-17 13:29:58", 2349: "2026-08-19 14:49:58", 2610: "2026-08-22 05:49:58",
                           3654: "2026-08-30 00:49:58"})}},
    rules=[Rule("2026-08-01", "2026-08-31", 360)],
    plan=[
        _plan(7, "Фасад и остекление", "2026-02-01", "2026-08-15",
              ["12.4.11.", "12.4.38."], {"facade_hoist": 1, "crane_manipulator": 1}),
        _plan(8, "Наружные сети и благоустройство", "2026-07-15", "2026-10-31",
              ["12.5.1.", "12.7.1.", "12.7.4."], {"excavator": 1, "dump_truck": 2, "roller": 1, "wheel_loader": 1}),
    ],
    fleet={"excavator": 1, "dump_truck": 2, "roller": 1, "wheel_loader": 1},
    demonstrates=["помехи камеры: грязь/разводы на объективе, блики, засвет, ночь",
                  "чужая высотка с краном за башнями", "этапы 7–8 офисного комплекса"],
    plan_note="Фасад по плану до 15.08, благоустройство с 15.07.",
)

site_spec(
    "ru_lesoparkovy",
    name="ЖК «Лесопарковый» — март: слякоть, туман, ночь (Москва, Чертаново)",
    address="Москва, ЮАО, Чертаново Южное, Варшавское ш., 168 (ЖК «Лесопарковый», Инград)",
    object_type="Жильё", floors_total=22,
    description=("Камера стройки ЖК «Лесопарковый» за март 2020 г. (ролик застройщика «ход строительства, "
                 "камера 1»): монолитные корпуса в 19–22 этажа на фасадных работах, во дворе — сети и "
                 "благоустройство, башенный кран. Московская ранняя весна: слякоть и остатки снега, ночной "
                 "туман (кадр почти белый), короткий световой день, прожекторы."),
    source={"title": "ЖК \"Лесопарковый\" Март 2020 - ход строительства (камера 1)",
            "url": "https://www.youtube.com/watch?v=Q3yjW_csgrc", "author": "magic people (YouTube)"},
    license="CC BY (YouTube: «Creative Commons Attribution license (reuse allowed)»)",
    timestamps="настоящие: дата и время — с экранной подписи камеры в ролике (tools/build_ru_demo.py)",
    cams={"cam_1": {"video": "yt_Q3yjW_csgrc.mp4", "stride": 20, "name": "Камера 1 — вид сверху во двор",
                    "osd": OsdSpec(x0=236.5, pitch=13.0, y0=1054, y1=1080, fmt="ymd", refs={
                        0: "2020-02-29 23:59:31", 10920: "2020-03-05 01:39:54", 21820: "2020-03-08 05:33:42",
                        27280: "2020-03-09 19:39:37", 38200: "2020-03-13 00:14:54", 49120: "2020-03-16 04:04:54",
                        65480: "2020-03-20 21:50:22", 76400: "2020-03-24 01:40:13", 87300: "2020-03-27 06:07:11",
                        92760: "2020-03-28 20:02:03", 103680: "2020-03-31 23:54:04"})}},
    rules=[Rule("2020-03-10", "2020-03-11", 20), Rule("2020-03-01", "2020-03-31", 180)],
    plan=[
        _plan(5, "Монолит надземной части", "2019-03-01", "2020-01-31",
              ["12.4.4.", "12.4.28."], {"tower_crane": 2, "concrete_pump": 1, "concrete_mixer": 3, "truck": 1}),
        _plan(7, "Фасад и остекление", "2019-10-01", "2020-04-15",
              ["12.4.11.", "12.4.38.", "12.4.54."], {"facade_hoist": 2, "tower_crane": 1, "crane_manipulator": 1}),
        _plan(8, "Наружные сети и благоустройство", "2020-03-01", "2020-06-30",
              ["12.5.1.", "12.7.1.", "12.7.4."], {"excavator": 1, "dump_truck": 2, "wheel_loader": 1, "roller": 1}),
    ],
    fleet={"tower_crane": 1, "facade_hoist": 2, "excavator": 1, "dump_truck": 2, "wheel_loader": 1},
    demonstrates=["российская весна: слякоть, остатки снега, ночной туман — отбраковка кадров для модели Б",
                  "этапы 7→8 (фасад → сети и благоустройство) на московском ЖК",
                  "моточасы: 2 рабочих дня (10–11.03) с шагом 20 мин, остальной месяц — раз в 3 ч"],
    plan_note="По плану фасад до 15.04, благоустройство с 01.03 (сдача корпусов — II кв. 2020).",
    notes=["Российская стройка, настоящая камера; время кадров — с экранной подписи камеры.",
           "План правдоподобный, в кодах перечня работ организаторов; не график застройщика.",
           "Детектор техники на этих кадрах не обучался."],
)

site_spec(
    "ru_first_moscow",
    name="Город-парк «Первый Московский» — от свай до каркаса за зиму (Новая Москва)",
    address="Москва, Новомосковский АО, г. Московский (город-парк «Первый Московский», «Абсолют Недвижимость»)",
    object_type="Жильё",
    description=("Таймлапс камеры на крыше соседнего дома, июль 2019 – март 2020 г., примерно кадр в сутки: "
                 "на переднем плане сваи и котлованы новых корпусов, за зиму они вырастают в монолитные "
                 "каркасы с башенными кранами; на заднем плане — уже построенные корпуса (не должны влиять на "
                 "этап). Осенние туманы, снег в январе–феврале, весенняя распутица — хронология этапов должна "
                 "только расти."),
    source={"title": "ЖК Первый Московский июль 2019 март 2020",
            "url": "https://www.youtube.com/watch?v=eU7t3AEBFdE", "author": "Видео Таймлапс (YouTube)"},
    license="CC BY (YouTube: «Creative Commons Attribution license (reuse allowed)»)",
    timestamps=("синтетические: 265 кадров ролика разложены равномерно на 01.07.2019–14.03.2020 (по названию "
                "ролика, примерно кадр в сутки), время суток — полдень"),
    cams={"cam_1": {"video": "yt_eU7t3AEBFdE.mp4", "name": "Камера на крыше — общий вид квартала",
                    "synthetic": {"start": "2019-07-01T12:00:00", "end": "2020-03-14T12:00:00"}}},
    rules=None,
    plan=[
        _plan(2, "Ограждение котлована, шпунт, сваи", "2019-07-01", "2019-08-31",
              ["12.3.6."], {"drilling_rig": 2, "crawler_crane": 1, "concrete_mixer": 2}),
        _plan(3, "Земляные работы, котлован", "2019-08-01", "2019-09-30",
              ["12.3.1.", "12.3.7."], {"excavator": 2, "dump_truck": 4}),
        _plan(4, "Монолит подземной части", "2019-09-15", "2019-11-30",
              ["12.3.4.", "12.3.9."], {"tower_crane": 2, "concrete_pump": 1, "concrete_mixer": 3}),
        _plan(5, "Монолит надземной части", "2019-11-15", "2020-08-31",
              ["12.4.4.", "12.4.28."], {"tower_crane": 3, "concrete_pump": 1, "concrete_mixer": 3, "truck": 1}),
    ],
    fleet={"tower_crane": 3, "concrete_pump": 1, "concrete_mixer": 3, "excavator": 2, "dump_truck": 4,
           "drilling_rig": 2},
    demonstrates=["хронология этапов 2 → 4 → 5 за 8 месяцев на московском квартале (монотонный рост)",
                  "российская зима: туман в ноябре–декабре, снег в январе–феврале — не откатывают этап",
                  "задний план: готовые корпуса соседних очередей"],
    plan_note="План правдоподобный: сваи — лето 2019, каркасы — с ноября 2019.",
    notes=["Российская стройка (Новая Москва), камера — таймлапс-бокс на крыше.",
           "Метки времени синтетические (кадр в сутки), моточасы на этом объекте не показательны.",
           "План правдоподобный, в кодах перечня работ организаторов; не график застройщика.",
           "Детектор техники на этих кадрах не обучался."],
)

site_spec(
    "ru_arhdf_school",
    name="Школа — камера российской площадки, датасет arh-df (6 июня 2023)",
    address="не указан (подпись камеры: «…shkol…», «…oskovsky, 7»; датасет arh-df — вероятно, Архангельск)",
    object_type="Образование",
    description=("Кадры одной широкоугольной камеры российской стройплощадки из открытого датасета "
                 "Dataset Ninja «Construction Equipment» (Kaggle arh-df) за 6 июня 2023 г., 10:40–16:50: "
                 "школа построена, спортплощадки готовы, справа монтируют металлокаркас пристройки; "
                 "экскаваторы, погрузчики, самосвалы и бортовые грузовики. Время — с экранной подписи камеры."),
    source={"title": "Dataset Ninja: Construction Equipment (Kaggle kartaviychert/arh-df)",
            "url": "https://datasetninja.com/construction-equipment",
            "kaggle": "https://www.kaggle.com/datasets/kartaviychert/arh-df"},
    license="GPL-2.0 (по Dataset Ninja)",
    timestamps="настоящие: дата и время — с экранной подписи камеры (tools/build_ru_demo.py)",
    cams={"cam_1": {"images": "dninja_construction_equipment/ds/img", "name": "Камера на мачте — вся площадка",
                    "osd": OsdSpec(x0=1278.0, pitch=32.0, y0=6, y1=50, fmt="dmy", refs={
                        "001ebfeb-frame288.jpg": "06-06-2023 11:51:23", "0bad1cea-frame156.jpg": "06-06-2023 10:41:47",
                        "1fee57d6-frame225.jpg": "06-06-2023 16:12:23", "2fc1855b-frame189.jpg": "06-06-2023 11:16:15",
                        "41403812-frame160.jpg": "06-06-2023 14:40:43", "54cf67e9-frame171.jpg": "06-06-2023 11:44:30",
                        "6656ad9e-frame292.jpg": "06-06-2023 12:02:49", "76f3f09b-frame167.jpg": "06-06-2023 14:57:51",
                        "86ecd498-frame220.jpg": "06-06-2023 16:40:44", "95ea97b5-frame44.jpg": "06-06-2023 16:50:12",
                        "a5a058c3-frame64.jpg": "06-06-2023 14:51:03", "b3f1974c-frame230.jpg": "06-06-2023 11:22:20",
                        "c1008335-frame10.jpg": "06-06-2023 13:48:02", "d33cefaa-frame5.jpg": "06-06-2023 11:14:15",
                        "e1a477dc-frame69.jpg": "06-06-2023 13:48:37", "ef71dad9-frame197.jpg": "06-06-2023 14:52:28"})}},
    rules=[Rule("2023-06-06", "2023-06-06", 5)],
    plan=[
        _plan(5, "Монолит надземной части", "2023-05-01", "2023-07-31",
              ["12.4.29.", "12.4.30."], {"mobile_crane": 1, "truck": 1, "crane_manipulator": 1}),
        _plan(8, "Наружные сети и благоустройство", "2023-05-15", "2023-08-31",
              ["12.5.1.", "12.7.1.", "12.7.4."], {"excavator": 2, "dump_truck": 2, "wheel_loader": 1, "roller": 1}),
    ],
    fleet={"excavator": 2, "dump_truck": 2, "wheel_loader": 1, "truck": 1, "mobile_crane": 1},
    demonstrates=["камера российской площадки с экранной подписью времени",
                  "много техники одновременно: экскаваторы, погрузчики, самосвалы, бортовые",
                  "параллельные работы: металлокаркас пристройки и благоустройство"],
    plan_note="План правдоподобный: каркас пристройки и благоустройство идут одновременно.",
    notes=["ВНИМАНИЕ: эти кадры были в обучении детектора техники (источник dninja в equipment_v1) — "
           "качество детекции на них оптимистично, для оценки детектора не использовать.",
           "Один день съёмки; оставлено по кадру на 5 минут."],
)


def _video_path(c: dict, raw: Path) -> Path:
    return raw / (c.get("video") or f"vc_{c['stream']}.mp4")


def _camera_timeline(c: dict, raw: Path, ext: Path | None) -> dict:
    """Кадры камеры и их метки: из подписи в ролике, синтетические или из подписи на снимках."""
    import cv2

    if c.get("images"):
        if ext is None:
            raise ValueError("для объекта из снимков нужен --ext (каталог с датасетами)")
        files = sorted((ext / c["images"]).glob("*.jpg"))
        osd: OsdSpec = c["osd"]
        strips = {i: osd.strip(cv2.imread(str(f))) for i, f in enumerate(files)}
        names = {f.name: i for i, f in enumerate(files)}
        spec = dataclasses.replace(osd, refs={names[k]: v for k, v in osd.refs.items()})
        idx = list(range(len(files)))
        stamps, texts, scores, margins = read_stamps(spec, idx, strips)
        # снимки не упорядочены по времени — хронологию не чистим, ненадёжные метки просто выбрасываем
        keep = [k for k, s in enumerate(stamps) if s is not None]
        order = sorted(keep, key=lambda k: stamps[k])
        return {"kind": "images", "files": [files[k] for k in order], "idx": [idx[k] for k in order],
                "stamps": [stamps[k] for k in order], "interp": [False] * len(order),
                "texts": [texts[k] for k in order], "scores": [scores[k] for k in order],
                "margins": [margins[k] for k in order], "total": len(files), "dropped": len(files) - len(keep),
                "sigs": None}
    video = _video_path(c, raw)
    osd = c.get("osd")
    idx, strips, sigs = first_pass(video, osd, c.get("stride", 1))
    if osd is None:
        syn = c["synthetic"]
        a, b = dt.datetime.fromisoformat(syn["start"]), dt.datetime.fromisoformat(syn["end"])
        n = len(idx)
        stamps = [a + (b - a) * (k / max(1, n - 1)) for k in range(n)]
        stamps = [s.replace(microsecond=0) for s in stamps]
        return {"kind": "synthetic", "video": video, "idx": idx, "stamps": stamps, "interp": [False] * n,
                "texts": [""] * n, "scores": [0.0] * n, "margins": [0.0] * n, "total": n, "dropped": 0,
                "sigs": [sigs[i] for i in idx]}
    stamps, texts, scores, margins = read_stamps(osd, idx, strips)
    fixed, interp = clean_timeline(stamps)
    return {"kind": "video", "video": video, "idx": idx, "stamps": fixed, "interp": interp, "texts": texts,
            "scores": scores, "margins": margins, "total": len(idx), "dropped": 0,
            "sigs": [sigs[i] for i in idx]}


def _save(fr: np.ndarray, c: dict, dst: Path, full: Path | None, max_side: int, quality: int) -> tuple[int, int]:
    import cv2

    if c.get("crop"):
        # чёрные поля: поток 4:3 вписан в кадр 16:9 — для маски и моделей это мусор
        x0, y0, x1, y1 = c["crop"]
        fr = fr[y0:y1, x0:x1]
    if full is not None:
        full.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = fr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return fr.shape[1], fr.shape[0]


def build_site(key: str, spec: dict, raw: Path, out_root: Path, full_dir: Path | None, max_side: int,
               quality: int = 80, ext: Path | None = None) -> dict:
    import cv2

    site_dir = out_root / key
    site_dir.mkdir(parents=True, exist_ok=True)
    cams_cfg, rows = [], []
    for cam, c in spec["cams"].items():
        tl = _camera_timeline(c, raw, ext)
        stamps = tl["stamps"]
        chosen = (select(stamps, spec["rules"], tl["sigs"]) if spec["rules"] else list(range(len(stamps))))
        cam_dir = site_dir / cam
        cam_dir.mkdir(exist_ok=True)
        for old in cam_dir.glob("*.jpg"):
            old.unlink()
        names: dict[int, str] = {}
        size = None

        def target(k: int) -> tuple[Path, Path | None]:
            name = f"{key}_{cam}_{stamps[k]:%Y_%m_%d_%H_%M_%S}.jpg"
            names[k] = name
            return cam_dir / name, (full_dir / key / cam / name) if full_dir is not None else None

        if tl["kind"] == "images":
            for k in chosen:
                size = _save(cv2.imread(str(tl["files"][k])), c, *target(k), max_side, quality)
        else:
            want = {tl["idx"][k]: k for k in chosen}
            cap = cv2.VideoCapture(str(tl["video"]))
            i, last = 0, max(want)
            while i <= last:
                if i not in want:
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    break
                size = _save(fr, c, *target(want[i]), max_side, quality)
                i += 1
            cap.release()
        source_name = tl["video"].name if tl["kind"] != "images" else c["images"]
        for k in chosen:
            rows.append({"file": f"{cam}/{names[k]}",
                         "source": tl["files"][k].name if tl["kind"] == "images" else source_name,
                         "frame_index": tl["idx"][k], "osd_text": tl["texts"][k],
                         "ocr_score": round(tl["scores"][k], 3), "ocr_margin": round(tl["margins"][k], 3),
                         "timestamp": stamps[k].isoformat(), "interpolated": int(tl["interp"][k])})
        sel = [stamps[k] for k in chosen]
        steps = [(b - a).total_seconds() / 60 for a, b in zip(sel, sel[1:])]
        n_bad = sum(tl["interp"])
        if tl["kind"] == "synthetic":
            note = f"синтетические: {tl['total']} кадров ролика равномерно на {sel[0]:%d.%m.%Y}–{sel[-1]:%d.%m.%Y}"
        elif tl["kind"] == "images":
            note = (f"настоящие — распознаны с экранной подписи камеры ({tl['total']} снимков, "
                    f"{tl['dropped']} с нечитаемой подписью выброшены)")
        else:
            note = (f"настоящие — распознаны с экранной подписи камеры ({tl['total']} кадров ролика, "
                    f"{n_bad} меток восстановлено интерполяцией)")
        cams_cfg.append({
            "key": cam, "dir": cam, "name": c["name"],
            "interval_min": min(r.step_min for r in spec["rules"]) if spec["rules"] else round(
                statistics.median(steps)) if steps else 20,
            "image_size": list(size) if size else None,
            "frames": len(chosen), "first_frame": sel[0].isoformat(), "last_frame": sel[-1].isoformat(),
            "days": len({t.date() for t in sel}),
            "median_step_min": round(statistics.median(steps), 1) if steps else None,
            "timestamps": note,
        })
        print(f"{key}/{cam}: {note}; отобрано {len(chosen)} ({sel[0]:%d.%m.%Y %H:%M} … {sel[-1]:%d.%m.%Y %H:%M})")
    with (site_dir / "frames_source.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    if spec.get("source"):
        source = spec["source"]
    else:
        streams = [c["stream"] for c in spec["cams"].values()]
        source = {**VIDEOCAM_SOURCE, "url": PAGE_URL.format(stream=streams[0]),
                  "video": [SOURCE_URL.format(stream=s) for s in streams]}
    cfg = {
        "name": spec["name"], "description": spec["description"], "source": source,
        "license": spec.get("license", LICENSE),
        "object_type": spec["object_type"], "address": spec["address"],
        "floors_total": spec.get("floors_total"),
        "timezone": "Europe/Moscow", "shift_hours": 10,
        "timestamps": spec.get("timestamps",
                               "настоящие: дата и время — с экранной подписи камеры (tools/build_ru_demo.py)"),
        "cameras": cams_cfg,
        "plan_note": spec["plan_note"],
        "suggested_plan": copy.deepcopy(spec["plan"]),
        "plan": copy.deepcopy(spec["plan"]),
        "fleet": spec["fleet"],
        "demonstrates": spec["demonstrates"],
        "notes": spec["notes"] if "source" in spec else VIDEOCAM_NOTES + spec.get("notes", []),
    }
    (site_dir / "site.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return cfg


def _dump_osd(args: argparse.Namespace) -> int:
    """Лист подписей N кадров с исходными номерами — прочитать глазами и вписать в refs."""
    import cv2

    key, n = args.dump_osd[0], int(args.dump_osd[1])
    c = next(iter(SITES[key]["cams"].values()))
    if c.get("images"):
        files = sorted((args.ext / c["images"]).glob("*.jpg"))
        picks = [(f.name, c["osd"].strip(cv2.imread(str(f)))) for f in files[:: max(1, len(files) // n)][:n]]
    else:
        idx, strips, _ = first_pass(_video_path(c, args.raw), c["osd"], c.get("stride", 1))
        picks = [(str(idx[round(j * (len(idx) - 1) / max(1, n - 1))]), None) for j in range(n)]
        picks = [(lab, strips[int(lab)]) for lab, _ in picks]
    rows = []
    for label, strip in picks:
        s = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        pad = np.zeros((s.shape[0], 330, 3), np.uint8)
        cv2.putText(pad, label, (5, s.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(np.hstack([pad, s]))
    cv2.imwrite(str(args.dump_to), np.vstack(rows))
    print(f"{key}: {len(rows)} подписей → {args.dump_to}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="каталог со скачанными роликами (vc_*.mp4, yt_*.mp4)")
    p.add_argument("--ext", type=Path, help="каталог внешних датасетов (для объекта из снимков arh-df)")
    p.add_argument("--out", type=Path, help="каталог sites/ демо-данных")
    p.add_argument("--only", action="append", help="ключ объекта (можно несколько раз)")
    p.add_argument("--full-dir", type=Path, help="сюда же — кадры в исходном размере (для оценки детектора)")
    p.add_argument("--max-side", type=int, default=1280, help="длинная сторона кадров демо-объекта")
    p.add_argument("--quality", type=int, default=80, help="качество JPEG кадров демо-объекта")
    p.add_argument("--dump-osd", nargs=2, metavar=("ОБЪЕКТ", "N"), help="лист подписей N кадров для чтения refs")
    p.add_argument("--dump-to", type=Path, default=Path("osd_dump.png"))
    args = p.parse_args(argv)
    if args.dump_osd:
        return _dump_osd(args)
    if args.out is None:
        p.error("нужен --out")
    for key, spec in SITES.items():
        if args.only and key not in args.only:
            continue
        build_site(key, spec, args.raw, args.out, args.full_dir, args.max_side, args.quality, args.ext)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
