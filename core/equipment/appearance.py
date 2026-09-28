"""Внешность рамки: изменение содержимого между кадрами, сдвиг камеры, цвет.

Зачем. Экскаватор копает, не сдвигая шасси: центр рамки на месте, форма
почти та же, а ковш и стрела — в другой позе. Смещение этого не видит, а
содержимое рамки видит. Но сравнивать «в лоб» нельзя: между кадрами 20–30
минут, за это время меняются солнце, облака, фонари, камера ночью
переключается в ИК, дождь добавляет шум. Поэтому:

1. Сравниваем не рамки двух кадров (детектор каждый раз рисует её чуть
   иначе), а ОДНУ И ТУ ЖЕ область пикселей — окрестность прошлой рамки — в
   прошлом и текущем кадре. Дрожание рамки детектора на это не влияет.
2. Серые уменьшенные кропы после локальной нормировки контраста (LCN):
   плавные изменения освещения (тень облака, закат, ореол прожектора)
   уходят, структура (кромки стрелы и ковша) остаётся.
3. Метрика — доля площади рамки, где структура изменилась. Не корреляция
   всего кропа: ковш занимает десятую часть рамки и в общей корреляции тонет.
4. Вычитаем фон: ту же долю считаем по 8 клеткам кольца вокруг рамки и
   берём медиану. Если поменялось всё вокруг (дождь, снег, засветка) — это
   не работа машины. Медиана, а не среднее: стрела, вышедшая за рамку в
   одну-две клетки кольца, фон не «поднимает».
5. Поиск лучшего совмещения в ±1 пиксель уменьшенного кропа — гасит
   микродрожание камеры на ветру.
6. Однотонный кроп (ночь без подсветки, засвет) — сравнивать нечего,
   возвращаем «нет данных», а не «изменилось».

Калибровка на синтетике (tests/core/test_equipment_tracker.py, кроп 64 px,
порог 0.75): статика с шумом, сменой освещения, JPEG, дождём, тенью облака —
0.000–0.012; поворот стрелы на 10–40° днём — 0.04–0.08, ночью — 0.02–0.05;
человек перед стоящей машиной — 0.04–0.07 (содержимое правда изменилось —
от таких единичных срабатываний защищает confirm_moves в hours).

Крупный сдвиг всей камеры (перевесили, повернули) ловим фазовой корреляцией
уменьшенных кадров: фон занимает большую часть кадра, поэтому пик
корреляции — это сдвиг камеры, а не отдельной машины.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import boxes

PATCH = 64            # сторона кропа окрестности рамки после уменьшения
RING_FRAC = 0.5       # окрестность = рамка, расширенная на 50 % в каждую сторону
SMALL_W = 320         # ширина уменьшенного кадра для оценки сдвига камеры


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# кроп окрестности рамки
# --------------------------------------------------------------------------


class _Pixels:
    """Кроп окрестности: сырые уровни серого (проверка «однотонно») и после LCN (сравнение)."""

    __slots__ = ("raw", "lcn")

    def __init__(self, raw: np.ndarray):
        self.raw = raw
        self.lcn = _lcn(raw)


class Patch:
    """Окрестность рамки из одного кадра: где взята (координаты кадра) и что там было."""

    __slots__ = ("region", "bbox", "pixels", "frame_hw")

    def __init__(self, region: tuple[int, int, int, int], bbox: boxes.Box,
                 pixels: _Pixels, frame_hw: tuple[int, int]):
        self.region = region          # x0, y0, x1, y1 в пикселях кадра
        self.bbox = bbox              # рамка, вокруг которой брали
        self.pixels = pixels
        self.frame_hw = frame_hw


def take_patch(gray: np.ndarray, bbox: boxes.Box) -> Patch | None:
    h, w = gray.shape[:2]
    region = _region(boxes.expand(bbox, RING_FRAC), w, h)
    pixels = _crop(gray, region)
    if pixels is None:
        return None
    return Patch(region, bbox, pixels, (h, w))


def appearance_delta(prev: Patch, gray: np.ndarray, shift: tuple[float, float] = (0.0, 0.0),
                     min_std: float = 3.0) -> float | None:
    """Изменение содержимого окрестности прошлой рамки сверх изменения фона, 0..1.

    None — сравнивать нечего (однотонный кроп, другой размер кадра, область
    ушла за край после сдвига камеры).
    """
    h, w = gray.shape[:2]
    if (h, w) != prev.frame_hw:
        return None
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    x0, y0, x1, y1 = prev.region
    region = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if region[0] < 0 or region[1] < 0 or region[2] > w or region[3] > h:
        return None
    cur = _crop(gray, region)
    if cur is None:
        return None
    cells = _cells(prev.region, prev.bbox)
    inner = _cell_delta(prev.pixels, cur, cells[4], min_std)
    if inner is None:
        return None
    ring = [d for i, c in enumerate(cells) if i != 4
            for d in [_cell_delta(prev.pixels, cur, c, min_std)] if d is not None]
    background = float(np.median(ring)) if len(ring) >= 3 else 0.0
    return float(max(0.0, inner - background))


def _region(b: boxes.Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = int(max(0, np.floor(b[0]))), int(max(0, np.floor(b[1])))
    x1, y1 = int(min(w, np.ceil(b[0] + b[2]))), int(min(h, np.ceil(b[1] + b[3])))
    return x0, y0, x1, y1


def _crop(gray: np.ndarray, region: tuple[int, int, int, int]) -> _Pixels | None:
    x0, y0, x1, y1 = region
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    # INTER_AREA усредняет пиксели — заодно гасит шум сенсора и JPEG.
    raw = cv2.resize(gray[y0:y1, x0:x1], (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _Pixels(raw)


def _cells(region: tuple[int, int, int, int], bbox: boxes.Box) -> list[tuple[int, int, int, int]]:
    """Сетка 3×3 в координатах кропа: центр — сама рамка, остальное — кольцо фона."""
    x0, y0, x1, y1 = region
    sx, sy = PATCH / max(1, x1 - x0), PATCH / max(1, y1 - y0)
    bx0 = int(np.clip(round((bbox[0] - x0) * sx), 0, PATCH))
    by0 = int(np.clip(round((bbox[1] - y0) * sy), 0, PATCH))
    bx1 = int(np.clip(round((bbox[0] + bbox[2] - x0) * sx), 0, PATCH))
    by1 = int(np.clip(round((bbox[1] + bbox[3] - y0) * sy), 0, PATCH))
    xs, ys = (0, bx0, bx1, PATCH), (0, by0, by1, PATCH)
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(3) for i in range(3)]


CHANGE_TAU = 0.75     # порог «пиксель изменился», в единицах локального контраста
LCN_SIGMA = PATCH / 16
LCN_FLOOR = 8.0       # уровни серого: ниже — это шум, а не структура (ночь)


def _lcn(p: np.ndarray) -> np.ndarray:
    """Локальная нормировка контраста: (I − среднее окрестности) / (σ окрестности + пол).

    Убирает то, что меняется плавно — тень облака, градиент заката, ореол
    прожектора, — и оставляет структуру: кромки стрелы, ковша, кабины. Пол в
    знаменателе не даёт раздуть шум ночных однотонных участков до «структуры».
    """
    mu = cv2.GaussianBlur(p, (0, 0), LCN_SIGMA)
    d = p - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), LCN_SIGMA))
    return d / (sd + LCN_FLOOR)


def _cell_delta(a: np.ndarray, b: np.ndarray, cell: tuple[int, int, int, int],
                min_std: float) -> float | None:
    """Доля площади клетки, где структура изменилась (лучшее совмещение в ±1 px)."""
    x0, y0, x1, y1 = cell
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ref_raw = b.raw[y0 + 1:y1 - 1, x0 + 1:x1 - 1]
    if ref_raw.size < 4 or ref_raw.std() < min_std:
        return None
    ref = b.lcn[y0 + 1:y1 - 1, x0 + 1:x1 - 1]
    best = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ya, xa = y0 + 1 + dy, x0 + 1 + dx
            cand_raw = a.raw[ya:ya + ref.shape[0], xa:xa + ref.shape[1]]
            if cand_raw.shape != ref.shape or cand_raw.std() < min_std:
                continue
            cand = a.lcn[ya:ya + ref.shape[0], xa:xa + ref.shape[1]]
            d = _changed_fraction(cand, ref)
            best = d if best is None else min(best, d)
    return best


def _changed_fraction(a_n: np.ndarray, b_n: np.ndarray) -> float:
    """Доля пикселей, где нормированная яркость разошлась больше чем на CHANGE_TAU.

    Доля площади, а не корреляция всего кропа: ковш занимает десятую часть
    рамки, и в общей корреляции его поворот тонет, а в доле изменившейся
    площади виден сразу. Шум сенсора после усреднения при уменьшении и
    нормировки до порога не дотягивает, поэтому на статике доля ≈ 0.
    """
    return float((np.abs(a_n - b_n) > CHANGE_TAU).mean())


# --------------------------------------------------------------------------
# сдвиг камеры
# --------------------------------------------------------------------------


def small_frame(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Уменьшенный кадр для фазовой корреляции и множитель обратно в пиксели кадра."""
    h, w = gray.shape[:2]
    scale = w / SMALL_W if w > SMALL_W else 1.0
    size = (max(1, int(round(w / scale))), max(1, int(round(h / scale))))
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return small, scale


def camera_shift(prev_small: np.ndarray, cur_small: np.ndarray, scale: float,
                 min_response: float = 0.1) -> tuple[float, float] | None:
    """Сдвиг текущего кадра относительно прошлого в пикселях кадра (dx, dy).

    None — сцена изменилась слишком сильно, чтобы сдвиг был надёжным
    (смена день/ночь, туман); вызывающий тогда не компенсирует ничего.
    """
    if prev_small.shape != cur_small.shape:
        return None
    win = cv2.createHanningWindow(cur_small.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(prev_small, cur_small, win)
    if response < min_response:
        return None
    return float(dx * scale), float(dy * scale)


# --------------------------------------------------------------------------
# цвет — для склейки камер и повторного сопоставления треков
# --------------------------------------------------------------------------


def color_hist(image_bgr: np.ndarray, bbox: boxes.Box) -> np.ndarray | None:
    """HSV-гистограмма центральной части рамки (края рамки — фон).

    Оттенок и насыщенность, без яркости: одна и та же жёлтая машина с двух
    камер и при разном солнце остаётся жёлтой.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        return None
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = boxes.clip(boxes.expand(bbox, -0.1), w, h)
    x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).astype(np.float32).ravel()
    s = float(hist.sum())
    return hist / s if s > 0 else None


def hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """1 − расстояние Бхаттачарьи: 1 — одинаковые цвета, 0 — ничего общего."""
    if a is None or b is None:
        return None
    return float(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
