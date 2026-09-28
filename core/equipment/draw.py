"""Кадр с рамками техники: для «Проверить снимок», /api/frames/{id}/annotated.jpg и доказательств.

Палитра `CLASS_COLORS` — единая для бэкенда и UI (легенда, «полоски»
моточасов, тепловая карта): одна и та же машина везде одного цвета.
Подписи по-русски («Экскаватор · 0.91 · работает»). Шрифты Hershey в
OpenCV кириллицу не умеют, поэтому текст рисуется через Pillow с TTF-шрифтом;
если подходящего шрифта в системе нет (тонкий Docker-образ), подпись
транслитерируется и рисуется средствами OpenCV — кадр отдаётся всегда.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from core import taxonomy
from core.contracts import Activity, Detection

# Палитра — ровно та же, что app/static/js/palette.js (источник — UI: восемь классов ТЗ
# в слотах проверенной категориальной палитры, различимой при дальтонизме). Рамки на
# annotated.jpg и SVG-оверлей в браузере одного цвета; tests/ui/test_ui_static.py сверяет таблицы.
CLASS_COLORS: dict[str, str] = {
    "excavator": "#3987e5",  # 1 синий
    "dump_truck": "#d95926",  # 2 оранжевый
    "bulldozer": "#199e70",  # 3 бирюзовый
    "mobile_crane": "#c98500",  # 4 жёлтый
    "concrete_mixer": "#d55181",  # 5 маджента
    "roller": "#008300",  # 6 зелёный
    "truck": "#9085e9",  # 7 фиолетовый
    "crane_manipulator": "#e66767",  # 8 красный
    "tower_crane": "#0ea5e9",
    "crawler_crane": "#14b8a6",
    "concrete_pump": "#c026d3",
    "drilling_rig": "#b45309",
    "pile_driver": "#a8a29e",
    "wheel_loader": "#84cc16",
    "skid_steer": "#65a30d",
    "backhoe_loader": "#6366f1",
    "telehandler": "#0891b2",
    "grader": "#a3a635",
    "asphalt_paver": "#78716c",
    "aerial_platform": "#f472b6",
    "facade_hoist": "#94a3b8",
}
DEFAULT_COLOR = "#a1a1aa"

ACTIVITY_RU = {Activity.WORKING: "работает", Activity.IDLE: "стоит", Activity.UNKNOWN: ""}
STATUS_RU = {"parked": "на стоянке", "departed": "уехала"}

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def color_bgr(cls: str) -> tuple[int, int, int]:
    h = CLASS_COLORS.get(cls, DEFAULT_COLOR).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def label_for(d: Detection, labels_ru: bool = True) -> str:
    name = taxonomy.equipment_name(d.cls) if labels_ru else d.cls
    parts = [name, f"{d.conf:.2f}"]
    act = ACTIVITY_RU.get(d.activity, "")
    status = STATUS_RU.get(str(d.extra.get("unit_status", "")), "")
    if status:
        parts.append(status)
    elif act:
        parts.append(act if labels_ru else d.activity.value)
    return " · ".join(parts)


def annotate(image_bgr: np.ndarray, detections: list[Detection], labels_ru: bool = True) -> np.ndarray:
    """Копия кадра с рамками и подписями; исходный массив не меняется."""
    img = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if not detections:
        return img
    h, w = img.shape[:2]
    thick = max(2, round(min(h, w) / 360))
    font_px = max(12, round(min(h, w) / 45))
    labels = []
    for d in sorted(detections, key=lambda d: d.conf):
        x, y, bw, bh = (int(round(v)) for v in d.bbox)
        col = color_bgr(d.cls)
        # Работающая — сплошная рамка, стоящая — тоньше: видно с первого взгляда.
        t = thick + 1 if d.activity == Activity.WORKING else thick
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, t, cv2.LINE_AA)
        labels.append((x, y, col, label_for(d, labels_ru)))
    return _draw_labels(img, labels, font_px)


def _draw_labels(img: np.ndarray, labels, font_px: int) -> np.ndarray:
    font = _font(font_px) if labels else None
    if font is None:
        for x, y, col, text in labels:
            _cv_label(img, x, y, col, _translit(text), font_px)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for x, y, col, text in labels:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        pad = max(2, font_px // 5)
        ty = y - th - 2 * pad if y - th - 2 * pad >= 0 else y
        rgb = (col[2], col[1], col[0])
        draw.rectangle((x, ty, x + tw + 2 * pad, ty + th + 2 * pad), fill=rgb)
        draw.text((x + pad - l, ty + pad - t), text, font=font, fill=_ink(rgb))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _cv_label(img, x, y, col, text, font_px) -> None:
    scale = font_px / 30
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = y - th - base - 4 if y - th - base - 4 >= 0 else y
    cv2.rectangle(img, (x, ty), (x + tw + 6, ty + th + base + 4), col, -1)
    ink = _ink((col[2], col[1], col[0]))
    cv2.putText(img, text, (x + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (ink[2], ink[1], ink[0]), 1,
                cv2.LINE_AA)


def _ink(rgb) -> tuple[int, int, int]:
    """Чёрный или белый текст — что читается на цвете плашки."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (17, 17, 17) if lum > 150 else (255, 255, 255)


@lru_cache(maxsize=8)
def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in filter(None, (os.environ.get("EQUIPMENT_FONT"), *_FONT_CANDIDATES)):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


_TR = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "·": "|", "№": "N",
})


def _translit(text: str) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        t = low.translate(_TR)
        out.append(t.capitalize() if ch != low and t else t)
    return "".join(out)
