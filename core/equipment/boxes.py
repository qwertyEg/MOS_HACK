"""Арифметика рамок XYWH (левый верх, ширина, высота — как в contracts.Detection).

Отдельный модуль, потому что одни и те же IoU/пересечение нужны
постобработке, трекеру и слиянию камер; держать три копии — ждать, что
одна из них разойдётся с остальными.
"""
from __future__ import annotations

import math

Box = tuple[float, float, float, float]


def area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def intersection(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa_min(a: Box, b: Box) -> float:
    """Пересечение, делённое на площадь меньшей рамки: 1.0 — одна целиком внутри другой."""
    m = min(area(a), area(b))
    return intersection(a, b) / m if m > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def foot(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3]


def diag(b: Box) -> float:
    return math.hypot(b[2], b[3])


def clip(b: Box, width: int, height: int) -> Box:
    x1, y1 = max(0.0, b[0]), max(0.0, b[1])
    x2, y2 = min(float(width), b[0] + b[2]), min(float(height), b[1] + b[3])
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def expand(b: Box, frac: float) -> Box:
    dx, dy = b[2] * frac, b[3] * frac
    return b[0] - dx, b[1] - dy, b[2] + 2 * dx, b[3] + 2 * dy


def union_box(a: Box, b: Box) -> Box:
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x1, y1, x2 - x1, y2 - y1


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Box:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)
