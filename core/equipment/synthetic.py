"""Синтетические кадры стройплощадки: проверка модели А без камер и без GPU.

Рисует грубые, но честные для алгоритмов сцены: текстурный грунт, машины с
кабиной и стрелой, поза стрелы задаётся углом. Этого достаточно, чтобы
проверить то, что модель А делает сама (трекинг, «работает / стоит»,
устойчивость к ночи, шуму и дрожанию камеры), не проверяя детектор.
Используется тестами и может использоваться для демо интерфейса.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from core.contracts import Detection

from .boxes import Box


def ground(width: int = 640, height: int = 360, seed: int = 0) -> np.ndarray:
    """Грунт с крупной и мелкой текстурой, дальний план светлее — как на реальной площадке."""
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.uniform(70, 150, (height // 24 + 1, width // 24 + 1)).astype(np.float32),
                        (width, height), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 12, (height, width)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 1.2)
    base = np.clip(coarse + fine, 0, 255)
    img = np.stack([base * 0.85, base * 0.95, base * 1.05], axis=-1)   # чуть тёплый оттенок
    # несколько «плит» и штабелей — статичный фон с чёткими краями
    for _ in range(6):
        x, y = int(rng.integers(0, width - 60)), int(rng.integers(0, height - 40))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 60)), y + int(rng.integers(10, 30))),
                      tuple(float(v) for v in rng.uniform(60, 200, 3)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_machine(img: np.ndarray, box: Box, color=(0, 170, 240), pose: float = 0.0,
                 kind: str = "excavator") -> Box:
    """Машина в рамке `box`. pose — угол стрелы в градусах (только для экскаватора/крана).

    Возвращает рамку, которую выдал бы идеальный детектор (с учётом стрелы).
    """
    x, y, w, h = (int(round(v)) for v in box)
    body_top = y + h // 2
    dark = tuple(int(c * 0.45) for c in color)
    # гусеницы/колёса и корпус
    cv2.rectangle(img, (x, y + int(h * 0.8)), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x + 2, body_top), (x + w - 2, y + int(h * 0.8)), color, -1)
    cv2.rectangle(img, (x + w // 6, y + int(h * 0.25)), (x + w // 2, body_top), color, -1)       # кабина
    cv2.rectangle(img, (x + w // 6 + 4, y + int(h * 0.3)), (x + w // 2 - 4, body_top - 4), (200, 220, 230), -1)
    cv2.line(img, (x + 4, body_top + 3), (x + w - 4, body_top + 3), dark, 2)
    if kind in ("excavator", "mobile_crane", "backhoe_loader"):
        # Стрела из точки на корпусе; поза меняет угол стрелы и рукояти.
        # Всё держим внутри рамки: реальный детектор обводит машину вместе со стрелой.
        def inside(px_, py_):
            return int(np.clip(px_, x + 3, x + w - 3)), int(np.clip(py_, y + 3, y + h - 3))
        p = (x + int(w * 0.55), body_top)
        a = math.radians(-55 + pose)
        e = inside(p[0] + w * 0.32 * math.cos(a), p[1] + w * 0.32 * math.sin(a))
        b = math.radians(80 - pose * 1.5)
        k = inside(e[0] + w * 0.28 * math.cos(b), e[1] + w * 0.28 * math.sin(b))
        cv2.line(img, p, e, dark, max(3, w // 16))
        cv2.line(img, e, k, dark, max(3, w // 20))
        cv2.circle(img, k, max(4, w // 12), (30, 30, 30), -1)                                        # ковш
    elif kind in ("dump_truck", "truck"):
        cv2.rectangle(img, (x + w // 2, y + int(h * 0.2)), (x + w - 2, body_top), dark, -1)            # кузов
    return float(x), float(y), float(w), float(h)


def darken(img: np.ndarray, gain: float = 0.25, noise: float = 6.0, seed: int = 0) -> np.ndarray:
    """Ночной кадр: темно и шумно (слабая подсветка площадки, высокий ISO)."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * gain + rng.normal(0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def relight(img: np.ndarray, gain: float = 1.0, offset: float = 0.0, noise: float = 3.0,
            seed: int = 0) -> np.ndarray:
    """Другое освещение и шум сенсора — то, что меняется между кадрами с интервалом 20–30 мин."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * gain + offset + rng.normal(0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Дрожание/сдвиг камеры: вся картинка смещается, края дополняются отражением."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def det(cls: str, box: Box, conf: float = 0.9, **extra) -> Detection:
    return Detection(cls=cls, conf=conf, bbox=tuple(float(v) for v in box), source="synthetic", extra=dict(extra))
