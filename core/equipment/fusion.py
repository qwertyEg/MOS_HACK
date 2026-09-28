"""Слияние камер: одна машина с разных ракурсов — одна единица техники.

Поля обзора камер пересекаются, и без слияния экскаватор, видимый тремя
камерами, превращается в три экскаватора: ложная «лишняя техника» и тройные
моточасы. Метод (docs/ARCHITECTURE.md §5.5):

1. Камера калибруется один раз: 4+ точки на кадре ↔ те же точки на плане
   площадки в метрах → гомография (`homography_from_points`).
2. Точка контакта рамки с землёй (середина нижней кромки) проецируется на
   план. Именно нижняя кромка: центр рамки высокой машины (кран) «висит» в
   воздухе, и его проекция уезжает на десятки метров.
3. Детекции разных камер одного окна времени (±15 мин) совместимого класса
   ближе `merge_radius_m` — одна машина. В «серой зоне» (до 1.6 радиуса)
   решает сходство цвета рамки. Совпавший номер — склейка безусловно, разные
   прочитанные номера — никогда.
4. Группировка — union-find по рёбрам от самых надёжных к менее надёжным с
   запретом «две детекции одной камеры в одной группе»: на одном кадре две
   рамки — это две разные машины, как бы близко они ни стояли.
5. Некалиброванная камера в слиянии не участвует — её машины считаются
   отдельно, а интерфейс подсказывает откалибровать камеру.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from core import taxonomy

from . import appearance
from .config import EquipmentConfig

# --------------------------------------------------------------------------
# калибровка
# --------------------------------------------------------------------------


def homography_from_points(image_pts: Sequence[Sequence[float]],
                           site_pts: Sequence[Sequence[float]]) -> tuple[list[list[float]], float]:
    """Гомография кадр → план (метры) и средняя ошибка репроекции в метрах.

    По 4–5 точкам — точное/МНК-решение; от 6 точек — RANSAC, чтобы одна
    ошибочно кликнутая точка не испортила калибровку (ошибка репроекции при
    этом считается по всем точкам — пользователь увидит, что что-то не так).
    """
    src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(site_pts, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError(f"точек на кадре {len(src)}, а на плане {len(dst)} — нужно поровну")
    if len(src) < 4:
        raise ValueError("для калибровки нужно минимум 4 пары точек (кадр ↔ план)")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("координаты точек должны быть числами")
    for name, pts in (("на кадре", src), ("на плане", dst)):
        if _degenerate(pts):
            raise ValueError(f"точки {name} лежат почти на одной прямой — возьмите углы площадки, "
                             "а не точки вдоль забора")
    method = cv2.RANSAC if len(src) >= 6 else 0
    H, _ = cv2.findHomography(src, dst, method, 1.0)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        raise ValueError("гомографию построить не удалось — проверьте соответствие точек")
    H = H / H[2, 2]
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    return [[float(v) for v in row] for row in H], err


def _degenerate(pts: np.ndarray) -> bool:
    centered = pts - pts.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return s[0] < 1e-9 or s[1] / s[0] < 1e-3


def project(H: Sequence[Sequence[float]] | None, point: tuple[float, float],
            image_size: tuple[int, int] | None = None,
            frame_size: tuple[int, int] | None = None) -> tuple[float, float] | None:
    """Точка кадра → план (метры). None — точка выше горизонта или нет калибровки.

    Если калибровали на кадре другого разрешения (снимок 1920×1080, а поток
    пришёл 1280×720), пиксели сначала пересчитываются в масштаб калибровки.
    """
    if H is None:
        return None
    x, y = point
    if (image_size and frame_size and min(frame_size) > 0 and min(image_size) > 0
            and tuple(image_size) != tuple(frame_size)):
        x *= image_size[0] / frame_size[0]
        y *= image_size[1] / frame_size[1]
    m = np.asarray(H, dtype=np.float64)
    v = m @ np.array([x, y, 1.0])
    if v[2] <= 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


# --------------------------------------------------------------------------
# номер
# --------------------------------------------------------------------------

_LOOKALIKE = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def normalize_plate(text: str | None) -> str | None:
    """«а 123 вс 77» и «A123BC77» — один номер: кириллица → латиница-двойник, без пробелов.

    Номер — ключ безусловной склейки, поэтому мусор («нет», «н/д», «UNKNOWN»)
    номером не считается: в любом российском номере (и автомобильном, и
    тракторном 1234 АВ 77) есть хотя бы три цифры и буква.
    """
    if not text:
        return None
    s = re.sub(r"[^0-9A-ZА-ЯЁ]", "", str(text).upper()).translate(_LOOKALIKE)
    digits = sum(ch.isdigit() for ch in s)
    if not 5 <= len(s) <= 10 or digits < 3 or digits == len(s):
        return None
    return s


# --------------------------------------------------------------------------
# группировка наблюдений разных камер
# --------------------------------------------------------------------------


@dataclass
class Observation:
    """Машина, увиденная одной камерой в момент `time` (или известная единица с последней позицией)."""
    camera_ids: frozenset[str]                 # камеры, которые уже «заняты» этим наблюдением
    time: dt.datetime
    cls: str
    site_xy: tuple[float, float] | None
    hist: np.ndarray | None = None
    plate: str | None = None
    unit_id: str | None = None
    key: object = None                          # что это за наблюдение для вызывающего
    extra: dict = field(default_factory=dict)


def link_score(a: Observation, b: Observation, config: EquipmentConfig) -> float | None:
    """Насколько надёжно a и b — одна машина: меньше — надёжнее, None — точно разные."""
    cfg = config
    if a.camera_ids & b.camera_ids:
        return None
    if a.plate and b.plate:
        # Номер важнее и окна времени, и класса: детектор может спутать тип,
        # но не номер. Разные прочитанные номера — точно разные машины.
        return 0.0 if a.plate == b.plate else None
    if abs((a.time - b.time).total_seconds()) > cfg.merge_window_min * 60:
        return None
    if not taxonomy.confusable(a.cls, b.cls):
        return None
    if a.site_xy is None or b.site_xy is None:
        return None
    d = math.dist(a.site_xy, b.site_xy)
    sim = appearance.hist_similarity(a.hist, b.hist)
    score = d / cfg.merge_radius_m + (cfg.appearance_weight * (1.0 - sim) if sim is not None else 0.0)
    if d <= cfg.merge_radius_m:
        return 0.01 + score
    if d <= cfg.merge_radius_m * cfg.merge_gray_factor and sim is not None and sim >= cfg.appearance_min_similarity:
        return 0.01 + score
    return None


def cluster(observations: Sequence[Observation], config: EquipmentConfig) -> list[list[int]]:
    """Union-find: группы наблюдений, которые считаются одной машиной."""
    n = len(observations)
    parent = list(range(n))
    cams = [set(o.camera_ids) for o in observations]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> bool:
        ri, rj = find(i), find(j)
        if ri == rj:
            return True
        if cams[ri] & cams[rj]:
            return False          # в группе оказались бы две рамки одной камеры
        parent[rj] = ri
        cams[ri] |= cams[rj]
        return True

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = link_score(observations[i], observations[j], config)
            if s is not None:
                edges.append((s, i, j))
    for _, i, j in sorted(edges):
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def count_units(observations: Sequence[Observation], config: EquipmentConfig | None = None) -> int:
    """Сколько разных машин в наборе одновременных наблюдений нескольких камер."""
    return len(cluster(observations, config or EquipmentConfig()))
