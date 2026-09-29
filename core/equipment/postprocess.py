"""Чистка детекций одного кадра: одна машина — одна рамка.

Детектор (и YOLO, и тем более VLM) на одной машине нередко выдаёт две рамки
разных, но похожих классов: «грузовик 0.61» и «самосвал 0.58», «грузовик» и
«кран-манипулятор». Если их не схлопнуть, аналитика насчитает две машины и
поднимет ложную «лишнюю технику». Поэтому, помимо обычного NMS, гасим
пересекающиеся рамки ВНУТРИ групп путаницы `taxonomy.CONFUSABLE_GROUPS`,
оставляя более уверенную. Классы из разных групп не трогаем: экскаватор,
грузящий самосвал, законно перекрывает его рамку.

Функция идемпотентна: её вызывают и детекторы (чтобы «Проверить снимок»
показывал чистую картинку), и движок перед трекингом.
"""
from __future__ import annotations

import dataclasses

from core import taxonomy
from core.contracts import Detection

from . import boxes
from .config import EquipmentConfig


def clean(detections: list[Detection], width: int, height: int,
          config: EquipmentConfig | None = None) -> list[Detection]:
    cfg = config or EquipmentConfig()
    known = taxonomy.equipment()
    frame_area = float(width * height) if width and height else 0.0

    kept: list[Detection] = []
    for d in detections:
        if d.cls not in known:
            continue                                    # класс не из словаря — ядро такого не знает
        if d.extra.get("manual"):
            # Рамку дорисовал или подтвердил оператор: пороги уверенности, размера и
            # края — это защита от шума детектора, а не от человека.
            kept.append(dataclasses.replace(d, bbox=boxes.clip(d.bbox, width, height) if frame_area else d.bbox,
                                            extra=dict(d.extra)))
            continue
        if d.conf < cfg.conf_for(d.cls):
            continue
        b = boxes.clip(d.bbox, width, height) if frame_area else d.bbox
        if b[2] < cfg.min_box_side_px or b[3] < cfg.min_box_side_px:
            continue
        if frame_area:
            a = boxes.area(b)
            if a < cfg.min_box_area_frac * frame_area:
                continue
            if _touches_edge(b, width, height, cfg.edge_margin_px) and a < cfg.edge_min_area_frac * frame_area:
                # Маленький обрезок у края: половина машины за кадром, класс и
                # положение ненадёжны, а соседняя камера увидит её целиком.
                continue
        # Копия, а не правка на месте: вход может ещё понадобиться вызывающему
        # (например, сырые рамки для отладки детектора).
        kept.append(dataclasses.replace(d, bbox=b, extra=dict(d.extra)))

    # Рамки оператора — первыми: при «одна машина — одна рамка» побеждает человек.
    # Две ручные рамки друг друга не гасят — их обе нарисовал или подтвердил он.
    kept.sort(key=lambda d: (bool(d.extra.get("manual")), d.conf), reverse=True)
    out: list[Detection] = []
    for d in kept:
        winner = next((k for k in out if not (k.extra.get("manual") and d.extra.get("manual"))
                       and _same_machine(k, d, cfg)), None)
        if winner is None:
            out.append(d)
            continue
        if winner.cls != d.cls and not winner.extra.get("manual"):
            # Проигравшая метка не выбрасывается бесследно: трекер добавит её
            # голос к треку, и при устойчивой путанице победит большинство.
            winner.extra["alt"] = [*winner.extra.get("alt", []), [d.cls, round(float(d.conf), 3)]]
    return out


def _same_machine(a: Detection, b: Detection, cfg: EquipmentConfig) -> bool:
    if a.cls == b.cls:
        return boxes.iou(a.bbox, b.bbox) >= cfg.same_class_iou
    if not taxonomy.confusable(a.cls, b.cls):
        return False
    if (a.cls, b.cls) in cfg.pair_exempt or (b.cls, a.cls) in cfg.pair_exempt:
        # Насос и миксер работают вплотную и в перспективе перекрываются почти
        # целиком — гасим только практически совпадающие рамки, иначе
        # сломаем правило пары «насос без миксера».
        return boxes.iou(a.bbox, b.bbox) >= cfg.pair_iou
    return (boxes.iou(a.bbox, b.bbox) >= cfg.cross_class_iou
            or boxes.ioa_min(a.bbox, b.bbox) >= cfg.cross_class_ioa)


def _touches_edge(b: boxes.Box, width: int, height: int, margin: float) -> bool:
    return (b[0] <= margin or b[1] <= margin
            or b[0] + b[2] >= width - margin or b[1] + b[3] >= height - margin)
