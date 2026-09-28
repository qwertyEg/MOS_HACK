"""Модель А: техника на снимках камер стройплощадки.

Что делает: находит и классифицирует технику (YOLO локально или GLM-4.6V
по API), сопоставляет её между кадрами одной камеры («работает / стоит» по
смещению, форме рамки и позе стрелы), склеивает одну машину, видимую
несколькими камерами, в одну единицу, ведёт статусы ACTIVE / IDLE /
PARKED / DEPARTED и списывает моточасы из плановых («временная полоска»).

Точки входа для веб-слоя (docs/ARCHITECTURE.md §5):
    get_detector("yolo" | "glm" | "local_vlm", **kw) → Detector
    EquipmentEngine(config).process(frame, image, detections, geometry, zones, plan) → EquipmentUpdate
    hours.planned_hours(...), hours.balances(...)
    fusion.homography_from_points(image_pts, site_pts)
    draw.annotate(image, detections)
"""
from __future__ import annotations

from core.contracts import Detector

from .config import EquipmentConfig
from .engine import EquipmentEngine, EquipmentUpdate

__all__ = ["get_detector", "EquipmentEngine", "EquipmentConfig", "EquipmentUpdate", "DETECTORS"]

# имя в настройках → что это (для страницы «Настройки»)
DETECTORS = {
    "yolo": "Локальная модель YOLO (без интернета)",
    "glm": "Внешний API: GLM-4.6V (z.ai)",
    "local_vlm": "Локальная VLM по OpenAI-совместимому API (Ollama/vLLM)",
}

_ALIASES = {"local": "yolo", "external": "glm", "zai": "glm", "glm-4.6v": "glm", "vlm_local": "local_vlm"}


def get_detector(name: str, **kw) -> Detector:
    """Детектор по имени из настроек. Тяжёлые зависимости грузятся при первом detect()."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key == "yolo":
        from .detect_yolo import YoloDetector
        return YoloDetector(**kw)
    if key == "glm":
        from .detect_vlm import VlmDetector
        return VlmDetector(**kw)
    if key == "local_vlm":
        from .detect_vlm import VlmDetector
        kw.setdefault("provider", "local")
        return VlmDetector(**kw)
    raise ValueError(f"неизвестный детектор {name!r}; доступны: {', '.join(DETECTORS)}")
