"""Режим работы и пороги (таблица `settings`, docs/ARCHITECTURE.md §10).

`mode = local | external | hybrid` — пресет пары провайдеров:
local = yolo + siglip (ноутбук без интернета), external = glm + glm,
hybrid = yolo + glm. Модели можно выбрать и по отдельности — тогда режим
выводится из пары (оба локальные → local, оба glm → external, иначе hybrid).

Пороги сгруппированы по потребителю: pipeline (отбор кадров для модели Б),
stage (пороги чек-листа), equipment (EquipmentConfig модели А), analytics
(окна правил). Пользователь правит их в UI; в БД лежат только отличия от
умолчаний, так что новые пороги из кода подхватываются без миграций.
"""
from __future__ import annotations

import copy
import threading
import time
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings as cfg
from app.models import Setting
from app.services import providers
from app.services.providers import MODEL_A, MODEL_B

PRESETS: dict[str, tuple[str, str]] = {
    "local": ("yolo", "siglip"),
    "external": ("glm", "glm"),
    "hybrid": ("yolo", "glm"),
}

CLOCK_MODES = ("auto", "wall", "last_frame")
MASK_MODES = ("darken", "gray", "black", "blur", "crop", "none")

DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "pipeline": {
        "stage_every_h": 1.0,         # модель Б не чаще раза в N часов съёмки на камеру
        "stage_mask_change": 0.05,    # внеочередной вызов, если доля маски сдвинулась сильнее
        "recent_window_h": 24.0,      # окно кадров для правил аналитики
        "clock": "auto",              # «сейчас» аналитики: auto | wall | last_frame
        "live_gap_days": 3.0,         # auto: последний кадр свежее N дней → живая площадка, иначе архив
    },
    "stage": {
        "yes_thr": 0.8,               # пороги SigLIP-чек-листа (k = 35) — калибровка core/stage/checklist_clip.py
        "no_thr": 0.46,
        "unsure_review_ratio": 0.5,   # доля «не уверен» выше — кадр в needs_review
        "equipment_weight": 1.0,      # вес техники модели А в определении этапа (core/stage/fusion.py); 0 — только чек-лист
        "mask_mode": "darken",        # как гасить фон перед моделью Б: darken | gray | black | blur | crop | none
    },
    "equipment": {},                  # заполняется из core.equipment.EquipmentConfig().to_dict()
    "analytics": {
        "pair_window_h": 2.0,
        "idle_alert_h": 4.0,
        "on_track_days": 3.0,
        "utilization": 0.7,
        "no_progress_days": 3.0,
    },
}

_cache: dict[str, Any] | None = None
_cache_at = 0.0
_lock = threading.Lock()
_TTL = 5.0


def _equipment_defaults() -> dict[str, Any]:
    mod = providers.optional_module("core.equipment")
    if mod is None or not hasattr(mod, "EquipmentConfig"):
        return {}
    try:
        return dict(mod.EquipmentConfig().to_dict())
    except Exception:  # noqa: BLE001 — чужой модуль не должен ронять настройки
        return {}


def default_thresholds() -> dict[str, dict[str, Any]]:
    out = copy.deepcopy(DEFAULT_THRESHOLDS)
    out["equipment"] = _equipment_defaults()
    return out


def _load(s: Session) -> dict[str, Any]:
    rows = {r.key: r.value for r in s.query(Setting).all()}
    mode = rows.get("mode") or cfg.default_mode
    if mode not in PRESETS:
        mode = cfg.default_mode
    model_a = rows.get("model_a") or PRESETS[mode][0]
    model_b = rows.get("model_b") or PRESETS[mode][1]
    if model_a not in MODEL_A:
        model_a = PRESETS[mode][0]
    if model_b not in MODEL_B:
        model_b = PRESETS[mode][1]
    thresholds = default_thresholds()
    stored = rows.get("thresholds") or {}
    if isinstance(stored, dict):
        for group, values in stored.items():
            if group in thresholds and isinstance(values, dict):
                thresholds[group].update(values)
    return {"mode": mode, "model_a": model_a, "model_b": model_b, "thresholds": thresholds}


def get_state(s: Session, fresh: bool = False) -> dict[str, Any]:
    """Текущие настройки. Кэш на несколько секунд: воркеры читают их на каждом кадре."""
    global _cache, _cache_at
    with _lock:
        if not fresh and _cache is not None and time.monotonic() - _cache_at < _TTL:
            return copy.deepcopy(_cache)
    state = _load(s)
    with _lock:
        _cache, _cache_at = state, time.monotonic()
    return copy.deepcopy(state)


def invalidate() -> None:
    global _cache
    with _lock:
        _cache = None


def mode_for(model_a: str, model_b: str) -> str:
    for mode, pair in PRESETS.items():
        if pair == (model_a, model_b):
            return mode
    local_a = model_a == "yolo"
    local_b = model_b in ("siglip", "local_vlm")
    if local_a and local_b:
        return "local"
    if not local_a and not local_b:
        return "external"
    return "hybrid"


def _validate_thresholds(patch: Any, current: dict[str, dict]) -> dict[str, dict]:
    if not isinstance(patch, dict):
        raise ValueError("thresholds: ожидается объект {группа: {порог: значение}}")
    out: dict[str, dict] = {}
    for group, values in patch.items():
        if group not in current:
            raise ValueError(f"thresholds: неизвестная группа «{group}» (есть: {', '.join(current)})")
        if not isinstance(values, dict):
            raise ValueError(f"thresholds.{group}: ожидается объект")
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if group == "pipeline" and key == "clock":
                if value not in CLOCK_MODES:
                    raise ValueError(f"thresholds.pipeline.clock: одно из {', '.join(CLOCK_MODES)}")
                clean[key] = value
                continue
            if group == "stage" and key == "mask_mode":
                if value not in MASK_MODES:
                    raise ValueError(f"thresholds.stage.mask_mode: одно из {', '.join(MASK_MODES)}")
                clean[key] = value
                continue
            if group != "equipment" and key not in current[group]:
                raise ValueError(f"thresholds.{group}: неизвестный порог «{key}»")
            if isinstance(value, bool):
                clean[key] = value
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"thresholds.{group}.{key}: ожидается число")
            if value < 0:
                raise ValueError(f"thresholds.{group}.{key}: не может быть отрицательным")
            clean[key] = value
        if group == "stage":
            merged = {**current["stage"], **clean}
            if not merged["no_thr"] < merged["yes_thr"] <= 1:
                raise ValueError("thresholds.stage: нужно no_thr < yes_thr ≤ 1")
        out[group] = clean
    return out


def update(s: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Применить изменения из PUT /api/settings. ValueError — невалидный ввод (→ 400)."""
    if not isinstance(payload, dict):
        raise ValueError("ожидается JSON-объект")
    current = _load(s)
    model_a, model_b = current["model_a"], current["model_b"]

    if "mode" in payload and payload["mode"] is not None:
        mode = payload["mode"]
        if mode not in PRESETS:
            raise ValueError(f"mode: одно из {', '.join(PRESETS)}")
        model_a, model_b = PRESETS[mode]
    if payload.get("model_a") is not None:
        if payload["model_a"] not in MODEL_A:
            raise ValueError(f"model_a: одно из {', '.join(MODEL_A)}")
        model_a = payload["model_a"]
    if payload.get("model_b") is not None:
        if payload["model_b"] not in MODEL_B:
            raise ValueError(f"model_b: одно из {', '.join(MODEL_B)}")
        model_b = payload["model_b"]

    stored_thr: dict[str, dict] = {}
    row = s.get(Setting, "thresholds")
    if row is not None and isinstance(row.value, dict):
        stored_thr = {k: dict(v) for k, v in row.value.items() if isinstance(v, dict)}
    if payload.get("thresholds") is not None:
        patch = _validate_thresholds(payload["thresholds"], current["thresholds"])
        for group, values in patch.items():
            stored_thr.setdefault(group, {}).update(values)

    mode = mode_for(model_a, model_b)
    for key, value in (("mode", mode), ("model_a", model_a), ("model_b", model_b),
                       ("thresholds", stored_thr)):
        row = s.get(Setting, key)
        if row is None:
            s.add(Setting(key=key, value=value))
        else:
            row.value = value
    s.commit()
    invalidate()
    return get_state(s, fresh=True)


def classes(model_a: str) -> list[dict[str, Any]]:
    """Словарь техники с пометкой, умеет ли её текущий детектор.

    Детектор может сообщить свои классы атрибутом `classes` / `supported_classes`;
    если не сообщает — GLM считаем знающим все 21 тип, YOLO — восемь из ТЗ.
    """
    from core import taxonomy

    supported: set[str] | None = None
    det = providers.registry.peek("detector", model_a)
    for attr in ("supported_classes", "classes"):
        value = getattr(det, attr, None) if det is not None else None
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001
                value = None
        if value:
            supported = set(value)
            break
    if supported is None:
        supported = set(taxonomy.equipment()) if model_a == "glm" else set(taxonomy.TZ_EQUIPMENT)
    return [{"key": e.key, "name": e.name, "tz": e.tz, "supported": e.key in supported}
            for e in taxonomy.equipment().values()]
