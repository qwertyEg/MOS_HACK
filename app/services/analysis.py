"""Разбор одиночного снимка без сохранения: /api/detect и /api/analyze.

«Проверить снимок» — главный сценарий для жюри: перетащил фото → рамки
техники и этап за секунды, с выбором провайдера. Здесь нет трекинга (не с чем
сравнивать), поэтому activity у рамок — UNKNOWN; для кадра из БД /api/detect
отдаёт уже посчитанные трекером поля.
"""
from __future__ import annotations

import base64
import datetime as dt
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app import storage
from app.models import Frame
from app.services import adapters, providers
from app.services import settings as settings_svc
from app.services.pipeline import assess_quality
from app.services.providers import registry
from core import contracts as c
from core import taxonomy


def adhoc_frame(img: np.ndarray, name: str = "upload") -> c.FrameInfo:
    h, w = img.shape[:2]
    return c.FrameInfo(frame_id=name, camera_id="adhoc", site_id="adhoc",
                       captured_at=dt.datetime.now(dt.UTC), width=w, height=h)


def detection_json(d: c.Detection) -> dict[str, Any]:
    """Контракт PLAN §4.3 + подписи для UI (поля контракта не трогаем, только добавляем)."""
    out = d.to_contract()
    out["name"] = taxonomy.equipment_name(d.cls)
    out["source"] = d.source
    out["appearance_delta"] = round(float(d.appearance_delta or 0.0), 3)
    return out


def annotate(img: np.ndarray, detections: list[c.Detection]) -> np.ndarray:
    """Рамки на кадре: core.equipment.draw, если есть, иначе простые рамки OpenCV
    (кириллицу OpenCV не рисует — подписи ключами)."""
    draw = providers.optional_module("core.equipment.draw")
    if draw is not None:
        try:
            return draw.annotate(img, detections, labels_ru=True)
        except Exception:  # noqa: BLE001 — картинка важнее красоты подписи
            pass
    out = img.copy()
    thick = max(2, round(max(img.shape[:2]) / 500))
    for d in detections:
        x, y, w, h = (int(round(v)) for v in d.bbox)
        color = (0, 200, 0) if d.activity == c.Activity.WORKING else (0, 170, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
        label = f"{d.cls} {d.conf:.2f}"
        cv2.putText(out, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick / 2 + 0.2,
                    color, max(1, thick - 1), cv2.LINE_AA)
    return out


def jpeg_data_url(img: np.ndarray, max_side: int = 1280) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def detect_image(img: np.ndarray, model_a: str) -> tuple[list[c.Detection], float]:
    """→ (рамки, мс). ProviderUnavailable — провайдер не готов (→ 503)."""
    detector = registry.require("detector", model_a)
    t0 = time.perf_counter()
    with registry.call_lock("detector", model_a):
        dets = detector.detect(img, adhoc_frame(img))
    return list(dets), (time.perf_counter() - t0) * 1000


def stored_detections(s: Session, fr: Frame, preferred: str) -> list[c.Detection] | None:
    """Детекции кадра из БД (с полями трекера) или None, если модель А его не видела."""
    if not fr.processed_a:
        return None
    rows = adapters.detections_by_frame(s, [fr.id], preferred).get(fr.id, [])
    return adapters.contract_detections(s, rows)


def load_frame_image(fr: Frame) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(storage.get().get(fr.key), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("файл кадра не читается")
    return img


def resolve_models(s: Session, provider: str | None, model_a: str | None, model_b: str | None) -> tuple[str, str]:
    state = settings_svc.get_state(s)
    a, b = state["model_a"], state["model_b"]
    if provider:
        if provider not in settings_svc.PRESETS:
            raise ValueError(f"provider: одно из {', '.join(settings_svc.PRESETS)}")
        a, b = settings_svc.PRESETS[provider]
    if model_a:
        if model_a not in providers.MODEL_A:
            raise ValueError(f"model_a: одно из {', '.join(providers.MODEL_A)}")
        a = model_a
    if model_b:
        if model_b not in providers.MODEL_B:
            raise ValueError(f"model_b: одно из {', '.join(providers.MODEL_B)}")
        b = model_b
    return a, b


def analyze_image(s: Session, img: np.ndarray, model_a: str, model_b: str,
                  annotate_image: bool = True, force_stage: bool = False) -> dict[str, Any]:
    t_start = time.perf_counter()
    errors: list[str] = []
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    try:
        quality = assess_quality(img, dt.datetime.now(dt.UTC))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"оценка качества: {exc}")
        from app.services.pipeline import basic_quality
        quality = basic_quality(img)
    timings["quality_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    detections: list[c.Detection] = []
    try:
        detections, ms = detect_image(img, model_a)
        timings["detect_ms"] = round(ms, 1)
    except providers.ProviderUnavailable as exc:
        errors.append(f"модель А ({model_a}) не готова: {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"модель А: {type(exc).__name__}: {exc}")

    checklist = stage = None
    if quality.usable_for_stage or force_stage:
        try:
            classifier = registry.require("classifier", model_b)
            t0 = time.perf_counter()
            with registry.call_lock("classifier", model_b):
                res = classifier.assess(img, adhoc_frame(img), keys=None, context=None)
            timings["stage_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            review = settings_svc.get_state(s)["thresholds"]["stage"]["unsure_review_ratio"]
            checklist = {
                "answers": {k: (v.value if hasattr(v, "value") else str(v)) for k, v in res.answers.items()},
                "scores": adapters.jsonable(res.scores),
                "unsure_ratio": round(res.unsure_ratio, 3),
                "needs_review": res.unsure_ratio > review,
                "model": res.model, "latency_ms": res.latency_ms, "cost_usd": res.cost_usd,
                "equipment_hint": adapters.jsonable(res.equipment_hint),
            }
            stage = _stage_from_answers(res, errors)
        except providers.ProviderUnavailable as exc:
            errors.append(f"модель Б ({model_b}) не готова: {exc.reason}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"модель Б: {type(exc).__name__}: {exc}")
    else:
        why = quality.reject_reason or ("ночной кадр" if quality.is_night else "кадр не годен")
        errors.append(f"модель Б не применяется: {why} (передайте force_stage=1, чтобы спросить всё равно)")

    h, w = img.shape[:2]
    out = {
        "model_a": model_a, "model_b": model_b,
        "mode": settings_svc.mode_for(model_a, model_b),
        "image": {"width": w, "height": h},
        "quality": {"quality_ok": quality.quality_ok, "is_night": quality.is_night,
                    "weather": adapters.jsonable(quality.weather), "reject_reason": quality.reject_reason,
                    "blur": adapters.jsonable(quality.blur), "brightness": adapters.jsonable(quality.brightness),
                    "usable_for_stage": quality.usable_for_stage},
        "detections": [detection_json(d) for d in detections],
        "counts": dict(Counter(d.cls for d in detections)),
        "checklist": checklist,
        "stage": stage,
        "errors": errors,
    }
    if annotate_image:
        out["annotated"] = jpeg_data_url(annotate(img, detections))
    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    out["timings"] = timings
    return out


def _stage_from_answers(res: c.ChecklistResult, errors: list[str]) -> dict[str, Any] | None:
    scoring = providers.optional_module("core.stage.scoring")
    if scoring is None:
        errors.append("этап не оценён: модуль core.stage.scoring не подключён")
        return None
    scores = scoring.evaluate(res.answers)
    front = getattr(scores, "front", None)
    return {
        "front": front,
        "name": taxonomy.stage_name(front) if front is not None else None,
        "evidence": adapters.jsonable(getattr(scores, "stage_evidence", {}) or {}),
        "substages": adapters.jsonable(getattr(scores, "substages", {}) or {}),
        "progress": adapters.jsonable(getattr(scores, "progress", {}) or {}),
        "stage_likelihood": adapters.jsonable(res.stage_likelihood or {}),
    }

