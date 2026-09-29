"""Динамическая маска камеры для веба: состояние, история, ручная кисть, сброс к автоматике.

Кисть оператора — `DynamicMask.set_background(bitmap, lock=True)`: нарисованное
неприкосновенно для автоматики (сжатие и «техника открывает фон» его не трогают).
PNG кисти хранится отдельно (`CameraState.initial_mask_key`) — переанализ объекта
состояние маски сбрасывает, а ручную маску подхватывает при первом же кадре.

Сброс к автоматике пересобирает маску по истории камеры — теми же годными
кадрами и рамками модели А, что видел конвейер, в том же порядке. Иначе у
архивной камеры маска после сброса не собралась бы вовсе: новых кадров нет.

После любой смены маски модель Б переспрашивается по кадрам, которые она уже
разбирала (`pipeline.requeue_stage_frames`) — каждый видит маску своего времени,
хронология этапов пересчитывается сама.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import db, storage
from app.models import Camera, CameraState, Frame, Site
from app.services import pipeline
from core.contracts import Weather

log = logging.getLogger(__name__)

_rebuild: dict[int, dict[str, Any]] = {}      # camera_id → {done, total, error}
_rebuild_guard = threading.Lock()


def _iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=dt.UTC)).isoformat()
    return str(value)


def rebuilding(camera_id: int) -> dict[str, Any] | None:
    with _rebuild_guard:
        st = _rebuild.get(camera_id)
        return dict(st) if st else None


def info(s: Session, cam: Camera) -> dict[str, Any]:
    """Честное состояние маски для интерфейса: построена ли, откуда, сколько накоплено, история."""
    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    mask = pipeline.load_mask(s, cam)
    out: dict[str, Any] = {
        "camera_id": cam.id, "initialized": False, "source": "none", "masked_ratio": 0.0,
        "locked_ratio": 0.0, "manual": bool(state and state.initial_mask_key),
        "frames": 0, "windows": 0, "history": [], "bootstrap": None, "init": None,
        "updated_at": _iso(state.updated_at) if state else None, "rebuilding": rebuilding(cam.id),
        "url": f"/api/cameras/{cam.id}/mask.png",
    }
    if mask is None:
        return out
    out.update({
        "initialized": bool(getattr(mask, "initialized", False)),
        "source": getattr(mask, "source", "auto"),
        "masked_ratio": round(float(mask.masked_ratio), 4),
        "locked_ratio": round(float(getattr(mask, "locked_ratio", 0.0)), 4),
        "frames": int(getattr(mask, "frames_seen", 0)),
        "windows": int(getattr(mask, "windows", 0)),
        "init": getattr(mask, "init_info", None) or None,
    })
    boot = getattr(mask, "bootstrap", None)
    if callable(boot) and not out["initialized"]:
        out["bootstrap"] = boot()
    hist = getattr(mask, "history", None) or []
    out["history"] = [{"i": i, "from": _iso(h.get("from")) or None, "event": h.get("event"),
                       "ratio": h.get("ratio")} for i, h in enumerate(hist)]
    return out


def background(mask: Any, at: dt.datetime | None = None, index: int | None = None) -> np.ndarray | None:
    """Фон на рабочем разрешении маски: текущий, на момент `at` или снимок истории `index`."""
    if mask is None or not getattr(mask, "initialized", True):
        return None
    if index is not None and hasattr(mask, "history_background"):
        hist = getattr(mask, "history", []) or []
        if not 0 <= index < len(hist):
            raise IndexError(index)
        return mask.history_background(index)
    if at is not None and hasattr(mask, "background_at"):
        return mask.background_at(at)
    return ~np.asarray(mask.visible(), dtype=bool)


def overlay_png(bg: np.ndarray, size: tuple[int, int] | None, style: str = "overlay") -> bytes:
    """overlay — погашенный фон полупрозрачным красным для слоя поверх кадра;
    bitmap — белое = фон (то, что принимает PUT: кисть начинает с текущей маски)."""
    if size and (bg.shape[1], bg.shape[0]) != size:
        bg = cv2.resize(bg.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    if style == "bitmap":
        img = bg.astype(np.uint8) * 255
    else:
        img = np.zeros(bg.shape[:2] + (4,), np.uint8)
        img[bg] = (40, 40, 220, 120)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def decode_bitmap(data: bytes) -> np.ndarray:
    """PNG/JPEG кисти → bool «фон». Прозрачность — тоже кисть: непрозрачное (alpha > 0) и
    светлое — фон; так принимается и чёрно-белая маска, и слой с альфой из canvas."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("маска не читается как изображение (нужен PNG)")
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., 3] > 127
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img > 127


def set_manual(s: Session, cam: Camera, bitmap: np.ndarray, reanalyze: bool = True) -> dict[str, Any]:
    """Сохранить маску кисти: неприкосновенна для автоматики, действует на все кадры камеры."""
    cls = pipeline.mask_class()
    if cls is None:
        raise RuntimeError("модуль маски (core.stage.mask) не подключён")
    ok, png = cv2.imencode(".png", bitmap.astype(np.uint8) * 255)
    with pipeline.mask_lock(cam.id):
        state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
        if state is None:
            state = CameraState(camera_id=cam.id)
            s.add(state)
        state.initial_mask_key = storage.get().put(pipeline.manual_mask_key(cam.id), png.tobytes(), "image/png")
        mask = pipeline.load_mask(s, cam)
        if mask is None:
            shape = (cam.image_h, cam.image_w) if cam.image_h and cam.image_w else bitmap.shape[:2]
            site = s.get(Site, cam.site_id)
            mask = pipeline.new_mask(cls, shape, site, dt.datetime.now(dt.UTC), None)
        mask.set_background(bitmap, lock=True)
        pipeline.save_mask(s, cam, mask)
        s.commit()
    job_id, n = (pipeline.requeue_stage_frames(s, cam, kind="mask", message="ручная маска — модель Б заново")
                 if reanalyze else (None, 0))
    return {**info(s, cam), "job_id": job_id, "requeued": n}


def reset_to_auto(s: Session, cam: Camera) -> dict[str, Any]:
    """Снять ручную маску и пересобрать автоматическую по истории камеры (в фоне)."""
    with _rebuild_guard:
        if cam.id in _rebuild and _rebuild[cam.id].get("running"):
            return {**info(s, cam), "job_id": None}
        _rebuild[cam.id] = {"running": True, "done": 0, "total": 0, "error": None}
    with pipeline.mask_lock(cam.id):
        state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
        if state is not None:
            state.initial_mask_key = ""
            state.mask_key = ""
            state.masked_ratio = 0.0
            state.retained = 1.0
            state.stage_mask_ratio = None
            s.commit()
        pipeline._masks.pop(cam.id, None)
    threading.Thread(target=_rebuild_thread, args=(cam.id,), name=f"mask-rebuild-{cam.id}", daemon=True).start()
    return {**info(s, cam), "job_id": None}


def _weather(value: str | None) -> Weather | None:
    try:
        return Weather(value) if value else None
    except ValueError:
        return None


def rebuild_from_history(camera_id: int, progress: dict[str, Any] | None = None) -> Any | None:
    """Прогнать годные кадры камеры (те, что видела маска: meta.mask_done) через новую маску."""
    cls = pipeline.mask_class()
    if cls is None:
        return None
    with db.session() as s:
        cam = s.get(Camera, camera_id)
        if cam is None:
            return None
        site = s.get(Site, cam.site_id)
        rows = [(f.id, f.key, f.captured_at, f.weather) for f in
                s.scalars(select(Frame).where(Frame.camera_id == camera_id).order_by(Frame.captured_at))
                if (f.meta or {}).get("mask_done")]
        if progress is not None:
            progress["total"] = len(rows)
        with pipeline.mask_lock(camera_id):
            mask = None
            params: set[str] = set()
            for i, (fid, key, when, weather) in enumerate(rows):
                try:
                    img = cv2.imdecode(np.frombuffer(storage.get().get(key), np.uint8), cv2.IMREAD_COLOR)
                except Exception as exc:  # noqa: BLE001 — пропавший файл не должен рвать пересборку
                    log.warning("пересборка маски: кадр %s не читается (%s)", fid, exc)
                    img = None
                if img is None:
                    continue
                if mask is None:
                    mask = pipeline.new_mask(cls, img.shape[:2], site, when, None)
                    params = pipeline._params(mask.update)
                kw: dict[str, Any] = {}
                if "weather" in params and _weather(weather) is not None:
                    kw["weather"] = _weather(weather)
                if "boxes" in params:
                    kw["boxes"] = pipeline.frame_boxes(s, fid)
                mask.update(img, when, **kw)
                if progress is not None:
                    progress["done"] = i + 1
            if mask is not None:
                pipeline.save_mask(s, cam, mask)
                s.commit()
        return mask


def _rebuild_thread(camera_id: int) -> None:
    with _rebuild_guard:
        progress = _rebuild.setdefault(camera_id, {"running": True, "done": 0, "total": 0, "error": None})
    try:
        mask = rebuild_from_history(camera_id, progress)
        if mask is not None:
            with db.session() as s:
                cam = s.get(Camera, camera_id)
                if cam is not None:
                    pipeline.requeue_stage_frames(s, cam, kind="mask",
                                                  message="маска пересобрана по истории — модель Б заново")
    except Exception as exc:  # noqa: BLE001
        log.exception("пересборка маски камеры %s", camera_id)
        progress["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        progress["running"] = False
    from app.services.queue import recomputer
    with db.session() as s:
        cam = s.get(Camera, camera_id)
        if cam is not None:
            recomputer.request(cam.site_id)


def preview(s: Session, cam: Camera, fr: Frame, mode: str) -> tuple[np.ndarray, bool]:
    """Кадр так, как его видит модель Б: маска на момент кадра, режим гашения классификатора."""
    from core.stage.mask import masked_for_model

    img = cv2.imdecode(np.frombuffer(storage.get().get(fr.key), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("файл кадра не читается")
    mask = pipeline.mask_for_frame(pipeline.load_mask(s, cam), fr.captured_at)
    return masked_for_model(img, {"mask": mask, "mask_mode": mode})
