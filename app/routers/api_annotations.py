"""Ручная разметка техники и её выгрузка (требование 3): /api/detections, /api/units, датасет.

Правка сохраняется как факт в таблице annotations и сразу видна: кадр
обновляется на месте, техника площадки перепрогоняется по сохранённым ответам
детектора (без детектора), затем пересчитывается аналитика. Переанализ правки
не стирает — накладывает их заново. Любое действие отменяется целиком по batch.

Тяжёлая работа (перепрогон, сборка zip) идёт в пуле потоков: обработчики либо
синхронные, либо отдают её в run_in_threadpool — цикл событий не замерзает.
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
import zipfile
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from app import auth, db
from app.config import settings
from app.models import Camera, Detection, EquipmentUnit, Frame, Site
from app.routers.common import bad, json_body, not_found, require_obj
from app.services import adapters, annotations, views
from app.services.queue import replayer

router = APIRouter(prefix="/api", tags=["ручная разметка"])


def _replay(ch: annotations.Change) -> dict[str, Any]:
    """Перепрогон техники после правки: склейку на небольшом объекте — сразу (оператор
    ждёт итог в ответе), остальное — в фоне с паузой (правки идут пачкой)."""
    if not ch.replay:
        return replayer.state(ch.site_id)
    if ch.sync:
        with db.session() as s:
            small = annotations.processed_frames(s, ch.site_id) <= annotations.SYNC_REPLAY_FRAMES
        if small and replayer.state(ch.site_id)["state"] != "running":
            replayer.run(ch.site_id)
            return replayer.state(ch.site_id)
    replayer.request(ch.site_id)
    return replayer.state(ch.site_id)


def _respond(fn: Callable[..., annotations.Change], *args) -> dict[str, Any]:
    with db.session() as s:
        try:
            ch = fn(s, *args)
        except LookupError as exc:
            raise not_found(str(exc)) from None
        except ValueError as exc:
            raise bad(str(exc)) from None
        except annotations.Stale as exc:
            raise HTTPException(409, str(exc)) from None
    out = ch.to_json()
    out["replay"] = _replay(ch)
    if ch.frame_id is not None:
        with db.session() as s:
            fr = s.get(Frame, ch.frame_id)
            if fr is not None:
                out["frame"] = views.frame_detail(s, fr)
    return out


# --------------------------------------------------------------------------
# рамки кадра
# --------------------------------------------------------------------------

@router.patch("/detections/{det_id}")
async def patch_detection(det_id: int, request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """{cls, scope: box|unit} — сменить класс рамки (или всей машины);
    {deleted: true, scope: frame|camera} — удалить ложную рамку (или место камеры)."""
    body = require_obj(await json_body(request))
    return await run_in_threadpool(_respond, annotations.patch_detection, det_id, body, user)


@router.delete("/detections/{det_id}")
def delete_detection(det_id: int, scope: str = "frame", user: str = Depends(auth.require_api_user)) -> dict:
    return _respond(annotations.patch_detection, det_id, {"deleted": True, "scope": scope}, user)


@router.post("/frames/{frame_id}/detections", status_code=201)
async def add_detection(frame_id: int, request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """Дорисовать пропущенную рамку: {cls, bbox: [x, y, w, h]} в пикселях кадра."""
    body = require_obj(await json_body(request))
    return await run_in_threadpool(_respond, annotations.add_box, frame_id, body, user)


@router.post("/frames/{frame_id}/verify")
async def verify_frame(frame_id: int, request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """{verified: true} — разметка кадра проверена (кадр — проверенный пример датасета)."""
    body = require_obj(await json_body(request, default={}))
    return await run_in_threadpool(_respond, annotations.verify_frame, frame_id, body, user)


@router.get("/frames/{frame_id}/annotations")
def frame_annotations(frame_id: int, _user: str = Depends(auth.require_api_user)) -> dict:
    with db.session() as s:
        fr = s.get(Frame, frame_id)
        if fr is None:
            raise not_found(f"кадр {frame_id} не найден")
        return annotations.frame_summary(s, fr)


# --------------------------------------------------------------------------
# машины (единицы техники)
# --------------------------------------------------------------------------

def _with_unit(fn: Callable) -> Callable:
    def call(s, unit_id, body, user):
        return fn(s, annotations.unit_from_body(s, unit_id, body), body, user)
    return call


@router.post("/units/merge")
async def merge_units(request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """Склеить машины: {unit_ids: [..], target_id?, cls?} — это одна машина."""
    body = require_obj(await json_body(request))
    return await run_in_threadpool(_respond, annotations.merge_units, body, user)


@router.post("/units/{unit_id}/split")
async def split_unit(unit_id: int, request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """Разделить машину: {frame_id, cls?} — с этого кадра (на его камере) это другая машина."""
    body = require_obj(await json_body(request))
    return await run_in_threadpool(_respond, _with_unit(annotations.split_unit), unit_id, body, user)


@router.patch("/units/{unit_id}")
async def patch_unit(unit_id: int, request: Request, user: str = Depends(auth.require_api_user)) -> dict:
    """{cls} — сменить тип машины целиком (рамки и моточасы)."""
    body = require_obj(await json_body(request))
    return await run_in_threadpool(_respond, annotations.patch_unit, unit_id, body, user)


@router.delete("/units/{unit_id}")
def delete_unit(unit_id: int, hide_place: bool | None = None, uid: str | None = None, site_id: int | None = None,
                user: str = Depends(auth.require_api_user)) -> dict:
    """«Это не техника»: рамки машины удаляются; hide_place — и её место на камерах.
    uid/site_id — чтобы не удалить другую машину, если список техники уже пересобран."""
    body: dict[str, Any] = {"uid": uid, "site_id": site_id}
    if hide_place is not None:
        body["hide_place"] = hide_place
    return _respond(_with_unit(annotations.delete_unit), unit_id, body, user)


@router.get("/units/{unit_id}/timeline")
def unit_timeline(unit_id: int, limit: int = 1000, _user: str = Depends(auth.require_api_user)) -> dict:
    """Все кадры машины по времени — выбрать, с какого кадра её разделить."""
    with db.session() as s:
        unit = s.get(EquipmentUnit, unit_id)
        if unit is None:
            raise not_found(f"машина {unit_id} не найдена")
        rows = s.execute(select(Detection, Frame).join(Frame, Frame.id == Detection.frame_id)
                         .where(Detection.unit_id == unit.id).order_by(Frame.captured_at, Detection.id)
                         .limit(max(1, min(limit, 5000)))).all()
        cams = dict(s.execute(select(Camera.id, Camera.name).where(Camera.site_id == unit.site_id)).all())
        return {"unit": views.unit_json(unit), "items": [
            {"frame_id": fr.id, "detection_id": d.id, "camera_id": fr.camera_id, "camera_name": cams.get(fr.camera_id),
             "captured_at": adapters.iso(fr.captured_at), "bbox": [d.x, d.y, d.w, d.h], "cls": d.cls,
             "activity": d.activity, "manual": bool((d.extra or {}).get(annotations.MANUAL)),
             "url": views.frame_urls(fr)["url"]} for d, fr in rows]}


# --------------------------------------------------------------------------
# журнал, отмена, перепрогон
# --------------------------------------------------------------------------

@router.get("/sites/{site_id}/annotations")
def site_annotations(site_id: int, limit: int = 100, _user: str = Depends(auth.require_api_user)) -> dict:
    """Журнал ручных правок объекта (что, кто, когда) + сводка + состояние перепрогона."""
    with db.session() as s:
        if s.get(Site, site_id) is None:
            raise not_found(f"объект {site_id} не найден")
        out = annotations.journal(s, site_id, limit)
    out["replay"] = replayer.state(site_id)
    return out


@router.get("/sites/{site_id}/annotations/status")
def annotations_status(site_id: int, _user: str = Depends(auth.require_api_user)) -> dict:
    return replayer.state(site_id)


@router.post("/sites/{site_id}/annotations/replay", status_code=202)
def replay_site(site_id: int, _user: str = Depends(auth.require_api_user)) -> dict:
    """Перепрогнать технику объекта по сохранённым рамкам с правками (без детектора)."""
    with db.session() as s:
        if s.get(Site, site_id) is None:
            raise not_found(f"объект {site_id} не найден")
    replayer.request(site_id)
    return replayer.state(site_id)


@router.delete("/annotations/batches/{batch}")
def undo_batch(batch: str, _user: str = Depends(auth.require_api_user)) -> dict:
    """Отменить действие оператора целиком (все строки пачки)."""
    return _respond(annotations.undo, batch)


# --------------------------------------------------------------------------
# датасет для дообучения
# --------------------------------------------------------------------------

def _export(site_ids: list[int], scope: str, val: float) -> FileResponse:
    tmp_dir = settings.path(settings.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="dataset-", suffix=".zip", dir=str(tmp_dir))
    os.close(fd)
    try:
        with db.session() as s, zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            stats = annotations.export_dataset(s, site_ids, zf, scope=scope, val_ratio=val)
    except LookupError as exc:
        os.unlink(path)
        raise not_found(str(exc)) from None
    except ValueError as exc:
        os.unlink(path)
        raise bad(str(exc)) from None
    except Exception:
        os.unlink(path)
        raise
    name = f"stroyvzor_dataset_{'_'.join(map(str, site_ids))}_{scope}_{dt.datetime.now():%Y%m%d_%H%M}.zip"
    return FileResponse(path, media_type="application/zip", filename=name,
                        headers={"X-Dataset-Frames": str(stats["frames"]), "X-Dataset-Boxes": str(stats["boxes"]),
                                 "Cache-Control": "no-store"},
                        background=BackgroundTask(os.unlink, path))


@router.get("/sites/{site_id}/dataset.zip")
def site_dataset(site_id: int, scope: str = "reviewed", val: float = 0.2,
                 _user: str = Depends(auth.require_api_user)) -> FileResponse:
    """Размеченные кадры объекта в формате YOLO (images/, labels/, data.yaml, manifest.json).
    scope=reviewed — только проверенные человеком кадры, all — все с псевдоразметкой модели."""
    return _export([site_id], scope, val)


@router.get("/dataset.zip")
def dataset(sites: str = "", scope: str = "reviewed", val: float = 0.2,
            _user: str = Depends(auth.require_api_user)) -> FileResponse:
    """То же по нескольким объектам: ?sites=1,2 (пусто — все объекты)."""
    if sites.strip():
        try:
            ids = [int(x) for x in sites.split(",") if x.strip()]
        except ValueError:
            raise bad("sites: номера объектов через запятую") from None
    else:
        with db.session() as s:
            ids = list(s.scalars(select(Site.id)))
    if not ids:
        raise not_found("объектов нет")
    return _export(ids, scope, val)
