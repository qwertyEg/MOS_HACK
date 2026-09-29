"""Камеры: CRUD, калибровка, зоны, загрузка файлов, лента кадров, маска, поток (simcam)."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import auth, netutil
from app.config import settings
from app.db import get_session
from app.models import Camera, CameraState, Frame, Site, Zone
from app.routers.common import bad, get_or_404, json_body, not_found, num_field, points_field, require_obj, str_field
from app.services import adapters, ingest, masks, pipeline, providers, sites, views
from app.services import settings as settings_svc

router = APIRouter(prefix="/api", tags=["камеры"], dependencies=[Depends(auth.require_api_user)])

CAMERA_KINDS = ("upload", "folder", "stream", "video")
ZONE_KINDS = ("work", "parking", "storage", "restricted")


def _homography(value) -> list[list[float]] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 3
            or any(not isinstance(r, list) or len(r) != 3 for r in value)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for r in value for v in r)):
        raise bad("homography: матрица 3×3 чисел или null")
    return [[float(v) for v in r] for r in value]


def _camera_fields(body: dict, partial: bool) -> dict:
    out: dict = {}
    if not partial or "name" in body:
        out["name"] = str_field(body, "name", required=True, max_len=128)
    if not partial or "kind" in body:
        kind = str_field(body, "kind", default="upload") or "upload"
        if kind not in CAMERA_KINDS:
            raise bad(f"kind: одно из {', '.join(CAMERA_KINDS)}")
        out["kind"] = kind
    if "interval_min" in body:
        out["interval_min"] = num_field(body, "interval_min", lo=1, hi=1440, integer=True)
    elif not partial:
        out["interval_min"] = settings.default_interval_min
    if "source_uri" in body:
        out["source_uri"] = str_field(body, "source_uri", max_len=1000)
    if "homography" in body:
        out["homography"] = _homography(body["homography"])
    return out


@router.get("/sites/{site_id}/cameras")
def list_cameras(site_id: int, s: Session = Depends(get_session)) -> list[dict]:
    site = get_or_404(s, Site, site_id, "объект")
    return [views.camera_json(s, cam) for cam in s.scalars(select(Camera).where(Camera.site_id == site.id)
                                                           .order_by(Camera.id))]


@router.post("/sites/{site_id}/cameras", status_code=201)
async def create_camera(site_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    cam = Camera(site_id=site.id, ingest_key=auth.new_ingest_key(),
                 **_camera_fields(require_obj(await json_body(request)), partial=False))
    s.add(cam)
    s.commit()
    return views.camera_json(s, cam)


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: int, s: Session = Depends(get_session)) -> dict:
    return views.camera_json(s, get_or_404(s, Camera, camera_id, "камера"))


@router.patch("/cameras/{camera_id}")
async def patch_camera(camera_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    body = require_obj(await json_body(request))
    fields = _camera_fields(body, partial=True)
    for k, v in fields.items():
        setattr(cam, k, v)
    if "homography" in fields and fields["homography"] is None:
        cam.calib_points = None
    if body.get("regenerate_key"):
        cam.ingest_key = auth.new_ingest_key()
    s.commit()
    if "homography" in fields:
        pipeline.reset_caches(cam.site_id)
    return views.camera_json(s, cam)


@router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: int, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    from app.services.queue import frame_queue
    frame_queue.cancel([cam.id])
    frame_queue.wait_cameras([cam.id], timeout=60)
    site_id = cam.site_id
    sites.delete_camera(s, cam.id)
    pipeline.reset_caches(site_id, [camera_id])
    return {"ok": True, "id": camera_id}


@router.post("/cameras/{camera_id}/calibration")
async def calibrate(camera_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    """4+ точки на кадре ↔ те же точки на плане площадки (метры) → гомография.
    Без калибровки камера считается независимой (без слияния с соседними)."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    body = require_obj(await json_body(request))
    image_pts = points_field(body, "image_points", 4)
    site_pts = points_field(body, "site_points", 4)
    if len(image_pts) != len(site_pts):
        raise bad("image_points и site_points: число точек должно совпадать")
    try:
        fusion = providers.module("core.equipment.fusion")
    except ImportError:
        raise HTTPException(503, "калибровка недоступна: модуль core.equipment не подключён") from None
    try:
        H, err = fusion.homography_from_points(image_pts, site_pts)
    except ValueError as exc:
        raise bad(f"гомография не построена: {exc}") from None
    if H is None:
        raise bad("гомография не построена: точки вырождены (лежат на одной прямой?)")
    cam.homography = [[float(v) for v in row] for row in H]
    cam.calib_points = {"image_points": image_pts, "site_points": site_pts, "reproj_error": float(err),
                        "image_size": [cam.image_w, cam.image_h]}
    s.commit()
    pipeline.reset_caches(cam.site_id)
    return {"homography": cam.homography, "reproj_error": float(err), "points": len(image_pts),
            "camera": views.camera_json(s, cam)}


# --------------------------------------------------------------------------
# зоны
# --------------------------------------------------------------------------

@router.get("/cameras/{camera_id}/zones")
def list_zones(camera_id: int, s: Session = Depends(get_session)) -> list[dict]:
    cam = get_or_404(s, Camera, camera_id, "камера")
    return [views.zone_json(z) for z in s.scalars(select(Zone).where(Zone.camera_id == cam.id).order_by(Zone.id))]


@router.post("/cameras/{camera_id}/zones", status_code=201)
async def create_zone(camera_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    body = require_obj(await json_body(request))
    kind = str_field(body, "kind", default="work") or "work"
    if kind not in ZONE_KINDS:
        raise bad(f"kind: одно из {', '.join(ZONE_KINDS)}")
    zone = Zone(site_id=cam.site_id, camera_id=cam.id, name=str_field(body, "name", required=True, max_len=128),
                kind=kind, polygon=points_field(body, "polygon", 3))
    s.add(zone)
    s.commit()
    return views.zone_json(zone)


@router.delete("/cameras/{camera_id}/zones/{zone_id}")
def delete_zone(camera_id: int, zone_id: int, s: Session = Depends(get_session)) -> dict:
    zone = s.get(Zone, zone_id)
    if zone is None or zone.camera_id != camera_id:
        raise not_found(f"зона {zone_id} у камеры {camera_id} не найдена")
    s.delete(zone)
    s.commit()
    return {"ok": True, "id": zone_id}


@router.delete("/cameras/{camera_id}/zones")
def delete_zones(camera_id: int, zone_id: int | None = None, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    if zone_id is not None:
        return delete_zone(cam.id, zone_id, s)
    rows = list(s.scalars(select(Zone).where(Zone.camera_id == cam.id)))
    for z in rows:
        s.delete(z)
    s.commit()
    return {"ok": True, "deleted": len(rows)}


# --------------------------------------------------------------------------
# загрузка и кадры
# --------------------------------------------------------------------------

@router.post("/cameras/{camera_id}/upload", status_code=202)
async def upload(camera_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    """multipart: files (или file) — изображения, zip, видео; interval_min, start_at,
    video_mode (realtime|timelapse), step_s, every_n. Разбор — в фоне, прогресс —
    GET /api/jobs/{job_id}."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    site = s.get(Site, cam.site_id)
    form = await request.form(max_files=10000, max_fields=100)
    uploads = [f for f in form.getlist("files") + form.getlist("file") if hasattr(f, "filename")]
    if not uploads:
        raise bad("не передано ни одного файла (поле files)")
    unsupported = [f.filename for f in uploads if not ingest.classify(f.filename or "")]
    if len(unsupported) == len(uploads):
        heic = any(Path(n or "").suffix.lower() in (".heic", ".heif") for n in unsupported)
        raise bad("формат не поддерживается: " + ", ".join(unsupported[:10]) +
                  f". Поддерживаются: {', '.join(sorted(ingest.SUPPORTED_EXT))}" +
                  (". Снимки iPhone (HEIC) сохраните как JPEG" if heic else ""))

    def form_num(key: str, cast, lo, hi, default):
        raw = form.get(key)
        if raw in (None, ""):
            return default
        try:
            value = cast(raw)
        except ValueError:
            raise bad(f"{key}: ожидается число") from None
        if not lo <= value <= hi:
            raise bad(f"{key}: допустимо от {lo} до {hi}")
        return value

    interval = form_num("interval_min", int, 1, 1440, cam.interval_min or settings.default_interval_min)
    step_s = form_num("step_s", float, 0.04, 86400, None)
    every_n = form_num("every_n", int, 1, 100000, 1)
    video_mode = str(form.get("video_mode") or "realtime")
    if video_mode not in ingest.VIDEO_MODES:
        raise bad(f"video_mode: одно из {', '.join(ingest.VIDEO_MODES)}")
    try:
        start_at = ingest.parse_datetime_input(form.get("start_at"))
    except ValueError as exc:
        raise bad(f"start_at: {exc}") from None
    params = ingest.UploadParams(interval_min=interval,
                                 start_at=adapters.to_utc(start_at, adapters.site_tz(site)) if start_at else None,
                                 video_mode=video_mode, step_s=step_s, every_n=every_n)

    tmp = settings.path(settings.tmp_dir) / "uploads" / uuid.uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    paths: list[Path] = []
    try:
        for i, up in enumerate(uploads):
            if not ingest.classify(up.filename or ""):
                continue
            # Каждый файл — в свой подкаталог: имя сохраняется как есть (в нём
            # метка времени), а одноимённые файлы не затирают друг друга.
            target = tmp / f"{i:05d}" / (Path(up.filename).name or f"file{i}")
            target.parent.mkdir()
            with target.open("wb") as dst:
                while chunk := await up.read(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise HTTPException(413, f"загрузка больше {settings.max_upload_mb} МБ")
                    dst.write(chunk)
            paths.append(target)
    except HTTPException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    job = ingest.new_job(s, "upload", cam, message=f"{len(uploads)} файл(ов)")
    if unsupported:
        job.errors = [f"{n}: {ingest.unsupported_reason(n)}" for n in unsupported]
        job.skipped = len(unsupported)
        s.commit()
    ingest.start_job(job.id, cam.id, paths, params, cleanup_dir=tmp)
    return {"job_id": job.id, "state": "ingesting", "files": len(uploads)}


@router.get("/cameras/{camera_id}/frames")
def list_frames(camera_id: int, limit: int = 50, before: str | None = None, after: str | None = None,
                order: str = "desc", status: str | None = None, s: Session = Depends(get_session)) -> list[dict]:
    cam = get_or_404(s, Camera, camera_id, "камера")
    if order not in ("asc", "desc"):
        raise bad("order: asc | desc")
    limit = max(1, min(limit, 5000))
    q = select(Frame).where(Frame.camera_id == cam.id)
    try:
        b = ingest.parse_datetime_input(before)
        a = ingest.parse_datetime_input(after)
    except ValueError as exc:
        raise bad(str(exc)) from None
    tz = adapters.site_tz(s.get(Site, cam.site_id))
    if b is not None:
        q = q.where(Frame.captured_at < adapters.to_utc(b, tz))
    if a is not None:
        q = q.where(Frame.captured_at > adapters.to_utc(a, tz))
    if status:
        q = q.where(Frame.status == status)
    q = q.order_by(Frame.captured_at.desc() if order == "desc" else Frame.captured_at).limit(limit)
    frames = list(s.scalars(q))
    counts = views.detection_counts(s, [f.id for f in frames], settings_svc.get_state(s)["model_a"])
    return [views.frame_summary(f, counts.get(f.id, 0)) for f in frames]


# --------------------------------------------------------------------------
# маска
# --------------------------------------------------------------------------

MASK_PNG_LIMIT = 8 * 1024 * 1024


def _mask_or_404(s: Session, cam: Camera):
    mask = pipeline.load_mask(s, cam)
    if mask is None or not getattr(mask, "initialized", True):
        raise not_found("маска ещё не построена: она копится по дневным кадрам камеры")
    return mask


@router.get("/cameras/{camera_id}/mask")
def mask_info(camera_id: int, s: Session = Depends(get_session)) -> dict:
    """Состояние маски: построена ли, откуда (auto | manual), сколько накоплено до первой
    маски, история изменений (для ползунка эволюции)."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    return masks.info(s, cam)


@router.get("/cameras/{camera_id}/mask.png")
def mask_png(camera_id: int, at: str | None = None, i: int | None = None, style: str = "overlay",
             s: Session = Depends(get_session)) -> Response:
    """Слой маски: overlay — погашенный фон полупрозрачным красным поверх кадра;
    bitmap — белое = фон (исходник для кисти). at — маска на момент кадра (ISO),
    i — снимок истории."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    if style not in ("overlay", "bitmap"):
        raise bad("style: overlay | bitmap")
    mask = _mask_or_404(s, cam)
    try:
        when = ingest.parse_datetime_input(at)
    except ValueError as exc:
        raise bad(str(exc)) from None
    if when is not None:
        site = s.get(Site, cam.site_id)
        when = adapters.to_utc(when, adapters.site_tz(site))
    try:
        bg = masks.background(mask, when, i)
    except IndexError:
        raise not_found(f"снимка маски {i} нет") from None
    if bg is None:
        bg = np.zeros((cam.image_h or 1, cam.image_w or 1), bool)     # на этот момент маски ещё не было
    size = (cam.image_w, cam.image_h) if cam.image_w and cam.image_h else None
    return Response(masks.overlay_png(bg, size, style), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@router.put("/cameras/{camera_id}/mask")
async def put_mask(camera_id: int, request: Request, reanalyze: bool = True,
                   s: Session = Depends(get_session)) -> dict:
    """Ручная маска кистью: тело — PNG (белое или непрозрачное = фон) в любом разрешении.

    Маска закрепляется (lock): автоматика её не стирает; переживает переанализ объекта.
    reanalyze — переспросить модель Б по уже разобранным кадрам камеры с новой маской."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    raw = await request.body()
    if not raw:
        raise bad("ожидается PNG маски в теле запроса")
    if len(raw) > MASK_PNG_LIMIT:
        raise HTTPException(413, "маска больше 8 МБ")
    try:
        bitmap = masks.decode_bitmap(raw)
    except ValueError as exc:
        raise bad(str(exc)) from None
    if bitmap.mean() >= 0.98:
        raise bad("маска закрывает весь кадр — модели Б нечего разбирать")
    try:
        return masks.set_manual(s, cam, bitmap, reanalyze=reanalyze)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None


@router.delete("/cameras/{camera_id}/mask")
def reset_mask(camera_id: int, s: Session = Depends(get_session)) -> dict:
    """Сброс к автоматической маске: ручная кисть снимается, маска пересобирается по истории
    камеры в фоне, модель Б затем переспрашивается по уже разобранным кадрам."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    return {"ok": True, **masks.reset_to_auto(s, cam)}


@router.get("/cameras/{camera_id}/mask/preview.jpg")
def mask_preview(camera_id: int, frame: int, s: Session = Depends(get_session)) -> Response:
    """Кадр так, как его видит модель Б: маска на момент кадра, режим гашения модели Б."""
    cam = get_or_404(s, Camera, camera_id, "камера")
    fr = get_or_404(s, Frame, frame, "кадр")
    if fr.camera_id != cam.id:
        raise bad("кадр другой камеры")
    mode = settings_svc.get_state(s)["thresholds"].get("stage", {}).get("mask_mode") or "darken"
    try:
        img, applied = masks.preview(s, cam, fr, mode)
    except ValueError as exc:
        raise not_found(str(exc)) from None
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        k = 1280 / max(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Mask-Applied": "1" if applied else "0"})


# --------------------------------------------------------------------------
# камера-поток (simcam Дениса): сообщить адрес приёмника и включить съёмку
# --------------------------------------------------------------------------

# Что из ответа камеры-потока (simcam) отдаём клиенту. Тело чужого ответа целиком не
# пересылаем: иначе source_uri = http://127.0.0.1:<порт>/… превращал сервис в прокси к
# внутренним сервисам сервера (SSRF с чтением ответа).
_CAMERA_FIELDS = ("ok", "name", "frames", "interval", "loop", "start_date", "first_capture", "last_capture",
                  "running", "sent", "failed", "finished", "last_error", "last_sent_at", "error")


def _camera_url(uri: str) -> str:
    """Адрес камеры: только http(s) и не служебные сети (метаданные облака, link-local,
    multicast). Loopback разрешён настройкой camera_allow_loopback (simcam на той же
    машине — демо), кроме порта самого сервиса."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(uri)
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise bad("source_uri: неверный адрес") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise bad("source_uri: нужен адрес вида http://10.0.0.5:8001")
    try:
        infos = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise HTTPException(502, f"камера: имя «{parts.hostname}» не разрешается") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved or \
                (ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")):
            raise bad(f"source_uri: адрес {ip} — служебная сеть, камерой быть не может")
        if ip.is_loopback and (not settings.camera_allow_loopback or port == settings.port):
            raise bad("source_uri: локальный адрес сервера не может быть камерой")
    return uri.rstrip("/")


def _remote(cam: Camera, path: str, method: str = "get", **kw) -> dict:
    import requests

    if not cam.source_uri:
        raise bad("у камеры не задан адрес (source_uri)")
    url = _camera_url(cam.source_uri) + path
    try:
        resp = getattr(requests, method)(url, timeout=5, proxies=netutil.proxies_for(url),
                                         allow_redirects=False, **kw)
    except requests.RequestException as exc:
        raise HTTPException(502, f"камера не отвечает: {exc.__class__.__name__}") from None
    if resp.status_code >= 400 or 300 <= resp.status_code < 400:
        raise HTTPException(502, f"камера ответила {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(502, "камера ответила не JSON — это не камера-поток (simcam)?") from None
    if not isinstance(data, dict):
        raise HTTPException(502, "камера ответила не объектом JSON")
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    merged = {**state, **data}
    return {k: merged[k] for k in _CAMERA_FIELDS if k in merged and isinstance(merged[k], (str, int, float, bool))}


@router.post("/cameras/{camera_id}/connect")
async def connect(camera_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    body = require_obj(await json_body(request, default={}))
    if body.get("source_uri"):
        cam.source_uri = str_field(body, "source_uri", max_len=1000)
    cam.kind = "stream"
    s.commit()
    payload = {"ingest_url": settings.public_base_url.rstrip("/") + "/api/ingest", "camera_id": cam.id,
               "api_key": cam.ingest_key, "restart": bool(body.get("restart", False))}
    if body.get("interval"):
        payload["interval"] = num_field(body, "interval", lo=0.1, hi=86400)
    # Сеть с таймаутом 5 с — в пуле потоков: цикл событий не замирает для всех пользователей.
    camera = await run_in_threadpool(_remote, cam, "/api/start", "post", json=payload)
    return {"ok": True, "camera": camera}


@router.post("/cameras/{camera_id}/disconnect")
def disconnect(camera_id: int, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    return {"ok": True, "camera": _remote(cam, "/api/stop", "post")}


@router.get("/cameras/{camera_id}/stream")
def stream_info(camera_id: int, s: Session = Depends(get_session)) -> dict:
    cam = get_or_404(s, Camera, camera_id, "камера")
    return _remote(cam, "/api/info")
