"""Кадры, файлы и разбор снимка: /api/frames, /media, /api/detect, /api/analyze."""
from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import auth, storage
from app.db import get_session
from app.models import Frame
from app.routers.common import bad, get_or_404, json_body, not_found
from app.services import analysis, ingest, views
from app.services import settings as settings_svc
from app.services.providers import ProviderUnavailable

router = APIRouter(tags=["кадры"], dependencies=[Depends(auth.require_api_user)])

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
                ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff", ".bin": "application/octet-stream"}


@router.get("/media/{key:path}")
def media(key: str) -> Response:
    """Файлы хранилища — только после входа (у Дениса /media был открыт всем)."""
    st = storage.get()
    try:
        data = st.get(key)
    except (ValueError, FileNotFoundError, OSError):
        raise not_found("файл не найден") from None
    except Exception:  # noqa: BLE001 — S3: нет объекта
        raise not_found("файл не найден") from None
    ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
    # Ключи содержат sha кадра и не переиспользуются — кэшировать можно смело.
    return Response(data, media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"),
                    headers={"Cache-Control": "private, max-age=86400"})


@router.get("/api/frames/{frame_id}")
def frame_detail(frame_id: int, s: Session = Depends(get_session)) -> dict:
    return views.frame_detail(s, get_or_404(s, Frame, frame_id, "кадр"))


@router.get("/api/frames/{frame_id}/annotated.jpg")
def annotated(frame_id: int, max_w: int = 1600, s: Session = Depends(get_session)) -> Response:
    fr = get_or_404(s, Frame, frame_id, "кадр")
    try:
        img = analysis.load_frame_image(fr)
    except (ValueError, FileNotFoundError, OSError):
        raise not_found("файл кадра недоступен") from None
    dets = analysis.stored_detections(s, fr, settings_svc.get_state(s)["model_a"]) or []
    out = analysis.annotate(img, dets)
    h, w = out.shape[:2]
    max_w = max(160, min(max_w, 4096))
    if w > max_w:
        out = cv2.resize(out, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def _read_image(upload) -> "cv2.typing.MatLike":
    data = await upload.read()
    try:
        img = await run_in_threadpool(ingest.decode_image, data)
    except ingest.ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from None
    if img is None:
        raise bad("файл не распознан как изображение (jpg/png/webp/bmp)")
    return img


@router.post("/api/detect")
async def detect(request: Request, s: Session = Depends(get_session)) -> dict:
    """Контракт модели А (PLAN §4.3): файл (multipart `file`) или JSON `{frame_id}` →
    `{frame_id, detections: [Detection.to_contract()]}`. Для кадра из БД отдаются
    посчитанные трекером поля (moved_since_prev, displacement_px, unit_id…)."""
    ctype = request.headers.get("content-type", "")
    upload = frame_id = provider = None
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file") if hasattr(form.get("file"), "read") else None
        frame_id, provider = form.get("frame_id"), form.get("provider") or form.get("model_a")
    elif ctype.startswith("application/json"):
        body = await json_body(request)          # NaN/Infinity/1e400 → 400, а не 500
        if not isinstance(body, dict):
            raise bad("ожидается объект {frame_id}")
        frame_id, provider = body.get("frame_id"), body.get("provider") or body.get("model_a")
    else:
        raise bad("нужен multipart с полем file или JSON {\"frame_id\": …}")

    try:
        model_a, _ = analysis.resolve_models(s, None, provider, None)
    except ValueError as exc:
        raise bad(str(exc)) from None

    if upload is not None:
        img = await _read_image(upload)
        try:
            # Детектор и общий замок моделей — в пуле потоков: цикл событий не ждёт их
            # и отдаёт остальным пользователям страницы и статику.
            dets, ms = await run_in_threadpool(analysis.detect_image, img, model_a)
        except ProviderUnavailable as exc:
            raise HTTPException(503, f"модель А ({model_a}) не готова: {exc.reason}") from None
        h, w = img.shape[:2]
        return {"frame_id": None, "detections": [d.to_contract() for d in dets], "provider": model_a,
                "source": "live", "latency_ms": round(ms, 1), "image": {"width": w, "height": h}}

    if frame_id in (None, ""):
        raise bad("нужен файл (file) или frame_id")
    fr = get_or_404(s, Frame, frame_id, "кадр")
    dets = analysis.stored_detections(s, fr, model_a)
    source = "stored"
    ms = None
    if dets is None:
        try:
            img = await run_in_threadpool(analysis.load_frame_image, fr)
        except (ValueError, FileNotFoundError, OSError):
            raise not_found("файл кадра недоступен") from None
        try:
            dets, ms = await run_in_threadpool(analysis.detect_image, img, model_a)
        except ProviderUnavailable as exc:
            raise HTTPException(503, f"модель А ({model_a}) не готова: {exc.reason}") from None
        source = "live"
    return {"frame_id": fr.id, "detections": [d.to_contract() for d in dets], "provider": model_a,
            "source": source, "latency_ms": round(ms, 1) if ms is not None else None,
            "image": {"width": fr.width, "height": fr.height}, "captured_at": views.iso(fr.captured_at)}


@router.post("/api/analyze")
async def analyze(request: Request, s: Session = Depends(get_session)) -> dict:
    """«Проверить снимок»: файл + provider (local|external|hybrid) или model_a/model_b →
    рамки, чек-лист, этап, время ответа. Ничего не сохраняет."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        raise bad("нужен multipart с полем file")
    form = await request.form()
    upload = form.get("file")
    if not hasattr(upload, "read"):
        raise bad("нужен файл снимка (поле file)")
    img = await _read_image(upload)
    try:
        model_a, model_b = analysis.resolve_models(s, form.get("provider") or None, form.get("model_a") or None,
                                                   form.get("model_b") or None)
    except ValueError as exc:
        raise bad(str(exc)) from None
    flag = lambda k, d: str(form.get(k, d)).lower() not in ("0", "false", "no", "")  # noqa: E731
    return await run_in_threadpool(analysis.analyze_image, s, img, model_a, model_b,
                                   annotate_image=flag("annotate", "1"), force_stage=flag("force_stage", "0"))
