"""POST /api/ingest — приём кадра от камеры-потока (контракт Дениса с simcam).

Запрос: заголовок `X-Camera-Key`, multipart `file`, `camera_id`, `captured_at`
(ISO 8601), `meta` (JSON-строка). Ответ 202 — кадр **сохранён** в БД и
хранилище и стоит в очереди (у Дениса 202 означал «в памяти процесса», и
рестарт терял уже подтверждённые кадры). 403 — неизвестная камера или ключ,
400 — пустой/битый кадр или метка, 503 — очередь камеры переполнена.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import auth
from app.config import settings
from app.db import get_session
from app.services import ingest
from app.services.queue import frame_queue

router = APIRouter(prefix="/api", tags=["приём кадров"])


def _err(status: int, text: str) -> JSONResponse:
    # Ключ `error` — как у Дениса (simcam печатает resp.text), `detail` — как у FastAPI.
    return JSONResponse({"error": text, "detail": text}, status_code=status)


@router.post("/ingest", status_code=202)
async def api_ingest(request: Request, s: Session = Depends(get_session)) -> JSONResponse:
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        return _err(400, "нужен multipart/form-data: file, camera_id, captured_at, meta")
    form = await request.form()
    try:
        camera_id = int(str(form.get("camera_id", "")).strip())
    except ValueError:
        return _err(403, "неизвестная камера или ключ")
    cam = auth.camera_by_key(s, camera_id, request.headers.get("X-Camera-Key", ""))
    if cam is None:
        return _err(403, "неизвестная камера или ключ")

    upload = form.get("file")
    if not hasattr(upload, "read"):
        return _err(400, "нет файла кадра (поле file)")
    data = await upload.read()
    if not data:
        return _err(400, "пустой кадр")
    try:
        captured_at = ingest.parse_datetime_input(form.get("captured_at"))
    except ValueError:
        return _err(400, f"метка времени не разобрана: {form.get('captured_at')}")

    raw_meta = str(form.get("meta") or "")
    try:
        meta = json.loads(raw_meta) if raw_meta else {}
    except ValueError:
        meta = {"raw": raw_meta[:500]}      # метаданные — дело камеры, ронять приём из-за них нельзя
    if not isinstance(meta, dict):
        meta = {"value": meta}

    if frame_queue.pending(cam.id) >= settings.queue_limit:
        return _err(503, "очередь камеры переполнена, повторите позже")

    res = ingest.ingest_stream_frame(s, cam, data, captured_at, meta)
    if res.status == "error":
        return _err(400, res.reason)
    if res.status == "duplicate":
        return JSONResponse({"ok": True, "queued": False, "duplicate": True,
                             "frame_id": res.frame.id if res.frame else None}, status_code=202)
    return JSONResponse({"ok": True, "queued": True, "frame_id": res.frame.id}, status_code=202)
