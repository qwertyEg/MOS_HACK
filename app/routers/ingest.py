"""POST /api/ingest — приём кадра от камеры-потока (контракт Дениса с simcam).

Запрос: заголовок `X-Camera-Key`, multipart `file`, `camera_id`, `captured_at`
(ISO 8601), `meta` (JSON-строка). Заголовок `X-Camera-Id` (или ?camera_id=)
необязателен, но с ним ключ проверяется до чтения тела. Кадр с меткой «из
будущего» (сбой часов камеры) не принимается — 400. Ответ 202 — кадр **сохранён** в БД и
хранилище и стоит в очереди (у Дениса 202 означал «в памяти процесса», и
рестарт терял уже подтверждённые кадры). 403 — неизвестная камера или ключ,
400 — пустой/битый кадр или метка, 503 — очередь камеры переполнена.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import auth
from app.config import settings
from app.db import get_session
from app.routers.common import MAX_ID
from app.services import ingest
from app.services.queue import frame_queue

router = APIRouter(prefix="/api", tags=["приём кадров"])


def _err(status: int, text: str) -> JSONResponse:
    # Ключ `error` — как у Дениса (simcam печатает resp.text), `detail` — как у FastAPI.
    return JSONResponse({"error": text, "detail": text}, status_code=status)


def _camera_id(raw) -> int | None:
    try:
        value = int(str(raw or "").strip())
    except ValueError:
        return None
    return value if 0 < value <= MAX_ID else None


@router.post("/ingest", status_code=202)
async def api_ingest(request: Request, s: Session = Depends(get_session)) -> JSONResponse:
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        return _err(400, "нужен multipart/form-data: file, camera_id, captured_at, meta")
    key = request.headers.get("X-Camera-Key", "")
    # Камера может назвать себя заголовком X-Camera-Id (simcam так и делает) — тогда ключ
    # проверяется ДО чтения тела: чужой с неверным ключом не заливает на диск сотни МБ.
    early = request.headers.get("X-Camera-Id") or request.query_params.get("camera_id")
    if early is not None:
        cam_id = _camera_id(early)
        if cam_id is None or auth.camera_by_key(s, cam_id, key) is None:
            return _err(403, "неизвестная камера или ключ")
    elif not key:
        return _err(403, "неизвестная камера или ключ")
    form = await request.form()
    camera_id = _camera_id(form.get("camera_id")) if early is None else _camera_id(early)
    if camera_id is None:
        return _err(403, "неизвестная камера или ключ")
    cam = auth.camera_by_key(s, camera_id, key)
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

    # Декодирование, превью и запись в SQLite — в пуле потоков, не в цикле событий.
    before = cam.last_frame_at
    res = await run_in_threadpool(ingest.ingest_stream_frame, s, cam, data, captured_at, meta)
    if res.status == "error":
        return _err(400, res.reason)
    if res.status == "duplicate":
        return JSONResponse({"ok": True, "queued": False, "duplicate": True,
                             "frame_id": res.frame.id if res.frame else None}, status_code=202)
    out = {"ok": True, "queued": True, "frame_id": res.frame.id}
    if before is not None and res.frame.captured_at < before:
        # Камера досылает накопленное после обрыва связи: кадр сохранён, но трекер его уже
        # не сравнит — моточасы по нему появятся после переанализа объекта.
        out.update(late=True, warning="кадр старше последнего принятого — моточасы по нему после переанализа")
    return JSONResponse(out, status_code=202)
