"""Камера как сервис.

Зачем он нужен. До сих пор система получала историю папкой и прогоняла её
целиком за раз. В жизни так не бывает: камера присылает кадр раз в двадцать
минут, и система обязана работать в этом ритме — принять кадр, обновить
маску, спросить модель, подвинуть графики, и успеть до следующего. Пока
кадры лежат папкой, этот режим не проверить: код прогона знает всю историю
заранее и потому устроен иначе.

Этот сервис поднимается отдельным процессом (в жизни — на другой машине) и
ведёт себя как настоящая камера: по команде начинает слать кадры по одному с
заданной паузой и метаданными. Всё, что отличает его от железной камеры, —
кадры он берёт из папки, а паузу можно поставить в полминуты вместо двадцати
минут, чтобы полгода стройки прошли за час.

Метки времени кадров берутся из имён файлов, то есть остаются настоящими:
сутки съёмки остаются сутками, даже если проходят за минуту. Это важно —
окно маски и разрежение вызовов модели считаются по времени съёмки, а не по
настенным часам. `--start-date` сдвигает всю серию целиком, сохраняя
интервалы: так архив 2005 года можно совместить с планом, который написан
на этот год.

Сервис намеренно ничего не знает про основное приложение: ни общих модулей,
ни общей базы. Он умеет только слать HTTP-запросы с картинкой — ровно
столько, сколько умеет камера.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RETRY_PAUSE = 5.0          # секунд до повтора, если приёмник не ответил
RETRY_LIMIT = 12           # столько неудач подряд — сдаёмся и останавливаемся


def local_proxies(url: str) -> dict | None:
    """Прокси из окружения мимо локальных адресов.

    Если в окружении прописан HTTP_PROXY без исключения для локалки, кадр на
    127.0.0.1:8000 уйдёт в прокси и не дойдёт. Приёмник кадров всегда рядом с
    камерой — в той же машине или в той же сети, — и через прокси ему ходить
    незачем. Проверено на этой машине: без этого «камера не отвечает» при
    живой камере.
    """
    host = (urlparse(url).hostname or "").lower()
    local = (not host or host in ("localhost", "host.docker.internal")
             or host.endswith(".local") or "." not in host)
    if not local:
        try:
            addr = ipaddress.ip_address(host)
            local = addr.is_private or addr.is_loopback or addr.is_unspecified
        except ValueError:
            local = False
    # Ключ "all" обязателен: requests дописывает прокси из окружения через
    # setdefault и только потом выбрасывает ключи со значением None, так что
    # ALL_PROXY переживает чистку и уводит кадр в прокси. Словарь каждый раз
    # новый — requests правит переданный ему на месте.
    return {"http": None, "https": None, "all": None} if local else None


def parse_stamp(name: str) -> dt.datetime | None:
    m = STAMP.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.UTC)
    except ValueError:
        return None


def list_frames(folder: Path) -> list[tuple[Path, dt.datetime]]:
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXT:
            when = parse_stamp(p.name)
            if when is not None:
                out.append((p, when))
    out.sort(key=lambda t: t[1])
    return out


@dataclass
class Config:
    name: str = "Камера"
    folder: Path = Path(".")
    interval: float = 30.0
    loop: bool = False
    start_date: dt.date | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class State:
    running: bool = False
    sent: int = 0
    failed: int = 0
    index: int = 0
    target: str = ""
    camera_id: int | None = None
    last_sent_at: str = ""
    last_capture: str = ""
    last_error: str = ""
    finished: bool = False


class StartRequest(BaseModel):
    """Что сообщает нам основной сервис, когда оператор жмёт «Подключить»."""
    ingest_url: str
    camera_id: int
    api_key: str
    interval: float | None = None
    restart: bool = True          # слать с начала или продолжить с места


cfg = Config()
state = State()
_frames: list[tuple[Path, dt.datetime]] = []
_thread: threading.Thread | None = None
_stop = threading.Event()
_lock = threading.Lock()

app = FastAPI(title="Имитатор камеры", docs_url="/api/docs")


def configure(config: Config) -> None:
    global _frames
    cfg.__dict__.update(config.__dict__)
    _frames = list_frames(cfg.folder)


def _shift(when: dt.datetime) -> dt.datetime:
    """Сдвиг серии к заданной дате с сохранением интервалов."""
    if cfg.start_date is None or not _frames:
        return when
    base = _frames[0][1]
    target = dt.datetime.combine(cfg.start_date, base.timetz())
    return when + (target - base)


def _send(req: StartRequest, path: Path, when: dt.datetime, seq: int) -> None:
    meta = dict(cfg.meta)
    meta.update({
        "camera": cfg.name,
        "source_file": path.name,
        "sequence": seq,
        "sent_at": dt.datetime.now(dt.UTC).isoformat(),
        "interval_sec": req.interval or cfg.interval,
    })
    with path.open("rb") as fh:
        resp = requests.post(
            req.ingest_url,
            # X-Camera-Id — сервис проверит ключ до чтения тела кадра
            headers={"X-Camera-Key": req.api_key, "X-Camera-Id": str(req.camera_id)},
            files={"file": (path.name, fh, "image/jpeg")},
            data={"camera_id": str(req.camera_id),
                  "captured_at": when.isoformat(),
                  "meta": json.dumps(meta, ensure_ascii=False)},
            proxies=local_proxies(req.ingest_url),
            timeout=30,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")


def _worker(req: StartRequest) -> None:
    pause = req.interval or cfg.interval
    misses = 0

    while not _stop.is_set():
        if state.index >= len(_frames):
            if not cfg.loop:
                state.finished = True
                break
            state.index = 0

        path, when = _frames[state.index]
        try:
            _send(req, path, _shift(when), state.index + 1)
            state.sent += 1
            state.index += 1
            state.last_sent_at = dt.datetime.now().strftime("%H:%M:%S")
            state.last_capture = _shift(when).strftime("%d.%m.%Y %H:%M")
            state.last_error = ""
            misses = 0
        except Exception as exc:                            # noqa: BLE001
            # Приёмник мог просто перезапускаться. Камера в такой ситуации
            # не выбрасывает кадр и не бежит дальше — она повторяет попытку,
            # иначе в истории появится дыра там, где съёмка шла исправно.
            state.failed += 1
            state.last_error = str(exc)[:300]
            misses += 1
            if misses >= RETRY_LIMIT:
                break
            _stop.wait(RETRY_PAUSE)
            continue

        _stop.wait(pause)

    state.running = False


@app.get("/api/info")
def info() -> JSONResponse:
    return JSONResponse({
        "name": cfg.name,
        "folder": str(cfg.folder),
        "frames": len(_frames),
        "interval": cfg.interval,
        "loop": cfg.loop,
        "start_date": cfg.start_date.isoformat() if cfg.start_date else None,
        "meta": cfg.meta,
        "first_capture": _frames[0][1].isoformat() if _frames else None,
        "last_capture": _frames[-1][1].isoformat() if _frames else None,
        "state": state.__dict__,
    })


@app.get("/api/preview")
def preview() -> Response:
    """Первый кадр. По нему оператор рисует маску ещё до начала съёмки."""
    if not _frames:
        return Response(status_code=404)
    return Response(_frames[0][0].read_bytes(), media_type="image/jpeg")


@app.post("/api/start")
def start(req: StartRequest) -> JSONResponse:
    global _thread
    with _lock:
        if state.running:
            return JSONResponse({"ok": False, "error": "уже идёт съёмка"},
                                status_code=409)
        if not _frames:
            return JSONResponse({"ok": False, "error": f"в папке {cfg.folder} "
                                 "нет кадров с меткой времени в имени"},
                                status_code=400)
        _stop.clear()
        if req.restart:
            state.index = 0
            state.sent = state.failed = 0
        state.running = True
        state.finished = False
        state.last_error = ""
        state.target = req.ingest_url
        state.camera_id = req.camera_id
        _thread = threading.Thread(target=_worker, args=(req,),
                                   name="simcam-sender", daemon=True)
        _thread.start()
    return JSONResponse({"ok": True, "frames": len(_frames),
                         "interval": req.interval or cfg.interval})


@app.post("/api/stop")
def stop() -> JSONResponse:
    _stop.set()
    state.running = False
    return JSONResponse({"ok": True, "sent": state.sent})


@app.get("/", response_class=HTMLResponse)
def page() -> HTMLResponse:
    """Страница на случай, если камеру открыли браузером. Служебная."""
    s = state
    rows = "".join(
        f"<tr><td style='padding:4px 16px 4px 0;color:#64748b'>{k}</td>"
        f"<td style='padding:4px 0'>{v}</td></tr>"
        for k, v in [
            ("папка", cfg.folder), ("кадров", len(_frames)),
            ("пауза", f"{cfg.interval:g} с"),
            ("состояние", "идёт съёмка" if s.running else
             ("серия закончена" if s.finished else "ожидает подключения")),
            ("отправлено", s.sent), ("ошибок", s.failed),
            ("последний кадр", s.last_capture or "—"),
            ("приёмник", s.target or "—"),
            ("ошибка", s.last_error or "—"),
        ])
    return HTMLResponse(
        f"<!doctype html><meta charset=utf-8><title>{cfg.name}</title>"
        "<body style='font:14px -apple-system,Segoe UI,Roboto,sans-serif;"
        "max-width:640px;margin:40px auto;padding:0 16px'>"
        f"<h1 style='font-size:18px'>{cfg.name}</h1>"
        "<p style='color:#64748b'>Имитатор камеры. Съёмку включает основной "
        "сервис на странице камеры.</p>"
        f"<table>{rows}</table>"
        "<p style='color:#94a3b8;font-size:12px'>Страница не обновляется сама — "
        "перезагрузите её.</p></body>")
