"""Макет бэкенда для разработки интерфейса без моделей и БД (запасной вариант).

Рендерит настоящие шаблоны app/templates и «проигрывает» записанные ответы
живого бэкенда (tests/ui/fixtures/api, пишет tests/ui/record_fixtures.py) —
формы JSON те же, что в app/services/views.py. Изменения (PUT/PATCH/POST/DELETE)
принимаются и отражаются эхом, в базу ничего не пишется.

    python tools/ui_mock.py                 # http://127.0.0.1:8765
    python tools/ui_mock.py --port 9000

Управление макетом (для снимков пустых состояний):
    GET /__mock?empty=1   — объектов нет (пустое состояние обзора)
    GET /__mock?reset=1   — вернуть всё как было
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "ui" / "fixtures"
sys.path.insert(0, str(ROOT / "tests" / "ui"))
from record_fixtures import key_for  # noqa: E402

TEMPLATES = ROOT / "app" / "templates"
STATIC = ROOT / "app" / "static"
VERSION = "mock"

app = FastAPI(title="СтройВзор — макет UI", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES))
STATE = {"empty": False}


def fixture(path: str):
    """Ответ по точному пути с запросом, иначе — по пути без запроса."""
    for p in (path, urllib.parse.urlsplit(path).path):
        f = FIX / "api" / f"{key_for(p)}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


# ------------------------------------------------------------------ страницы

def page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, f"pages/{name}.html",
                                      {"page": name, "user": "admin", "app_version": VERSION, **ctx})


@app.get("/", response_class=HTMLResponse)
def overview_page(request: Request):
    return page(request, "overview")


@app.get("/sites/{site_id}", response_class=HTMLResponse)
def site_page(request: Request, site_id: int):
    return page(request, "site", site_id=site_id)


@app.get("/cameras/{camera_id}", response_class=HTMLResponse)
def camera_page(request: Request, camera_id: int):
    return page(request, "camera", camera_id=camera_id)


@app.get("/frames/{frame_id}", response_class=HTMLResponse)
def frame_page(request: Request, frame_id: int):
    return page(request, "frame", frame_id=frame_id)


@app.get("/try", response_class=HTMLResponse)
def try_page(request: Request):
    return page(request, "try")


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return page(request, "settings")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", error: str | None = None):
    return templates.TemplateResponse(request, "pages/login.html", {
        "page": "login", "user": None, "app_version": VERSION, "next": next,
        "error": "Неверный логин или пароль" if error else None})


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    if not form.get("login") or not form.get("password"):
        return RedirectResponse("/login?error=1", status_code=303)
    nxt = str(form.get("next") or "/")
    return RedirectResponse(nxt if nxt.startswith("/") else "/", status_code=303)


@app.get("/logout")
def logout():
    return RedirectResponse("/login", status_code=303)


@app.get("/__mock")
def mock_control(empty: int | None = None, reset: int | None = None):
    if reset:
        STATE["empty"] = False
    if empty is not None:
        STATE["empty"] = bool(empty)
    return dict(STATE)


# ------------------------------------------------------------------ картинки

@app.get("/media/{key:path}")
def media(key: str):
    f = FIX / "media" / key_for("/media/" + key)
    if not f.exists():
        # Кадр, превью которого не записано, — любой записанный кадр той же камеры.
        cam = key.split("/")[1] if key.count("/") > 1 else ""
        same = sorted((FIX / "media").glob(f"media__frames__{cam}__*")) or sorted((FIX / "media").glob("media__*"))
        if not same:
            return Response(status_code=404)
        f = same[len(same) // 2]
    return Response(f.read_bytes(), media_type="image/png" if f.suffix == ".png" else "image/jpeg")


@app.get("/api/frames/{frame_id}/annotated.jpg")
def annotated(frame_id: int):
    f = FIX / "media" / f"annotated_{frame_id}.jpg"
    if not f.exists():
        detail = fixture(f"/api/frames/{frame_id}")
        return media(detail["url"].removeprefix("/media/")) if detail else Response(status_code=404)
    return Response(f.read_bytes(), media_type="image/jpeg")


# ------------------------------------------------------------------ JSON API

@app.post("/api/login")
def api_login():
    return {"ok": True, "user": "admin"}


@app.post("/api/analyze")
def api_analyze():
    data = fixture("/api/analyze")
    return data if data else JSONResponse({"detail": "в макете нет записанного ответа /api/analyze"}, 503)


@app.post("/api/demo/seed")
def api_seed():
    STATE["empty"] = False
    return {"sites": [], "jobs": [], "warnings": ["Макет: засев отключён — объекты уже записаны в фикстурах"]}


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def api(rest: str, request: Request):
    path = "/api/" + rest + (f"?{request.url.query}" if request.url.query else "")
    if request.method == "GET":
        if STATE["empty"] and rest == "sites":
            return []
        data = fixture(path)
        if data is None and rest.startswith("jobs/"):
            # Задание, запущенное в макете (загрузка, переанализ), «завершается» сразу.
            return {"id": rest[5:], "state": "done", "total": 0, "done": 0, "failed": 0, "postponed": 0,
                    "pending": 0, "errors": [], "duplicates": 0, "skipped": 0}
        if data is None:
            return JSONResponse({"detail": f"макет: нет записи для GET {path}"}, 404)
        return data
    # Изменения: эхо тела поверх записанного ресурса (для PATCH/PUT) — UI видит успех.
    body = None
    if "json" in request.headers.get("content-type", ""):
        try:
            body = await request.json()
        except ValueError:
            body = None
    if request.method == "DELETE":
        return {"ok": True}
    current = fixture(urllib.parse.urlsplit(path).path)
    if isinstance(current, dict) and isinstance(body, dict):
        return {**current, **body}
    if isinstance(body, (list, dict)):
        return body
    return current if current is not None else {"ok": True, "job_id": "mock"}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not (FIX / "api").is_dir():
        sys.exit("нет tests/ui/fixtures/api — запишите: python tests/ui/record_fixtures.py --base <бэкенд>")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
