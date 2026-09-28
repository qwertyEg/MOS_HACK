"""Сборка приложения FastAPI: роутеры, сессии, обработчики ошибок, жизненный цикл.

Старт: схема БД → учётка администратора → ключи моделей в окружение →
очередь (подбирает кадры, не обработанные до рестарта) → прогрев статусов
провайдеров в фоне (загрузка весов не должна задерживать старт).

Невалидный ввод — 400 с текстом по-русски (а не 422 со списком словарей и
не 500); непойманная ошибка — 500 с коротким текстом, стек — в лог.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app import __version__, auth, db
from app.config import BASE_DIR, settings
from app.routers import api_cameras, api_frames, api_sites, api_system, ingest, pages
from app.services.providers import registry
from app.services.queue import frame_queue

log = logging.getLogger("app")


def _warm_up() -> None:
    try:
        registry.status()
    except Exception:  # noqa: BLE001
        log.exception("прогрев провайдеров")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.export_env()
    db.init_db()
    with db.session() as s:
        auth.ensure_admin(s)
    if settings.workers_enabled:
        frame_queue.start()
    threading.Thread(target=_warm_up, name="providers-warmup", daemon=True).start()
    try:
        yield
    finally:
        frame_queue.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="СтройВзор — мониторинг стройплощадки", version=__version__,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, same_site="lax",
                       max_age=settings.session_max_age_h * 3600, session_cookie="stroyvzor_session")

    @app.exception_handler(auth.LoginRedirect)
    async def _login_redirect(request: Request, exc: auth.LoginRedirect):
        return RedirectResponse(f"/login?next={quote(exc.next_path)}", status_code=303)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path"))
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return JSONResponse({"detail": "; ".join(parts) or "невалидный запрос"}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("необработанная ошибка %s %s", request.method, request.url.path)
        return JSONResponse({"detail": f"внутренняя ошибка: {type(exc).__name__}: {exc}"}, status_code=500)

    static_dir = BASE_DIR / "app" / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(api_system.public)
    app.include_router(api_system.router)
    app.include_router(ingest.router)
    app.include_router(api_sites.router)
    app.include_router(api_cameras.router)
    app.include_router(api_frames.router)
    app.include_router(pages.router)
    return app


app = create_app()
