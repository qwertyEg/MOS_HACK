"""HTML-оболочки страниц. Данные страницы берут сами из JSON API (fetch),
поэтому контекст шаблона минимальный: {request, page, user, site_id?,
camera_id?, frame_id?, app_version}. Шаблоны — зона UI (app/templates/pages/);
если шаблона ещё нет, отдаём простую заглушку, а не 500."""
from __future__ import annotations

from html import escape

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from sqlalchemy.orm import Session

from app import __version__, auth
from app.config import BASE_DIR
from app.db import get_session

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

_TITLES = {
    "overview": "Обзор объектов", "site": "Объект", "camera": "Камера", "frame": "Кадр",
    "try": "Проверить снимок", "settings": "Настройки", "login": "Вход",
}


def _stub(page: str, ctx: dict, status_code: int = 200) -> HTMLResponse:
    title = _TITLES.get(page, page)
    if page == "login":
        error = ctx.get("error")
        body = (
            f"<h1>СтройВзор — вход</h1>{f'<p style=color:#b91c1c>{escape(error)}</p>' if error else ''}"
            f"<form method=post action=/login><input type=hidden name=next value='{escape(ctx.get('next', '/'))}'>"
            "<p><input name=login placeholder=Логин autofocus></p>"
            "<p><input name=password type=password placeholder=Пароль></p><button>Войти</button></form>"
        )
    else:
        ids = ", ".join(f"{k}={escape(str(ctx[k]))}" for k in ("site_id", "camera_id", "frame_id") if ctx.get(k))
        body = (f"<h1>{escape(title)}</h1><p>Шаблон pages/{page}.html ещё не готов. {ids}</p>"
                "<p><a href=/api/docs>JSON API</a> · <a href=/>Обзор</a> · <a href=/logout>Выйти</a></p>")
    return HTMLResponse(
        f"<!doctype html><meta charset=utf-8><title>{escape(title)} — СтройВзор</title>"
        f"<body style='font:15px system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px'>{body}",
        status_code=status_code)


def render(request: Request, page: str, user: str | None, status_code: int = 200, **extra) -> HTMLResponse:
    ctx = {"request": request, "page": page, "user": user, "app_version": __version__, **extra}
    try:
        return templates.TemplateResponse(request, f"pages/{page}.html", ctx, status_code=status_code)
    except TemplateNotFound:
        return _stub(page, ctx, status_code)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if auth.current_user(request):
        return RedirectResponse(auth.safe_next(next), status_code=303)
    return render(request, "login", None, next=auth.safe_next(next), error=None)


@router.post("/login")
def login_submit(request: Request, login: str = Form(""), password: str = Form(""), next: str = Form("/"),
                 s: Session = Depends(get_session)):
    try:
        user = auth.login_attempt(request, s, login, password)
    except auth.TooManyAttempts as exc:
        resp = render(request, "login", None, status_code=429, next=auth.safe_next(next),
                      error=f"Слишком много неверных попыток — повторите через {exc.wait_s} с")
        resp.headers["Retry-After"] = str(exc.wait_s)
        return resp
    if user is None:
        return render(request, "login", None, status_code=401, next=auth.safe_next(next),
                      error="Неверный логин или пароль")
    auth.login_user(request, user)
    return RedirectResponse(auth.safe_next(next), status_code=303)


@router.get("/logout")
def logout(request: Request, s: Session = Depends(get_session)):
    # Ссылка «Выйти» в шапке — обычный GET. Чужая страница (картинка, ссылка) не должна
    # разлогинивать: браузер помечает такие запросы Sec-Fetch-Site: cross-site / same-site.
    if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return RedirectResponse("/", status_code=303)
    auth.logout_user(request, s)
    return RedirectResponse("/login", status_code=303)


@router.post("/logout")
def logout_post(request: Request, s: Session = Depends(get_session)):
    auth.logout_user(request, s)
    return RedirectResponse("/login", status_code=303)


@router.get("/", response_class=HTMLResponse)
def overview_page(request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "overview", user)


@router.get("/sites/{site_id}", response_class=HTMLResponse)
def site_page(site_id: int, request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "site", user, site_id=site_id)


@router.get("/cameras/{camera_id}", response_class=HTMLResponse)
def camera_page(camera_id: int, request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "camera", user, camera_id=camera_id)


@router.get("/frames/{frame_id}", response_class=HTMLResponse)
def frame_page(frame_id: int, request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "frame", user, frame_id=frame_id)


@router.get("/try", response_class=HTMLResponse)
def try_page(request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "try", user)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: str = Depends(auth.require_page_user)):
    return render(request, "settings", user)
