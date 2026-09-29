"""Защита веб-слоя для доступа из сети (находки ревью «надёжность», 28.09).

Один ASGI-слой поверх приложения, до разбора запроса FastAPI:

- лимит тела запроса — по Content-Length сразу и по фактическому потоку
  (chunked): 300 МБ «пароля» на /login или кадра с неверным ключом камеры
  раньше целиком ложились во временный файл до ответа 401/403;
- CSRF: изменяющий запрос с чужой страницы отклоняется по Sec-Fetch-Site /
  Origin / Referer (SameSite=Lax не закрывает запросы с соседних поддоменов
  и простые формы text/plain на JSON-API — те дополнительно режет json_body);
- заголовки безопасности: CSP, запрет встраивания во фрейм, nosniff,
  Referrer-Policy, HSTS за TLS;
- Range с несколькими диапазонами к /static (Starlette 0.41 разбирает их
  квадратично, запрос на 60 КБ держал цикл событий ~5 с) — отбрасывается,
  файл отдаётся целиком.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

from app.config import settings

_MB = 1024 * 1024
_UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
_UPLOAD = re.compile(r"^/api/cameras/\d+/upload/?$")
_FRAME = re.compile(r"^/api/(ingest|analyze|detect)/?$|^/api/sites/\d+/plan/import/?$")
_LOGIN = {"/login", "/api/login"}

# Alpine.js (стандартная сборка) вычисляет выражения через Function → 'unsafe-eval';
# тема и словарь иконок — встроенные <script>; ECharts и :style ставят атрибут style.
# Всё остальное — только своё: внешних скриптов, фреймов и плагинов у сервиса нет.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
]


def body_limit(path: str) -> int:
    if _UPLOAD.match(path):
        return int(settings.max_upload_mb * _MB) + _MB          # + поля формы и границы multipart
    if _FRAME.match(path):
        return int(settings.max_frame_mb * _MB) + _MB
    if path in _LOGIN:
        return 64 * 1024
    return int(settings.max_body_mb * _MB)


class _TooLarge(Exception):
    pass


def _header(scope, name: bytes) -> str | None:
    for k, v in scope.get("headers") or ():
        if k == name:
            return v.decode("latin-1")
    return None


def _netloc(url: str | None) -> str | None:
    if not url or url == "null":
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    return parts.netloc.lower() or None


def _own_hosts(scope) -> set[str]:
    hosts = {h.lower() for h in (_header(scope, b"host"), _header(scope, b"x-forwarded-host")) if h}
    base = _netloc(settings.public_base_url)
    if base:
        hosts.add(base)
    return hosts


def cross_site(scope) -> str | None:
    """Почему изменяющий запрос считается чужим (None — свой или не из браузера)."""
    fetch_site = _header(scope, b"sec-fetch-site")
    if fetch_site in ("same-origin", "none"):
        # Заголовок ставит сам браузер, скрипт страницы подделать его не может: свой запрос
        # (в том числе за прокси, который переписывает Host) — пропускаем без сверки Origin.
        return None
    if fetch_site:
        return f"запрос со стороннего сайта (Sec-Fetch-Site: {fetch_site})"
    origin = _header(scope, b"origin")
    if origin is not None:
        if origin == "null" or _netloc(origin) not in _own_hosts(scope):
            return f"запрос со стороннего сайта (Origin: {origin[:80]})"
        return None
    referer = _header(scope, b"referer")
    if referer is not None and _netloc(referer) not in _own_hosts(scope):
        return "запрос со стороннего сайта (Referer)"
    return None     # curl, камеры, скрипты: у них нет куки браузера — CSRF к ним не относится


async def _send_json(send, status: int, detail: str, extra: list | None = None) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode()), *_HEADERS, *(extra or [])]})
    await send({"type": "http.response.body", "body": body})


class Guard:
    """ASGI-слой: лимит тела, CSRF, Range к статике, заголовки безопасности."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method, path = scope["method"], scope["path"]

        if path.startswith("/static/"):
            rng = _header(scope, b"range")
            if rng is not None and ("," in rng or len(rng) > 64):
                scope = dict(scope)
                scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"range"]

        if method in _UNSAFE:
            why = cross_site(scope)
            if why:
                return await _send_json(send, 403, f"{why} — отклонено (защита от CSRF)")
            limit = body_limit(path)
            declared = _header(scope, b"content-length")
            try:
                if declared is not None and int(declared) > limit:
                    return await _send_json(send, 413, f"тело запроса больше {limit // _MB or 1} МБ — отклонено")
            except ValueError:
                return await _send_json(send, 400, "неверный Content-Length")
            receive = self._counting(receive, limit)

        started = False
        https = scope.get("scheme") == "https" or _header(scope, b"x-forwarded-proto") == "https"

        async def send_wrapped(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = list(message.get("headers") or [])
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _HEADERS if k not in present]
                if b"content-security-policy" not in present and not path.startswith("/api/docs"):
                    headers.append((b"content-security-policy", CSP.encode()))
                if https:
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapped)
        except _TooLarge:
            if not started:
                await _send_json(send, 413, f"тело запроса больше {body_limit(path) // _MB or 1} МБ — отклонено")

    @staticmethod
    def _counting(receive, limit: int):
        seen = 0

        async def wrapped():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body") or b"")
                if seen > limit:
                    raise _TooLarge()
            return message
        return wrapped
