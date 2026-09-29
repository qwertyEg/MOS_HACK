"""Общие помощники роутеров: 4xx с понятным текстом вместо 500."""
from __future__ import annotations

import json
import math
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session


def bad(message: str) -> HTTPException:
    return HTTPException(400, message)


def not_found(message: str) -> HTTPException:
    return HTTPException(404, message)


MAX_ID = 2**63 - 1          # INTEGER SQLite/BIGINT: больше — OverflowError в драйвере (было 500)


class _BadNumber(ValueError):
    pass


def _no_constant(name: str) -> Any:
    raise _BadNumber(f"{name} — не число")


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _BadNumber(f"{text} — вне диапазона чисел")
    return value


def _bounded(text: str) -> int:
    value = int(text)
    if abs(value) > MAX_ID:
        raise _BadNumber(f"{text[:24]}… — слишком большое число")
    return value


def parse_json(raw: bytes | str) -> Any:
    """JSON без NaN/Infinity/1e400 и 20-значных целых: они проходили валидацию
    (сравнения с NaN всегда ложны) и роняли пересчёт, SQLite и JSON-ответы."""
    return json.loads(raw, parse_constant=_no_constant, parse_float=_finite, parse_int=_bounded)


async def json_body(request: Request, default: Any = ...) -> Any:
    raw = await request.body()
    if not raw.strip():
        if default is not ...:
            return default
        raise bad("ожидается JSON в теле запроса")
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        # text/plain и формы шлёт любая чужая страница без CORS-проверки (CSRF) — JSON-API их не принимает.
        raise HTTPException(415, "нужен Content-Type: application/json")
    try:
        return parse_json(raw)
    except _BadNumber as exc:
        raise bad(f"недопустимое число в JSON: {exc}") from None
    except ValueError:
        raise bad("тело запроса — не JSON") from None


def get_or_404(s: Session, model, ident: Any, what: str):
    try:
        ident = int(ident)
    except (TypeError, ValueError, OverflowError):
        raise bad(f"{what}: id должен быть числом") from None
    if not 0 < ident <= MAX_ID:
        raise not_found(f"{what} {ident} не найден")
    obj = s.get(model, ident)
    if obj is None:
        raise not_found(f"{what} {ident} не найден")
    return obj


def require_obj(body: Any) -> dict:
    if not isinstance(body, dict):
        raise bad("ожидается JSON-объект")
    return body


def str_field(body: dict, key: str, *, required: bool = False, max_len: int = 256, default: str = "") -> str:
    value = body.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise bad(f"{key}: ожидается строка")
    value = value.strip()
    if required and not value:
        raise bad(f"{key}: обязательное поле")
    if len(value) > max_len:
        raise bad(f"{key}: не длиннее {max_len} символов")
    return value


def num_field(body: dict, key: str, *, lo: float | None = None, hi: float | None = None,
              integer: bool = False, allow_none: bool = False) -> float | int | None:
    value = body.get(key)
    if value is None:
        if allow_none:
            return None
        raise bad(f"{key}: обязательное число")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise bad(f"{key}: ожидается число")
    if integer and int(value) != value:
        raise bad(f"{key}: ожидается целое число")
    if lo is not None and value < lo or hi is not None and value > hi:
        raise bad(f"{key}: допустимо от {lo} до {hi}")
    return int(value) if integer else float(value)


def points_field(body: dict, key: str, min_len: int) -> list[list[float]]:
    pts = body.get(key)
    if not isinstance(pts, list) or len(pts) < min_len:
        raise bad(f"{key}: нужен список минимум из {min_len} точек [x, y]")
    out = []
    for p in pts:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in p)):
            raise bad(f"{key}: каждая точка — пара чисел [x, y]")
        out.append([float(p[0]), float(p[1])])
    return out
