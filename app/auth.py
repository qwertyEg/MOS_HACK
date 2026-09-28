"""Авторизация: пользователь в таблице `users`, сессия в подписанной куке.

Ролей нет — в демо один оператор. Камеры входят не сессией, а своим ключом
(`X-Camera-Key`): у машины нет пароля пользователя, и ключ у каждой камеры
свой — утёк один, отключается одна камера (решение Дениса).

Пароль хранится PBKDF2-хешем (stdlib, без лишних зависимостей). Учётка
администратора создаётся/обновляется при старте из ADMIN_LOGIN/ADMIN_PASSWORD:
окружение — источник истины для единственного пользователя.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Camera, User

SESSION_KEY = "user"
_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(calc.hex(), digest)


def ensure_admin(s: Session) -> None:
    user = s.scalar(select(User).where(User.login == settings.admin_login))
    if user is None:
        s.add(User(login=settings.admin_login, password_hash=hash_password(settings.admin_password)))
    elif not verify_password(settings.admin_password, user.password_hash):
        user.password_hash = hash_password(settings.admin_password)
    s.commit()


def authenticate(s: Session, login: str, password: str) -> User | None:
    user = s.scalar(select(User).where(User.login == (login or "").strip()))
    if user is None or not verify_password(password or "", user.password_hash):
        return None
    return user


def login_user(request: Request, user: User) -> None:
    request.session[SESSION_KEY] = user.login


def logout_user(request: Request) -> None:
    request.session.pop(SESSION_KEY, None)


def current_user(request: Request) -> str | None:
    return request.session.get(SESSION_KEY)


def require_api_user(request: Request) -> str:
    """Зависимость для /api и /media: без сессии — 401 JSON, а не редирект."""
    user = current_user(request)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "нужна авторизация: войдите через /login")
    return user


class LoginRedirect(Exception):
    """Страница без входа: перехватывается в main и превращается в редирект на /login."""

    def __init__(self, next_path: str) -> None:
        self.next_path = next_path


def require_page_user(request: Request) -> str:
    user = current_user(request)
    if not user:
        raise LoginRedirect(request.url.path)
    return user


def safe_next(path: str | None) -> str:
    """Куда вернуть после входа: только свой относительный путь (без open redirect)."""
    if not path or not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/"
    return path


def camera_by_key(s: Session, camera_id: int, key: str) -> Camera | None:
    cam = s.get(Camera, camera_id)
    if cam is None or not cam.ingest_key or not key:
        return None
    if not secrets.compare_digest(key.encode(), cam.ingest_key.encode()):
        return None
    return cam


def new_ingest_key() -> str:
    return secrets.token_urlsafe(24)
