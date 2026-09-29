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
import logging
import secrets
import threading
import time

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Camera, Setting, User

log = logging.getLogger(__name__)
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
    if settings.admin_password in WEAK_PASSWORDS or len(settings.admin_password) < 8:
        log.warning("ADMIN_PASSWORD по умолчанию или короче 8 символов — для доступа из сети задайте свой в .env")
    load_session_state(s)


# --------------------------------------------------------------------------
# сессии: подпись, отзыв при выходе, смена пароля
# --------------------------------------------------------------------------

WEAK_SECRETS = {"", "dev-secret-change-me", "change-me-long-random-string", "change-me", "secret"}
WEAK_PASSWORDS = {"admin", "password", "123456", "change-me"}
_random_secret = secrets.token_urlsafe(32)
_pw_tags: dict[str, str] = {}          # логин → отпечаток хеша пароля (сменили пароль — старые сессии мертвы)
_revoked: dict[str, float] = {}        # id сессии → до какого времени помнить (вышли — кука больше не пускает)
_REVOKED_KEY = "auth.revoked_sessions"


def session_secret() -> str:
    """Ключ подписи куки. Ключ по умолчанию из config/.env.example известен всем — с ним
    куку «admin» можно подделать без пароля; тогда берём случайный ключ процесса
    (сессии сбрасываются при перезапуске) и предупреждаем в журнале."""
    if settings.secret_key in WEAK_SECRETS or len(settings.secret_key) < 16:
        return _random_secret
    return settings.secret_key


def _pw_tag(password_hash: str) -> str:
    return hmac.new(session_secret().encode(), password_hash.encode(), "sha256").hexdigest()[:16]


def load_session_state(s: Session) -> None:
    _pw_tags.clear()
    for user in s.scalars(select(User)):
        _pw_tags[user.login] = _pw_tag(user.password_hash)
    row = s.get(Setting, _REVOKED_KEY)
    now = time.time()
    _revoked.clear()
    if row is not None and isinstance(row.value, dict):
        _revoked.update({k: float(v) for k, v in row.value.items() if isinstance(v, (int, float)) and v > now})


def _revoke(s: Session | None, sid: str) -> None:
    now = time.time()
    _revoked[sid] = now + settings.session_max_age_h * 3600
    for k in [k for k, v in _revoked.items() if v <= now]:
        _revoked.pop(k, None)
    if s is None:
        return
    row = s.get(Setting, _REVOKED_KEY)
    if row is None:
        s.add(Setting(key=_REVOKED_KEY, value=dict(_revoked)))
    else:
        row.value = dict(_revoked)
    s.commit()


def authenticate(s: Session, login: str, password: str) -> User | None:
    user = s.scalar(select(User).where(User.login == (login or "").strip()))
    with _hashing:
        # Неизвестный логин тоже считает хеш: по времени ответа не узнать, есть ли такой пользователь.
        ok = verify_password(password or "", user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok:
        return None
    return user


# PBKDF2 на 200 000 итераций — ~0.1 с CPU. Одновременно считаем не больше двух хешей:
# поток неверных паролей не должен съедать процессор, нужный моделям и страницам.
_hashing = threading.BoundedSemaphore(2)
_DUMMY_HASH = hash_password(secrets.token_hex(8))


class TooManyAttempts(Exception):
    def __init__(self, wait_s: int) -> None:
        super().__init__(wait_s)
        self.wait_s = wait_s


class LoginLimiter:
    """Лимит неверных паролей: после N подряд с одного адреса на логин — пауза.

    Пока пауза, пароль не проверяется вовсе (ответ 429 сразу), так что перебор
    не грузит процессор и не подбирает пароль. Отдельно — общий лимит на адрес
    (перебор логинов). В памяти процесса: сервис один, после рестарта счёт с нуля.
    """

    def __init__(self) -> None:
        self._fails: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def _window(self) -> float:
        return float(settings.login_lockout_s)

    def _recent(self, key: tuple[str, str], now: float) -> list[float]:
        return [t for t in self._fails.get(key, []) if now - t < self._window()]

    def wait_s(self, ip: str, login: str) -> int:
        now = time.monotonic()
        limit = max(1, settings.login_max_failures)
        with self._lock:
            per_login = self._recent((ip, login), now)
            per_ip = self._recent((ip, "*"), now)
        blocked = [ts[-limit] for ts in (per_login,) if len(ts) >= limit]
        blocked += [per_ip[-4 * limit]] if len(per_ip) >= 4 * limit else []
        if not blocked:
            return 0
        return max(1, int(max(blocked) + self._window() - now + 0.999))

    def failed(self, ip: str, login: str) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._fails) > 10_000:          # память ограничена: старые записи выбрасываем
                self._fails = {k: v for k, v in self._fails.items() if v and now - v[-1] < self._window()}
            for key in ((ip, login), (ip, "*")):
                self._fails[key] = self._recent(key, now)[-64:] + [now]

    def succeeded(self, ip: str, login: str) -> None:
        with self._lock:
            self._fails.pop((ip, login), None)
            self._fails.pop((ip, "*"), None)

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()


limiter = LoginLimiter()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def login_attempt(request: Request, s: Session, login: str, password: str) -> User | None:
    """Вход с лимитом попыток. TooManyAttempts — пауза (→ 429), None — неверный пароль."""
    ip, name = client_ip(request), (login or "").strip().lower()[:64]
    wait = limiter.wait_s(ip, name)
    if wait:
        raise TooManyAttempts(wait)
    user = authenticate(s, login, password)
    if user is None:
        limiter.failed(ip, name)
    else:
        limiter.succeeded(ip, name)
    return user


def login_user(request: Request, user: User) -> None:
    request.session.clear()             # новая сессия на вход: старый id (фиксация сессии) не переживает логин
    request.session[SESSION_KEY] = user.login
    request.session["sid"] = secrets.token_urlsafe(16)
    request.session["pw"] = _pw_tags.setdefault(user.login, _pw_tag(user.password_hash))


def logout_user(request: Request, s: Session | None = None) -> None:
    """Выход отзывает сессию на сервере: копия куки (перехваченная, из другого браузера)
    после выхода тоже не пускает."""
    sid = request.session.get("sid")
    if sid:
        _revoke(s, str(sid))
    request.session.clear()


def current_user(request: Request) -> str | None:
    login = request.session.get(SESSION_KEY)
    if not login:
        return None
    sid, tag = request.session.get("sid"), request.session.get("pw")
    if not sid or sid in _revoked or tag is None or tag != _pw_tags.get(login):
        request.session.clear()
        return None
    return login


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
