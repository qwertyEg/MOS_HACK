"""Авторизация. Для прототипа — один аккаунт из конфигурации, сессия в куке.

Ролей нет намеренно: в демо один пользователь, а ролевая модель не относится
к тому, что оценивают в этой задаче.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from app.config import settings

SESSION_KEY = "user"


def check_credentials(login: str, password: str) -> bool:
    return login == settings.admin_login and password == settings.admin_password


def login_user(request: Request, login: str) -> None:
    request.session[SESSION_KEY] = login


def logout_user(request: Request) -> None:
    request.session.pop(SESSION_KEY, None)


def current_user(request: Request) -> str | None:
    return request.session.get(SESSION_KEY)


def require_user(request: Request) -> str:
    """Зависимость FastAPI. Для страниц — редирект, для API — 401."""
    user = current_user(request)
    if user:
        return user
    if request.url.path.startswith("/api/"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "нужна авторизация")
    raise HTTPException(
        status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/login"},
    )


def redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=status.HTTP_303_SEE_OTHER)
