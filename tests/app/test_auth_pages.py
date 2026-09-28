"""Вход, сессия, защита API и файлов, HTML-оболочки."""
from __future__ import annotations

from tests.app.conftest import series


def test_api_requires_login(env):
    env.client.post("/api/logout")
    r = env.client.get("/api/sites")
    assert r.status_code == 401
    assert "авторизац" in r.json()["detail"]
    # health открыт — им пользуется healthcheck Docker
    assert env.client.get("/api/health").status_code == 200


def test_wrong_password_rejected(env):
    env.client.post("/api/logout")
    assert env.client.post("/api/login", json={"login": "admin", "password": "nope"}).status_code == 401
    r = env.client.post("/login", data={"login": "admin", "password": "nope"}, follow_redirects=False)
    assert r.status_code == 401
    assert env.client.get("/api/me").status_code == 401


def test_form_login_redirects_to_next_and_logout(env):
    env.client.post("/api/logout")
    r = env.client.get("/sites/5", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login?next=/sites/5")
    r = env.client.post("/login", data={"login": "admin", "password": "admin", "next": "/sites/5"},
                        follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/sites/5"
    assert env.client.get("/api/me").json()["user"] == "admin"
    # open redirect не проходит
    r = env.client.post("/login", data={"login": "admin", "password": "admin", "next": "//evil.com"},
                        follow_redirects=False)
    assert r.headers["location"] == "/"
    env.client.get("/logout")
    assert env.client.get("/api/me").status_code == 401


def test_pages_render_without_templates(env):
    """Шаблоны пишет UI-агент; пока их нет — заглушка 200, а не 500."""
    for path in ("/", "/sites/1", "/cameras/2", "/frames/3", "/try", "/settings"):
        r = env.client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"]
    env.client.post("/api/logout")
    r = env.client.get("/login")
    assert r.status_code == 200 and "password" in r.text


def test_media_requires_login(env):
    site = env.site()
    cam = env.camera(site["id"])
    env.upload(cam["id"], series(1))
    url = env.frames(cam["id"])[0]["url"]
    assert env.client.get(url).status_code == 200
    env.client.post("/api/logout")
    assert env.client.get(url).status_code == 401
