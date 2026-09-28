"""Проверки UI без браузера: палитра классов против справочника и сборка шаблонов.

    python -m pytest -q tests/ui/test_ui_static.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("overview", "site", "camera", "frame", "try", "settings", "login")


def test_palette_covers_checklist_classes():
    """Ключи app/static/js/palette.js — ровно 21 класс техники из reference/checklist.json."""
    js = (ROOT / "app/static/js/palette.js").read_text(encoding="utf-8")
    block = js[js.index("CLASS_COLORS = {"):js.index("};", js.index("CLASS_COLORS = {"))]
    keys = set(re.findall(r"^\s*([a-z_]+):\s*\"#[0-9a-f]{6}\"", block, re.M))
    ref = json.loads((ROOT / "reference/checklist.json").read_text(encoding="utf-8"))
    assert keys == {e["key"] for e in ref["equipment"]}


def test_palette_colors_unique():
    js = (ROOT / "app/static/js/palette.js").read_text(encoding="utf-8")
    block = js[js.index("CLASS_COLORS = {"):js.index("};", js.index("CLASS_COLORS = {"))]
    colors = re.findall(r"\"(#[0-9a-f]{6})\"", block)
    assert len(colors) == 21 and len(colors) == len(set(colors))


class _Req:
    query_params: dict = {}
    url = "/"


@pytest.mark.parametrize("page", PAGES)
def test_page_templates_render(page):
    """Каждая страница собирается с контекстом бэкенда {request, page, user, *_id, app_version}."""
    env = Environment(loader=FileSystemLoader(str(ROOT / "app/templates")), autoescape=True)
    ctx = {"request": _Req(), "page": page, "user": None if page == "login" else "admin", "app_version": "test",
           "site_id": 1, "camera_id": 1, "frame_id": 1, "next": "/", "error": "Неверный логин или пароль"}
    html = env.get_template(f"pages/{page}.html").render(**ctx)
    assert "/static/css/app.css" in html and "alpine.min.js" in html
    if page == "login":
        assert 'action="/login"' in html and 'name="password"' in html and "Неверный логин" in html
    else:
        assert 'x-data="svHeader"' in html


def test_css_built():
    css = (ROOT / "app/static/css/app.css").read_text(encoding="utf-8")
    assert len(css) > 20000 and "--bg" in css
