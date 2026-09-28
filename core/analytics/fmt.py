"""Русские формулировки для объяснений: длительности, даты, списки, склонения.

Отклонение читает прораб, а не разработчик: «2 ч 10 мин», «камеры 1, 2», «3 снимка».
"""
from __future__ import annotations

import datetime as dt
import re

from core import taxonomy


def plural(n: int | float, forms: tuple[str, str, str]) -> str:
    """plural(3, ("снимок", "снимка", "снимков")) → «снимка»."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def count(n: int, forms: tuple[str, str, str]) -> str:
    return f"{n} {plural(n, forms)}"


SHOTS = ("снимок", "снимка", "снимков")
DAYS = ("день", "дня", "дней")
FRAMES = ("кадр", "кадра", "кадров")
FRAMES_LOC = ("кадре", "кадрах", "кадрах")   # «на 6 кадрах»


def hours(h: float) -> str:
    """1.1667 → «1 ч 10 мин»; 0.75 → «45 мин»; 30.2 → «30 ч»."""
    minutes = int(round(max(0.0, h) * 60))
    hh, mm = divmod(minutes, 60)
    if hh >= 24:
        return f"{hh} ч"
    if hh and mm:
        return f"{hh} ч {mm} мин"
    if hh:
        return f"{hh} ч"
    return f"{mm} мин"


def days(n: float) -> str:
    """Модуль числа дней словами: «1 день», «3 дня», «11 дней» (знак передаёт текст вокруг)."""
    k = int(round(abs(n)))
    return f"{k} {plural(k, DAYS)}"


def date(d: dt.date | dt.datetime | None) -> str:
    if d is None:
        return "—"
    if isinstance(d, dt.datetime):
        d = d.date()
    return f"{d:%d.%m.%Y}"


def moment(t: dt.datetime | None) -> str:
    """Локальное время уже переведено вызывающим; здесь только формат «28.09 14:20»."""
    return "—" if t is None else f"{t:%d.%m %H:%M}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{round(100 * x)} %"


def eq(cls: str) -> str:
    """Название техники со строчной буквы и без пояснений в скобках: «грузовик»."""
    name = re.sub(r"\s*\(.*?\)", "", taxonomy.equipment_name(cls)).strip()
    return name[:1].lower() + name[1:] if name else cls


def eq_list(classes) -> str:
    return ", ".join(eq(c) for c in classes)


def stage(stage_id: int | None) -> str:
    return f"«{taxonomy.stage_name(stage_id)}»"


def cameras(ids, names: dict | None = None) -> str:
    ids = [i for i in ids if i is not None]
    if not ids:
        return "камера не указана"
    names = names or {}
    shown = [str(names.get(str(i), i)) for i in ids]
    return ("камера " if len(shown) == 1 else "камеры ") + ", ".join(shown)
