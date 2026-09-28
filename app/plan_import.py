"""Загрузка календарного плана из текстового файла.

Формат — тот же, в котором план удобно записать и прочитать глазами:

    Подготовка территории: 12.10.2005 - 06.11.2005
    Ограждение котлована, шпунт, сваи: 07.11.2005 - 10.02.2006

Настоящий CSV тоже принимается (Excel сохраняет его с `;` и кавычками):

    "Ограждение котлована, шпунт, сваи";07.11.2005;10.02.2006

Разбор нарочно терпимый к разделителям, потому что файл пишет человек, а не
программа. Но терпимость не должна вести к тихим потерям: строка, которую не
удалось понять, попадает в отчёт с номером, а не пропадает. Оператор загрузил
план из восьми строк и увидел, что применилось семь, — ему нужно знать, какая
осталась и почему, а не гадать по диаграмме.

Даты читаются день-первым (25.09.2026 — двадцать пятое сентября), как и
показываются в интерфейсе. ISO (2026-09-25) тоже принимается: его пишут
программы, и терять его из-за формата глупо.

Модуль ничего не знает про базу — на входе текст и названия этапов объекта,
на выходе строки и отчёт. Так его можно проверить тестом, а не прогоном.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

_DATE = r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{4}"
_SEP = r"\s*(?:[-–—]|;|,|\t|\bдо\b|\bпо\b)\s*"
_LINE = re.compile(
    rf"^\s*(?P<name>.+?)\s*[:;,\t]\s*(?P<a>{_DATE}){_SEP}(?P<b>{_DATE})\s*$",
    re.IGNORECASE)

# Опечаток терпим по одной на слово, и только если слово от пяти букв. Ошибка
# в коротком слове («под» → «над») — уже другое слово, а не опечатка.
TYPO_MIN_LEN = 5


def parse_date(raw: str) -> dt.date | None:
    """`25/09/2026`, `25.09.2026`, `25-09-2026` или `2026-09-25` → дата.

    None — не разобрано. Возвращаем None, а не бросаем: вызывающему решать,
    ошибка это или пустое поле.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", raw)
        if not m:
            return None
        d, mo, y = (int(x) for x in m.groups())
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None


def norm(name: str) -> str:
    """Название для сравнения: без регистра, ё, знаков и лишних пробелов."""
    name = name.casefold().replace("ё", "е")
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def decode(data: bytes) -> str:
    """Файл из Excel на русской Windows приходит в cp1251, а не в UTF-8."""
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


@dataclass(slots=True)
class Row:
    line: int
    name: str
    start: dt.date
    end: dt.date


@dataclass(slots=True)
class Parsed:
    rows: list[Row] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse(text: str) -> Parsed:
    """Текст → строки плана и список того, что понять не удалось."""
    out = Parsed()
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().strip("﻿")
        if not line or line.startswith("#"):
            continue
        # Шапка таблицы и пояснения — строки вовсе без цифр. Строка, в
        # которой цифры есть, но разобрать её не вышло, — уже ошибка.
        if not re.search(r"\d", line):
            continue

        m = _LINE.match(line)
        if not m:
            out.errors.append(f"строка {no}: не разобрана «{line[:70]}» — "
                              "ждём «Название: дд.мм.гггг - дд.мм.гггг»")
            continue

        name = m.group("name").strip().strip('"«»\' ').strip()
        a, b = parse_date(m.group("a")), parse_date(m.group("b"))
        if a is None or b is None:
            bad = m.group("a") if a is None else m.group("b")
            out.errors.append(f"строка {no}: нет такой даты — {bad}")
            continue
        if b < a:
            out.errors.append(f"строка {no}: окончание раньше начала "
                              f"({name}: {a:%d.%m.%Y} — {b:%d.%m.%Y})")
            continue
        out.rows.append(Row(no, name, a, b))
    return out


def _distance(a: str, b: str) -> int:
    """Число правок (вставка, удаление, замена) между двумя словами."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _same_with_typos(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    return all(x == y or (max(len(x), len(y)) >= TYPO_MIN_LEN
                          and _distance(x, y) <= 1)
               for x, y in zip(ta, tb))


def match(name: str, known: dict[str, object]) -> tuple[object | None, str]:
    """Название из файла → этап. known: {нормализованное название: этап}.

    Возвращает (этап, как нашли): "exact", "fuzzy" либо (None, ""), если
    подходящего нет. Приблизительное совпадение отдельной пометкой, чтобы
    отчёт мог сказать, что именно было понято по догадке.

    Нечёткое сравнение — пословное, а не по общей похожести строк. Общая
    похожесть склеивает «подземной» с «надземной»: слова различаются двумя
    буквами из девяти, строки целиком похожи на девяносто процентов, а этапы
    совсем разные. Ошибиться здесь значит молча поставить даты не тому этапу,
    и лучше попросить поправить название, чем угадать.
    """
    key = norm(name)
    if key in known:
        return known[key], "exact"

    hits = [k for k in known if _same_with_typos(key, k)]
    if len(hits) == 1:                  # неоднозначное — не угадываем
        return known[hits[0]], "fuzzy"
    return None, ""
