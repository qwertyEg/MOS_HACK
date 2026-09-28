"""Каталог работ из «Сводного перечня строительных работ» организаторов.

Источник — `reference/work_map.csv` (раскладка Никиты, собрана tools/build_reference.py
из reference/works_catalog.xlsx и checklist.json). Это канон: каждая строка перечня
привязана к статусу `substage | unobservable | out_of_scope | header`, макроэтапу и
подэтапу чек-листа. Почему канон именно этот, а не reference/work_types.csv Дениса, —
docs/methodology.md, раздел «Каталог работ» (таблица 47 расхождений и решения).

Раскладка Дениса используется в одном месте — как запасной этап для строк, которые
в перечне отмечены «только для дорог» (out_of_scope). В проекте здания такая строка
всё же может встретиться (например, «10.11.1. Устройство ограждения стройплощадки»),
и импорт графика не должен её терять: этап берём по коду у Дениса и предупреждаем.

xlsx здесь не читаем: work_map.csv уже содержит исправленные коды (баг Excel, который
превратил «10.1.» в 10 января), а openpyxl на каждый запуск сервиса не нужен.
Та же починка кодов нужна при импорте чужих графиков — функция `normalize_code`.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.taxonomy import REFERENCE_DIR

WORK_MAP_PATH = REFERENCE_DIR / "work_map.csv"
LEGACY_WORK_TYPES_PATH = REFERENCE_DIR / "work_types.csv"

STATUSES = ("substage", "unobservable", "out_of_scope", "header")

# Excel-сериал дат: 1 = 1900-01-01, отсчёт от 1899-12-30 (учитывает ложный 29.02.1900).
_EXCEL_EPOCH = dt.date(1899, 12, 30)
_SERIAL_RANGE = (20000, 80000)   # 1954 … 2119 — всё, что правдоподобно как дата

_RU_MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5, "июн": 6, "июл": 7,
    "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_CODE_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")


@dataclass(frozen=True)
class WorkItem:
    """Строка перечня работ.

    `code` — собственный код строки («12.3.7.»); у строки-детализации без кода — "" и
    родитель в `parent_code`. Уникальный ключ строки — `key` (код или «родитель/название»),
    именно его кладём в PlanItem.work_codes.
    """
    code: str
    name: str
    level: int
    status: str                       # substage | unobservable | out_of_scope | header
    stage_id: int | None
    substage_id: str | None
    object_types: tuple[str, ...]
    parent_code: str = ""
    row: int = 0                      # номер строки в xlsx организаторов
    reason: str = ""
    fallback_stage_id: int | None = None  # этап по раскладке Дениса — только для out_of_scope

    @property
    def key(self) -> str:
        return self.code or f"{self.parent_code}/{self.name}"

    @property
    def label(self) -> str:
        """«12.3.7. Земляные работы» — так работа показывается в UI рядом с этапом."""
        return f"{self.code} {self.name}" if self.code else self.name

    @property
    def observable(self) -> bool:
        return self.status == "substage"

    @property
    def plan_stage_id(self) -> int | None:
        """Этап, в который работа сворачивается в календарном плане.

        Невидимая с камеры работа с этапом (геодезия, отселение) всё равно стоит в графике —
        её даты нужны, чтобы план этапа не «похудел». Внутренние работы без этапа (отделка,
        внутренние сети) в сравнение план/факт не входят: камера их не видит никогда.
        """
        if self.status in ("substage", "unobservable"):
            return self.stage_id
        if self.status == "out_of_scope":
            return self.fallback_stage_id
        return None


# --------------------------------------------------------------------------
# нормализация кодов и названий (общая для каталога и импорта графиков)
# --------------------------------------------------------------------------


def norm_name(text: object) -> str:
    """Название для сравнения: регистр, ё/е, пробелы и переносы строк не важны."""
    s = str(text or "").replace(" ", " ").replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip().lower()


def code_key(code: str) -> str:
    """Код для сравнения: всегда с точкой на конце («12» → «12.», «12.3.1» → «12.3.1.»)."""
    code = (code or "").strip()
    if code and _CODE_RE.match(code) and not code.endswith("."):
        code += "."
    return code


def _date_to_code(d: dt.date) -> str:
    return f"{d.day}.{d.month}."


def normalize_code(value: object) -> tuple[str, str]:
    """Код работы из ячейки → (код, пометка о починке или "").

    Excel превращает коды второго уровня («10.1.», «12.3.») в даты 10 января / 12 марта
    (в xlsx организаторов так 19 строк). Чиним в обе стороны:
    - ячейка-дата (datetime/date) → «d.m.»;
    - число-сериал Excel (45667) → дата → «d.m.»;
    - текстовая дата из CSV-выгрузки («10.01.2025», «2025-01-10», «10.янв») → «d.m.»;
    - компонент с ведущим нулём («10.01») — тоже дата: у организаторов нулей в кодах нет;
    - число 12.3 (float) → «12.3.», но с пометкой: «12.30» Excel уже необратимо превратил в 12.3.
    Неизвестный формат возвращаем как есть — пусть его увидит проверка по каталогу.
    """
    if value is None:
        return "", ""
    if isinstance(value, dt.datetime):
        return _date_to_code(value.date()), "код восстановлен из даты Excel"
    if isinstance(value, dt.date):
        return _date_to_code(value), "код восстановлен из даты Excel"
    if isinstance(value, bool):
        return str(value), ""
    if isinstance(value, int):
        if _SERIAL_RANGE[0] < value < _SERIAL_RANGE[1]:
            return _date_to_code(_EXCEL_EPOCH + dt.timedelta(days=value)), "код восстановлен из сериала даты Excel"
        return f"{value}.", ""
    if isinstance(value, float):
        if value != value:  # NaN
            return "", ""
        if value.is_integer():
            return normalize_code(int(value))
        return f"{value!r}.", "код прочитан как число — проверьте (например, «12.30» Excel хранит как 12.3)"

    s = str(value).replace(" ", " ").strip()
    if not s or s.lower() in ("nan", "nat", "none", "-", "—"):
        return "", ""
    compact = s.replace(" ", "").replace(",", ".")
    if _CODE_RE.match(compact):
        parts = [p for p in compact.rstrip(".").split(".") if p]
        # У организаторов нет ни ведущих нулей, ни четырёхзначных компонентов: «10.01» и
        # «10.01.2025» — это дата, в которую Excel превратил код «10.1.».
        datey = any(len(p) > 1 and p.startswith("0") for p in parts) or (len(parts) == 3 and len(parts[2]) == 4)
        if datey and len(parts) in (2, 3):
            day, month = int(parts[0]), int(parts[1])
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{day}.{month}.", "код восстановлен из даты"
        return code_key(compact), ""
    # текстовые даты целиком: 10.01.2025, 10.01.25, 2025-01-10, 10/01/2025 (+ время)
    head = re.split(r"[ T]", s)[0]
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return _date_to_code(dt.datetime.strptime(head, fmt).date()), "код восстановлен из даты"
        except ValueError:
            pass
    # «10.янв», «10-янв», «янв.10», «10 jan»
    m = re.fullmatch(r"(\d{1,2})[.\- ]?([a-zа-я]{3})[a-zа-я]*\.?", s.lower())
    m2 = re.fullmatch(r"([a-zа-я]{3})[a-zа-я]*[.\- ]?(\d{1,2})\.?", s.lower())
    if m and m.group(2) in _RU_MONTHS:
        return f"{int(m.group(1))}.{_RU_MONTHS[m.group(2)]}.", "код восстановлен из даты «дд.ммм»"
    if m2 and m2.group(1) in _RU_MONTHS:
        return f"{int(m2.group(2))}.{_RU_MONTHS[m2.group(1)]}.", "код восстановлен из даты «ммм.дд»"
    return s, ""


def ancestors(code: str) -> list[str]:
    """«12.3.7.5.» → [«12.3.7.», «12.3.», «12.»] — от ближнего к дальнему."""
    parts = [p for p in code_key(code).rstrip(".").split(".") if p]
    return [".".join(parts[:i]) + "." for i in range(len(parts) - 1, 0, -1)]


def code_level(code: str) -> int:
    return len([p for p in code_key(code).rstrip(".").split(".") if p])


# --------------------------------------------------------------------------
# загрузка
# --------------------------------------------------------------------------


def _int_or_none(value: str) -> int | None:
    value = (value or "").strip()
    try:
        return int(value)
    except ValueError:
        return None


@lru_cache(maxsize=4)
def _legacy_stage_by_row(path: Path = LEGACY_WORK_TYPES_PATH) -> dict[int, int]:
    """Номер строки xlsx → макроэтап по раскладке Дениса (только 1..8)."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row, stage = _int_or_none(r.get("row", "")), _int_or_none(r.get("macro_stage_id", ""))
            if row and stage and 1 <= stage <= 8:
                out[row] = stage
    return out


@lru_cache(maxsize=4)
def _load(path: Path = WORK_MAP_PATH) -> tuple[WorkItem, ...]:
    legacy = _legacy_stage_by_row()
    items: list[WorkItem] = []
    levels: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            code = (r.get("code") or "").strip()
            parent = (r.get("parent_code") or "").strip()
            if code:
                level = code_level(code)
                levels[code_key(code)] = level
            else:
                level = levels.get(code_key(parent), code_level(parent)) + 1
            row = _int_or_none(r.get("row", "")) or 0
            status = (r.get("status") or "").strip()
            items.append(WorkItem(
                code=code,
                name=re.sub(r"\s+", " ", r.get("name") or "").strip(),
                level=level,
                status=status,
                stage_id=_int_or_none(r.get("stage_id", "")),
                substage_id=(r.get("substage_id") or "").strip() or None,
                object_types=tuple(t for t in (r.get("object_types") or "").split("|") if t),
                parent_code=parent,
                row=row,
                reason=(r.get("reason") or "").strip(),
                fallback_stage_id=legacy.get(row) if status == "out_of_scope" else None,
            ))
    return tuple(items)


def load() -> list[WorkItem]:
    """Все 377 строк перечня в порядке xlsx."""
    return list(_load())


@lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, WorkItem], dict[str, WorkItem], dict[str, WorkItem]]:
    by_code: dict[str, WorkItem] = {}
    by_key: dict[str, WorkItem] = {}
    names: dict[str, list[WorkItem]] = {}
    for it in _load():
        if it.code:
            by_code.setdefault(code_key(it.code), it)
        by_key.setdefault(_key_norm(it.code or it.parent_code, it.name, bool(it.code)), it)
        names.setdefault(norm_name(it.name), []).append(it)
    # По одному названию ищем только однозначные строки: «Армокаркас» есть в трёх разделах.
    by_name = {n: lst[0] for n, lst in names.items() if len(lst) == 1}
    return by_code, by_key, by_name


def _key_norm(code: str, name: str, own: bool) -> str:
    return code_key(code) if own else f"{code_key(code)}/{norm_name(name)}"


def get(key: str) -> WorkItem | None:
    """Строка по ключу из PlanItem.work_codes: код («12.3.7.») или «родитель/название»."""
    by_code, by_key, _ = _indexes()
    if "/" in key:
        parent, _, name = key.partition("/")
        return by_key.get(_key_norm(parent, name, False))
    return by_code.get(code_key(key))


def label(key: str) -> str:
    """Подпись работы для UI; неизвестный ключ возвращается как есть."""
    it = get(key)
    return it.label if it else key


def resolve(code: str = "", name: str = "", parent_code: str = "") -> tuple[WorkItem | None, str]:
    """Найти строку перечня для строки чужого графика → (строка, как нашли).

    Порядок: точный код → «родитель/название» для строки-детализации → ближайший
    предок кода (код глубже перечня: «12.3.7.5.») → однозначное название.
    `how` пуст для точного совпадения, иначе объясняет, как нашли, — импорт превращает
    это в предупреждение, чтобы приблизительная привязка не была молчаливой.
    """
    by_code, by_key, by_name = _indexes()
    ck = code_key(code)
    if ck and ck in by_code:
        return by_code[ck], ""
    if not ck and parent_code and name:
        hit = by_key.get(_key_norm(parent_code, name, False))
        if hit:
            return hit, ""
    if ck:
        for anc in ancestors(ck):
            hit = by_code.get(anc)
            if hit and hit.status != "header":
                return hit, f"кода {code} нет в перечне — отнесён по родительскому коду {hit.code}"
    if name:
        hit = by_name.get(norm_name(name))
        if hit:
            return hit, f"найден по названию: {hit.label}"
    return None, ""


def stages_under(code: str) -> set[int]:
    """Этапы, в которые сворачиваются потомки раздела («12.3.» → {2, 3, 4}).

    Нужно, когда в графике стоит сам раздел без детализации: «10. Подготовка территории»
    однозначно этап 1, а «12.3. Устройство подземной части» — три этапа, и молча выбрать
    один из них нельзя.
    """
    key = code_key(code)
    out = set()
    for it in _load():
        own = code_key(it.code) if it.code else code_key(it.parent_code)
        if own.startswith(key) and it.plan_stage_id is not None:
            out.add(it.plan_stage_id)
    return out


def works_for_stage(stage_id: int, substage_id: str | None = None, *,
                    include_unobservable: bool = False) -> list[WorkItem]:
    """Работы перечня, которые на камере означают этап (или подэтап).

    По умолчанию — только наблюдаемые (substage): это их показывает UI рядом с этапом
    («12.3.1. Устройство котлована»). Невидимые работы этапа — по флагу.
    """
    wanted = {"substage", "unobservable"} if include_unobservable else {"substage"}
    out = []
    for it in _load():
        if it.status not in wanted or it.stage_id != int(stage_id):
            continue
        if substage_id is not None and it.substage_id != substage_id:
            continue
        out.append(it)
    return out


def legacy_discrepancies() -> list[dict]:
    """Строки, которые Никита (work_map.csv) и Денис (work_types.csv) разложили по-разному.

    Считаем только расхождения по существу: разные этапы у видимой работы, видимая у одного
    и исключённая у другого. Строки «только для дорог» (у Никиты out_of_scope по колонкам
    типа объекта, у Дениса — этап по коду) сюда не входят: это не разная раскладка, а разный
    охват, он решён запасным этапом (`fallback_stage_id`). Итог — 47 строк, их разбор —
    docs/methodology.md.
    """
    if not LEGACY_WORK_TYPES_PATH.exists():
        return []
    legacy = {}
    with LEGACY_WORK_TYPES_PATH.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            legacy[_int_or_none(r["row"])] = r
    out = []
    for it in _load():
        d = legacy.get(it.row)
        if d is None:
            continue
        d_stage = _int_or_none(d.get("macro_stage_id", ""))
        d_obs = d_stage is not None and 1 <= d_stage <= 8
        kind = None
        if it.status == "substage" and d_obs and it.stage_id != d_stage:
            kind = "разные этапы"
        elif it.status == "substage" and not d_obs:
            kind = "Никита: видно, Денис: исключено"
        elif it.status == "unobservable" and d_obs:
            kind = "Никита: не видно с камеры, Денис: видно"
        if kind:
            out.append({
                "row": it.row, "key": it.key, "name": it.name, "kind": kind,
                "nikita": f"{it.status} {it.stage_id or ''} {it.substage_id or ''}".strip(),
                "denis": d.get("macro_stage_id", ""), "denis_reason": d.get("reason", ""),
            })
    return out
