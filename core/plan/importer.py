"""Импорт календарного графика (CSV/XLSX) и демо-план.

Реальный график подрядчика — это не 8 строк «этап, начало, окончание», а сотни строк
с многоуровневыми кодами перечня организаторов («12.3.7.»), строками-детализациями без
кода, сводными строками разделов, пустыми датами и техникой в свободной форме. Импорт:

1. находит строку заголовков по названиям колонок (рус./англ. варианты, не обязательно
   первая строка — у организаторов заголовок в третьей);
2. чинит коды, которые Excel превратил в даты («10.1.» → 10 января);
3. отбрасывает сводные строки, у которых есть датированные потомки (иначе раздел
   «12.3.7. Земляные работы» растянул бы котлован на даты обратной засыпки — а она этап 4);
4. относит каждую строку к макроэтапу через каталог (core.plan.catalog), а если кода нет —
   по названию и ключевым словам;
5. сворачивает в 8 макроэтапов: начало = min, окончание = max, work_codes — ключи работ.

Принцип: плохой файл не роняет сервис и ничего не теряет молча. Любая строка, которую
не удалось понять или пришлось додумать, даёт предупреждение с номером строки.
Пустая дата (NaT в pandas, "" в CSV) — это None, а не строка 'NaT' в базе (баг ветки
api-solution, после которого объект переставал открываться).

pandas здесь намеренно не используется: CSV читает stdlib, XLSX — openpyxl.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from dataclasses import dataclass, field
from functools import lru_cache

from core import taxonomy
from core.contracts import PlanItem
from core.plan import catalog, norms

MAX_HEADER_SCAN = 30
_EXCEL_EPOCH = dt.date(1899, 12, 30)
_NULLS = {"", "nan", "nat", "none", "null", "-", "—", "–", "н/д", "нет", "n/a"}

# --------------------------------------------------------------------------
# колонки
# --------------------------------------------------------------------------

# (роль, точные названия, подстроки). Порядок ролей важен: «кол-во техники» — это техника,
# а не количество; «наименование этапа» — название, а не номер этапа.
_ROLES: list[tuple[str, set[str], tuple[str, ...]]] = [
    ("equipment", {"техника", "механизмы", "машины", "equipment", "machines", "machinery"},
     ("техник", "механизм", "equipment", "machiner")),
    ("name", {"наименование", "вид работ", "виды работ", "работа", "работы", "название", "описание",
              "name", "task", "work", "activity", "задача", "наименование работ"},
     ("наименован", "вид работ", "виды работ", "название", "task name", "description")),
    ("start", {"начало", "старт", "start", "plan_start", "start date", "с", "дата начала", "begin"},
     ("начал", "старт", "start", "begin")),
    ("end", {"окончание", "финиш", "конец", "end", "finish", "plan_end", "end date", "по",
             "дата окончания", "завершение"},
     ("окончан", "финиш", "конец", "заверш", "finish", "end date", "plan_end")),
    ("duration", {"длительность", "продолжительность", "дней", "дн", "duration", "days", "срок"},
     ("длительн", "продолжительн", "duration")),
    ("qty", {"кол-во", "количество", "qty", "count", "шт", "кол во"}, ("кол-во", "количеств")),
    ("stage", {"этап", "stage", "stage_id", "макроэтап", "stage id"}, ()),
    ("code", {"код", "№ п/п", "№п/п", "n п/п", "номер", "№", "шифр", "code", "wbs", "id", "п/п", "no", "#"},
     ("п/п", "шифр", "код работ", "wbs")),
]
# Факт — не план: колонки «Начало (факт)» пропускаем, чтобы не принять их за плановые.
_SKIP_MARKERS = ("факт", "actual", "fact")


def _norm_header(value: object) -> str:
    s = catalog.norm_name(value)
    return s.strip(" .:;,*()[]")


def _role_of(header: str) -> str | None:
    if not header or any(m in header for m in _SKIP_MARKERS):
        return None
    for role, exact, subs in _ROLES:
        if header in exact:
            return role
    for role, exact, subs in _ROLES:
        if any(s in header for s in subs):
            return role
    return None


def _detect_header(rows: list[list]) -> tuple[int, dict[str, int]] | None:
    """Строка заголовков → (индекс, {роль: колонка}). Нужны (название|код|этап) и (начало|окончание)."""
    best = None
    for i, row in enumerate(rows[:MAX_HEADER_SCAN]):
        roles: dict[str, int] = {}
        for j, cell in enumerate(row):
            role = _role_of(_norm_header(cell))
            if role and role not in roles:
                roles[role] = j
        has_id = any(r in roles for r in ("name", "code", "stage"))
        has_date = any(r in roles for r in ("start", "end"))
        if has_id and has_date and (best is None or len(roles) > len(best[1])):
            best = (i, roles)
    return best


# --------------------------------------------------------------------------
# значения ячеек
# --------------------------------------------------------------------------


def _is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip().lower() in _NULLS


def parse_date(value: object) -> tuple[dt.date | None, bool]:
    """Ячейка → (дата | None, была ли в ячейке непустая, но нераспознанная запись).

    Понимает datetime/date, сериал Excel, «дд.мм.гггг», «дд.мм.гг», ISO, «дд/мм/гггг»,
    с временем после пробела или «T», и «10 января 2026». Пустое, NaN, 'NaT' → None.
    """
    if _is_null(value):
        return None, False
    if isinstance(value, dt.datetime):
        return value.date(), False
    if isinstance(value, dt.date):
        return value, False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 20000 < float(value) < 80000:
            return _EXCEL_EPOCH + dt.timedelta(days=int(float(value))), False
        return None, True
    s = str(value).strip()
    head = re.split(r"[ T]", s)[0]
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(head, fmt).date(), False
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})\s+([а-яa-z]+)\.?\s+(\d{4})(?:\s*г\.?)?", s.lower())
    if m:
        month = catalog._RU_MONTHS.get(m.group(2)[:3])
        if month:
            try:
                return dt.date(int(m.group(3)), month, int(m.group(1))), False
            except ValueError:
                pass
    return None, True


def _parse_number(value: object) -> float | None:
    if _is_null(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(m.group(0).replace(",", ".")) if m else None


# --------------------------------------------------------------------------
# техника в свободной форме
# --------------------------------------------------------------------------

_AMBIGUOUS = {"кран"}  # башенный? автокран? манипулятор? — лучше спросить, чем угадать


@lru_cache(maxsize=1)
def _equipment_aliases() -> tuple[tuple[str, str], ...]:
    """(псевдоним, ключ), длинные первыми: «экскаватор-погрузчик» раньше «экскаватора»."""
    pairs: dict[str, str] = {}
    for key, e in taxonomy.equipment().items():
        pairs[catalog.norm_name(key)] = key
        pairs[catalog.norm_name(key.replace("_", " "))] = key
        for part in re.split(r"[()/,]", e.name):
            part = catalog.norm_name(part)
            if len(part) >= 4:
                pairs[part] = key
    for ru, key in taxonomy.RU_ALIASES.items():
        pairs[catalog.norm_name(ru)] = key
    pairs.update({
        "миксер": "concrete_mixer", "автомиксер": "concrete_mixer", "бетоновоз": "concrete_mixer",
        "бетононасос": "concrete_pump", "манипулятор": "crane_manipulator", "кму": "crane_manipulator",
        "автовышка": "aerial_platform", "автогидроподъемник": "aerial_platform", "вышка": "aerial_platform",
        "фасадный подъемник": "facade_hoist", "строительный подъемник": "facade_hoist", "люлька": "facade_hoist",
        "грузопассажирский подъемник": "facade_hoist", "подъемник": "facade_hoist",
        "катки": "roller", "катка": "roller", "каток": "roller",
        "грейдер": "grader", "автогрейдер": "grader", "копер": "pile_driver", "вибропогружатель": "pile_driver",
        "сваебой": "pile_driver", "буровая": "drilling_rig", "бортовой": "truck", "длинномер": "truck",
        "трал": "truck", "тягач": "truck", "грузовой автомобиль": "truck", "грузовик": "truck",
        "телескопический погрузчик": "telehandler", "телескопический": "telehandler",
        "мини-погрузчик": "skid_steer", "минипогрузчик": "skid_steer", "бобкэт": "skid_steer",
        "фронтальный погрузчик": "wheel_loader", "погрузчик": "wheel_loader",
        "экскаватор-погрузчик": "backhoe_loader", "jcb": "backhoe_loader",
        "асфальтоукладчик": "asphalt_paver", "укладчик": "asphalt_paver",
        "гусеничный кран": "crawler_crane", "башенный кран": "tower_crane", "автокран": "mobile_crane",
    })
    return tuple(sorted(pairs.items(), key=lambda kv: -len(kv[0])))


def _stem(alias: str) -> str:
    # Русские падежи и числа меняют 1–2 последние буквы: «экскаватора», «самосвалов».
    return alias if len(alias) <= 5 or not re.search(r"[а-я]$", alias) else alias[:-2]


# Количество — отдельно стоящее число 1–99 («экскаватор 2», «×4», «3 шт»), а не номер модели
# («КБ-405», «ZX330»): иначе кран КБ-405 превратился бы в 405 кранов.
_COUNT_RE = re.compile(r"(?:^|[\s×x*:=(])(\d{1,2})(?:\s*(?:шт|ед|единиц)[а-я.]*)?(?=$|[\s),.;])")


def equipment_key(text: str) -> str | None:
    """Название техники в свободной форме → ключ словаря (None, если не узнали)."""
    t = catalog.norm_name(text)
    t = re.sub(r"\d+|[×*:=]|\bx\b|\bшт\b\.?|\bед\b\.?|\bединиц[аы]?\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -–—.()")
    if not t or t in _AMBIGUOUS:
        return None
    for alias, key in _equipment_aliases():
        if alias == t or re.search(r"(?:^|[\s\-])" + re.escape(_stem(alias)), t):
            return key
    return None


def parse_equipment(text: object, qty: object = None) -> tuple[dict[str, int], list[str]]:
    """«Экскаватор 2; самосвал ×4, каток» → ({excavator: 2, dump_truck: 4, roller: 1}, нераспознанные)."""
    if _is_null(text):
        return {}, []
    out: dict[str, int] = {}
    unknown: list[str] = []
    tokens = [t.strip() for t in re.split(r"[;,\n/+]|\s+и\s+", str(text)) if t.strip()]
    qty_n = _parse_number(qty)
    for tok in tokens:
        key = equipment_key(tok)
        if key is None:
            unknown.append(tok)
            continue
        m = _COUNT_RE.search(catalog.norm_name(tok))
        n = int(m.group(1)) if m else (int(qty_n) if qty_n and len(tokens) == 1 else 1)
        out[key] = max(out.get(key, 0), max(1, n))
    return out, unknown


# --------------------------------------------------------------------------
# отнесение строки к макроэтапу
# --------------------------------------------------------------------------

# Если у строки нет кода перечня (свой WBS подрядчика), а название не совпало с перечнем —
# относим по ключевым словам. Порядок важен: «засыпка пазух» раньше «котлован»,
# «кровл» раньше «монолит».
_KEYWORDS: list[tuple[str, int]] = [
    ("обратная засыпка", 4), ("засыпк", 4), ("гидроизоляц", 4),
    ("шпунт", 2), ("свай", 2), ("сваи", 2), ("стена в грунте", 2), ("ограждение котлован", 2),
    ("котлован", 3), ("разработка грунта", 3), ("земляные", 3), ("выемк", 3),
    ("фундаментн", 4), ("подземн", 4), ("ниже отм", 4), ("ниже нуля", 4),
    ("кровл", 6), ("фасад", 7), ("остеклен", 7), ("витраж", 7), ("окон", 7), ("облицовк", 7),
    ("наружные сети", 8), ("благоустр", 8), ("озеленен", 8), ("асфальт", 8), ("проезд", 8), ("тротуар", 8),
    ("надземн", 5), ("каркас", 5), ("монолит", 5), ("перекрыти", 5), ("этаж", 5),
    ("снос", 1), ("демонтаж", 1), ("вырубк", 1), ("подготовк", 1), ("ограждение площадк", 1),
    ("бытов", 1), ("вынос сетей", 1),
]


def _stage_by_keywords(name: str) -> int | None:
    n = catalog.norm_name(name)
    for kw, stage in _KEYWORDS:
        if kw in n:
            return stage
    return None


# Другие названия макроэтапов: справочник Дениса, PLAN.md, обиходные формулировки графиков.
_STAGE_ALIASES = {
    "подготовка": 1, "подготовительные работы": 1, "подготовительный период": 1,
    "сваи": 2, "свайное поле": 2, "шпунт": 2, "ограждение котлована": 2, "свайные работы": 2,
    "котлован": 3, "устройство котлована": 3, "разработка котлована": 3, "земляные работы": 3,
    "подземная часть": 4, "монолит подземной части": 4, "нулевой цикл": 4,
    "надземная часть": 5, "монолит надземной части": 5, "каркас": 5, "монолитный каркас": 5,
    "кровля": 6, "кровельные работы": 6,
    "фасад": 7, "фасадные работы": 7, "остекление": 7,
    "благоустройство": 8, "наружные сети": 8, "благоустройство территории": 8,
}


def _stage_by_name(value: str) -> int | None:
    """Номер или название макроэтапа: 1..8, «4.1», «Этап 3», «Кровля», ключ этапа."""
    s = catalog.norm_name(value).strip(" .")
    if not s:
        return None
    m = re.match(r"^(?:этап\s*)?(\d+)(?:[.,]\d+)*\.?$", s)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 8 else None
    if s in _STAGE_ALIASES:
        return _STAGE_ALIASES[s]
    for st in taxonomy.stages().values():
        if s in (catalog.norm_name(st.name), st.key):
            return st.id
    # «Земляные работы» — начало названия этапа 3; префикс принимаем, только если он однозначен
    hits = {st.id for st in taxonomy.stages().values() if len(s) >= 8 and catalog.norm_name(st.name).startswith(s)}
    return hits.pop() if len(hits) == 1 else None


# --------------------------------------------------------------------------
# чтение файлов
# --------------------------------------------------------------------------


def _read_xlsx(data: bytes) -> tuple[list[list], list[str]]:
    import openpyxl  # ленивый импорт: модулю плана он нужен только для xlsx

    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    best: tuple[int, list[list]] | None = None
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        found = _detect_header(rows)
        score = len(found[1]) if found else 0
        if best is None or score > best[0]:
            best = (score, rows)
    wb.close()
    return (best[1] if best else []), []


def _decode(data: bytes) -> tuple[str, list[str]]:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc), []
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), ["кодировка файла не распознана (не UTF-8 и не Windows-1251) — "
                                    "русские буквы могут быть искажены, сохраните CSV в UTF-8"]


def _read_csv(data: bytes) -> tuple[list[list], list[str]]:
    text, warnings = _decode(data)
    sample = text[:8192]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        delim = max(";,\t", key=first.count)
    return [row for row in csv.reader(io.StringIO(text), delimiter=delim)], warnings


# --------------------------------------------------------------------------
# разбор
# --------------------------------------------------------------------------


@dataclass
class _Row:
    line: int                     # номер строки в файле, как его видит пользователь
    code: str
    parent: str                   # ближайший код выше (для строк-детализаций)
    name: str
    start: dt.date | None
    end: dt.date | None
    equipment: dict[str, int]
    stage_hint: int | None
    notes: list[str] = field(default_factory=list)

    @property
    def dated(self) -> bool:
        return self.start is not None or self.end is not None


def _cell(row: list, idx: int | None):
    return row[idx] if idx is not None and idx < len(row) else None


def _extract_rows(rows: list[list], header_idx: int, cols: dict[str, int], warnings: list[str]) -> list[_Row]:
    out: list[_Row] = []
    parent = ""
    for i in range(header_idx + 1, len(rows)):
        raw = rows[i]
        line = i + 1
        if not raw or all(_is_null(c) for c in raw):
            continue
        code, code_note = catalog.normalize_code(_cell(raw, cols.get("code")))
        name = re.sub(r"\s+", " ", str(_cell(raw, cols.get("name")) or "")).strip()
        if _is_null(name):
            name = ""
        # код в начале названия: «12.3.1. Устройство котлована» в одной колонке.
        # Одноуровневый «1. Подготовка» принимаем за код, только если отдельной колонки кода нет,
        # и родителем следующих строк он не становится: это нумерация списка, а не раздел.
        list_number = False
        if not code and name:
            m = re.match(r"^(\d+(?:\.\d+)*\.)\s+(.+)$", name)
            if m and ("." in m.group(1).rstrip(".") or "code" not in cols):
                code, name = catalog.code_key(m.group(1)), m.group(2).strip()
                list_number = catalog.code_level(code) == 1
        notes = [code_note] if code_note else []

        start, bad_s = parse_date(_cell(raw, cols.get("start")))
        end, bad_e = parse_date(_cell(raw, cols.get("end")))
        label = f"строка {line}" + (f" ({code} {name})".rstrip() if code or name else "")
        if bad_s or bad_e:
            warnings.append(f"{label}: дата не распознана — «{_cell(raw, cols.get('start')) if bad_s else _cell(raw, cols.get('end'))}»; "
                            "строка учтена без этой даты")
        duration = _parse_number(_cell(raw, cols.get("duration")))
        if duration and duration > 0:
            span = dt.timedelta(days=max(0, round(duration) - 1))
            if start and not end:
                end = start + span
            elif end and not start:
                start = end - span
        if start and end and start > end:
            warnings.append(f"{label}: начало {start:%d.%m.%Y} позже окончания {end:%d.%m.%Y} — даты переставлены местами, проверьте")
            start, end = end, start

        eq, unknown = parse_equipment(_cell(raw, cols.get("equipment")), _cell(raw, cols.get("qty")))
        if unknown:
            warnings.append(f"{label}: не распознана техника: {', '.join(unknown)}")

        hint = None
        stage_raw = _cell(raw, cols.get("stage"))
        if not _is_null(stage_raw):
            hint = _stage_by_name(str(stage_raw))
            if hint is None:
                # в колонке «этап» могут стоять коды перечня — это не номер этапа (баг: «10.2.» → этап 10)
                c, _ = catalog.normalize_code(stage_raw)
                if catalog.get(c) is not None and not code:
                    code = c

        if code:
            parent = "" if list_number else code
            row_parent = ""
        else:
            row_parent = parent
        out.append(_Row(line, code, row_parent, name, start, end, eq, hint, notes))
    return out


def _summary_codes(rows: list[_Row]) -> set[str]:
    """Коды сводных строк — тех, у которых в файле есть датированные потомки.

    Потомок — строка с кодом-продолжением («12.3.7.» → «12.3.7.2.») или строка-детализация
    под этим кодом. Порядок строк не важен: график могли отсортировать по датам.
    """
    out: set[str] = set()
    for r in rows:
        if not r.dated:
            continue
        if r.code:
            out.update(catalog.ancestors(r.code))
        elif r.parent:
            key = catalog.code_key(r.parent)
            out.add(key)
            out.update(catalog.ancestors(key))
    return out


_INTERNAL = "внутренние работы"
_HEADER = "заголовок раздела"
_UNKNOWN = "не определён"


def _classify(row: _Row) -> tuple[int | None, str | None, str, str]:
    """Строка → (этап | None, ключ работы, причина пропуска, пометка для предупреждения).

    Порядок — от надёжного к приблизительному; всё приблизительное помечается:
    колонка «этап» → точный код перечня (или «родитель/название») → название макроэтапа
    («Кровля», «1. Подготовка территории») → предок кода / однозначное название работы из
    перечня → детализация наследует родителя → ключевые слова → номер макроэтапа в колонке кода.
    """
    if row.stage_hint is not None:
        return row.stage_hint, row.code or None, "", ""
    item, how = catalog.resolve(row.code, row.name, row.parent)
    if item is None or how:
        by_name = _stage_by_name(row.name) if row.name else None
        if by_name is not None:
            organizer_code = bool(row.code) and catalog.code_level(row.code) > 1
            note = f"кода «{row.code}» нет в перечне организаторов — этап {by_name} по названию" if organizer_code else ""
            return by_name, row.code or None, "", note
    if item is not None:
        if item.status == "header":
            stages = catalog.stages_under(item.code or item.parent_code)
            if len(stages) == 1:
                stage = stages.pop()
                return stage, item.key, "", f"раздел без детализации целиком отнесён к этапу {stage}"
            return None, item.key, _HEADER, (f"раздел охватывает этапы {', '.join(map(str, sorted(stages)))} — "
                                             "распишите его по работам, иначе сроки не учесть")
        stage = item.plan_stage_id
        if stage is None:
            return None, item.key, _INTERNAL, ""
        note = how
        if item.status == "out_of_scope":
            note = (f"{how + '; ' if how else ''}в перечне работа отмечена только для дорог — "
                    f"отнесена к этапу {stage} по коду")
        return stage, item.key, "", note
    # строка-детализация с неизвестным названием наследует родителя (как в самом перечне)
    if not row.code and row.parent:
        parent_item, _ = catalog.resolve(row.parent)
        if parent_item is not None and parent_item.plan_stage_id is not None:
            return parent_item.plan_stage_id, parent_item.key, "", ""
    stage = _stage_by_keywords(row.name)
    if stage is not None:
        why = f"кода «{row.code}» нет в перечне организаторов; " if row.code else ""
        return stage, row.code or None, "", f"{why}отнесена к этапу {stage} по названию — проверьте"
    # простой план из 8 строк: «№ 3, начало, окончание» без названий
    m = re.fullmatch(r"(\d)\.", catalog.code_key(row.code)) if row.code else None
    if m and 1 <= int(m.group(1)) <= 8 and not row.name:
        return int(m.group(1)), None, "", f"код «{row.code}» понят как номер макроэтапа"
    return None, None, _UNKNOWN, ""


def parse(data: bytes, filename: str) -> tuple[list[PlanItem], list[str]]:
    """Календарный график CSV/XLSX → (макроэтапы, предупреждения).

    Не бросает исключений на плохом файле: вернёт пустой список и объяснение.
    PlanItem.equipment — только то, что указано в файле (максимум по строкам этапа:
    строки этапа обычно идут последовательно, и сумма посчитала бы одну машину дважды).
    Плановые часы не заполняются — их считает core.equipment.hours из парка площадки.
    """
    warnings: list[str] = []
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            rows, w = _read_xlsx(data)
        elif name.endswith(".xls"):
            return [], ["формат .xls (Excel 97–2003) не поддерживается — сохраните график как .xlsx или .csv"]
        else:
            rows, w = _read_csv(data)
    except Exception as exc:  # битый zip, не xlsx под видом xlsx и т.п. — не роняем сервис
        return [], [f"файл не читается как {'XLSX' if name.endswith(('.xlsx', '.xlsm')) else 'CSV'}: {exc}"]
    warnings += w

    found = _detect_header(rows)
    if found is None:
        return [], warnings + ["не найдена строка заголовков: нужны колонки «Наименование» (или «Код» / «Этап») "
                               "и «Начало» / «Окончание»"]
    header_idx, cols = found
    parsed = _extract_rows(rows, header_idx, cols, warnings)
    if not parsed:
        return [], warnings + ["в файле нет строк с работами под заголовком"]

    by_stage: dict[int, dict] = {}
    skipped_internal = 0
    summaries = _summary_codes(parsed)
    for row in parsed:
        if row.code and catalog.code_key(row.code) in summaries:
            continue
        stage, key, skip, note = _classify(row)
        label = f"строка {row.line}" + (f" ({row.code} {row.name})".rstrip() if row.code or row.name else "")
        for n in row.notes:
            if not n.startswith("код восстановлен"):
                warnings.append(f"{label}: {n}")
        if stage is None:
            if skip == _UNKNOWN:
                what = f"код «{row.code}» не найден в перечне организаторов, " if row.code else ""
                warnings.append(f"{label}: {what}этап не определён по названию — строка не учтена")
            elif skip == _HEADER and row.dated and note:
                warnings.append(f"{label}: {note}")
            elif skip == _INTERNAL and row.dated:
                skipped_internal += 1
            continue
        if note:
            warnings.append(f"{label}: {note}")
        acc = by_stage.setdefault(stage, {"starts": [], "ends": [], "codes": [], "equipment": {}, "rows": []})
        if row.start:
            acc["starts"].append(row.start)
        if row.end:
            acc["ends"].append(row.end)
        if key and key not in acc["codes"]:
            acc["codes"].append(key)
        for k, v in row.equipment.items():
            acc["equipment"][k] = max(acc["equipment"].get(k, 0), v)
        acc["rows"].append(row.line)
        if not row.dated:
            warnings.append(f"{label}: нет дат начала и окончания — строка отнесена к этапу {stage}, но в сроки не вошла")

    if skipped_internal:
        warnings.append(f"пропущено строк внутренних работ: {skipped_internal} — с камеры не видны, "
                        "в сравнение план/факт не входят (организаторы разрешили их не учитывать)")

    items: list[PlanItem] = []
    for stage in sorted(by_stage):
        acc = by_stage[stage]
        start = min(acc["starts"]) if acc["starts"] else None
        end = max(acc["ends"]) if acc["ends"] else None
        title = f"этап {stage} «{taxonomy.stage_name(stage)}»"
        if start and end and start > end:
            warnings.append(f"{title}: самое раннее начало позже самого позднего окончания — даты переставлены, проверьте")
            start, end = end, start
        if not start and not end:
            warnings.append(f"{title}: ни у одной строки нет дат — этап без сроков")
        elif not start or not end:
            warnings.append(f"{title}: нет даты {'начала' if not start else 'окончания'} — "
                            "в сравнение план/факт этап не войдёт, пока её не укажут")
        items.append(PlanItem(
            stage_id=stage, planned_start=start, planned_end=end, name=taxonomy.stage_name(stage),
            work_codes=acc["codes"], equipment=acc["equipment"],
        ))
    if not items:
        warnings.append("ни одна строка не отнесена к макроэтапам — проверьте коды и названия работ")
    return items, warnings


# --------------------------------------------------------------------------
# демо-план
# --------------------------------------------------------------------------

# Типовой монолитный жилой дом ~17 этажей, одна секция: (этап, сдвиг начала, длительность), дни.
# Этапы перекрываются, как на реальной стройке: котлован идёт вслед за ограждением,
# фасад стартует, когда каркас прошёл две трети, благоустройство — на хвосте фасада.
DEMO_TEMPLATE: tuple[tuple[int, int, int], ...] = (
    (1, 0, 30),
    (2, 20, 45),
    (3, 45, 45),
    (4, 80, 75),
    (5, 150, 240),
    (6, 380, 50),
    (7, 300, 170),
    (8, 420, 90),
)
# Кадры должны покрывать не больше этой доли плана: иначе на дату последнего кадра план
# ожидает ~100 % и вердикт становится тавтологичным «отставанием» (баг автоплана api-solution).
DEMO_MAX_COVERAGE = 0.85


def demo_plan(start: dt.date, end: dt.date | None = None, stage_ids=None) -> list[PlanItem]:
    """Правдоподобный график под диапазон кадров — для демонстрации, с явной пометкой в UI.

    start — дата первого кадра: с неё начинается первый из выбранных этапов;
    end — дата последнего кадра (необязательно): если кадры охватывают больше
    DEMO_MAX_COVERAGE типового срока, длительности растягиваются, но план всё равно
    кончается позже последнего кадра — никакой подгонки плана под факт;
    stage_ids — какие этапы включить (например, начиная с текущего по кадрам); по умолчанию все 8.

    Техника этапа предзаполняется типичным парком из норм (core.plan.norms.default_equipment).
    """
    ids = sorted({int(s) for s in stage_ids}) if stage_ids else [s for s, _, _ in DEMO_TEMPLATE]
    tpl = [(s, off, dur) for s, off, dur in DEMO_TEMPLATE if s in ids]
    if not tpl:
        return []
    base = min(off for _, off, _ in tpl)
    total = max(off + dur for _, off, dur in tpl) - base
    scale = 1.0
    if end is not None and end > start:
        span = (end - start).days
        if span > DEMO_MAX_COVERAGE * total:
            scale = span / (DEMO_MAX_COVERAGE * total)
    items = []
    for s, off, dur in tpl:
        a = start + dt.timedelta(days=round((off - base) * scale))
        b = a + dt.timedelta(days=max(1, round(dur * scale)) - 1)
        items.append(PlanItem(
            stage_id=s, planned_start=a, planned_end=b, name=taxonomy.stage_name(s),
            work_codes=[w.key for w in catalog.works_for_stage(s) if w.code][:6],
            equipment=norms.default_equipment(s),
        ))
    return items
