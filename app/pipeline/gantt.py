"""Диаграмма Ганта: план против того, что увидели камеры.

**Полос две, и только две.** Плановая — то, что задал человек в календарном
плане. Фактическая — то, что происходит на площадке на самом деле. Вторая
берётся не с камеры, а из сведения ответов всех камер разом: одна за день
могла сказать «да, да», другая «нет, да», третья не разглядеть вовсе, и
ровно в том, чтобы собрать это в один ответ, и состоит смысл системы.
Полоса на камеру превращала бы вывод обратно в сырые данные и перекладывала
сведение на глаз смотрящего.

Сведение делает `aggregate`: ответы по кадрам сворачиваются в ответ дня
большинством голосов, «не видно» не голосует, дальше сглаживание и отрезки
активности. Сюда приходит уже готовая хронология по объекту целиком.

**Шкала — сетка, а не резиновая полоса.** Даты читаются только тогда, когда
есть на что смотреть: колонки недель внутри месяцев, месяцы шапкой над ними.
Границы колонок нарисованы через всю таблицу, и любую плашку можно отнести к
неделе, не прикладывая линейку. На длинных сроках недельные колонки
вырождаются в частокол, поэтому там шаг — месяц, а шапка — год.

Совпадение считается по дням: пересечение календарных множеств к их
объединению. Мера выбрана из-за того, как ошибается система. Сравнение одних
только дат начала объявило бы полным совпадением этап, который начался
вовремя и тянулся вдвое дольше; сравнение длительностей — этап, который шёл
столько же, но на месяц позже. Пересечение штрафует и то и другое.

Модуль намеренно ничего не знает ни о базе, ни о шаблонах: на вход даты, на
выход проценты. Геометрию тогда можно проверить тестом, а не глазами.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass, field

Interval = tuple[dt.date, dt.date]

ON_TIME_DAYS = 3   # расхождение меньше — считаем «в срок»
# Потолок числа колонок. Шаг сетки выбирается не по длине срока в днях, а по
# тому, сколько колонок получится: так стройка на полгода и архив на двадцать
# лет одинаково остаются читаемыми, и ни один срок не даёт частокол.
MAX_COLS = 40

MONTHS = ("Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль",
          "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь")
MONTHS_SHORT = ("янв", "фев", "мар", "апр", "май", "июн", "июл",
                "авг", "сен", "окт", "ноя", "дек")


@dataclass(slots=True)
class Box:
    left: float           # проценты от левого края шкалы
    width: float
    label: str = ""


@dataclass(slots=True)
class Col:
    """Колонка сетки: неделя внутри месяца либо целый месяц."""
    left: float
    width: float
    label: str
    major: bool = False   # начало месяца (года) — разделитель жирнее


@dataclass(slots=True)
class Group:
    """Шапка над колонками: месяц с годом либо год."""
    left: float
    width: float
    label: str


@dataclass(slots=True)
class Row:
    stage_id: int
    title: str
    plan: Box | None
    fact: list[Box] = field(default_factory=list)
    reached: bool = False
    match: float | None = None    # доля совпадения плана и факта по дням
    shift: int | None = None      # сдвиг начала, дней; плюс — позже плана
    verdict: str = ""


@dataclass(slots=True)
class Chart:
    rows: list[Row]
    groups: list[Group]
    cols: list[Col]
    start: dt.date
    end: dt.date
    today_left: float | None
    match: float | None        # среднее совпадение по этапам, где есть и то и то
    detected: int              # этапов, которые камеры увидели
    planned: int               # этапов с проставленными датами
    unit: str                  # шаг сетки: week | month | quarter | year
    # Разрыв в днях между планом и фактом, если они лежат в разных эпохах и не
    # пересекаются вовсе. Ноль — пересекаются. Такая картинка почти всегда
    # значит ошибку в датах (архив 2005 года против плана 2026), а не
    # настоящее отставание на семь тысяч дней.
    gap_days: int = 0


# ---------------------------------------------------------------------------
# счёт по дням
# ---------------------------------------------------------------------------

def _days(intervals: list[Interval]) -> set[dt.date]:
    out: set[dt.date] = set()
    for a, b in intervals:
        for i in range((b - a).days + 1):
            out.add(a + dt.timedelta(days=i))
    return out


def _jaccard(plan: set[dt.date], fact: set[dt.date]) -> float | None:
    if not plan or not fact:
        return None
    return len(plan & fact) / len(plan | fact)


def _verdict(shift: int | None, has_fact: bool, plan_over: bool) -> str:
    if not has_fact:
        return "не наблюдался" if plan_over else "ещё не наблюдался"
    if shift is None:
        return "вне плана"
    if shift <= -ON_TIME_DAYS:
        return f"раньше на {abs(shift)} дн."
    if shift >= ON_TIME_DAYS:
        return f"позже на {shift} дн."
    return "в срок"


# ---------------------------------------------------------------------------
# сетка
# ---------------------------------------------------------------------------

def _month_end(day: dt.date) -> dt.date:
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def _next_month(day: dt.date) -> dt.date:
    return _month_end(day) + dt.timedelta(days=1)


def _quarter(day: dt.date) -> int:
    return (day.month - 1) // 3 + 1


def _weeks(lo: dt.date, hi: dt.date, place) -> tuple[list[Group], list[Col]]:
    """Недели колонками, месяцы шапкой."""
    groups, cols = [], []
    month = lo
    while month <= hi:
        end = _month_end(month)
        left, width = place(month, end)
        groups.append(Group(left, width, f"{MONTHS[month.month - 1]}, {month.year}"))

        # Неделя обрывается на границе месяца: иначе колонка принадлежала бы
        # сразу двум месяцам шапки, и подпись над ней врала бы.
        start = month
        while start <= end:
            stop = min(start + dt.timedelta(days=6 - start.weekday()), end)
            cl, cw = place(start, stop)
            cols.append(Col(cl, cw,
                            str(start.day) if start == stop
                            else f"{start.day}–{stop.day}",
                            major=start == month))
            start = stop + dt.timedelta(days=1)
        month = _next_month(month)
    return groups, cols


def _months(lo: dt.date, hi: dt.date, place) -> tuple[list[Group], list[Col]]:
    """Месяцы колонками, годы шапкой."""
    cols = []
    month = lo
    while month <= hi:
        left, width = place(month, _month_end(month))
        cols.append(Col(left, width, MONTHS_SHORT[month.month - 1],
                        major=month.month == 1))
        month = _next_month(month)
    return _years_header(lo, hi, place), cols


def _quarters(lo: dt.date, hi: dt.date, place) -> tuple[list[Group], list[Col]]:
    """Кварталы колонками, годы шапкой."""
    cols = []
    start = dt.date(lo.year, (_quarter(lo) - 1) * 3 + 1, 1)
    while start <= hi:
        last = _month_end(start + dt.timedelta(days=62))
        stop = min(last, hi)
        left, width = place(max(start, lo), stop)
        cols.append(Col(left, width, f"{_quarter(start)} кв.",
                        major=start.month == 1))
        start = stop + dt.timedelta(days=1)
    return _years_header(lo, hi, place), cols


def _years(lo: dt.date, hi: dt.date, place) -> tuple[list[Group], list[Col]]:
    """Годы колонками. Шапка одна на весь срок — группировать уже нечем."""
    cols = []
    year = lo
    while year <= hi:
        last = min(dt.date(year.year, 12, 31), hi)
        left, width = place(year, last)
        cols.append(Col(left, width, str(year.year), major=True))
        year = last + dt.timedelta(days=1)
    label = (str(lo.year) if lo.year == hi.year else f"{lo.year}–{hi.year}")
    return [Group(0.0, 100.0, label)], cols


def _years_header(lo: dt.date, hi: dt.date, place) -> list[Group]:
    groups = []
    year = lo
    while year <= hi:
        last = min(dt.date(year.year, 12, 31), hi)
        left, width = place(year, last)
        groups.append(Group(left, width, str(year.year)))
        year = last + dt.timedelta(days=1)
    return groups


def _grid(lo: dt.date, hi: dt.date, span: int) -> tuple[list[Group], list[Col], str]:
    """Шапка и колонки самого мелкого шага, который ещё читается."""
    def place(a: dt.date, b: dt.date) -> tuple[float, float]:
        return (a - lo).days / span * 100, ((b - a).days + 1) / span * 100

    steps = (("week", _weeks), ("month", _months),
             ("quarter", _quarters), ("year", _years))
    for unit, make in steps:
        groups, cols = make(lo, hi, place)
        if len(cols) <= MAX_COLS or unit == "year":
            return groups, cols, unit
    raise AssertionError("недостижимо: годовой шаг возвращается всегда")


# ---------------------------------------------------------------------------
# сборка
# ---------------------------------------------------------------------------

def build(stages: list[dict], fact: dict[int, list[Interval]] | None = None,
          reached: set[int] | None = None,
          today: dt.date | None = None) -> Chart | None:
    """План этапов и сведённая хронология наблюдений → готовая диаграмма.

    stages: [{"id", "title", "planned_start", "planned_end"}] в порядке плана.
    fact:   {id этапа: отрезки активности} — уже сведённые по всем камерам.
    """
    fact = fact or {}
    reached = reached or set()

    dates: list[dt.date] = []
    for st in stages:
        if st.get("planned_start") and st.get("planned_end"):
            dates += [st["planned_start"], st["planned_end"]]
    for ivs in fact.values():
        for a, b in ivs:
            dates += [a, b]
    if not dates:
        return None

    # Шкала растягивается до целых месяцев: сетка с обрубленным первым и
    # последним месяцем читается хуже, чем чуть более широкая, но ровная.
    lo = min(dates).replace(day=1)
    hi = _month_end(max(dates))
    span = max(1, (hi - lo).days + 1)

    def place(a: dt.date, b: dt.date, label: str = "") -> Box:
        # Минимальная ширина: однодневный отрезок на годовой шкале — это
        # 0.3% полосы, то есть невидимая полоска. Лучше показать заметную
        # метку не совсем в масштабе, чем не показать наблюдение вовсе.
        return Box(left=(a - lo).days / span * 100,
                   width=max(0.8, ((b - a).days + 1) / span * 100),
                   label=label or f"{a:%d.%m.%Y} — {b:%d.%m.%Y}")

    rows, matches, detected, planned = [], [], 0, 0
    for st in stages:
        has_plan = bool(st.get("planned_start") and st.get("planned_end"))
        plan_days = (_days([(st["planned_start"], st["planned_end"])])
                     if has_plan else set())
        planned += 1 if has_plan else 0

        ivs = fact.get(st["id"], [])
        fact_days = _days(ivs)
        if fact_days:
            detected += 1

        match = _jaccard(plan_days, fact_days)
        if match is not None:
            matches.append(match)

        shift = ((min(a for a, _ in ivs) - st["planned_start"]).days
                 if ivs and has_plan else None)
        plan_over = bool(has_plan and today and st["planned_end"] < today)

        rows.append(Row(
            stage_id=st["id"], title=st["title"],
            plan=place(st["planned_start"], st["planned_end"]) if has_plan else None,
            fact=[place(a, b) for a, b in ivs],
            reached=st["id"] in reached,
            match=match, shift=shift,
            verdict=_verdict(shift, bool(fact_days), plan_over),
        ))

    groups, cols, unit = _grid(lo, hi, span)

    gap = 0
    plan_span = [d for st in stages for d in (st.get("planned_start"), st.get("planned_end")) if d]
    fact_span = [d for ivs in fact.values() for a, b in ivs for d in (a, b)]
    if plan_span and fact_span:
        if min(fact_span) > max(plan_span):
            gap = (min(fact_span) - max(plan_span)).days
        elif max(fact_span) < min(plan_span):
            gap = (min(plan_span) - max(fact_span)).days
    today_left = (((today - lo).days / span * 100)
                  if today and lo <= today <= hi else None)

    return Chart(rows=rows, groups=groups, cols=cols, start=lo, end=hi,
                 today_left=today_left,
                 match=sum(matches) / len(matches) if matches else None,
                 detected=detected, planned=planned, unit=unit, gap_days=gap)
