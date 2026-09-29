"""Диаграмма Ганта: план против того, что увидели камеры.

**Полос две, и только две.** Плановая — то, что задал человек в календарном
плане. Фактическая — то, что происходит на площадке на самом деле. Вторая
берётся не с камеры, а из сведения ответов всех камер разом: одна за день
могла сказать «да, да», другая «нет, да», третья не разглядеть вовсе, и
ровно в том, чтобы собрать это в один ответ, и состоит смысл системы.
Полоса на камеру превращала бы вывод обратно в сырые данные и перекладывала
сведение на глаз смотрящего.

Сведение делает `aggregate`: один ответ «да» с любой камеры подтверждает
признак, «не видно» не голосует. Для каждого этапа берётся собственный сигнал
начала; соседние этапы перекрываются на короткий период.

**Шкала — сетка, а не резиновая полоса.** Даты читаются только тогда, когда
есть на что смотреть: колонки недель внутри месяцев, месяцы шапкой над ними.
Границы колонок нарисованы через всю таблицу, и любую плашку можно отнести к
неделе, не прикладывая линейку. На длинных сроках недельные колонки
вырождаются в частокол, поэтому там шаг — месяц, а шапка — год.

Вместо процента совпадения с планом строка показывает оценку вероятности
окончить этап в срок и отклонение текущего прогресса от планового в днях.
Вероятность — предварительная эвристика по темпу видимых вех, не калиброванная
на большой выборке.

Модуль намеренно ничего не знает ни о базе, ни о шаблонах: на вход даты, на
выход проценты. Геометрию тогда можно проверить тестом, а не глазами.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
from dataclasses import dataclass, field

from app.pipeline.progress import StageProgress, object_progress

Interval = tuple[dt.date, dt.date]

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
    progress: int = 0
    coverage: int = 0
    variance_days: int | None = None
    on_time_probability: int | None = None
    verdict: str = ""


@dataclass(slots=True)
class Chart:
    rows: list[Row]
    groups: list[Group]
    cols: list[Col]
    start: dt.date
    end: dt.date
    today_left: float | None
    object_progress: int
    observation_coverage: int
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


def _planned_progress(start: dt.date, end: dt.date,
                      today: dt.date) -> int:
    if today < start:
        return 0
    if today >= end:
        return 100
    return round(100 * (today - start).days / max(1, (end - start).days))


def _estimate(progress: int, coverage: int, start: dt.date | None,
              planned_start: dt.date, planned_end: dt.date,
              today: dt.date, completed_at: dt.date | None
              ) -> tuple[int | None, int | None, str]:
    """Грубая эвристика темпа: вероятность и отклонение в днях."""
    duration = max(1, (planned_end - planned_start).days + 1)
    if completed_at is not None:
        delta = (planned_end - completed_at).days
        probability = 100 if delta >= 0 else 0
        label = "раньше срока" if delta >= 0 else "позже срока"
        return probability, delta, label

    if coverage and start and today >= planned_start:
        actual = max(0, min(100, progress))
        expected = _planned_progress(planned_start, planned_end, today)
        variance = round((actual - expected) * duration / 100)
    else:
        variance = None

    if not coverage or not start or today < planned_start:
        return None, variance, "пока мало наблюдений"

    elapsed = max(1, (today - start).days + 1)
    rate = max(0.5, progress) / elapsed
    remaining = max(0, 100 - progress) / rate
    forecast_finish = today + dt.timedelta(days=round(remaining))
    buffer_days = (planned_end - forecast_finish).days
    scale = max(7.0, duration * 0.20)
    probability = round(100 / (1 + math.exp(max(-30, min(30, -buffer_days / scale)))))
    label = "предварительная эвристика"
    return max(1, min(99, probability)), variance, label


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
          stage_progress: dict[int, StageProgress] | None = None,
          work_weights: dict[int, float] | None = None,
          today: dt.date | None = None) -> Chart | None:
    """План и факт → диаграмма, прогноз срока и прогресс вех."""
    fact = fact or {}
    reached = reached or set()
    stage_progress = stage_progress or {}
    work_weights = work_weights or {
        st["id"]: st.get("work_weight", 1.0) for st in stages
    }

    dates: list[dt.date] = []
    for st in stages:
        if st.get("planned_start") and st.get("planned_end"):
            dates += [st["planned_start"], st["planned_end"]]
    for intervals in fact.values():
        for a, b in intervals:
            dates += [a, b]
    if not dates:
        return None

    lo = min(dates).replace(day=1)
    hi = _month_end(max(dates))
    span = max(1, (hi - lo).days + 1)

    def place(a: dt.date, b: dt.date, label: str = "") -> Box:
        return Box(left=(a - lo).days / span * 100,
                   width=max(0.8, ((b - a).days + 1) / span * 100),
                   label=label or f"{a:%d.%m.%Y} — {b:%d.%m.%Y}")

    rows, detected, planned = [], 0, 0
    for index, st in enumerate(stages):
        has_plan = bool(st.get("planned_start") and st.get("planned_end"))
        planned += int(has_plan)
        intervals = fact.get(st["id"], [])
        fact_days = _days(intervals)
        detected += int(bool(fact_days))
        metric = stage_progress.get(st["id"], StageProgress(0, 0, 0, 0))
        actual_start = min((a for a, _ in intervals), default=None)
        # Начало следующего этапа не равно завершению текущего: соседние
        # работы перекрываются. Завершение следует за концом фактической полосы.
        completed_at = (max((b for _, b in intervals), default=None)
                        if st["id"] in reached else None)

        probability, variance, verdict = (None, None, "нет календарного плана")
        if has_plan and today:
            probability, variance, verdict = _estimate(
                metric.percent, metric.coverage, actual_start,
                st["planned_start"], st["planned_end"], today, completed_at)

        rows.append(Row(
            stage_id=st["id"], title=st["title"],
            plan=place(st["planned_start"], st["planned_end"]) if has_plan else None,
            fact=[place(a, b) for a, b in intervals],
            reached=st["id"] in reached,
            progress=metric.percent, coverage=metric.coverage,
            variance_days=variance,
            on_time_probability=probability, verdict=verdict,
        ))

    groups, cols, unit = _grid(lo, hi, span)
    today_left = (((today - lo).days / span * 100)
                  if today and lo <= today <= hi else None)
    object_percent, coverage = object_progress(
        stage_progress, [st["id"] for st in stages], work_weights)
    planned_dates = [d for st in stages
                     for d in (st.get("planned_start"), st.get("planned_end")) if d]
    fact_dates = [d for intervals in fact.values() for pair in intervals for d in pair]
    gap = 0
    if planned_dates and fact_dates:
        if min(fact_dates) > max(planned_dates):
            gap = (min(fact_dates) - max(planned_dates)).days
        elif max(fact_dates) < min(planned_dates):
            gap = (min(planned_dates) - max(fact_dates)).days
    return Chart(
        rows=rows, groups=groups, cols=cols, start=lo, end=hi,
        today_left=today_left, object_progress=object_percent,
        observation_coverage=coverage, detected=detected, planned=planned,
        unit=unit, gap_days=gap,
    )
