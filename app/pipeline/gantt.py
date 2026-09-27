"""Диаграмма Ганта: план против того, что увидели камеры.

Смысл картинки — сравнение, поэтому шкала времени одна на всё. Плановая
полоса этапа и фактические полосы лежат в одной строке друг под другом: так
сдвиг читается глазом без арифметики, а несколько камер видно сразу — если
они расходятся, это само по себе диагностика, а не шум.

Совпадение считается по дням, а не по границам: пересечение календарных
множеств к их объединению. Мера выбрана из-за того, как ошибается система.
Сравнение одних только дат начала объявило бы полным совпадением этап,
который начался вовремя и тянулся вдвое дольше; сравнение длительностей —
этап, который шёл столько же, но на месяц позже. Пересечение штрафует и то
и другое, и делает это соразмерно.

Модуль намеренно ничего не знает ни о базе, ни о шаблонах: на вход даты, на
выход проценты. Геометрию тогда можно проверить тестом, а не глазами.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

Interval = tuple[dt.date, dt.date]

# Цвета камер. Плановая полоса всегда серая — она фон, на котором смотрят
# факт, и спорить за внимание с ним не должна.
PALETTE = ("#2563eb", "#0d9488", "#d946ef", "#ea580c", "#65a30d")
ON_TIME_DAYS = 3          # расхождение меньше — считаем «в срок»


@dataclass(slots=True)
class Box:
    left: float           # проценты от левого края шкалы
    width: float
    label: str = ""


@dataclass(slots=True)
class Lane:
    """Одна строка полос внутри этапа: план либо конкретная камера."""
    key: str
    label: str
    color: str
    boxes: list[Box] = field(default_factory=list)


@dataclass(slots=True)
class Source:
    """Наблюдатель: что он увидел по каждому этапу."""
    key: str
    label: str
    color: str
    intervals: dict[int, list[Interval]] = field(default_factory=dict)
    reached: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Row:
    stage_id: int
    title: str
    plan: Box | None
    lanes: list[Lane]
    reached: bool
    match: float | None       # доля совпадения плана и факта по дням
    shift: int | None         # сдвиг начала, дней; плюс — позже плана
    verdict: str


@dataclass(slots=True)
class Chart:
    rows: list[Row]
    ticks: list[dict]
    sources: list[dict]
    start: dt.date
    end: dt.date
    today_left: float | None
    match: float | None        # среднее совпадение по этапам, где есть и то и то
    detected: int              # этапов, которые камеры вообще увидели
    planned: int               # этапов с проставленными датами


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
        return f"раньше плана на {abs(shift)} дн."
    if shift >= ON_TIME_DAYS:
        return f"позже плана на {shift} дн."
    return "в срок"


def _ticks(lo: dt.date, hi: dt.date, span: int) -> list[dict]:
    """Подписи оси: первые числа месяцев, прореженные под длину диапазона."""
    out, cur = [], dt.date(lo.year, lo.month, 1)
    step = max(1, span // 300)
    while cur <= hi:
        if cur >= lo:
            out.append({"label": cur.strftime("%m.%y"),
                        "left": (cur - lo).days / span * 100})
        month = cur.month + step
        cur = dt.date(cur.year + (month - 1) // 12, (month - 1) % 12 + 1, 1)

    # Короткий диапазон может не зацепить ни одного первого числа — тогда
    # подписываем хотя бы границы, иначе шкала остаётся без единой отметки.
    if len(out) < 2:
        out = [{"label": lo.strftime("%d.%m.%y"), "left": 0.0},
               {"label": hi.strftime("%d.%m.%y"), "left": 100.0}]
    return out


def build(stages: list[dict], sources: list[Source],
          today: dt.date | None = None) -> Chart | None:
    """Этапы плана и наблюдения камер → готовая к отрисовке диаграмма.

    stages: [{"id", "title", "planned_start", "planned_end"}] в порядке плана.
    """
    dates: list[dt.date] = []
    for st in stages:
        if st.get("planned_start") and st.get("planned_end"):
            dates += [st["planned_start"], st["planned_end"]]
    for src in sources:
        for ivs in src.intervals.values():
            for a, b in ivs:
                dates += [a, b]
    if not dates:
        return None

    lo, hi = min(dates), max(dates)
    # Шкала кончается не в последний день, а сразу за ним. Отрезок включает
    # обе границы, поэтому его ширина — это (конец − начало + 1) день; без
    # этого лишнего дня в знаменателе полоса последнего этапа вылезала за
    # правый край дорожки ровно на сутки.
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

        lanes, fact_days, starts = [], set(), []
        for src in sources:
            ivs = src.intervals.get(st["id"], [])
            if not ivs:
                continue
            lanes.append(Lane(key=src.key, label=src.label, color=src.color,
                              boxes=[place(a, b) for a, b in ivs]))
            fact_days |= _days(ivs)
            starts.append(min(a for a, _ in ivs))

        if fact_days:
            detected += 1
        match = _jaccard(plan_days, fact_days)
        if match is not None:
            matches.append(match)

        shift = ((min(starts) - st["planned_start"]).days
                 if starts and has_plan else None)
        plan_over = bool(has_plan and today and st["planned_end"] < today)

        rows.append(Row(
            stage_id=st["id"], title=st["title"],
            plan=place(st["planned_start"], st["planned_end"], "по плану")
                 if has_plan else None,
            lanes=lanes,
            reached=any(st["id"] in src.reached for src in sources),
            match=match, shift=shift,
            verdict=_verdict(shift, bool(fact_days), plan_over),
        ))

    today_left = (((today - lo).days / span * 100)
                  if today and lo <= today <= hi else None)

    return Chart(
        rows=rows, ticks=_ticks(lo, hi, span),
        sources=[{"key": s.key, "label": s.label, "color": s.color}
                 for s in sources if any(s.intervals.values())],
        start=lo, end=hi, today_left=today_left,
        match=sum(matches) / len(matches) if matches else None,
        detected=detected, planned=planned,
    )
