"""Моточасы: сколько каждая машина должна отработать на этапе и сколько отработала.

План (docs/ARCHITECTURE.md §5.6). Плановые часы по типу техники на этапе =
число единиц × рабочие дни этапа × смена (10 ч) × коэффициент использования
(0.7). Число единиц — из строки плана (сколько нужно на этапе), иначе из
заявленного парка площадки, иначе минимум из норм «этап → техника». Ручные
часы в строке плана (`hours_manual`) важнее расчёта — пользователь знает
свой проект лучше формулы.

Факт. Если единица двигалась между кадрами t₀ → t₁, засчитываем
min(t₁ − t₀, 45 мин): кадры идут раз в 20–30 минут, и при пропуске кадров
(камера молчала два часа) мы знаем только, что машина где-то в этом окне
поработала, — поэтому зачёт ограничен. Часы идут на этап, который по плану
идёт в этот день (московский день, не UTC). Ночью работы идут — ночные
интервалы засчитываются наравне с дневными.

«Временная полоска» (HoursBalance) = план − факт по (этап, тип техники).
"""
from __future__ import annotations

import bisect
import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from functools import lru_cache

from core import taxonomy
from core.contracts import ActivityInterval, HoursBalance, PlanItem

log = logging.getLogger(__name__)

DEFAULT_TZ = "Europe/Moscow"


@lru_cache(maxsize=8)
def local_tz(name: str = DEFAULT_TZ) -> dt.tzinfo:
    """Часовой пояс площадки. В тонких Docker-образах нет базы tzdata — тогда
    для Москвы фиксированные UTC+3 (переходов на летнее время с 2014 г. нет)."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — ZoneInfoNotFoundError, ImportError, битая база
        if name == DEFAULT_TZ:
            return dt.timezone(dt.timedelta(hours=3), "MSK")
        log.warning("часовой пояс %r не найден, используется UTC", name)
        return dt.timezone.utc


def local_date(t: dt.datetime, tz: str | dt.tzinfo = DEFAULT_TZ) -> dt.date:
    zone = local_tz(tz) if isinstance(tz, str) else tz
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)   # внутри всё UTC — наивное время считаем UTC
    return t.astimezone(zone).date()


def credit_window(start: dt.datetime, end: dt.datetime, max_gap_min: float = 45.0) -> tuple[dt.datetime, dt.datetime]:
    """Какой отрезок засчитать за интервал между кадрами с движением: последние
    не более max_gap_min минут перед кадром, на котором движение увидели."""
    return max(start, end - dt.timedelta(minutes=max_gap_min)), end


def credit_hours(start: dt.datetime, end: dt.datetime, max_gap_min: float = 45.0) -> float:
    """Сколько часов засчитать за интервал между кадрами, на котором машина двигалась."""
    s, e = credit_window(start, end, max_gap_min)
    return max(0.0, (e - s).total_seconds()) / 3600.0


def workdays_in(start: dt.date, end: dt.date, workdays: Iterable[int] = (0, 1, 2, 3, 4, 5)) -> int:
    days = set(workdays)
    if end < start:
        return 0
    total = (end - start).days + 1
    full_weeks, rest = divmod(total, 7)
    n = full_weeks * len(days)
    n += sum(1 for k in range(rest) if (start + dt.timedelta(days=full_weeks * 7 + k)).weekday() in days)
    return n


# --------------------------------------------------------------------------
# план
# --------------------------------------------------------------------------


def _default_units(stage_id: int) -> dict[str, int]:
    """Минимальный состав техники этапа из норм (core/plan/norms.py), если модуль есть;
    иначе — по одной единице каждого обязательного типа из справочника."""
    try:
        from core.plan import norms  # модуль плана пишется параллельно — зависимость мягкая
        got = norms.default_equipment(stage_id)
        if got:
            return {str(k): int(v) for k, v in got.items()}
    except Exception:  # noqa: BLE001 — нет модуля или норм для этапа: берём справочник
        pass
    st = taxonomy.stages().get(int(stage_id))
    return {cls: 1 for cls in (st.equipment_expected if st else ())}


def planned_hours(plan: list[PlanItem], fleet: Mapping[str, int],
                  shift_hours: float = 10.0, utilization: float = 0.7,
                  workdays: Iterable[int] = (0, 1, 2, 3, 4, 5)) -> dict[int, dict[str, float]]:
    """Плановые моточасы: этап → тип техники → часы."""
    wd = tuple(workdays)
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for item in plan:
        days = 0
        if item.planned_start and item.planned_end:
            days = workdays_in(item.planned_start, item.planned_end, wd)
        base = _default_units(item.stage_id)
        counts = {cls: int(item.equipment.get(cls, fleet.get(cls, n))) for cls, n in base.items()}
        # Необязательная по нормам техника попадает в план, только если её
        # явно заложили в строку плана: парк площадки общий на все этапы.
        counts.update({cls: int(n) for cls, n in item.equipment.items() if cls not in counts})
        hours = {cls: n * days * shift_hours * utilization for cls, n in counts.items() if n > 0 and days > 0}
        if item.hours_manual:
            hours.update({cls: float(v) for cls, v in item.planned_hours.items()})
        stage = out[item.stage_id]
        for cls, h in hours.items():
            stage[cls] = round(stage.get(cls, 0.0) + h, 2)
    return dict(out)


def stage_for(plan: list[PlanItem], cls: str, day: dt.date) -> int | None:
    """На какой этап списать работу машины класса `cls` в день `day`.

    Если по плану в этот день идут несколько этапов, выбираем тот, где этот
    тип техники нужнее: явно заложен в строку плана → обязателен по нормам →
    допустим → не запрещён. При равенстве — этап, который должен закончиться
    раньше (работа обычно идёт на то, что горит).
    """
    active = [it for it in plan if it.planned_start and it.planned_end
              and it.planned_start <= day <= it.planned_end]
    if not active:
        return None

    def tier(it: PlanItem) -> int:
        if it.equipment.get(cls, 0) > 0 or it.planned_hours.get(cls, 0) > 0:
            return 0
        st = taxonomy.stages().get(int(it.stage_id))
        if st is None:
            return 3
        if cls in st.equipment_expected:
            return 1
        if cls in st.equipment_optional:
            return 2
        return 4 if cls in st.equipment_forbidden else 3

    best = min(active, key=lambda it: (tier(it), it.planned_end, it.stage_id))
    return best.stage_id


def balances(plan: list[PlanItem], intervals: list[ActivityInterval], *,
             fleet: Mapping[str, int] | None = None, tz: str = DEFAULT_TZ,
             shift_hours: float = 10.0, utilization: float = 0.7,
             workdays: Iterable[int] = (0, 1, 2, 3, 4, 5)) -> list[HoursBalance]:
    """«Полоски» по (этап, тип).

    Плановые часы: из строки плана (их туда кладёт веб-слой — авто или
    вручную); если в строке пусто или передан `fleet` — считаем здесь.
    Этап интервала выводится из ТЕКУЩЕГО плана по дате интервала: если
    пользователь сдвинул даты этапов, полоски пересчитаются, а не останутся
    привязанными к старому плану. Без плана — этап, записанный в интервале.
    """
    planned: dict[tuple[int | None, str], float] = defaultdict(float)
    for item in plan:
        if item.planned_hours and fleet is None:
            ph = {cls: float(h) for cls, h in item.planned_hours.items()}
        else:
            ph = planned_hours([item], fleet or {}, shift_hours, utilization, workdays).get(item.stage_id, {})
        for cls, h in ph.items():
            planned[(item.stage_id, cls)] += h

    worked: dict[tuple[int | None, str], float] = defaultdict(float)
    last: dict[tuple[int | None, str], dt.datetime] = {}
    for iv, h in _without_overlaps(intervals):
        mid = iv.start + (iv.end - iv.start) / 2
        stage = stage_for(plan, iv.cls, local_date(mid, tz)) if plan else iv.stage_id
        key = (stage, iv.cls)
        worked[key] += h
        if key not in last or iv.end > last[key]:
            last[key] = iv.end

    order = {k: i for i, k in enumerate(taxonomy.equipment())}
    keys = sorted(set(planned) | set(worked),
                  key=lambda k: (k[0] is None, k[0] or 0, order.get(k[1], len(order)), k[1]))
    return [HoursBalance(stage_id=k[0], cls=k[1], planned_hours=round(planned.get(k, 0.0), 2),
                         worked_hours=worked.get(k, 0.0), last_worked_at=last.get(k))
            for k in keys]


def _without_overlaps(intervals: list[ActivityInterval]):
    """(интервал, часы без перекрытия с другими интервалами той же единицы).

    Движок сам не пишет перекрывающихся интервалов, но в журнале они могут
    появиться: склейка дубля (веб-слой переписывает unit_id старых строк на
    новый), повторная обработка кадров. Полоска не должна считать одно и то
    же время дважды.
    """
    by_unit: dict[str, list[ActivityInterval]] = defaultdict(list)
    for iv in intervals:
        by_unit[iv.unit_id].append(iv)
    for ivs in by_unit.values():
        covered: dt.datetime | None = None
        for iv in sorted(ivs, key=lambda i: (i.start, i.end)):
            span = (iv.end - iv.start).total_seconds()
            start = iv.start if covered is None else max(iv.start, covered)
            fresh = max(0.0, (iv.end - start).total_seconds())
            yield iv, (iv.hours * fresh / span if span > 0 else 0.0)
            covered = iv.end if covered is None else max(covered, iv.end)


# --------------------------------------------------------------------------
# учёт уже засчитанного времени единицы
# --------------------------------------------------------------------------


class IntervalSet:
    """Уже засчитанные отрезки времени одной единицы.

    Одну машину видят две камеры, и обе видят её движение в пересекающихся
    интервалах (10:00–10:25 и 10:10–10:35) — засчитать нужно 35 минут, а не
    50. Кадры разных камер к тому же могут обрабатываться не по порядку
    (загрузили папку одной камеры после другой), поэтому храним множество
    отрезков, а не «засчитано до».
    """

    def __init__(self, floor: dt.datetime | None = None):
        self._iv: list[tuple[dt.datetime, dt.datetime]] = []
        # После рестарта всё до floor уже засчитано в прошлой жизни движка.
        # Храним отдельно, а не отрезком (−∞, floor]: иначе к нему прилипали бы
        # новые настоящие интервалы и терялись при переносе (склейка единиц).
        self.floor = floor

    def __iter__(self):
        """Засчитанные этим движком отрезки (без границы рестарта)."""
        return iter(list(self._iv))

    def add(self, start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, dt.datetime]]:
        """Добавить отрезок; вернуть его куски, которых раньше не было."""
        if self.floor is not None:
            start = max(start, self.floor)
        if end <= start:
            return []
        fresh, cur = [], start
        i = bisect.bisect_left(self._iv, (start, start))
        if i > 0 and self._iv[i - 1][1] > start:
            i -= 1
        j = i
        while j < len(self._iv) and self._iv[j][0] < end:
            s, e = self._iv[j]
            if s > cur:
                fresh.append((cur, s))
            cur = max(cur, e)
            j += 1
        if cur < end:
            fresh.append((cur, end))
        merged_start = min(start, self._iv[i][0]) if i < j else start
        merged_end = max(end, self._iv[j - 1][1]) if i < j else end
        self._iv[i:j] = [(merged_start, merged_end)]
        return fresh
