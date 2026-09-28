"""Геометрия и арифметика диаграммы «план против факта».

Модуль раскладки не знает ни про базу, ни про шаблоны, поэтому проверяется
напрямую: на вход даты, на выход проценты. Ошибка здесь не падает, а тихо
рисует неверную картинку — именно такие и стоит ловить тестом.
"""

import datetime as dt

import pytest

from app.pipeline import gantt

D = dt.date


def stage(sid, title, start=None, end=None):
    return {"id": sid, "title": title, "planned_start": start, "planned_end": end}


PLAN = [
    stage(1, "Подготовка", D(2026, 1, 1), D(2026, 1, 31)),
    stage(2, "Котлован", D(2026, 2, 1), D(2026, 3, 2)),
    stage(3, "Монолит", D(2026, 3, 3), D(2026, 5, 1)),
]


# ---------------------------------------------------------------------------
# раскладка полос
# ---------------------------------------------------------------------------

def test_empty_returns_nothing() -> None:
    """Ни дат плана, ни наблюдений — рисовать нечего, и это не ошибка."""
    assert gantt.build([stage(1, "Без дат")], {}) is None


def test_scale_covers_plan_and_fact_in_whole_months() -> None:
    """Шкала вмещает обе стороны сравнения и равна целому числу месяцев.

    Обрубленный первый месяц ломает шапку: подпись стоит над куском, а
    выглядит как подпись над всем месяцем.
    """
    chart = gantt.build(PLAN, {1: [(D(2025, 12, 20), D(2026, 1, 20))]})
    assert chart.start == D(2025, 12, 1)
    assert chart.end == D(2026, 5, 31)
    for row in chart.rows:
        for box in [row.plan] + row.fact:
            if box:
                assert 0 <= box.left <= 100
                assert box.left + box.width <= 100.5


def test_match_penalises_shift_and_stretch() -> None:
    """Совпадение должно падать и от сдвига, и от растянутости.

    Сравнение одних только дат начала объявило бы вторую пару полным
    совпадением, а сравнение длительностей — третью.
    """
    exact = gantt.build([PLAN[0]], {1: [(D(2026, 1, 1), D(2026, 1, 31))]})
    shifted = gantt.build([PLAN[0]], {1: [(D(2026, 1, 15), D(2026, 2, 14))]})
    stretched = gantt.build([PLAN[0]], {1: [(D(2026, 1, 1), D(2026, 3, 1))]})

    assert exact.rows[0].match == pytest.approx(1.0)
    assert shifted.rows[0].match < 0.6
    assert stretched.rows[0].match < 0.6


def test_shift_sign_reads_as_delay() -> None:
    """Плюс — позже плана. Знак важнее величины: по нему пишется вердикт."""
    late = gantt.build([PLAN[0]], {1: [(D(2026, 1, 11), D(2026, 2, 10))]})
    early = gantt.build([PLAN[0]], {1: [(D(2025, 12, 22), D(2026, 1, 21))]})

    assert late.rows[0].shift == 10
    assert "позже" in late.rows[0].verdict
    assert early.rows[0].shift == -10
    assert "раньше" in early.rows[0].verdict


def test_small_deviation_is_on_time() -> None:
    chart = gantt.build([PLAN[0]], {1: [(D(2026, 1, 2), D(2026, 1, 30))]})
    assert chart.rows[0].verdict == "в срок"


def test_fact_is_one_bar_per_stage() -> None:
    """Факт сведён по всем камерам заранее: здесь он приходит одним набором.

    Полоса на камеру вернула бы на экран сырые данные — а сведение и есть
    то, ради чего система существует.
    """
    chart = gantt.build(PLAN, {2: [(D(2026, 2, 1), D(2026, 2, 20)),
                                   (D(2026, 2, 25), D(2026, 3, 5))]})
    row = next(r for r in chart.rows if r.stage_id == 2)
    assert len(row.fact) == 2          # разрыв в наблюдении — два отрезка
    assert all(isinstance(b, gantt.Box) for b in row.fact)


def test_unobserved_stage_is_stated_not_hidden() -> None:
    """Этап без факта остаётся в списке: «не увидели» — тоже результат."""
    chart = gantt.build(PLAN, {1: [(D(2026, 1, 1), D(2026, 1, 31))]})
    row = next(r for r in chart.rows if r.stage_id == 3)
    assert row.fact == []
    assert row.match is None
    assert "наблюдался" in row.verdict
    assert chart.detected == 1 and chart.planned == 3


def test_past_and_future_stages_read_differently() -> None:
    """Ненаблюдавшийся этап в прошлом — тревога, в будущем — просто рано."""
    past = gantt.build([stage(1, "Прошлый", D(2026, 1, 1), D(2026, 1, 31))],
                       {}, today=D(2026, 3, 1))
    future = gantt.build([stage(1, "Будущий", D(2026, 4, 1), D(2026, 4, 30))],
                         {}, today=D(2026, 3, 1))
    assert past.rows[0].verdict == "не наблюдался"
    assert future.rows[0].verdict == "ещё не наблюдался"


def test_today_marker_only_inside_range() -> None:
    """Архив 2005 года и сегодняшняя дата на одной шкале — это не шкала."""
    inside = gantt.build(PLAN, {}, today=D(2026, 2, 1))
    outside = gantt.build(PLAN, {}, today=D(2030, 1, 1))
    assert inside.today_left is not None
    assert outside.today_left is None


def test_one_day_interval_stays_visible() -> None:
    """Однодневное наблюдение на годовой шкале — 0.3% ширины, то есть ничто."""
    chart = gantt.build([stage(1, "Год", D(2026, 1, 1), D(2026, 12, 31))],
                        {1: [(D(2026, 6, 1), D(2026, 6, 1))]})
    assert chart.rows[0].fact[0].width >= 0.8


# ---------------------------------------------------------------------------
# сетка дат
# ---------------------------------------------------------------------------

def test_weeks_break_at_month_boundary() -> None:
    """Неделя не может принадлежать двум месяцам шапки сразу.

    Иначе подпись месяца стоит над колонкой, часть которой относится
    к соседнему, и дату по сетке прочесть нельзя.
    """
    chart = gantt.build([stage(1, "Сентябрь", D(2026, 9, 1), D(2026, 9, 30))], {})
    assert chart.unit == "week"
    assert [c.label for c in chart.cols] == ["1–6", "7–13", "14–20", "21–27", "28–30"]
    assert [g.label for g in chart.groups] == ["Сентябрь, 2026"]
    assert chart.cols[0].major and not chart.cols[1].major


def test_columns_tile_the_scale_without_gaps() -> None:
    """Колонки обязаны покрывать шкалу встык: дыра в сетке — это ложь о датах."""
    chart = gantt.build(PLAN, {})
    edge = 0.0
    for col in chart.cols:
        assert col.left == pytest.approx(edge, abs=0.01)
        edge = col.left + col.width
    assert edge == pytest.approx(100.0, abs=0.01)


def test_long_range_switches_to_months() -> None:
    """На двух годах недельные колонки вырождаются в частокол."""
    chart = gantt.build([stage(1, "Долгий", D(2026, 1, 1), D(2027, 12, 31))], {})
    assert chart.unit == "month"
    assert len(chart.cols) == 24
    assert [g.label for g in chart.groups] == ["2026", "2027"]
    assert chart.cols[0].major and chart.cols[12].major


@pytest.mark.parametrize("start, end, unit", [
    (D(2026, 9, 1), D(2026, 11, 30), "week"),
    (D(2026, 1, 1), D(2026, 12, 31), "month"),
    (D(2026, 1, 1), D(2028, 6, 30), "month"),
    (D(2020, 1, 1), D(2026, 12, 31), "quarter"),
    (D(2005, 10, 1), D(2026, 11, 30), "year"),
])
def test_grid_step_keeps_columns_readable(start, end, unit) -> None:
    """Шаг сетки выбирается по числу колонок, а не по длине срока.

    Архив двадцатилетней давности на объекте — не выдумка: старая камера с
    метками 2005 года растягивает шкалу, и сетка обязана это пережить,
    а не выродиться в двести пятьдесят полосок.
    """
    chart = gantt.build([stage(1, "Срок", start, end)], {})
    assert chart.unit == unit
    assert len(chart.cols) <= gantt.MAX_COLS
    assert chart.groups and chart.cols


# ---------------------------------------------------------------------------
# план и факт в разных эпохах
# ---------------------------------------------------------------------------

def test_disjoint_plan_and_fact_are_flagged() -> None:
    """Архив 2005 года против плана 2026 — ошибка в датах, а не отставание.

    Диаграмма всё равно рисуется честно (растянутая шкала), но разрыв
    выносится в поле, чтобы страница могла сказать об этом словами.
    """
    plan = [stage(1, "Подготовка", D(2005, 10, 12), D(2005, 11, 6))]
    chart = gantt.build(plan, {1: [(D(2026, 9, 25), D(2026, 10, 18))]})
    assert chart.gap_days == (D(2026, 9, 25) - D(2005, 11, 6)).days

    reverse = gantt.build([stage(1, "Позже", D(2026, 9, 25), D(2026, 10, 18))],
                          {1: [(D(2005, 10, 12), D(2005, 11, 6))]})
    assert reverse.gap_days > 0


def test_overlapping_plan_and_fact_have_no_gap() -> None:
    plan = [stage(1, "Подготовка", D(2026, 1, 1), D(2026, 1, 31))]
    chart = gantt.build(plan, {1: [(D(2026, 1, 20), D(2026, 2, 20))]})
    assert chart.gap_days == 0


def test_no_fact_no_gap() -> None:
    assert gantt.build(PLAN, {}).gap_days == 0


def test_today_does_not_stretch_the_scale() -> None:
    """Сегодняшняя дата шкалу не растягивает — только рисует черту, если попала."""
    far = gantt.build(PLAN, {}, today=D(2031, 1, 1))
    near = gantt.build(PLAN, {}, today=D(2026, 2, 1))
    assert (far.start, far.end) == (near.start, near.end)
    assert far.today_left is None and near.today_left is not None
