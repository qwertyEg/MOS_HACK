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


def src(key, intervals, reached=(), color="#000"):
    return gantt.Source(key=key, label=key, color=color,
                        intervals=intervals, reached=set(reached))


def test_empty_returns_nothing() -> None:
    """Ни дат плана, ни наблюдений — рисовать нечего, и это не ошибка."""
    assert gantt.build([stage(1, "Без дат")], []) is None


def test_scale_spans_plan_and_fact() -> None:
    """Шкала обязана вместить обе стороны сравнения, иначе полосы уедут."""
    chart = gantt.build(PLAN, [src("c1", {1: [(D(2025, 12, 20), D(2026, 1, 20))]})])
    assert chart.start == D(2025, 12, 20)
    assert chart.end == D(2026, 5, 1)
    for row in chart.rows:
        for box in [row.plan] + [b for lane in row.lanes for b in lane.boxes]:
            if box:
                assert 0 <= box.left <= 100
                assert box.left + box.width <= 100.5


def test_match_penalises_shift_and_stretch() -> None:
    """Совпадение должно падать и от сдвига, и от растянутости.

    Сравнение одних только дат начала объявило бы вторую пару полным
    совпадением, а сравнение длительностей — первую.
    """
    exact = gantt.build([PLAN[0]], [src("c", {1: [(D(2026, 1, 1), D(2026, 1, 31))]})])
    shifted = gantt.build([PLAN[0]], [src("c", {1: [(D(2026, 1, 15), D(2026, 2, 14))]})])
    stretched = gantt.build([PLAN[0]], [src("c", {1: [(D(2026, 1, 1), D(2026, 3, 1))]})])

    assert exact.rows[0].match == pytest.approx(1.0)
    assert shifted.rows[0].match < 0.6
    assert stretched.rows[0].match < 0.6


def test_shift_sign_reads_as_delay() -> None:
    """Плюс — позже плана. Знак важнее величины: по нему пишется вердикт."""
    late = gantt.build([PLAN[0]], [src("c", {1: [(D(2026, 1, 11), D(2026, 2, 10))]})])
    early = gantt.build([PLAN[0]], [src("c", {1: [(D(2025, 12, 22), D(2026, 1, 21))]})])

    assert late.rows[0].shift == 10
    assert "позже" in late.rows[0].verdict
    assert early.rows[0].shift == -10
    assert "раньше" in early.rows[0].verdict


def test_small_deviation_is_on_time() -> None:
    chart = gantt.build([PLAN[0]], [src("c", {1: [(D(2026, 1, 2), D(2026, 1, 30))]})])
    assert chart.rows[0].verdict == "в срок"


def test_cameras_keep_separate_lanes() -> None:
    """Две камеры видят по-своему — усреднять их нельзя, это сведение."""
    chart = gantt.build(PLAN, [
        src("c1", {2: [(D(2026, 2, 1), D(2026, 2, 20))]}),
        src("c2", {2: [(D(2026, 2, 10), D(2026, 3, 5))]}),
    ])
    row = next(r for r in chart.rows if r.stage_id == 2)
    assert [lane.key for lane in row.lanes] == ["c1", "c2"]
    # Объединение, а не пересечение: работы, видные только одной камере,
    # всё равно шли.
    # план 01.02–02.03 (30 дней) целиком лежит внутри объединения
    # 01.02–05.03 (33 дня): 30/33
    assert row.match == pytest.approx(30 / 33, abs=0.01)


def test_unobserved_stage_is_stated_not_hidden() -> None:
    """Этап без факта остаётся в списке: «не увидели» — тоже результат."""
    chart = gantt.build(PLAN, [src("c1", {1: [(D(2026, 1, 1), D(2026, 1, 31))]})])
    row = next(r for r in chart.rows if r.stage_id == 3)
    assert row.lanes == []
    assert row.match is None
    assert "наблюдался" in row.verdict
    assert chart.detected == 1 and chart.planned == 3


def test_past_and_future_stages_read_differently() -> None:
    """Ненаблюдавшийся этап в прошлом — тревога, в будущем — просто рано."""
    past = gantt.build([stage(1, "Прошлый", D(2026, 1, 1), D(2026, 1, 31))],
                       [src("c", {})], today=D(2026, 3, 1))
    future = gantt.build([stage(1, "Будущий", D(2026, 4, 1), D(2026, 4, 30))],
                         [src("c", {})], today=D(2026, 3, 1))
    assert past.rows[0].verdict == "не наблюдался"
    assert future.rows[0].verdict == "ещё не наблюдался"


def test_today_marker_only_inside_range() -> None:
    """Архив 2005 года и сегодняшняя дата на одной шкале — это не шкала."""
    inside = gantt.build(PLAN, [], today=D(2026, 2, 1))
    outside = gantt.build(PLAN, [], today=D(2030, 1, 1))
    assert inside.today_left is not None
    assert outside.today_left is None


def test_one_day_interval_stays_visible() -> None:
    """Однодневное наблюдение на годовой шкале — 0.3% ширины, то есть ничто."""
    chart = gantt.build(
        [stage(1, "Год", D(2026, 1, 1), D(2026, 12, 31))],
        [src("c", {1: [(D(2026, 6, 1), D(2026, 6, 1))]})])
    assert chart.rows[0].lanes[0].boxes[0].width >= 0.8
