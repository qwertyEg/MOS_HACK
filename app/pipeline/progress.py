"""Наблюдаемые вехи, объём работ и предварительный темп этапов.

Модель Б сообщает только видимые признаки. Процент строится из этих
подтверждений: завершённая веха даёт полный вес, работающий сейчас признак —
половину веса. Отдельно показывается, какая доля вех вообще наблюдалась.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.models import Answer

DEFAULT_STAGE_WEIGHTS = {1: 5.0, 2: 8.0, 3: 7.0, 4: 12.0,
                         5: 35.0, 6: 23.0, 7: 10.0}

# Вес внутри этапа, признаки текущей работы и признак её завершения.
# Перманентный признак закрывает веху, временный даёт половину веса только
# пока камеры продолжают его видеть.
MILESTONES = {
    1: [
        ("Расчистка площадки", 50, ("old_building", "tree_felling"), ("cleared",)),
        ("Подготовленная поверхность", 50, ("flat_ground",), ("flat_ground",)),
    ],
    2: [
        ("Погружение свай и шпунта", 55, ("pile_rig", "pile_stock"), ("pile_heads", "sheet_pile")),
        ("Завершение ограждения котлована", 45, ("pile_heads", "sheet_pile"), ("capping_beam",)),
    ],
    3: [
        ("Разработка котлована", 70, ("pit", "earthwork", "soil_pile"), ("pit_bottom_bare",)),
        ("Подготовка дна и начало бетонирования", 30,
         ("pit_bottom_bare",), ("basement_concrete_start",)),
    ],
    4: [
        ("Основание и подземные конструкции", 50,
         ("basement_concrete_start", "formwork_basement", "rebar"),
         ("basement_walls",)),
        ("Закрытие подземной части", 50, ("basement_walls", "backfill"), ("backfill", "above_grade")),
    ],
    5: [
        ("Монолитный каркас", 70, ("above_grade", "formwork_floor", "unfinished_top"), ("masonry",)),
        ("Заполнение проёмов", 30, ("masonry",), ("glazing",)),
    ],
    6: [
        ("Окна и утепление", 35, ("scaffold", "insulation"), ("glazing",)),
        ("Облицовка фасада", 45, ("insulation", "cladding"), ("cladding",)),
        ("Кровля", 20, ("roof_cover",), ("roof_cover",)),
    ],
    7: [
        ("Проезды и дорожки", 50, ("asphalt_work", "paving"), ("paving",)),
        ("Озеленение и площадки", 50,
         ("landscaping_work", "landscaping", "amenities"),
         ("landscaping", "amenities")),
    ],
}


@dataclass(slots=True)
class StageProgress:
    percent: int
    coverage: int
    observed_milestones: int
    total_milestones: int


def stage_progress(days: dict, stage_starts: dict[int, object] | None = None,
                   as_of=None) -> dict[int, StageProgress]:
    """Считает вехи только внутри периода соответствующего этапа."""
    all_days = {day: answers for day, answers in days.items()
                if as_of is None or day <= as_of}
    ordered_stages = sorted((stage_id, start) for stage_id, start
                            in (stage_starts or {}).items())
    windows = {}
    for index, (stage_id, start) in enumerate(ordered_stages):
        end = max(all_days, default=start)
        if index + 1 < len(ordered_stages):
            end = min(end, ordered_stages[index + 1][1] + dt.timedelta(days=14))
        windows[stage_id] = {day: answers for day, answers in all_days.items()
                             if start <= day <= end}

    out = {}
    for stage_id, milestones in MILESTONES.items():
        relevant_days = windows.get(stage_id, all_days if stage_starts is None else {})
        recent = set(sorted(relevant_days)[-14:])
        total_weight = sum(item[1] for item in milestones)
        done_weight = 0.0
        seen_weight = 0.0
        observed = 0

        for _name, weight, active_keys, done_keys in milestones:
            keys = set(active_keys) | set(done_keys)
            seen = any(answers.get(key) in (Answer.YES, Answer.NO)
                       for answers in relevant_days.values() for key in keys)
            if seen:
                seen_weight += weight
                observed += 1

            completed = any(
                answers.get(key) is Answer.YES
                for answers in relevant_days.values() for key in done_keys)
            active = any(
                relevant_days[day].get(key) is Answer.YES
                for day in recent for key in active_keys)
            if completed:
                done_weight += weight
            elif active:
                done_weight += weight * 0.5

        out[stage_id] = StageProgress(
            percent=round(100 * done_weight / total_weight) if total_weight else 0,
            coverage=round(100 * seen_weight / total_weight) if total_weight else 0,
            observed_milestones=observed,
            total_milestones=len(milestones),
        )
    return out


def object_progress(progress_by_stage: dict[int, StageProgress], stages: list,
                    weights: dict[int, float]) -> tuple[int, int]:
    """Сворачивает этапы по объёму работ и возвращает прогресс и покрытие."""
    total = sum(weights.get(stage_id, 0.0) for stage_id in stages)
    if not total:
        return 0, 0
    percent = sum(weights.get(sid, 0.0) * value.percent
                  for sid, value in progress_by_stage.items() if sid in stages) / total
    coverage = sum(weights.get(sid, 0.0) * value.coverage
                   for sid, value in progress_by_stage.items() if sid in stages) / total
    return round(percent), round(coverage)
