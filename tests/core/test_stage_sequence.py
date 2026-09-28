"""Монотонная хронология этапов: выбросы, откаты, ручные якоря, latching, needs_review.

Кадры строятся из «профилей» этапа — ответов, которые модель Б дала бы на кадре
этого этапа (признаки своего этапа — «да», отличительные признаки поздних — «нет»).
"""
import datetime as dt

import pytest

from core.contracts import Answer, ChecklistResult, StageObservation, StageState, StageStatus
from core.stage import sequence

Y, N, U = Answer.YES, Answer.NO, Answer.UNSURE
T0 = dt.datetime(2026, 5, 1, 9, 0, tzinfo=dt.timezone.utc)   # 12:00 по Москве

LATER_NO = {"above_grade": N, "formwork_floor": N, "unfinished_top": N, "cladding": N, "glazing": N,
            "paving": N, "landscaping": N, "amenities": N, "roof_cover": N, "roof_work": N}
PROFILES = {
    1: {"fence": Y, "cabins": Y, "cleared": Y, "pit": N, "rig": N, "sheet_pile": N, **LATER_NO},
    3: {"pit": Y, "soil_pile": Y, "earthwork": Y, "slab": N, "below_grade": N, "formwork": N, "rebar": N,
        **LATER_NO},
    4: {"formwork": Y, "rebar": Y, "slab": Y, "below_grade": Y, "pit": Y, **LATER_NO},
    5: {"above_grade": Y, "formwork_floor": Y, "unfinished_top": Y, "crane": Y, "pit": N, "cladding": N,
        "glazing": N, "scaffold": N, "paving": N, "landscaping": N, "amenities": N},
    8: {"paving": Y, "landscaping": Y, "amenities": Y, "bare_ground": N, "above_grade": Y},
}
_ids = iter(range(1, 100000))


def obs(day: float, answers: dict, camera: int = 1, frame_id=None) -> StageObservation:
    return StageObservation(frame_id=frame_id if frame_id is not None else next(_ids), camera_id=camera,
                            captured_at=T0 + dt.timedelta(days=day),
                            result=ChecklistResult(answers=dict(answers)))


def day(n: int) -> dt.date:
    return (T0 + dt.timedelta(days=n)).date()


def series(stage: int, days: range, per_day: int = 2) -> list[StageObservation]:
    return [obs(d + k * 0.1, PROFILES[stage], camera=k + 1) for d in days for k in range(per_day)]


def fronts(tl) -> list[int]:
    return [f for _, f in tl.daily_front]


def test_clean_progression_gives_monotone_front_and_dates():
    tl = sequence.infer(series(3, range(0, 10)) + series(4, range(10, 20)) + series(5, range(20, 30)))
    f = fronts(tl)
    assert f == sorted(f) and f[0] == 3 and tl.current_stage == 5
    assert tl.rejected_outliers == [] and tl.needs_review == []
    s = tl.states
    assert s[1].status is s[2].status is s[3].status is s[4].status is StageStatus.DONE
    assert s[5].status is StageStatus.ACTIVE and s[6].status is StageStatus.NOT_STARTED
    assert s[3].actual_start is None                   # этап шёл уже на первом кадре — дату не выдумываем
    assert abs((s[4].actual_start - day(10)).days) <= 2 and abs((s[5].actual_start - day(20)).days) <= 2
    assert s[3].actual_end == s[4].actual_start and s[4].actual_end == s[5].actual_start
    assert s[5].confidence > 0.5 and s[5].evidence_frame_ids
    assert 0.35 < tl.overall_progress < 0.6


def test_single_outlier_stage8_among_stage3_is_rejected():
    frames = series(3, range(0, 12))
    ghost = obs(5.5, PROFILES[8], camera=2, frame_id="ghost")      # соседний готовый дом / галлюцинация
    tl = sequence.infer(frames + [ghost])
    assert tl.rejected_outliers == ["ghost"]
    assert set(fronts(tl)) == {3} and tl.current_stage == 3
    assert tl.states[8].status is StageStatus.NOT_STARTED and tl.states[8].progress == 0


def test_rollback_attempt_is_ignored():
    """Этапы не откатываются: кадры «котлована» после каркаса — выбросы (соседняя стройка)."""
    late = series(3, range(6, 9))
    tl = sequence.infer(series(5, range(0, 6)) + late)
    assert fronts(tl) == [5] * 9 and tl.current_stage == 5
    assert sorted(tl.rejected_outliers) == sorted(o.frame_id for o in late)


def test_front_does_not_jump_on_one_day_but_long_gap_allows_it():
    tl = sequence.infer(series(3, range(0, 8)) + series(5, range(60, 66)))
    f = fronts(tl)
    assert f[:8] == [3] * 8 and f[-1] == 5 and tl.rejected_outliers == []
    assert tl.states[4].status is StageStatus.DONE     # пройден в пропуске между наблюдениями


def test_manual_anchor_is_respected():
    manual = {5: StageState(stage_id=5, status=StageStatus.ACTIVE, progress=0.3,
                            actual_start=day(6), manual=True)}
    tl = sequence.infer(series(3, range(0, 12)), manual=manual)
    for d, f in tl.daily_front:
        assert (f >= 5) if d >= day(6) else (f < 5), (d, f)
    s5 = tl.states[5]
    assert s5.manual and s5.status is StageStatus.ACTIVE and s5.progress == 0.3 and s5.actual_start == day(6)
    assert tl.states[3].status is StageStatus.DONE and tl.current_stage >= 5


def test_manual_not_started_caps_the_front():
    manual = {4: StageState(stage_id=4, status=StageStatus.NOT_STARTED, progress=0.0, manual=True)}
    tl = sequence.infer(series(3, range(0, 5)) + series(4, range(5, 12)), manual=manual)
    assert max(fronts(tl)) == 3 and tl.states[4].manual and tl.states[4].status is StageStatus.NOT_STARTED


def test_latching_sign_needs_two_days():
    """Засыпка пазух (latching) один раз «увидена» — не засчитана; два разных дня — навсегда."""
    once = series(4, range(0, 10))
    once.append(obs(3.5, {**PROFILES[4], "backfill": Y}))
    tl1 = sequence.infer(once)
    assert tl1.states[4].progress == pytest.approx(0.75)
    assert tl1.states[4].status is StageStatus.ACTIVE

    twice = series(4, range(0, 10)) + [obs(3.5, {**PROFILES[4], "backfill": Y}),
                                       obs(4.5, {**PROFILES[4], "backfill": Y})]
    twice += [obs(8.5, {**PROFILES[4], "backfill": N})]            # потом засыпку «не видно» — неважно
    tl2 = sequence.infer(twice)
    assert tl2.states[4].progress == pytest.approx(1.0)
    assert tl2.states[4].status is StageStatus.DONE and tl2.states[4].actual_end == day(4)


def test_progress_is_monotone_and_one_day_spike_is_not_counted():
    base = {"pit": Y, "soil_pile": Y, "earthwork": N, "struts": N, "pit_bottom_prepared": N,
            "slab": N, "below_grade": N, **LATER_NO}
    frames = [obs(d, base) for d in range(10)]
    frames[4] = obs(4, {k: U for k in base})                          # «провал»: всё не видно
    frames[6] = obs(6, {**base, "earthwork": Y})                      # одиночный всплеск активности
    previous = 0.0
    for k in range(1, len(frames) + 1):
        progress = sequence.infer(frames[:k]).states[3].progress
        assert progress >= previous, k
        previous = progress
    assert previous == pytest.approx(0.6)                             # всплеск 0.7 одним днём не засчитан
    tl = sequence.infer(frames, config={"progress_confirm_days": 1})
    assert tl.states[3].progress == pytest.approx(0.7)                # без подтверждения — засчитан бы


def test_needs_review_lists_frames_with_many_unsure():
    frames = series(3, range(0, 6))
    doubtful = obs(2.5, {"pit": Y, "soil_pile": U, "earthwork": U, "slab": U}, frame_id="doubt")
    empty = obs(3.5, {}, frame_id="empty")                             # сбой классификатора — тоже проверить
    tl = sequence.infer(frames + [doubtful, empty])
    assert tl.needs_review == ["doubt", "empty"] and tl.current_stage == 3
    tl2 = sequence.infer(frames + [doubtful], config={"needs_review_ratio": 0.8})
    assert tl2.needs_review == []


def test_unsure_camera_does_not_dilute_the_day():
    blind = {k: U for k in PROFILES[3]}
    both = [obs(d, PROFILES[3], camera=1) for d in range(5)] + [obs(d + 0.2, blind, camera=2) for d in range(5)]
    alone = [obs(d, PROFILES[3], camera=1) for d in range(5)]
    a, b = sequence.infer(both), sequence.infer(alone)
    assert fronts(a) == fronts(b) and a.states[3].progress == b.states[3].progress


def test_day_boundary_is_moscow_time():
    late_evening_utc = dt.datetime(2026, 5, 1, 22, 30, tzinfo=dt.timezone.utc)   # 01:30 2 мая по Москве
    assert sequence.local_day(late_evening_utc) == dt.date(2026, 5, 2)
    assert sequence.local_day(dt.datetime(2026, 5, 1, 22, 30)) == dt.date(2026, 5, 2)   # наивное = UTC


def test_no_observations_uses_manual_only():
    empty = sequence.infer([])
    assert empty.current_stage is None and empty.daily_front == []
    assert all(s.status is StageStatus.NOT_STARTED for s in empty.states.values())
    blind = sequence.infer([obs(0, {"pit": U, "slab": U}, frame_id="blind")])
    assert blind.current_stage is None and blind.needs_review == ["blind"]   # не «подготовка» по равенству шансов
    manual = {1: StageState(1, StageStatus.DONE, 1.0, manual=True),
              2: StageState(2, StageStatus.ACTIVE, 0.4, manual=True)}
    tl = sequence.infer([], manual=manual)
    assert tl.current_stage == 2 and tl.states[2].manual and tl.states[1].progress == 1.0
    assert tl.overall_progress == pytest.approx((5 * 1.0 + 8 * 0.4) / 100)
