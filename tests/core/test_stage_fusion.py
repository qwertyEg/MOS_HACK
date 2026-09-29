"""Этап с учётом техники (core/stage/fusion.py + sequence.infer(equipment=…)).

Сценарии — с демо-объектов, где этап по одному чек-листу был неправдоподобен: асфальтирование
парковки (чек-лист «подготовка», на площадке каток и самосвалы), ранний «каркас» по соседней
башне при копре на площадке (Эдинбург), и обратное — российский сборный каркас, где ложные рамки
(бытовка-«самосвал») не должны сбивать уверенный чек-лист.
"""
import datetime as dt

import pytest

from core.contracts import ActivityInterval, Answer, ChecklistResult, StageObservation, StageStatus
from core.stage import fusion, sequence

Y, N, U = Answer.YES, Answer.NO, Answer.UNSURE
T0 = dt.datetime(2026, 5, 1, 7, 0, tzinfo=dt.timezone.utc)    # 10:00 по Москве
_ids = iter(range(1, 10 ** 6))

LATER_NO = {"above_grade": N, "formwork_floor": N, "unfinished_top": N, "cladding": N, "glazing": N,
            "paving": N, "landscaping": N, "amenities": N, "roof_cover": N, "roof_work": N}
PIT = {"pit": Y, "soil_pile": Y, "earthwork": Y, "slab": N, "below_grade": N, "formwork": N, "rebar": N, **LATER_NO}
FRAME = {"above_grade": Y, "formwork_floor": Y, "unfinished_top": Y, "crane": Y, "pit": N, "cladding": N,
         "glazing": N, "scaffold": N, "paving": N, "landscaping": N, "amenities": N}
# Парковка издалека: SigLIP видит только «расчищено» и «планировка», этап не выдаёт.
PARKING = {"cleared": Y, "grading": Y, "fence": N, "cabins": N, "pit": N, "above_grade": N, "paving": N,
           "landscaping": N, "amenities": N, "asphalt_work": N, "curbs": N}


def obs(day: float, answers: dict, camera: int = 1) -> StageObservation:
    return StageObservation(frame_id=next(_ids), camera_id=camera, captured_at=T0 + dt.timedelta(days=day),
                            result=ChecklistResult(answers=dict(answers)))


def work(day: int, cls: str, hours: float, at_h: float = 0.0) -> ActivityInterval:
    start = T0 + dt.timedelta(days=day, hours=at_h)
    return ActivityInterval(unit_id=f"{cls}-1", cls=cls, start=start, end=start + dt.timedelta(hours=hours),
                            hours=hours, stage_id=None, frame_ids=[f"{cls}@{day}"])


def seen(day: int, cls: str, activity: str = "unknown", conf: float = 0.8, k: int = 0) -> fusion.Sighting:
    return fusion.Sighting(frame_id=f"{cls}@{day}.{k}", captured_at=T0 + dt.timedelta(days=day, hours=k * 0.3),
                           cls=cls, conf=conf, activity=activity)


# --------------------------------------------------------------------------
# нормы → правдоподобия
# --------------------------------------------------------------------------


def best(cls: str) -> list[int]:
    lam = fusion.likelihoods()[cls]
    top = max(lam.values())
    return sorted(s for s, v in lam.items() if v >= top - 1e-9)


def test_specific_equipment_points_to_its_stage():
    assert best("roller") == [8] and best("asphalt_paver") == [8]
    assert best("pile_driver") == [2] and best("drilling_rig") == [2]
    assert set(best("concrete_pump")) == {4, 5}
    assert set(best("tower_crane")) == {4, 5, 6}
    assert set(best("excavator")) == {1, 3, 8}     # экскаватор этап не выдаёт — выбор за чек-листом


def test_forbidden_equipment_argues_against_stage_but_earlier_stage_work_is_softened():
    lam = fusion.likelihoods()
    assert lam["roller"][5] < lam["roller"][4] < lam["roller"][8]
    assert lam["pile_driver"][5] < 0 < lam["pile_driver"][2]
    # экскаватор на каркасе — засыпка пазух, сети: против фронта 5 слабее, чем копёр
    assert lam["excavator"][5] > lam["pile_driver"][5]


def test_days_aggregate_work_and_presence_by_local_day():
    ev = fusion.EquipmentEvidence(
        intervals=[work(0, "excavator", 2.0), work(0, "excavator", 1.0, 3), work(1, "dump_truck", 0.5),
                   ActivityInterval("manual:excavator", "excavator", T0, T0, -3.0, None)],   # ручная поправка
        sightings=[seen(0, "tower_crane", "idle"), seen(0, "tower_crane", "idle"),           # одна рамка дважды
                   seen(0, "tower_crane", "idle", k=1), seen(1, "bulldozer", conf=0.3)])   # неуверенная рамка
    days = fusion.days(ev, tz_offset_hours=3.0)
    d0, d1 = days[T0.date()], days[T0.date() + dt.timedelta(days=1)]
    assert d0.working_h == {"excavator": 3.0} and d0.seen == {"tower_crane": 2} and d0.judged == {"tower_crane": 2}
    assert d1.working_h == {"dump_truck": 0.5} and "bulldozer" not in d1.seen
    a = fusion.strengths(d0)
    assert a["excavator"] > 0.9 and a["tower_crane"] == pytest.approx(0.2)   # стоит на плотной съёмке
    daily = fusion.days(fusion.EquipmentEvidence(sightings=[seen(0, "pile_driver")]))
    assert fusion.strengths(daily[T0.date()])["pile_driver"] == pytest.approx(0.5)   # снимок раз в сутки


# --------------------------------------------------------------------------
# хронология с техникой
# --------------------------------------------------------------------------


def test_paving_site_is_external_works_by_roller_and_trucks():
    """«Асфальтирование парковки»: чек-лист выдаёт «подготовку», каток с самосвалами — благоустройство."""
    frames = [obs(d + k * 0.1, PARKING, k) for d in range(2) for k in range(5)]
    ev = fusion.EquipmentEvidence(intervals=[x for d in range(2) for x in (
        work(d, "roller", 2.5), work(d, "dump_truck", 2.3, 1), work(d, "wheel_loader", 7.0, 2))])
    alone = sequence.infer(frames)
    fused = sequence.infer(frames, equipment=ev)
    assert alone.current_stage != 8
    assert fused.current_stage == 8 and fused.states[8].status is StageStatus.ACTIVE
    assert fused.states[7].status is StageStatus.DONE
    b = fused.basis
    assert b["decided_by"] == "equipment" and b["stage"] == 8 and b["checklist_stage"] is None
    assert "каток" in b["text"] and "Наружные сети и благоустройство" in b["text"]
    assert [r["cls"] for r in b["equipment"]][:1] == ["wheel_loader"] and b["equipment"][0]["hours"] == 14.0
    # этап, выданный техникой, получает снимки-доказательства — кадры с её работой
    assert set(fused.states[8].evidence_frame_ids) & {"roller@1", "dump_truck@1", "wheel_loader@1"}


def test_confident_checklist_is_not_overridden_by_false_boxes():
    """Российский сборный каркас: автокран и башенный кран работают, бытовка — «самосвал», прицеп —
    «экскаватор» (стоят). Уверенный чек-лист каркаса остаётся, техника его подтверждает."""
    frames = [obs(k * 0.05, FRAME, k) for k in range(8)]
    ev = fusion.EquipmentEvidence(
        intervals=[work(0, "mobile_crane", 9.9), work(0, "tower_crane", 1.5, 1)],
        sightings=[seen(0, "excavator", "idle", 0.55, k) for k in range(4)]
        + [seen(0, "dump_truck", "idle", 0.6, k) for k in range(4)])
    tl = sequence.infer(frames, equipment=ev)
    assert tl.current_stage == 5
    assert tl.basis["equipment_relation"] == "agree" and tl.basis["decided_by"] == "both"
    assert "башенный кран" in tl.basis["text"] and "подтверждает" in tl.basis["text"]


def test_pile_driver_on_site_holds_back_premature_superstructure():
    """Соседняя готовая башня даёт чек-листу «каркас», а на площадке ещё копёр (Эдинбург, 2005–2006)."""
    frames = [obs(d + k * 0.1, PIT, k) for d in range(10) for k in range(2)] + [obs(d, FRAME) for d in range(10, 14)]
    ev = fusion.EquipmentEvidence(sightings=[seen(d, "pile_driver") for d in range(14)]
                                  + [seen(d, "excavator", k=1) for d in range(14)])
    assert sequence.infer(frames).current_stage == 5
    tl = sequence.infer(frames, equipment=ev)
    assert tl.current_stage == 3 and max(f for _, f in tl.daily_front) == 3
    assert tl.rejected_outliers == []     # выбросы — по одному чек-листу: техника голос модели Б не отнимает
    assert tl.basis["checklist_stage"] == 5 and tl.basis["decided_by"] == "equipment"


def test_tower_crane_and_concrete_move_front_to_monolith():
    """«Башенный кран + бетон → монолит»: чек-лист видит только котлован и арматуру."""
    weak = {"pit": Y, "soil_pile": U, "earthwork": U, "slab": U, "rebar": U, "formwork": U, "below_grade": U,
            **LATER_NO}
    frames = [obs(d + k * 0.1, weak, k) for d in range(10) for k in range(2)]
    ev = fusion.EquipmentEvidence(intervals=[x for d in range(3, 10) for x in (
        work(d, "concrete_pump", 3), work(d, "concrete_mixer", 2, 1), work(d, "tower_crane", 4, 2))])
    assert sequence.infer(frames).current_stage == 3
    tl = sequence.infer(frames, equipment=ev)
    assert tl.current_stage == 4
    assert "автобетононасос" in tl.basis["text"]


def test_equipment_alone_does_not_assign_a_stage():
    tl = sequence.infer([obs(0, {"pit": U, "slab": U})],
                        equipment=fusion.EquipmentEvidence(intervals=[work(0, "roller", 5.0)]))
    assert tl.current_stage is None and tl.daily_front == []


def test_zero_weight_and_no_equipment_mean_checklist_only():
    frames = [obs(d + k * 0.1, PARKING, k) for d in range(2) for k in range(5)]
    ev = fusion.EquipmentEvidence(intervals=[work(d, "roller", 3.0) for d in range(2)])
    alone = sequence.infer(frames)
    off = sequence.infer(frames, config={"equipment_weight": 0.0}, equipment=ev)
    assert off.current_stage == alone.current_stage and off.daily_front == alone.daily_front
    assert off.basis["equipment"] == [] and alone.basis["equipment"] == []


def test_equipment_outside_checklist_span_is_ignored_but_night_day_inside_counts():
    frames = [obs(0, PIT), obs(4, PIT)]
    ev = fusion.EquipmentEvidence(intervals=[work(2, "excavator", 3.0), work(9, "roller", 8.0)])
    tl = sequence.infer(frames, equipment=ev)
    days = [d for d, _ in tl.daily_front]
    assert (T0 + dt.timedelta(days=2)).date() in days            # ночь/дождь: только техника — день учтён
    assert (T0 + dt.timedelta(days=9)).date() not in days        # после последнего ответа модели Б — нет
    assert tl.current_stage == 3


def test_basis_without_equipment_says_so():
    tl = sequence.infer([obs(d, PIT) for d in range(3)], equipment=fusion.EquipmentEvidence())
    assert tl.current_stage == 3 and tl.basis["decided_by"] == "checklist"
    assert "котлован" in tl.basis["text"] and "камеры не видели" in tl.basis["text"]
