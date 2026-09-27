"""Анализатор, оценка кадра, хронология и отклонения — на поддельных ответах модели."""

import importlib.util
from datetime import date, datetime, timedelta

from conftest import FakeClient, triage
from core import config, timeline
from core.analyzer import Analyzer, normalize_answers, pick_candidates
from core.scoring import evaluate
from core.storage import Storage

FRAME_5 = dict(likelihood={1: 0.2, 4: 0.3, 5: 0.9, 7: 0.2},
               yes={"above_grade", "crane", "formwork_floor", "unfinished_top", "fence", "cabins"},
               no={"cladding", "glazing"})


def analyze(checklist, likelihood, yes=(), no=(), equipment=(), strategy="two_step", storage=None, **measure):
    client = FakeClient(triage(likelihood, equipment, **measure), yes, no)
    result, cached = Analyzer(checklist, storage, client, strategy).analyze("sha", lambda: "data:")
    return result, client


# --- анализатор ---

def test_candidates_threshold_and_minimum():
    assert pick_candidates({1: 0.1, 2: 0.05, 3: 0.0}) == [1, 2]
    assert pick_candidates({3: 0.8, 4: 0.6, 5: 0.4, 6: 0.35}) == [3, 4, 5]


def test_answers_normalized_and_missing_are_unsure():
    got = normalize_answers({"answers": {"pit": "Да", "crane": "no", "fence": "может быть"}}, ["pit", "crane", "fence", "slab"])
    assert got == {"pit": "yes", "crane": "no", "fence": "unsure", "slab": "unsure"}


def test_strategy_changes_only_number_of_checklist_calls(checklist):
    _, two = analyze(checklist, {4: 0.6, 5: 0.9})
    _, per = analyze(checklist, {4: 0.6, 5: 0.9}, strategy="per_stage")
    assert len(two.requests) == 2 and len(per.requests) == 3


def test_cache_hit_spends_nothing(checklist, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    storage = Storage(tmp_path / "db.sqlite")
    client = FakeClient(triage({5: 0.9}))
    an = Analyzer(checklist, storage, client)
    first, cached1 = an.analyze("abc", lambda: "data:")
    calls = len(client.requests)
    second, cached2 = an.analyze("abc", lambda: (_ for _ in ()).throw(AssertionError("картинку не кодируем")))
    assert not cached1 and cached2 and len(client.requests) == calls and first == second
    assert storage.spend()["calls"] == calls


# --- оценка кадра ---

def test_shared_sign_does_not_trigger_unasked_stage(checklist):
    # Траншея с трубами — признак и выноса сетей (1.3), и наружных сетей (8.1).
    # На подготовке территории она не должна перебросить объект на благоустройство.
    a, _ = analyze(checklist, {1: 0.9, 2: 0.3}, {"utility_trench", "pipes_stock", "fence"})
    assert a["candidates"] == [1, 2]
    score = evaluate(checklist, a)
    assert score["front"] == 1 and score["stages"][1]["substages"]["1.3"] == "active"


def test_superstructure_frame(checklist):
    a, _ = analyze(checklist, FRAME_5["likelihood"], FRAME_5["yes"], FRAME_5["no"], floors_built=8)
    score = evaluate(checklist, a, floors_total=16)
    assert score["front"] == 5
    for sid in (1, 2, 3, 4):
        assert score["stages"][sid]["progress"] == 1.0
    # 5.1 идёт на 8 из 16 этажей, 5.3 ещё не видна → 0.7 × 0.5
    assert score["stages"][5]["progress"] == 0.35
    assert score["stages"][7]["progress"] == 0
    assert score["overall_pct"] == round(5 + 8 + 7 + 12 + 35 * 0.35, 1)


def test_not_a_construction_site(checklist):
    a, _ = analyze(checklist, {s: 0.0 for s in range(1, 9)})
    score = evaluate(checklist, a)
    assert score["front"] is None and score["overall_pct"] == 0


# --- хронология ---

def frame(i, day, analysis):
    return {"id": i, "taken_at": datetime(2025, 1, 1) + timedelta(days=day), "filename": f"{i}.jpg",
            "analysis": analysis}


def test_progress_never_goes_back_and_latching_holds(checklist):
    pit, _ = analyze(checklist, {3: 0.9}, {"pit", "soil_pile", "earthwork"},
                     equipment=[("excavator", 1, 1), ("dump_truck", 2, 0)])
    blind, _ = analyze(checklist, {1: 0.6, 3: 0.3})  # ракурс, где котлован не виден
    tl = timeline.build(checklist, [frame(1, 0, pit), frame(2, 1, blind)])
    assert tl["series"][1][1] >= tl["series"][0][1]
    assert tl["frames"][1]["score"]["front"] == 3
    assert "pit" in tl["latched"]


def test_tz_example_excavator_without_trucks(checklist):
    a, _ = analyze(checklist, {3: 0.9}, {"pit", "earthwork", "soil_pile"}, equipment=[("excavator", 1, 1)])
    tl = timeline.build(checklist, [frame(1, 0, a)])
    assert any(d["rule"] == "excavator_no_trucks" for d in tl["deviations"])


def test_forbidden_equipment_on_stage(checklist):
    a, _ = analyze(checklist, FRAME_5["likelihood"], FRAME_5["yes"], FRAME_5["no"],
                   equipment=[("tower_crane", 1, 1), ("pile_driver", 1, 1)])
    tl = timeline.build(checklist, [frame(1, 0, a)])
    assert [d["title"] for d in tl["deviations"] if d["rule"] == "forbidden_equipment"] == [
        "Техника не по этапу: Копёр / вибропогружатель"]


def test_idle_streak(checklist):
    busy, _ = analyze(checklist, {3: 0.9}, {"pit"}, equipment=[("excavator", 1, 1), ("dump_truck", 2, 1)])
    idle, _ = analyze(checklist, {3: 0.9}, {"pit"}, equipment=[("excavator", 1, 0)], workers_count=0)
    tl = timeline.build(checklist, [frame(1, 0, busy)] + [frame(i, i, idle) for i in range(1, 5)])
    assert tl["idle_streak"] == 4
    assert any(d["rule"] == "idle_streak" for d in tl["deviations"])


def test_schedule_lag_verdict_and_forecast(checklist):
    plan = {s["id"]: (date(2024, 1, 1), date(2024, 12, 31)) for s in checklist.stages}
    early, _ = analyze(checklist, {1: 0.9}, {"fence", "cabins", "cleared"})
    later, _ = analyze(checklist, {3: 0.9}, {"pit", "soil_pile", "earthwork"})
    frames = [frame(1, 0, early), frame(2, 60, later)]  # 2025-01-01 и 2025-03-02 — план уже весь позади
    tl = timeline.build(checklist, frames, plan)
    sch = tl["schedule"]
    assert sch["verdict"] == "отставание" and sch["lag_days"] > 60
    assert any(d["rule"] == "stage_overdue" and d["severity"] == "critical" for d in tl["deviations"])
    fc = tl["forecast"]
    assert fc["finish"] > date(2025, 3, 2) and fc["delay_days"] > 0


def test_ahead_and_tolerance(checklist):
    a, _ = analyze(checklist, {1: 0.9}, {"fence", "cabins", "cleared"})  # готова подготовка = 5%
    start = date(2025, 1, 1)
    for span, verdict in ((100, "в срок"), (1000, "опережение")):
        # 5% плана — это 5 дней при плане на 100 дней (в допуске) и 50 дней при плане на 1000
        plan = {s["id"]: (start, start + timedelta(days=span)) for s in checklist.stages}
        tl = timeline.build(checklist, [frame(1, 0, a)], plan)
        assert tl["schedule"]["verdict"] == verdict


# --- справочник ---

def test_checklist_references_are_consistent(checklist):
    spec = importlib.util.spec_from_file_location("build", config.ROOT / "reference" / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert build.check_references(checklist.data) == []


# --- регрессии с прогона на test_photos ---

def test_candidates_include_stage_after_latest():
    # 2006-05-08: разведка дала этапу 4 лишь 0.2 — плиту не проверили, этап занижен
    assert pick_candidates({2: 0.3, 3: 0.9, 4: 0.2}, latest=3) == [2, 3, 4]
    assert pick_candidates({7: 0.0, 8: 0.0}, latest=8) == [7, 8]


def test_not_visible_counts_as_unsure():
    assert normalize_answers({"answers": {"pit": "not_visible"}}, ["pit"]) == {"pit": "unsure"}


def test_finished_building_is_not_lost(checklist):
    # 2007-11-28: готовый фасад, «работы идут» по всем этапам ≈ 0
    a, _ = analyze(checklist, {s: 0.0 for s in range(1, 9)}, {"above_grade", "cladding", "glazing"},
                   latest_stage=7)
    assert a["candidates"] == [7, 8]
    assert evaluate(checklist, a)["front"] == 7


def test_idle_needs_evidence_and_consecutive_days(checklist):
    facade, _ = analyze(checklist, {7: 0.9}, {"cladding"})  # вид сбоку, рабочих не посчитать
    overview, _ = analyze(checklist, {3: 0.9}, {"pit"}, view="top")
    tl = timeline.build(checklist, [frame(i, 30 * i, facade) for i in range(3)])
    assert not any(d["idle"] for d in tl["days"]) and tl["idle_streak"] == 0
    # обзорные кадры без работ, но с разрывом в месяц — не серия
    tl = timeline.build(checklist, [frame(i, 30 * i, overview) for i in range(3)])
    assert all(d["idle"] for d in tl["days"]) and tl["idle_streak"] == 1


def test_missing_equipment_only_on_overview(checklist):
    side, _ = analyze(checklist, FRAME_5["likelihood"], FRAME_5["yes"], FRAME_5["no"], workers_count=5)
    top, _ = analyze(checklist, FRAME_5["likelihood"], FRAME_5["yes"], FRAME_5["no"], workers_count=5, view="top")
    assert not any(d["rule"] == "missing_equipment" for d in timeline.build(checklist, [frame(1, 0, side)])["deviations"])
    assert any(d["rule"] == "missing_equipment" for d in timeline.build(checklist, [frame(1, 0, top)])["deviations"])


def test_parked_tower_crane_alone_is_not_idle_site(checklist):
    a, _ = analyze(checklist, FRAME_5["likelihood"], FRAME_5["yes"], FRAME_5["no"],
                   equipment=[("tower_crane", 2, 0)])
    assert not any(d["rule"] == "all_idle" for d in timeline.build(checklist, [frame(1, 0, a)])["deviations"])


def test_shared_earthwork_signs_do_not_open_next_stage(checklist):
    # 2005-12-23, прогон p2: этап 4 — кандидат как «следующий», «идёт разработка грунта»
    # включала засыпку пазух (4.6), и котлован превращался в подземный монолит
    a, _ = analyze(checklist, {3: 0.9}, {"pit", "earthwork", "soil_pile"}, latest_stage=3)
    assert a["candidates"] == [3, 4]
    assert evaluate(checklist, a)["front"] == 3


def test_single_sign_does_not_jump_over_triage(checklist):
    # 2006-07-17, прогон с контекстом: этап 5 — кандидат из истории, «опалубка наверху» = да
    # (модель приняла опалубку стен подвала), разведка этапу 5 дала 0 → остаётся этап 4
    yes = {"below_grade", "rebar", "formwork", "basement_walls", "formwork_floor", "crane"}
    a, _ = analyze(checklist, {4: 1.0}, yes, latest_stage=4)
    assert a["candidates"] == [4, 5]
    assert evaluate(checklist, a)["front"] == 4
    # та же картина, но разведка допускает каркас — один признак этап открывает
    a, _ = analyze(checklist, {4: 1.0, 5: 0.4}, yes, latest_stage=4)
    assert evaluate(checklist, a)["front"] == 5


def test_tower_crane_alone_is_not_superstructure(checklist):
    # PLAN.md §3.1: башенный кран стоит с нулевого цикла — это не признак каркаса
    assert "crane" not in checklist.stage_by_id[5]["must_have"]
