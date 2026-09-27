"""База знаний по стройке: контекст для модели, сводка, выгрузка CSV."""

import csv
import io
from datetime import date, datetime, timedelta

from conftest import FakeClient, triage
from core import timeline
from core.analyzer import Analyzer, pick_candidates
from core.site import ContextBuilder, frames_table, knowledge, to_csv


def analyze(checklist, likelihood, yes=(), equipment=(), context=None, **measure):
    client = FakeClient(triage(likelihood, equipment, **measure), yes)
    result, _ = Analyzer(checklist, None, client).analyze("sha", lambda: "data:", context)
    return result, client


def frame(i, day, analysis):
    return {"id": i, "taken_at": datetime(2006, 1, 1) + timedelta(days=day), "filename": f"cam_{i}.jpg",
            "analysis": analysis}


def series(checklist):
    pit, _ = analyze(checklist, {3: 0.9}, {"pit", "earthwork", "soil_pile"}, latest_stage=3,
                     equipment=[("excavator", 2, 1), ("dump_truck", 1, 1)])
    slab, _ = analyze(checklist, {4: 0.9}, {"slab", "rebar", "formwork", "below_grade"}, latest_stage=4,
                      equipment=[("tower_crane", 1, 0), ("concrete_pump", 1, 1), ("concrete_mixer", 2, 1)])
    frame5, _ = analyze(checklist, {5: 0.9}, {"above_grade", "crane", "formwork_floor", "unfinished_top"},
                        latest_stage=5, equipment=[("tower_crane", 2, 1)])
    return [frame(1, 0, pit), frame(2, 30, slab), frame(3, 60, frame5)]


def test_context_accumulates_front_facts_and_equipment(checklist):
    ctx = ContextBuilder(checklist)
    assert ctx.text() == "" and ctx.digest() == ""
    for f in series(checklist):
        ctx.add(f["analysis"], f["taken_at"].date())
    text = ctx.text()
    assert ctx.front == 5 and "достигнутый этап: 5" in text and "этапы 1–4 выполнены" in text
    assert "открытый котлован (01.01.2006)" in text  # latching-факт с датой
    assert "Башенный кран (до 2)" in text
    assert "context_conflict" in text


def test_context_goes_into_prompt_and_cache_key(checklist):
    ctx = ContextBuilder(checklist)
    for f in series(checklist)[:2]:
        ctx.add(f["analysis"], f["taken_at"].date())
    _, client = analyze(checklist, {5: 0.9}, context=ctx)
    assert all("Контекст этой стройки" in r for r in client.requests)  # и разведка, и чек-лист
    an = Analyzer(checklist, None, client)
    assert an.cache_key("sha", ctx) != an.cache_key("sha")
    assert an.cache_key("sha") == an.cache_key("sha", ContextBuilder(checklist))  # пустой контекст = без контекста


def test_previous_front_is_always_checked():
    # разведка сочла кадр котлованом, а по истории стройка уже на каркасе — чек-лист каркаса проверяем
    likelihood = {s: 0.0 for s in range(1, 9)} | {3: 0.9}
    assert pick_candidates(likelihood, latest=3, prev_front=5) == [3, 4, 5]


def test_knowledge_over_whole_site(checklist):
    tl = timeline.build(checklist, series(checklist))
    kb = knowledge(checklist, tl)
    assert kb["photos"] == 3 and kb["front"] == 5 and kb["period"] == (date(2006, 1, 1), date(2006, 3, 2))
    eq = {e["type"]: e for e in kb["equipment"]}
    assert eq["tower_crane"]["photos"] == 2 and eq["tower_crane"]["max_at_once"] == 2
    assert eq["tower_crane"]["first_seen"] == date(2006, 1, 31) and eq["tower_crane"]["stages"] == [4, 5]
    assert eq["excavator"]["working_share"] == 0.5
    assert [f["key"] for f in kb["facts"]][:1] == ["pit"]
    assert kb["stages"][2]["first_seen"] == date(2006, 1, 1)  # этап 3


def test_csv_opens_in_excel_and_has_equipment_columns(checklist):
    tl = timeline.build(checklist, series(checklist))
    data = to_csv(frames_table(checklist, tl))
    assert data.startswith("﻿".encode())  # BOM — Excel иначе ломает кириллицу
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 3
    assert rows[0]["Экскаватор: всего"] == "2" and rows[0]["Экскаватор: в работе"] == "1"
    assert rows[2]["Башенный кран: всего"] == "2" and rows[0]["Башенный кран: всего"] == "0"
    assert rows[2]["этап_кадра"] == "5"
    assert to_csv(knowledge(checklist, tl)["equipment"])
