"""Реальная модель GLM на test_photos против ручной разметки.

    pytest -m live -s                         # glm-4.6v, two_step
    GLM_EVAL_MODEL=glm-4.6v-flash pytest -m live -s
    GLM_EVAL_STRATEGY=per_stage pytest -m live -s
    GLM_EVAL_PROVIDER=local pytest -m live -s # локальная модель (docs/local-model.md)

Ответы пишутся в tests/fixtures/glm_cassette.json: повторный прогон с теми же
промптами бесплатен и работает без ключа. Отчёт — tests/fixtures/live_report.json.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from cassette import CassetteClient, CassetteMiss
from core import config, service, timeline
from core.analyzer import Analyzer
from core.providers import PROVIDERS, make_client as make_provider_client
from core.images import data_url, date_from_name, prepare, sha256
from core.plan import typical_plan
from core.scoring import evaluate
from core.site import ContextBuilder
from core.storage import Storage

pytestmark = pytest.mark.live

FIXTURES = Path(__file__).parent / "fixtures"
PHOTOS = config.ROOT / "test_photos"
TRUTH = json.loads((FIXTURES / "ground_truth.json").read_text())
FILES = sorted(k for k in TRUTH if not k.startswith("_"))
PROVIDER = os.getenv("GLM_EVAL_PROVIDER", "zai")
MODEL = os.getenv("GLM_EVAL_MODEL", PROVIDERS[PROVIDER]["default_model"])
STRATEGY = os.getenv("GLM_EVAL_STRATEGY", "two_step")
THINKING = os.getenv("GLM_EVAL_THINKING") == "1"
CONTEXT = os.getenv("GLM_EVAL_CONTEXT", "1") == "1"   # 0 — без истории стройки, для A/B


def make_client(model=MODEL, provider=PROVIDER):
    ready, _ = PROVIDERS[provider]["status"]()
    real = make_provider_client(provider, model, THINKING) if ready else None
    return CassetteClient(FIXTURES / "glm_cassette.json", real, model=model, thinking=THINKING, provider=provider)


def test_connectivity_on_free_model():
    """Ключ, адрес и формат ответа — на бесплатной модели и самом маленьком кадре."""
    client = make_client("glm-4.6v-flash", "zai")
    img = data_url(prepare((PHOTOS / "doric_2007_11_28_12_30_37.jpg").read_bytes()))
    try:
        reply = client.ask_json("Отвечай одним JSON-объектом.", img,
                                'Есть ли на снимке здание? Верни {"building": true} или {"building": false}.', 30)
    except CassetteMiss:
        pytest.skip("нет ключа ZAI_API_KEY и нет записи в кассете")
    assert isinstance(reply.data.get("building"), bool)


def test_model_against_ground_truth(tmp_path, monkeypatch, checklist):
    client = make_client()
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    storage = Storage(tmp_path / "db.sqlite")
    oid = storage.create_object("Doric", "Офисно-деловой центр", 7)
    storage.save_plan(oid, typical_plan(datetime(2005, 10, 1).date()), "manual")
    for n in FILES:
        service.ingest(storage, oid, n, (PHOTOS / n).read_bytes())

    analyzer = Analyzer(checklist, storage, client, STRATEGY, use_context=CONTEXT)
    try:
        _, errors = service.analyze_frames(storage, analyzer, storage.list_objects()[0])
    except CassetteMiss:
        pytest.skip(f"провайдер {PROVIDER} не готов и нет записи в кассете")
    assert errors == [], errors

    obj = storage.list_objects()[0]
    frames, _, tl = service.report(checklist, storage, analyzer, obj)

    rows, cost, hits, halluc, missed = [], 0.0, 0, 0, 0
    for f, scored in zip(frames, tl["frames"]):
        t, a = TRUTH[f["filename"]], f["analysis"]
        single = evaluate(checklist, a, floors_total=7)  # оценка кадра без хронологии
        eq = {e["type"] for e in a["triage"]["equipment"]}
        ok = single["front"] in t["accept"]
        hits += ok
        h = sorted(eq & set(t["equipment_absent"]))
        m = sorted(set(t["equipment_present"]) - eq)
        halluc += bool(h)
        missed += len(m)
        cost += a["usage"]["cost_usd"]
        rows.append({
            "photo": f["filename"][6:16], "truth": t["stage"], "accept": t["accept"],
            "frame_stage": single["front"], "timeline_stage": scored["score"]["front"], "ok": ok,
            "candidates": a["candidates"], "latest_stage": a["triage"].get("latest_stage"),
            "context_conflict": a["triage"].get("context_conflict"),
            "likelihood": {k: v for k, v in a["triage"]["stage_likelihood"].items() if v >= 0.2},
            "view": a["triage"]["view"], "view_ok": a["triage"]["view"] == t["view"],
            "equipment": {e["type"]: f"{e['working']}/{e['total']}" for e in a["triage"]["equipment"]},
            "hallucinated": h, "missed": m,
            "yes": sorted(k for k, v in a["answers"].items() if v == "yes"),
            "unsure_share": round(sum(v == "unsure" for v in a["answers"].values()) / max(len(a["answers"]), 1), 2),
            "usage": a["usage"], "comment": a["comments"],
            "description": a["triage"]["description"],
        })

    report = {
        "provider": PROVIDER, "model": MODEL, "strategy": STRATEGY, "thinking": THINKING, "context": CONTEXT,
        "stage_accuracy": f"{hits}/{len(FILES)}",
        "exact_stage": sum(r["frame_stage"] == r["truth"] for r in rows),
        "view_accuracy": sum(r["view_ok"] for r in rows),
        "frames_with_hallucinated_equipment": halluc, "missed_equipment": missed,
        "cost_usd": round(cost, 5), "cost_per_photo": round(cost / len(FILES), 5),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
        "cached_tokens": sum(r["usage"]["cached_tokens"] for r in rows),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
        "recorded_calls": client.recorded, "replayed_calls": client.replayed,
        "timeline": {"overall_pct": tl["overall_pct"], "front": tl["front"],
                     "verdict": tl["schedule"]["verdict"] if tl["schedule"] else None,
                     "deviations": [f"{d['date']} {d['title']}" for d in tl["deviations"]]},
        "rows": rows,
    }
    name = (f"live_report_{PROVIDER}_{MODEL.replace(':', '-')}_{STRATEGY}"
            f"{'_thinking' if THINKING else ''}{'' if CONTEXT else '_nocontext'}.json")
    (FIXTURES / name).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))

    print(f"\n{PROVIDER}:{MODEL} / {STRATEGY}{'' if CONTEXT else ' / без контекста'}: этап {report['stage_accuracy']} (точно {report['exact_stage']}), "
          f"ракурс {report['view_accuracy']}/10, ${report['cost_usd']} "
          f"(${report['cost_per_photo']}/фото), записано {client.recorded}, из кассеты {client.replayed}")
    for r in rows:
        print(f"  {r['photo']} эталон {r['truth']} кадр {r['frame_stage']} хрон. {r['timeline_stage']} "
              f"{'OK ' if r['ok'] else 'ERR'} канд. {r['candidates']} {r['equipment']} "
              f"{'лишнее ' + str(r['hallucinated']) if r['hallucinated'] else ''}")

    assert hits >= 7, "этап определён верно меньше чем на 7 из 10 фото"
    assert halluc <= 2
    assert report["cost_per_photo"] < 0.01


def test_wrong_history_is_flagged_not_followed(checklist):
    """Каскад ошибки: фото с неверной датой → история врёт.

    Ранним фото (сваи, котлован) даём историю от поздних (готовый фасад).
    Ожидаем: модель не «дотягивает» кадр до фасада и отмечает противоречие.
    """
    client = make_client()
    analyzer = Analyzer(checklist, None, client, STRATEGY)
    blind = Analyzer(checklist, None, client, STRATEGY, use_context=False)
    ctx = ContextBuilder(checklist)
    try:
        for n in ("doric_2007_07_12_12_30_07.jpg", "doric_2007_09_03_12_30_42.jpg", "doric_2007_11_28_12_30_37.jpg"):
            raw = (PHOTOS / n).read_bytes()
            a, _ = blind.analyze(sha256(raw), lambda: data_url(prepare(raw)))
            ctx.add(a, date_from_name(n).date())
        assert ctx.front == 7
        out = []
        for n in ("doric_2005_12_23_12_30_10.jpg", "doric_2006_05_08_12_30_49.jpg"):
            raw = (PHOTOS / n).read_bytes()
            a, _ = analyzer.analyze(sha256(raw), lambda: data_url(prepare(raw)), ctx)
            alone, _ = blind.analyze(sha256(raw), lambda: data_url(prepare(raw)))  # тот же кадр без истории
            out.append((n, evaluate(checklist, a)["front"], evaluate(checklist, alone)["front"],
                        a["triage"]["context_conflict"]))
    except CassetteMiss:
        pytest.skip("нет ключа и нет записи в кассете")
    for n, front, alone, conflict in out:
        print(f"\n  {n[6:16]}: эталон {TRUTH[n]['stage']}, с ложной историей {front}, без истории {alone}, "
              f"противоречие: {conflict}")
    for n, front, alone, _ in out:
        # 2005-12-23 модель и без истории считает котлованом (спорный кадр) — сравниваем с этим
        assert front == alone or front in TRUTH[n]["accept"], f"{n}: ложная история утащила этап в {front}"
    assert all(conflict for *_, conflict in out), "модель не отметила противоречие"
