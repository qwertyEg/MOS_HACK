"""Реальная модель GLM на test_photos против ручной разметки.

    pytest -m live -s                         # glm-4.6v, two_step
    GLM_EVAL_MODEL=glm-4.6v-flash pytest -m live -s
    GLM_EVAL_STRATEGY=per_stage pytest -m live -s

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
from core.glm import GLMClient
from core.images import data_url, prepare
from core.plan import typical_plan
from core.scoring import evaluate
from core.storage import Storage

pytestmark = pytest.mark.live

FIXTURES = Path(__file__).parent / "fixtures"
PHOTOS = config.ROOT / "test_photos"
TRUTH = json.loads((FIXTURES / "ground_truth.json").read_text())
FILES = sorted(k for k in TRUTH if not k.startswith("_"))
MODEL = os.getenv("GLM_EVAL_MODEL", "glm-4.6v")
STRATEGY = os.getenv("GLM_EVAL_STRATEGY", "two_step")
THINKING = os.getenv("GLM_EVAL_THINKING") == "1"


def make_client(model=MODEL):
    real = GLMClient(model=model, thinking=THINKING) if config.API_KEY else None
    return CassetteClient(FIXTURES / "glm_cassette.json", real, model=model, thinking=THINKING)


def test_connectivity_on_free_model():
    """Ключ, адрес и формат ответа — на бесплатной модели и самом маленьком кадре."""
    client = make_client("glm-4.6v-flash")
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
    storage.save_plan(oid, typical_plan(datetime(2005, 10, 1).date()))
    for n in FILES:
        service.ingest(storage, oid, n, (PHOTOS / n).read_bytes())

    analyzer = Analyzer(checklist, storage, client, STRATEGY)
    try:
        _, errors = service.analyze_frames(analyzer, storage.list_frames(oid))
    except CassetteMiss:
        pytest.skip("нет ключа ZAI_API_KEY и нет записи в кассете")
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
            "candidates": a["candidates"],
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
        "model": MODEL, "strategy": STRATEGY, "thinking": THINKING,
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
    name = f"live_report_{MODEL}_{STRATEGY}{'_thinking' if THINKING else ''}.json"
    (FIXTURES / name).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))

    print(f"\n{MODEL} / {STRATEGY}: этап {report['stage_accuracy']} (точно {report['exact_stage']}), "
          f"ракурс {report['view_accuracy']}/10, ${report['cost_usd']} "
          f"(${report['cost_per_photo']}/фото), записано {client.recorded}, из кассеты {client.replayed}")
    for r in rows:
        print(f"  {r['photo']} эталон {r['truth']} кадр {r['frame_stage']} хрон. {r['timeline_stage']} "
              f"{'OK ' if r['ok'] else 'ERR'} канд. {r['candidates']} {r['equipment']} "
              f"{'лишнее ' + str(r['hallucinated']) if r['hallucinated'] else ''}")

    assert hits >= 7, "этап определён верно меньше чем на 7 из 10 фото"
    assert halluc <= 2
    assert report["cost_per_photo"] < 0.01
