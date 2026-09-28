"""Модель Б на vision-LLM: двухшаговый GLM (фейк и кассета с реальными ответами) и локальная VLM.

Кассета — записанные ответы GLM-4.6V на 10 снимков стройки Edinburgh (test_photos
Никиты). Промпты перенесены побайтно, поэтому ответы воспроизводятся без ключа и
сети, а сквозной путь «кадр → GLM → scoring → хронология» проверяется на реальных
ответах модели.
"""
import base64
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pytest

from core.contracts import Answer, FrameInfo, Provider, StageObservation
from core.stage import get_classifier, prompts, scoring, sequence
from core.stage.checklist_vlm import (GlmChecklistClassifier, LocalVlmClassifier,
                                      normalize_triage, parse_answer, pick_candidates)
from core.vlm_client import CassetteClient, Reply, Usage, VLMError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PHOTOS = FIXTURES / "stage_photos"
FRAME = FrameInfo(frame_id=1, camera_id=1, site_id=1, captured_at=dt.datetime(2026, 6, 1, 9, tzinfo=dt.timezone.utc),
                  width=640, height=480)
IMG = np.full((480, 640, 3), 120, np.uint8)


def triage(likelihood, latest=None, equipment=()):
    return {"quality": "good", "view": "side", "description": "тест",
            "equipment": [{"type": t, "total": n, "working": w, "evidence": ""} for t, n, w in equipment],
            "workers_count": 3, "floors_built": None, "floors_glazed": None, "facade_clad_pct": None,
            "pit_area_pct": 40, "latest_stage": latest,
            "stage_likelihood": {str(k): v for k, v in likelihood.items()}}


class FakeGLM:
    """Разведка возвращает заданный triage, чек-лист — «yes»/«no» на заданные признаки, остальное not_visible."""
    provider, model, thinking = "zai", "glm-4.6v", False

    def __init__(self, tri, yes=(), no=(), fail_on=None):
        self.tri, self.yes, self.no, self.fail_on = tri, set(yes), set(no), fail_on
        self.requests = []

    def ready(self):
        return True, ""

    def ask_json(self, system, image_url, prompt, max_tokens, schema=None):
        text = prompt if isinstance(prompt, str) else "\n\n".join(p for p in prompt if p)
        self.requests.append(text)
        usage = Usage(prompt_tokens=3000, cached_tokens=1500, completion_tokens=300, cost_usd=0.001, latency_ms=900)
        step = "triage" if text.startswith("Шаг 1") else "checklist"
        if self.fail_on == step:
            raise KeyError("choices")                     # не VLMError — как в баге A7
        if step == "triage":
            return Reply(self.tri, json.dumps(self.tri), usage, 900.0, [usage])
        keys = [line[2:].split(":")[0] for line in text.splitlines() if line.startswith("- ")]
        answers = {k: "yes" if k in self.yes else "no" if k in self.no else "not_visible" for k in keys}
        return Reply({"answers": answers, "comment": "тест"}, "{}", usage, 900.0, [usage])


def test_two_step_asks_candidates_and_normalises_answers():
    fake = FakeGLM(triage({3: 0.9, 4: 0.2}, latest=3, equipment=[("excavator", 1, 1), ("dump_truck", 0, 0)]),
                   yes={"pit", "earthwork"}, no={"slab"})
    r = GlmChecklistClassifier(client=fake).assess(IMG, FRAME)
    assert r.raw["candidates"] == [3, 4] and len(fake.requests) == 2
    assert fake.requests[1] == prompts.checklist_step([3, 4])
    assert r.answers["pit"] is Answer.YES and r.answers["slab"] is Answer.NO
    assert r.answers["formwork"] is Answer.UNSURE                    # not_visible не голосует против
    assert set(r.answers) == set(prompts.sign_keys_for([3, 4]))
    assert r.equipment_hint == {"excavator": 1}                      # total=0 — не фантомная машина
    assert r.provider is Provider.EXTERNAL and r.cost_usd == pytest.approx(0.002)
    assert r.stage_likelihood[3] == 0.9 and scoring.evaluate(r.answers).front == 3


def test_likelihood_alone_never_opens_a_stage():
    """Раньше этап вне кандидатов открывался одной вероятностью разведки ≥ 0.5."""
    fake = FakeGLM(triage({3: 0.9, 4: 0.4, 5: 0.35, 7: 0.6}, latest=3), yes={"pit", "soil_pile"})
    r = GlmChecklistClassifier(client=fake).assess(IMG, FRAME, context={"front": 5})
    assert r.raw["candidates"] == [3, 4, 5] and r.stage_likelihood[7] == 0.6
    assert scoring.evaluate(r.answers).front == 3


def test_context_front_is_always_a_candidate_and_text_goes_after_static_part():
    fake = FakeGLM(triage({3: 0.9}, latest=3))
    GlmChecklistClassifier(client=fake).assess(IMG, FRAME, context={"front": 6, "text": "Контекст: этап 6."})
    assert fake.requests[1].startswith(prompts.checklist_step([3, 4, 6]))    # этап по истории сверяется всегда
    assert fake.requests[0].startswith("Шаг 1") and fake.requests[0].endswith("Контекст: этап 6.")


def test_explicit_keys_are_asked_as_is():
    fake = FakeGLM(triage({5: 0.9}, latest=5), yes={"crane"})
    r = GlmChecklistClassifier(client=fake).assess(IMG, FRAME, keys=["crane", "pit"])
    assert set(r.answers) == {"crane", "pit"} and r.answers["crane"] is Answer.YES


def test_any_failure_becomes_vlm_error_with_spent_usage():
    for step in ("triage", "checklist"):
        fake = FakeGLM(triage({3: 0.9}, latest=3), fail_on=step)
        with pytest.raises(VLMError) as e:
            GlmChecklistClassifier(client=fake).assess(IMG, FRAME)
        if step == "checklist":
            assert e.value.usage.cost_usd == pytest.approx(0.001)       # разведка оплачена — не теряем


def test_malformed_triage_is_normalised_not_crashing():
    t = normalize_triage({"stage_likelihood": [0.1, 0.2], "equipment": "экскаватор", "latest_stage": "пять",
                          "quality": "отличное", "facade_clad_pct": 250,
                          "context_conflict": "null"})
    assert set(t["stage_likelihood"]) == set(range(1, 9)) and not any(t["stage_likelihood"].values())
    assert t["equipment"] == [] and t["latest_stage"] is None and t["quality"] == "good"
    assert t["facade_clad_pct"] == 100 and t["context_conflict"] is None
    t2 = normalize_triage({"equipment": [{"type": "UFO", "total": None, "working": 2}, "мусор",
                                         {"type": "crane_manipulator", "total": 1, "working": 5}]})
    assert t2["equipment"][0]["type"] == "other" and t2["equipment"][0]["total"] == 2
    assert t2["equipment"][1]["working"] == 1
    assert normalize_triage("не словарь")["stage_likelihood"][1] == 0.0


def test_pick_candidates_bounds():
    assert pick_candidates({s: 0.0 for s in range(1, 9)}) == [1, 2]       # минимум два
    lk = {s: 0.9 for s in range(1, 9)}
    assert len(pick_candidates(lk, latest=4, prev_front=7)) == 3          # максимум три
    top = pick_candidates({**{s: 0.0 for s in range(1, 9)}, 8: 0.9}, latest=8)
    assert len(top) == 2 and top[-1] == 8                                  # «8 + 1» не существует


def _nikita_url(path: Path) -> str:
    """Кадр → data-URL ровно как в api-solution/core/images.py (PIL, 1280, JPEG q85) — ключ кассеты."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def cassette_results():
    tape = CassetteClient(FIXTURES / "glm_cassette.json", record=False)
    clf = GlmChecklistClassifier(client=tape)
    out = {}
    for p in sorted(PHOTOS.glob("*.jpg")):
        a = clf.analyzer.analyze(_nikita_url(p))
        out[p.name] = clf.result_from_analysis(a)
    return out


def test_cassette_real_glm_answers_match_ground_truth(cassette_results):
    """Реальные ответы GLM-4.6V → наш scoring: этап кадра в допустимом диапазоне разметки."""
    truth = json.loads((PHOTOS / "ground_truth.json").read_text())
    hits = 0
    for name, r in cassette_results.items():
        front = scoring.evaluate(r.answers).front
        hits += front in truth[name]["accept"]
        assert not set(r.equipment_hint) & set(truth[name]["equipment_absent"]), (name, r.equipment_hint)
        assert set(truth[name]["equipment_present"]) <= set(r.equipment_hint), (name, r.equipment_hint)
    assert hits >= 7, hits          # у Никиты на этих кадрах 9/10 «в допуске»; без контекста и с новым скорингом


def test_cassette_timeline_is_monotone_and_ends_at_facade(cassette_results):
    """Хронология по реальным ответам на кадры раз в 2–4 месяца: без выбросов, монотонно,
    и точнее покадровой оценки — кадры без открытого этапа достраиваются непрерывностью."""
    truth = json.loads((PHOTOS / "ground_truth.json").read_text())
    names = sorted(cassette_results)
    observations = [StageObservation(frame_id=name, camera_id=1, result=cassette_results[name],
                                     captured_at=dt.datetime.strptime(name[6:25], "%Y_%m_%d_%H_%M_%S")
                                     .replace(tzinfo=dt.timezone.utc)) for name in names]
    tl = sequence.infer(observations)
    fronts = [f for _, f in tl.daily_front]
    assert fronts == sorted(fronts) and tl.rejected_outliers == []
    assert sum(f in truth[n]["accept"] for n, f in zip(names, fronts)) >= 8
    assert tl.current_stage == 7 and tl.states[5].status.value == "done"
    assert tl.states[7].status.value == "active" and tl.states[8].status.value == "not_started"


def test_parse_answer_does_not_turn_not_visible_into_no():
    """Баг D5 Дениса: «Не видно» (ровно то, что просит промпт) давало NO."""
    for raw in ("Не видно", "не вижу", "Кран не виден", "не видно, перекрыто", "Нет данных", "затрудняюсь",
                "not_visible", "", None, "Здание не строится"):
        assert parse_answer(raw) is Answer.UNSURE, raw
    assert parse_answer("Да.") is Answer.YES and parse_answer("да, виден") is Answer.YES
    assert parse_answer("Нет") is Answer.NO and parse_answer("no") is Answer.NO


# Перенесено интегратором из tests/test_parse_answer.py Дениса (тестировал удалённый
# app/pipeline/model_b.py). Сознательно изменились три случая: «Нет, конструкций не
# видно» теперь UNSURE (маркер «не видно» проверяется первым — баг D5 выше), «На
# изображении да, присутствует» — UNSURE (развёрнутый ответ не угадываем), а пары
# «Стройка»/«завершено» нет — такого вопроса в новом чек-листе нет.
@pytest.mark.parametrize("raw,expected", [
    ("Да", Answer.YES), ("да.", Answer.YES), ("ДА", Answer.YES),
    ("Нет", Answer.NO), ("нет,", Answer.NO), ("No", Answer.NO),
    ("Не уверен", Answer.UNSURE), ("не уверен.", Answer.UNSURE), ("Не уверена", Answer.UNSURE),
    ("Я не могу определить", Answer.UNSURE), ("Сложно сказать, кадр засвечен", Answer.UNSURE),
    ("**Да**", Answer.YES), ("Да, виден котлован", Answer.YES),
    ("Нет, конструкций не видно", Answer.UNSURE), ("На изображении да, присутствует", Answer.UNSURE),
    ("", Answer.UNSURE), ("Изображение показывает строительную площадку", Answer.UNSURE),
])
def test_parse_answer_legacy_cases(raw, expected):
    assert parse_answer(raw) is expected


class FakeLocal:
    provider, model, thinking = "local", "qwen2.5vl:3b", False

    def __init__(self, yes=(), reject_schema=False):
        self.yes = set(yes)
        self.reject_schema = reject_schema
        self.calls = []

    def ready(self):
        return True, ""

    def ask_json(self, system, image_url, prompt, max_tokens, schema=None):
        keys = [line.split(":")[0] for line in prompt.splitlines()[2:] if ":" in line]
        self.calls.append({"keys": keys, "schema": schema, "system": system})
        if schema is not None and self.reject_schema:
            raise VLMError("HTTP 400: response_format not supported")
        data = {k: "да" if k in self.yes else "не видно" for k in keys}
        return Reply(data, json.dumps(data, ensure_ascii=False), Usage(), 0.0, [])


def test_local_vlm_batches_by_stage_with_schema():
    fake = FakeLocal(yes={"pit", "earthwork"})
    clf = LocalVlmClassifier(client=fake, group_size=5)
    r = clf.assess(IMG, FRAME, context={"front": 3})
    assert all(1 <= len(c["keys"]) <= 5 for c in fake.calls)
    asked = [k for c in fake.calls for k in c["keys"]]
    assert set(asked) == set(prompts.sign_keys_for([2, 3, 4, 5])) and len(asked) == len(set(asked))
    first = fake.calls[0]
    assert first["schema"]["required"] == first["keys"]
    assert first["schema"]["properties"][first["keys"][0]]["enum"] == ["да", "нет", "не видно"]
    assert "не видно" in first["system"]
    assert r.answers["pit"] is Answer.YES and r.answers["slab"] is Answer.UNSURE
    assert r.provider is Provider.LOCAL and scoring.evaluate(r.answers).front == 3


def test_local_vlm_falls_back_without_schema_and_fails_loudly():
    fake = FakeLocal(yes={"crane"}, reject_schema=True)
    r = LocalVlmClassifier(client=fake).assess(IMG, FRAME, keys=["crane", "pit"])
    assert r.answers == {"crane": Answer.YES, "pit": Answer.UNSURE}
    assert [c["schema"] is None for c in fake.calls] == [False, True, False, True]   # по группе: схема → без неё

    class Dead(FakeLocal):
        def ask_json(self, *a, **k):
            raise VLMError("сервер не отвечает")

    with pytest.raises(VLMError, match="не ответила"):              # не молчаливое «всё не видно» (D6)
        LocalVlmClassifier(client=Dead()).assess(IMG, FRAME, keys=["crane"])


def test_get_classifier_names():
    fake = FakeGLM(triage({3: 0.9}, latest=3))
    assert isinstance(get_classifier("glm", client=fake), GlmChecklistClassifier)
    assert isinstance(get_classifier("local_vlm", client=FakeLocal()), LocalVlmClassifier)
    assert get_classifier("siglip").name == "siglip"                # без загрузки весов
    with pytest.raises(ValueError):
        get_classifier("yolo")


def test_classifiers_follow_the_protocol():
    from core.contracts import StageClassifier

    for clf in (GlmChecklistClassifier(client=FakeGLM(triage({3: 0.9}))), LocalVlmClassifier(client=FakeLocal()),
                get_classifier("siglip")):
        assert isinstance(clf, StageClassifier)
