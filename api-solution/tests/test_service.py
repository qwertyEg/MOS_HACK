"""Весь сервис без GLM на реальных фото из test_photos.

Модель подменена «эталонной»: отвечает по ручной разметке ground_truth.json.
Проверяется всё остальное — даты, сжатие, хранилище, кэш, разбор ошибок,
оценка кадров, хронология, план, прогноз, отклонения.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from PIL import Image

from conftest import triage
from core import config, service
from core.analyzer import Analyzer
from core.vlm import Reply, Usage, VLMError as GLMError
from core.images import date_from_name, prepare
from core.plan import typical_plan
from core.storage import Storage

PHOTOS = config.ROOT / "test_photos"
TRUTH = json.loads((Path(__file__).parent / "fixtures" / "ground_truth.json").read_text())
FILES = sorted(k for k in TRUTH if not k.startswith("_"))

pytestmark = pytest.mark.skipif(not PHOTOS.exists(), reason="нет папки test_photos")


class GroundTruthClient:
    """Отвечает как идеальная модель: этап и техника — из разметки."""

    model = "glm-4.6v"
    thinking = False

    def __init__(self, checklist, by_url, fail=()):
        self.checklist, self.by_url, self.fail = checklist, by_url, set(fail)
        self.calls = 0
        self.prompts = []  # (имя файла, текст запроса) — чтобы проверить контекст

    def ask_json(self, system, image_url, prompt, max_tokens):
        text = prompt if isinstance(prompt, str) else "\n\n".join(p for p in prompt if p)
        self.calls += 1
        name = self.by_url[image_url]
        self.prompts.append((name, text))
        if name in self.fail:
            raise GLMError("HTTP 500: тест")
        t = TRUTH[name]
        usage = Usage(2000, 800, 200, 0.001)
        if text.startswith("Шаг 1"):
            data = triage({t["stage"]: 0.9}, [(e, 1, 1) for e in t["equipment_present"]])
            data["view"] = t["view"]
            return Reply(data, json.dumps(data), usage, [usage])
        stage = self.checklist.stage_by_id[t["stage"]]
        keys = [line[2:].split(":")[0] for line in text.splitlines() if line.startswith("- ")]
        answers = {k: "yes" if k in stage["must_have"] else "unsure" for k in keys}
        return Reply({"answers": answers}, "{}", usage, [usage])


@pytest.fixture
def env(tmp_path, monkeypatch, checklist):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    storage = Storage(tmp_path / "db.sqlite")
    oid = storage.create_object("Doric", "Офисно-деловой центр", 7)
    storage.save_plan(oid, typical_plan(date(2005, 10, 1)), "manual")
    return storage, oid


def site_obj(storage):
    return storage.list_objects()[0]


def ingest_all(storage, oid):
    return {service.ingest(storage, oid, n, (PHOTOS / n).read_bytes()): n for n in FILES}


def client_for(storage, oid, checklist, fail=()):
    from core.images import data_url
    by_url = {data_url(open(f["image_path"], "rb").read()): f["filename"] for f in storage.list_frames(oid)}
    return GroundTruthClient(checklist, by_url, fail)


# --- приём фото ---

def test_dates_from_real_filenames():
    assert date_from_name(FILES[0]) == datetime(2005, 12, 23, 12, 30, 10)
    assert all(date_from_name(n) for n in FILES)


def test_prepare_real_photos_never_upscales():
    for n in FILES:
        src = Image.open(PHOTOS / n)
        out = Image.open(__import__("io").BytesIO(prepare((PHOTOS / n).read_bytes())))
        assert max(out.size) <= config.IMAGE_MAX_SIDE and out.size[0] <= src.size[0]


def test_ingest_is_idempotent_and_orders_by_date(env):
    storage, oid = env
    ingest_all(storage, oid)
    ingest_all(storage, oid)
    frames = storage.list_frames(oid)
    assert [f["filename"] for f in frames] == FILES  # имена совпадают с хронологией
    assert all(Path(f["image_path"]).exists() and f["date_source"] == "имя файла" for f in frames)


def test_ingest_without_date_is_rejected(env):
    storage, oid = env
    with pytest.raises(ValueError, match="дата"):
        service.ingest(storage, oid, "photo.jpg", (PHOTOS / FILES[0]).read_bytes())


# --- разбор и отчёт ---

def test_full_series_report(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)
    client = client_for(storage, oid, checklist)
    analyzer = Analyzer(checklist, storage, client)

    spent, errors = service.analyze_frames(storage, analyzer, site_obj(storage))
    assert errors == [] and spent > 0
    assert service.pending(storage, analyzer, site_obj(storage)) == []

    obj = storage.list_objects()[0]
    frames, plan, tl = service.report(checklist, storage, analyzer, obj)
    fronts = [f["score"]["front"] for f in tl["frames"]]
    assert fronts == [TRUTH[n]["stage"] for n in FILES]

    progress = [p for _, p in tl["series"]]
    assert progress == sorted(progress) and 60 < progress[-1] < 100
    assert tl["front"] == 7
    assert tl["schedule"]["verdict"] in ("отставание", "в срок", "опережение")
    assert tl["forecast"]["finish"] is not None
    # Этапы до фасада закрыты, благоустройство не начато.
    assert all(tl["stage_progress"][s] == 1.0 for s in (1, 2, 3, 4))
    assert tl["stage_progress"][8] == 0


def test_second_run_is_free(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)
    client = client_for(storage, oid, checklist)
    analyzer = Analyzer(checklist, storage, client)
    service.analyze_frames(storage, analyzer, site_obj(storage))
    calls = client.calls
    spent, _ = service.analyze_frames(storage, analyzer, site_obj(storage))
    assert spent == 0 and client.calls == calls


def test_one_failed_frame_does_not_stop_the_rest(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)
    client = client_for(storage, oid, checklist, fail={FILES[3]})
    analyzer = Analyzer(checklist, storage, client)
    _, errors = service.analyze_frames(storage, analyzer, site_obj(storage))
    assert len(errors) == 1 and FILES[3] in errors[0]
    # сбойный кадр и все после него: когда он разберётся, их история изменится
    assert [f["filename"] for f in service.pending(storage, analyzer, site_obj(storage))] == FILES[3:]
    # Отчёт строится по разобранным кадрам, неразобранный просто пропускается.
    obj = storage.list_objects()[0]
    _, _, tl = service.report(checklist, storage, analyzer, obj)
    assert len(tl["frames"]) == len(FILES) - 1


def test_changing_model_keeps_old_results_visible(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)
    client = client_for(storage, oid, checklist)
    service.analyze_frames(storage, Analyzer(checklist, storage, client), site_obj(storage))
    client.model = "glm-4.6v-flash"
    other = Analyzer(checklist, storage, client)
    assert len(service.pending(storage, other, site_obj(storage))) == len(FILES)
    frames = service.frames_with_analysis(storage, other, site_obj(storage))
    assert all(f["analysis"] for f in frames)


# --- автоплан по датам фото ---

def test_auto_plan_spans_photo_dates(tmp_path, monkeypatch, checklist):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    storage = Storage(tmp_path / "db.sqlite")
    oid = storage.create_object("Doric", "Офисно-деловой центр", 7)
    assert storage.get_plan_source(oid) == "auto" and storage.get_plan(oid) == {}

    service.ingest(storage, oid, FILES[0], (PHOTOS / FILES[0]).read_bytes())
    assert storage.get_plan(oid) == {}  # один день — периода нет

    ingest_all(storage, oid)
    plan = {k: (date.fromisoformat(s), date.fromisoformat(e)) for k, (s, e) in storage.get_plan(oid).items()}
    first, last = date(2005, 12, 23), date(2007, 11, 28)
    assert plan[1][0] == first and max(e for _, e in plan.values()) == last
    assert all(first <= s <= e <= last for s, e in plan.values())
    assert plan[5][1] - plan[5][0] > plan[6][1] - plan[6][0]  # каркас дольше кровли — типовые пропорции

    last_frame = storage.list_frames(oid)[-1]
    service.delete_frame(storage, oid, last_frame["id"])
    assert max(date.fromisoformat(e) for _, e in storage.get_plan(oid).values()) == date(2007, 9, 3)


def test_manual_plan_is_not_overwritten(env):
    storage, oid = env  # в env план сохранён как manual
    before = storage.get_plan(oid)
    ingest_all(storage, oid)
    assert storage.get_plan(oid) == before and storage.get_plan_source(oid) == "manual"


# --- контекст стройки между разборами ---

def test_each_frame_sees_history_of_earlier_frames(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)
    client = client_for(storage, oid, checklist)
    service.analyze_frames(storage, Analyzer(checklist, storage, client), site_obj(storage))
    triage = [(n, t) for n, t in client.prompts if t.startswith("Шаг 1")]
    assert [n for n, _ in triage] == FILES  # строго по датам
    assert "Контекст этой стройки" not in triage[0][1]
    assert "по 5 предыдущим снимкам" in triage[5][1]
    assert "достигнутый этап: 5" in triage[7][1]  # после трёх кадров каркаса


def test_backdated_photo_reopens_later_frames(env, checklist):
    storage, oid = env
    for n in FILES[1:]:
        service.ingest(storage, oid, n, (PHOTOS / n).read_bytes())
    client = client_for(storage, oid, checklist)
    analyzer = Analyzer(checklist, storage, client)
    service.analyze_frames(storage, analyzer, site_obj(storage))
    assert service.pending(storage, analyzer, site_obj(storage)) == []

    service.ingest(storage, oid, FILES[0], (PHOTOS / FILES[0]).read_bytes())  # фото задним числом
    client.by_url.update(client_for(storage, oid, checklist).by_url)
    # у всех более поздних кадров изменилась история → их разборы устарели
    assert [f["filename"] for f in service.pending(storage, analyzer, site_obj(storage))] == FILES
    service.analyze_frames(storage, analyzer, site_obj(storage))
    assert service.pending(storage, analyzer, site_obj(storage)) == []


def test_history_conflict_becomes_deviation(env, checklist):
    storage, oid = env
    ingest_all(storage, oid)

    class Conflicting(GroundTruthClient):
        def ask_json(self, system, image_url, prompt, max_tokens):
            reply = super().ask_json(system, image_url, prompt, max_tokens)
            if self.by_url[image_url] == FILES[-1] and "stage_likelihood" in reply.data:
                reply.data["context_conflict"] = "на снимке снова открытый котлован"
            return reply

    base = client_for(storage, oid, checklist)
    analyzer = Analyzer(checklist, storage, Conflicting(checklist, base.by_url))
    service.analyze_frames(storage, analyzer, site_obj(storage))
    _, _, tl = service.report(checklist, storage, analyzer, site_obj(storage))
    assert any(d["rule"] == "history_conflict" and "котлован" in d["detail"] for d in tl["deviations"])
