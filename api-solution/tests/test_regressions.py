"""Регрессии после добавления провайдеров, контекста стройки и автоплана.

Каждый тест — сценарий, который мог сломаться или стать хуже; подробно —
docs/testing.md, раздел «Проверка рисков».
"""

import io
import sqlite3
import time
from datetime import date, datetime, timedelta

import pytest
from PIL import Image

from conftest import FakeClient, triage
from core import config, service, site, vlm
from core.analyzer import Analyzer
from core.glm import GLMClient
from core.scoring import evaluate
from core.site import ContextBuilder
from core.storage import Storage


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    return Storage(tmp_path / "db.sqlite")


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (seed % 255, (seed * 7) % 255, 90)).save(buf, "JPEG")
    return buf.getvalue()


# --- транспорт z.ai не изменился ---

def test_zai_keeps_system_proxy(monkeypatch):
    """Рефакторинг в общий клиент: обход прокси — только у локальной модели.
    z.ai из корпоративной сети должен идти через системный прокси, как раньше."""
    sent = {}

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"a": 1}'}}], "usage": {}}

    def post(url, **kw):
        sent.update(kw)
        return R()

    monkeypatch.setattr(vlm.requests, "post", post)
    GLMClient(api_key="k").ask_json("S", "data:", "t", 10)
    assert "proxies" not in sent


# --- база старого формата ---

def test_old_database_migrates_without_losing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as c:  # схема до появления plan_source
        c.executescript("""
            CREATE TABLE objects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, object_type TEXT NOT NULL,
                                  floors_total INTEGER, created_at TEXT NOT NULL);
            CREATE TABLE plan (object_id INTEGER NOT NULL, stage_id INTEGER NOT NULL, start TEXT, end TEXT,
                               PRIMARY KEY (object_id, stage_id));
            INSERT INTO objects VALUES (1, 'Старый', 'Жильё', 9, '2026-09-01T00:00:00');
            INSERT INTO plan VALUES (1, 5, '2025-01-01', '2025-06-01');
        """)
    s = Storage(db)
    assert s.list_objects()[0]["name"] == "Старый"
    assert s.get_plan(1) == {5: ("2025-01-01", "2025-06-01")}
    assert s.get_plan_source(1) == "auto"
    Storage(db)  # повторное открытие не падает на уже добавленной колонке


# --- крайние случаи автоплана ---

def test_deleting_all_photos_clears_plan_and_report_still_works(storage, checklist):
    oid = storage.create_object("A", "Жильё")
    for i in range(2):
        service.ingest(storage, oid, f"f{i}.jpg", jpeg(i), datetime(2025, 1, 1) + timedelta(days=10 * i))
    assert storage.get_plan(oid)
    for f in storage.list_frames(oid):
        service.delete_frame(storage, oid, f["id"])
    assert storage.get_plan(oid) == {}
    obj = storage.list_objects()[0]
    frames, plan, tl = service.report(checklist, storage, Analyzer(checklist, storage, FakeClient(triage({}))), obj)
    assert frames == [] and tl["schedule"] is None and tl["overall_pct"] == 0


def test_two_photos_one_day_apart_give_valid_plan(storage, checklist):
    oid = storage.create_object("A", "Жильё")
    for i in range(2):
        service.ingest(storage, oid, f"f{i}.jpg", jpeg(i), datetime(2025, 1, 1) + timedelta(days=i))
    plan = {k: (date.fromisoformat(s), date.fromisoformat(e)) for k, (s, e) in storage.get_plan(oid).items()}
    assert len(plan) == 8 and all(s <= e for s, e in plan.values())
    assert min(s for s, _ in plan.values()) == date(2025, 1, 1) and max(e for _, e in plan.values()) == date(2025, 1, 2)
    an = Analyzer(checklist, storage, FakeClient(triage({3: 0.9}, latest_stage=3), {"pit"}))
    service.analyze_frames(storage, an, storage.list_objects()[0])
    _, _, tl = service.report(checklist, storage, an, storage.list_objects()[0])
    assert tl["schedule"]["verdict"] in ("отставание", "в срок", "опережение")


# --- разборы, сохранённые до изменений ---

def test_analysis_from_previous_version_still_renders(storage, checklist):
    """В SQLite могут лежать разборы без provider / context / latest_stage / context_conflict."""
    oid = storage.create_object("A", "Жильё")
    sha = service.ingest(storage, oid, "f.jpg", jpeg(1), datetime(2025, 1, 1))
    an = Analyzer(checklist, storage, FakeClient(triage({5: 0.9}), {"above_grade", "formwork_floor"}))
    old, _ = an.analyze("x", lambda: "data:")
    for k in ("provider", "context"):
        old.pop(k)
    for k in ("latest_stage", "context_conflict"):
        old["triage"].pop(k)
    storage.save_analysis("legacy", sha, "glm-4.6v", False, "two_step", old)

    obj = storage.list_objects()[0]
    frames, _, tl = service.report(checklist, storage, an, obj)
    assert frames[0]["analysis"] and not frames[0]["current"]  # показан, помечен устаревшим
    assert evaluate(checklist, old)["front"] == 5
    kb = site.knowledge(checklist, tl)
    assert kb["photos"] == 1 and site.to_csv(site.frames_table(checklist, tl))
    ctx = ContextBuilder(checklist)
    ctx.add(old, date(2025, 1, 1))
    assert "достигнутый этап: 5" in ctx.text()


# --- производительность: страница теперь проходит по всем кадрам ---

def test_report_on_300_frames_is_fast(storage, checklist):
    oid = storage.create_object("A", "Жильё")
    for i in range(300):
        service.ingest(storage, oid, f"f{i}.jpg", jpeg(i), datetime(2025, 1, 1) + timedelta(hours=8 * i))
    obj = storage.list_objects()[0]
    an = Analyzer(checklist, storage, FakeClient(triage({5: 0.9}, [("tower_crane", 1, 1)], latest_stage=5),
                                                 {"above_grade", "crane"}))
    service.analyze_frames(storage, an, obj)
    t0 = time.monotonic()
    service.report(checklist, storage, an, obj)
    service.pending(storage, an, obj)
    elapsed = time.monotonic() - t0
    # Столько делает одна перерисовка страницы (отчёт + счётчик ожидающих).
    assert elapsed < 5, f"{elapsed:.1f} с на 300 кадров"


# --- выключенный контекст: для A/B и как настройка в UI ---

def test_context_can_be_switched_off(checklist):
    ctx = ContextBuilder(checklist)
    a, _ = Analyzer(checklist, None, FakeClient(triage({5: 0.9}), {"above_grade"})).analyze("s", lambda: "d:")
    ctx.add(a, date(2025, 1, 1))
    client = FakeClient(triage({5: 0.9}))
    off = Analyzer(checklist, None, client, use_context=False)
    result, _ = off.analyze("s2", lambda: "d:", ctx)
    assert result["context"] == "" and not any("Контекст" in r for r in client.requests)
    assert off.cache_key("s2", ctx) == off.cache_key("s2")


def test_wrong_history_does_not_rewrite_the_frame_itself(checklist):
    """Каскад ошибки: история утверждает «фасад», кадр — котлован. Этап из истории
    добавляется в кандидаты, но этап самого кадра определяют его ответы."""
    ctx = ContextBuilder(checklist)
    facade, _ = Analyzer(checklist, None, FakeClient(triage({7: 0.9}, latest_stage=7),
                                                     {"cladding", "glazing", "above_grade"})).analyze("f", lambda: "d:")
    ctx.add(facade, date(2025, 6, 1))
    assert ctx.front == 7
    pit, _ = Analyzer(checklist, None, FakeClient(triage({3: 0.9}, latest_stage=3), {"pit", "earthwork", "soil_pile"},
                                                  {"cladding", "glazing"})).analyze("p", lambda: "d:", ctx)
    assert 7 in pit["candidates"]
    assert evaluate(checklist, pit)["front"] == 3


def test_storage_survives_deleted_data_dir(tmp_path, monkeypatch):
    """Streamlit кэширует Storage на весь процесс; папку data/ удалили на ходу —
    раньше: OperationalError «unable to open database file» при каждой перерисовке."""
    import shutil
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "data" / "img")
    s = Storage(tmp_path / "data" / "db.sqlite")
    s.create_object("A", "Жильё")
    shutil.rmtree(tmp_path / "data")
    assert s.list_objects() == []  # база пересоздана пустой, интерфейс не падает
    oid = s.create_object("B", "Жильё")
    service.ingest(s, oid, "f.jpg", jpeg(1), datetime(2025, 1, 1))
    assert len(s.list_frames(oid)) == 1
