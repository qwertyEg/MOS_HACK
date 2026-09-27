"""Интерфейс Streamlit целиком (AppTest) — на временной базе, рабочую data/ не трогает."""

from datetime import datetime, timedelta

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakeClient, triage
from core import config, service
from core.analyzer import Analyzer
from core.storage import Storage
from test_regressions import jpeg

APP = str(config.ROOT / "app.py")


@pytest.fixture
def seeded(tmp_path, monkeypatch, checklist):
    real_data = config.ROOT / "data"
    existed = real_data.exists()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "img")
    st.cache_resource.clear()  # Storage кэшируется на процесс — иначе возьмётся от прошлого теста
    storage = Storage()
    oid = storage.create_object("ЖК Тест", "Жильё", 17)
    scenes = [({3: 0.9}, {"pit", "earthwork", "soil_pile"}, [("excavator", 2, 1)], 3),
              ({4: 0.9}, {"slab", "rebar", "formwork", "below_grade"}, [("concrete_pump", 1, 1)], 4),
              ({5: 0.9}, {"above_grade", "crane", "formwork_floor", "unfinished_top"}, [("tower_crane", 1, 1)], 5)]
    for i, (lk, yes, eq, latest) in enumerate(scenes):
        service.ingest(storage, oid, f"cam_{i}.jpg", jpeg(i), datetime(2025, 5, 1) + timedelta(days=30 * i))
        frame = storage.list_frames(oid)[i]
        an = Analyzer(checklist, storage, FakeClient(triage(lk, eq, latest_stage=latest, view="top"), yes))
        ctx = service._walk(storage, an, storage.list_objects()[0])  # история до этого кадра
        from core.site import ContextBuilder
        builder = ContextBuilder(checklist)
        for f in ctx[0][:i]:
            builder.add(f["analysis"], datetime.fromisoformat(f["taken_at"]).date())
        an.analyze(frame["sha256"], lambda: "d:", builder)
    yield storage
    st.cache_resource.clear()
    assert real_data.exists() == existed, "тест интерфейса тронул рабочую папку data/"


def test_all_tabs_render(seeded):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Загрузка", "Сводка", "Стройка", "Кадры", "План"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Текущий этап"].startswith("5.")
    headers = [h.value for h in at.subheader]
    assert "Техника за весь период" in headers and "Выгрузка CSV" in headers
    assert any("План построен автоматически" in i.value for i in at.info)


def test_local_provider_is_selectable_but_blocked(seeded):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("local").run()
    assert not at.exception, at.exception
    assert any("docs/local-model.md" in e.value for e in at.error)
    # результаты, разобранные z.ai, остаются видны
    assert {m.label: m.value for m in at.metric}["Текущий этап"].startswith("5.")


def test_context_toggle_marks_frames_for_reanalysis(seeded):
    at = AppTest.from_file(APP, default_timeout=60).run()
    toggle = next(t for t in at.sidebar.toggle if t.label == "Учитывать историю стройки")
    assert not any("Ожидают разбора" in m.value for m in at.markdown)
    toggle.set_value(False).run()
    assert not at.exception, at.exception
    assert any("Ожидают разбора" in m.value for m in at.markdown)
