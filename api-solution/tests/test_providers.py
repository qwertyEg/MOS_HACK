"""Контракт провайдера модели: любой провайдер из реестра проходит через ту же логику.

Сеть подменена: сервер «отвечает» заготовленным JSON. Проверяется, что
провайдер правильно зовёт свой /chat/completions и что его ответ проходит
весь путь — разведка, чек-лист, оценка этапа — так же, как у z.ai.

Для локальной модели Денис после реализации запускает ещё и live-тест:
GLM_EVAL_PROVIDER=local pytest -m live -s (см. docs/local-model.md).
"""

import json

import pytest

from core import vlm
from core.analyzer import Analyzer
from core.local import LocalVLMClient
from core.providers import PROVIDERS, make_client
from core.scoring import evaluate
from core.vlm import VLMError, cache_id

TRIAGE = {"quality": "good", "view": "side", "description": "каркас", "equipment": [
    {"type": "tower_crane", "total": 1, "working": 1, "evidence": "груз на крюке"}],
    "workers_count": 4, "floors_built": 6, "floors_glazed": None, "facade_clad_pct": None, "pit_area_pct": None,
    "latest_stage": 5, "context_conflict": None,
    "stage_likelihood": {"1": 0, "2": 0, "3": 0, "4": 0.2, "5": 0.9, "6": 0, "7": 0, "8": 0}}


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self._p = {"choices": [{"message": {"content": content}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
        self.text = json.dumps(self._p)

    def json(self):
        return self._p


@pytest.fixture
def server(monkeypatch):
    """Отвечает на разведку TRIAGE, на чек-лист — «yes» на признаки каркаса."""
    sent = []

    def post(url, json=None, headers=None, timeout=None, **kw):
        sent.append({"url": url, "body": json, "kwargs": kw})
        texts = " ".join(p["text"] for p in json["messages"][1]["content"] if p["type"] == "text")
        if texts.startswith("Шаг 1"):
            return FakeResponse("```json\n" + __import__("json").dumps(TRIAGE) + "\n```")
        yes = {"above_grade", "crane", "formwork_floor", "unfinished_top"}
        keys = [line[2:].split(":")[0] for line in texts.splitlines() if line.startswith("- ")]
        return FakeResponse(__import__("json").dumps({"answers": {k: "yes" if k in yes else "no" for k in keys}}))

    monkeypatch.setattr(vlm.requests, "post", post)
    monkeypatch.setattr(LocalVLMClient, "IMPLEMENTED", True)  # проверяем контракт, а не готовность
    monkeypatch.setattr("core.glm.config.API_KEY", "k")
    return sent


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_every_provider_runs_the_same_pipeline(server, checklist, provider):
    client = make_client(provider)
    assert client.provider == provider and client.model
    result, _ = Analyzer(checklist, None, client).analyze("sha", lambda: "data:image/jpeg;base64,AA")
    assert server[0]["url"].endswith("/chat/completions")
    content = server[0]["body"]["messages"][1]["content"]
    assert [p["type"] for p in content][-1] == "image_url"  # картинка последней
    assert result["provider"] == provider
    score = evaluate(checklist, result)
    assert score["front"] == 5 and result["triage"]["equipment"][0]["type"] == "tower_crane"


def test_providers_do_not_share_cache(checklist):
    ids = {cache_id(make_client(p)) for p in PROVIDERS}
    assert len(ids) == len(PROVIDERS)
    keys = {Analyzer(checklist, None, make_client(p)).cache_key("sha") for p in PROVIDERS}
    assert len(keys) == len(PROVIDERS)


def test_local_provider_bypasses_proxy(server):
    make_client("local").ask_json("S", "data:", "t", 10)
    assert server[0]["kwargs"]["proxies"] == {"http": None, "https": None, "all": None}
    assert "thinking" not in server[0]["body"]


def test_unimplemented_local_fails_before_network(monkeypatch):
    monkeypatch.setattr(LocalVLMClient, "IMPLEMENTED", False)
    monkeypatch.setattr(vlm.requests, "post", lambda *a, **k: pytest.fail("не должно быть запроса"))
    ok, reason = PROVIDERS["local"]["status"]()
    assert not ok and "docs/local-model.md" in reason
    with pytest.raises(VLMError, match="не подключена"):
        make_client("local").ask_json("S", "data:", "t", 10)


def test_thinking_only_where_supported():
    assert make_client("zai", thinking=True).thinking is True
    assert make_client("local", thinking=True).thinking is False
