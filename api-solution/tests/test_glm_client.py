"""HTTP-клиент GLM на подменённом requests.post — без сети и без токенов."""

import json

import pytest

from core import glm
from core.glm import GLMClient, GLMError


class FakeResponse:
    def __init__(self, status, content="", usage=None):
        self.status_code = status
        self._payload = {"choices": [{"message": {"content": content}}],
                         "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 100,
                                            "prompt_tokens_details": {"cached_tokens": 400}}}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Очередь ответов и журнал запросов вместо сети."""
    queue, sent = [], []

    def post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "body": json, "headers": headers})
        return queue.pop(0)

    monkeypatch.setattr(glm.requests, "post", post)
    monkeypatch.setattr(glm.time, "sleep", lambda s: None)
    return queue, sent


def client(**kw):
    return GLMClient(api_key="k", base_url="https://api.test/v4", model="glm-4.6v", **kw)


def test_request_shape(http):
    queue, sent = http
    queue.append(FakeResponse(200, '{"ok": 1}'))
    reply = client().ask_json("SYS", "data:image/jpeg;base64,AAA", "вопрос", max_tokens=500)
    body = sent[0]["body"]
    assert sent[0]["url"] == "https://api.test/v4/chat/completions"
    assert sent[0]["headers"]["Authorization"] == "Bearer k"
    assert body["model"] == "glm-4.6v" and body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 500
    # текст раньше картинки — статичный префикс для кэша z.ai
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert [p["type"] for p in body["messages"][1]["content"]] == ["text", "image_url"]
    assert reply.data == {"ok": 1}


def test_thinking_adds_token_budget(http):
    queue, sent = http
    queue.append(FakeResponse(200, '{"ok": 1}'))
    client(thinking=True).ask_json("S", "u", "t", max_tokens=500)
    assert sent[0]["body"]["thinking"] == {"type": "enabled"} and sent[0]["body"]["max_tokens"] > 500


def test_cost_counts_cached_tokens_cheaper(http):
    queue, _ = http
    queue.append(FakeResponse(200, '{"ok": 1}'))
    u = client().ask_json("S", "u", "t", 100).usage
    # 600 свежих × 0.30 + 400 из кэша × 0.05 + 100 выходных × 0.90, за 1M
    assert u.cost_usd == pytest.approx((600 * 0.30 + 400 * 0.05 + 100 * 0.90) / 1e6)


def test_retries_on_rate_limit_and_server_error(http):
    queue, sent = http
    queue += [FakeResponse(429), FakeResponse(503), FakeResponse(200, '{"ok": 1}')]
    assert client().ask_json("S", "u", "t", 100).data == {"ok": 1}
    assert len(sent) == 3


def test_auth_error_is_not_retried(http):
    queue, sent = http
    queue.append(FakeResponse(401))
    with pytest.raises(GLMError, match="401"):
        client().ask_json("S", "u", "t", 100)
    assert len(sent) == 1


def test_bad_json_gets_one_reminder_and_both_calls_are_billed(http):
    queue, sent = http
    queue += [FakeResponse(200, "Извините, вот описание без JSON"), FakeResponse(200, '```json\n{"ok": 2}\n```')]
    reply = client().ask_json("S", "u", "t", 100)
    assert reply.data == {"ok": 2} and len(reply.calls) == 2
    assert reply.usage.prompt_tokens == 2000
    assert "JSON" in sent[1]["body"]["messages"][-1]["content"]


def test_gives_up_after_second_bad_json(http):
    queue, _ = http
    queue += [FakeResponse(200, "нет json"), FakeResponse(200, "всё ещё нет")]
    with pytest.raises(GLMError, match="JSON"):
        client().ask_json("S", "u", "t", 100)


def test_missing_key_fails_before_network(http):
    _, sent = http
    with pytest.raises(GLMError, match="ZAI_API_KEY"):
        GLMClient(api_key="").ask_json("S", "u", "t", 100)
    assert sent == []
