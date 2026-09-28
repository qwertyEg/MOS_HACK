"""Клиент VLM на подменённом транспорте и кассете — без сети и без токенов.

Проверяем то, что ломалось в наследии: мусорный ответ не должен пролетать мимо
VLMError (баг A7), худший случай не должен висеть 12 минут (A14), пустой ответ
рассуждающей модели не должен молча стать «не видно» (Денис), прокси не должен
перехватывать локальный адрес.
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from core.vlm_client import (CassetteClient, CassetteMiss, LocalVLMClient, OpenAICompatibleClient, VLMError,
                             ZaiClient, extract_json, image_to_data_url, make_client)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class FakeResponse:
    def __init__(self, status=200, content="", usage=None, finish="stop", body=None, raise_json=False):
        self.status_code = status
        self._raise = raise_json
        self._payload = body if body is not None else {
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 100,
                               "prompt_tokens_details": {"cached_tokens": 400}},
        }
        self.text = "<html>502</html>" if raise_json else json.dumps(self._payload)

    def json(self):
        if self._raise:
            raise ValueError("not json")
        return self._payload


class FakeClock:
    """Время идёт только когда «сервер думает» или клиент спит — проверяем бюджет без ожидания."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeSession:
    def __init__(self, clock, queue=None, cost=0.5):
        self.queue = list(queue or [])
        self.sent = []
        self.clock = clock
        self.cost = cost
        self.trust_env = True

    def post(self, url, json=None, headers=None, timeout=None):
        self.sent.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        item = self.queue.pop(0) if self.queue else FakeResponse(200, '{"ok": 1}')
        if isinstance(item, Exception):
            # сервер «висел» до таймаута
            self.clock.t += timeout[1] if isinstance(timeout, tuple) else timeout
            raise item
        self.clock.t += self.cost
        return item

    def get(self, url, headers=None, timeout=None):
        self.sent.append({"url": url, "get": True})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def wire(client, queue=None, cost=0.5):
    clock = FakeClock()
    client.session = FakeSession(clock, queue, cost)
    client._clock = clock
    client._sleep = clock.sleep
    return client.session, clock


def zai(**kw):
    return ZaiClient(base_url="https://api.test/v4", model="glm-4.6v", api_key="k", **kw)


def test_request_shape_text_before_image_for_zai():
    c = zai()
    s, _ = wire(c, [FakeResponse(200, '{"ok": 1}')])
    reply = c.ask_json("SYS", "data:image/jpeg;base64,AAA", ["статичная часть", "", "контекст"], max_tokens=500)
    sent = s.sent[0]
    assert sent["url"] == "https://api.test/v4/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    body = sent["body"]
    assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 500
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    # текст раньше картинки — неизменный префикс для кэша z.ai; пустые части выброшены
    assert [p["type"] for p in body["messages"][1]["content"]] == ["text", "text", "image_url"]
    assert "response_format" not in body          # у GLM vision нет json_schema
    assert reply.data == {"ok": 1} and reply.text == '{"ok": 1}' and reply.raw_text == reply.text
    assert c.model_id == "zai:glm-4.6v:plain"


def test_cost_counts_cached_tokens_cheaper():
    c = zai()
    wire(c, [FakeResponse(200, '{"ok": 1}')])
    u = c.ask_json("S", "u", "t", 100).usage
    assert u.cost_usd == pytest.approx((600 * 0.30 + 400 * 0.05 + 100 * 0.90) / 1e6)


def test_retries_rate_limit_and_server_error_then_succeeds():
    c = zai()
    s, _ = wire(c, [FakeResponse(429), FakeResponse(503), FakeResponse(200, '{"ok": 1}')])
    assert c.ask_json("S", "u", "t", 100).data == {"ok": 1}
    assert len(s.sent) == 3


def test_auth_error_is_not_retried():
    c = zai()
    s, _ = wire(c, [FakeResponse(401)])
    with pytest.raises(VLMError, match="401"):
        c.ask_json("S", "u", "t", 100)
    assert len(s.sent) == 1


def test_bad_json_gets_one_reminder_and_both_calls_are_billed():
    c = zai()
    s, _ = wire(c, [FakeResponse(200, "Извините, вот описание без JSON"),
                    FakeResponse(200, '```json\n{"ok": 2}\n```')])
    reply = c.ask_json("S", "u", "t", 100)
    assert reply.data == {"ok": 2} and len(reply.calls) == 2
    assert reply.usage.prompt_tokens == 2000
    assert "JSON" in s.sent[1]["body"]["messages"][-1]["content"]


@pytest.mark.parametrize("responses, match", [
    ([FakeResponse(200, "нет json"), FakeResponse(200, "всё ещё нет")], "JSON"),
    ([FakeResponse(200, raise_json=True)], "не JSON"),                  # 200 с HTML-телом прокси
    ([FakeResponse(200, body={"error": "oops"})], "choices"),           # нет choices
    ([FakeResponse(200, body={"choices": []})], "choices"),
    ([FakeResponse(200, body={"choices": [{"message": {"content": 42}}]})], "content"),
    ([FakeResponse(200, "", finish="length")], "рассуждение"),          # весь бюджет ушёл в thinking
])
def test_garbage_answers_always_become_vlm_error(responses, match):
    c = zai()
    wire(c, responses)
    with pytest.raises(VLMError, match=match):
        c.ask_json("S", "u", "t", 100)


def test_unexpected_exception_inside_is_wrapped_into_vlm_error():
    c = zai()
    wire(c)

    def boom(*a, **k):
        raise AttributeError("внезапно")

    c._post = boom
    with pytest.raises(VLMError, match="AttributeError"):
        c.ask_json("S", "u", "t", 100)


def test_worst_case_is_bounded_by_timeout_budget():
    """Сервер каждый раз висит до таймаута: раньше это было 180 с × 4 × 2 ≈ 12 минут."""
    c = zai(timeout=60)
    s, clock = wire(c, [requests.Timeout("read timed out")] * 10)
    with pytest.raises(VLMError):
        c.ask_json("S", "u", "t", 100)
    assert clock.t <= 60 + 1
    assert len(s.sent) <= c.max_retries + 1


def test_missing_key_fails_before_network_and_ready_explains():
    c = ZaiClient(base_url="https://api.test/v4", model="glm-4.6v", api_key=None)
    s, _ = wire(c)
    ok, reason = c.ready()
    assert not ok and "ZAI_API_KEY" in reason
    with pytest.raises(VLMError, match="ZAI_API_KEY"):
        c.ask_json("S", "u", "t", 100)
    assert s.sent == []


def test_make_client_reads_env_and_local_bypasses_proxy(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("VLM_MODEL", "qwen3-vl:30b-a3b-instruct")
    monkeypatch.setenv("ZAI_API_KEY", "secret")
    local = make_client("local")
    assert isinstance(local, LocalVLMClient) and isinstance(local, OpenAICompatibleClient)
    assert local.model == "qwen3-vl:30b-a3b-instruct" and local.base_url == "http://127.0.0.1:11435/v1"
    assert local.session.trust_env is False           # HTTP_PROXY не должен уводить 127.0.0.1 в Squid
    remote = make_client("zai")
    assert remote.session.trust_env is True and remote.model == "glm-4.6v" and remote.api_key == "secret"
    assert remote.base_url == "https://api.z.ai/api/paas/v4"
    with pytest.raises(ValueError):
        make_client("openai-magic")


def test_local_client_sends_schema_and_image_first():
    c = LocalVLMClient(base_url="http://localhost:11434/v1", model="qwen2.5vl:3b")
    s, _ = wire(c, [FakeResponse(200, '{"pit": "да"}')])
    schema = {"type": "object", "properties": {"pit": {"type": "string"}}}
    c.ask_json("S", "data:image/jpeg;base64,AA", "вопросы", 64, schema=schema)
    body = s.sent[0]["body"]
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert "thinking" not in body
    # картинка первой — Ollama переиспользует KV-кэш изображения между группами вопросов
    assert [p["type"] for p in body["messages"][1]["content"]] == ["image_url", "text"]


def test_local_ready_checks_server_and_model_list():
    c = LocalVLMClient(base_url="http://localhost:11434/v1", model="qwen2.5vl:3b")
    listing = FakeResponse(200, body={"data": [{"id": "qwen2.5vl:3b"}, {"id": "llava:latest"}]})
    wire(c, [listing])
    assert c.ready() == (True, "")
    c2 = LocalVLMClient(base_url="http://localhost:11434/v1", model="llava")
    wire(c2, [listing])
    assert c2.ready()[0]                               # ":latest" — то же самое
    c3 = LocalVLMClient(base_url="http://localhost:11434/v1", model="qwen3-vl:30b")
    wire(c3, [listing])
    ok, reason = c3.ready()
    assert not ok and "не загружена" in reason
    c4 = LocalVLMClient(base_url="http://localhost:11434/v1", model="x")
    wire(c4, [requests.ConnectionError("refused")])
    ok, reason = c4.ready()
    assert not ok and "не отвечает" in reason


def test_extract_json_survives_typical_vlm_formatting():
    assert extract_json('<think>думаю {"a": 0}</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<|begin_of_box|>{"a": 2}<|end_of_box|>') == {"a": 2}
    assert extract_json('Вот ответ: {"a": 3, "b": [1, 2,],}') == {"a": 3, "b": [1, 2]}
    assert extract_json('{"floors_built": <int или null>, "x": 1}') == {"floors_built": None, "x": 1}
    for bad in ("просто текст", "[1, 2, 3]", "{сломано"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_image_to_data_url_downscales_and_accepts_any_channels():
    img = np.random.default_rng(0).integers(0, 255, (1500, 2000, 3), dtype=np.uint8)
    url = image_to_data_url(img, max_side=1280, quality=80)
    assert url.startswith("data:image/jpeg;base64,")
    back = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert max(back.shape[:2]) == 1280 and back.shape[1] > back.shape[0]
    assert image_to_data_url(img[..., 0]).startswith("data:image/jpeg")                  # серый
    assert image_to_data_url(np.dstack([img, img[..., :1]])).startswith("data:image/jpeg")  # BGRA
    with pytest.raises(ValueError):
        image_to_data_url(np.zeros((0, 0, 3), np.uint8))


def test_cassette_replays_recorded_glm_answer_without_network():
    """Кассета Никиты: ответ на проверку связи записан — воспроизводится без ключа и сети."""
    from PIL import Image, ImageOps
    import io

    photo = FIXTURES / "stage_photos" / "doric_2007_11_28_12_30_37.jpg"
    img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    tape = CassetteClient(FIXTURES / "glm_cassette.json", model="glm-4.6v-flash")
    reply = tape.ask_json("Отвечай одним JSON-объектом.", url,
                          'Есть ли на снимке здание? Верни {"building": true} или {"building": false}.', 30)
    assert reply.data == {"building": True} and tape.replayed == 1
    with pytest.raises(CassetteMiss):
        tape.ask_json("S", url, "вопрос, которого нет в кассете", 30)
    assert isinstance(CassetteMiss("x"), VLMError)     # промах кассеты ловится как обычный сбой модели


def test_cassette_records_through_real_client(tmp_path):
    real = zai()
    wire(real, [FakeResponse(200, '{"answers": {"pit": "yes"}}')])
    tape = CassetteClient(tmp_path / "tape.json", real=real)
    first = tape.ask_json("S", "data:x", ["шаг", "контекст"], 50)
    again = CassetteClient(tmp_path / "tape.json").ask_json("S", "data:x", ["шаг", "контекст"], 50)
    assert first.data == again.data and tape.recorded == 1
    assert json.loads((tmp_path / "tape.json").read_text())      # записано сразу, на диск
