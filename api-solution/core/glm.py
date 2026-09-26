"""Клиент GLM (z.ai, OpenAI-совместимый /chat/completions).

У vision-моделей GLM нет режима принудительного JSON (response_format только
text), поэтому JSON вырезается из текста ответа, а при неудаче делается один
повторный запрос с напоминанием о формате.
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests

from . import config


class GLMError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class Reply:
    data: dict
    raw_text: str
    usage: Usage
    calls: list = field(default_factory=list)  # Usage каждого HTTP-вызова, включая повтор


def extract_json(text: str) -> dict:
    """Первый JSON-объект из ответа модели: без <think>, без ```-ограждений."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    text = text[start:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(_repair(text))
    if not isinstance(obj, dict):
        raise ValueError("JSON в ответе — не объект")
    return obj


def _repair(text):
    """Частые поломки JSON у VLM — чиним до того, как платить за повторный запрос.

    Заглушки из схемы, скопированные как есть (`"floors_built": <int или null>`),
    и висячие запятые перед закрывающей скобкой.
    """
    text = re.sub(r":\s*<[^<>\n]*>", ": null", text)
    return re.sub(r",\s*([}\]])", r"\1", text)


class GLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, thinking=False, timeout=180):
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.base_url = base_url or config.BASE_URL
        self.model = model or config.DEFAULT_MODEL
        self.thinking = thinking
        self.timeout = timeout

    def _post(self, messages, max_tokens):
        if not self.api_key:
            raise GLMError("Не задан ZAI_API_KEY (файл .env)")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            # Рассуждение добавляет к ответу тысячи выходных токенов — самых дорогих.
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "max_tokens": max_tokens + (6000 if self.thinking else 0),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body,
                                  headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == 3:
                    raise GLMError(f"сеть: {e}") from e
                time.sleep(delay)
                delay *= 2
                continue
            latency = int((time.monotonic() - t0) * 1000)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 3:
                    raise GLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                # 401 — неверный ключ, 400/1113 — нет баланса: повторять бессмысленно.
                raise GLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            payload = r.json()
            return payload, latency
        raise GLMError("не удалось получить ответ")

    def ask_json(self, system: str, image_url: str, text: str, max_tokens: int) -> Reply:
        # Текст до картинки: z.ai кэширует только текстовый префикс (замер: при
        # картинке первой в кэш попадал лишь системный промпт). Инструкции разведки
        # одинаковы для всех фото — со второго кадра они идут по цене cached_input.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        calls = []
        last_error = None
        for _ in range(2):
            payload, latency = self._post(messages, max_tokens)
            usage = self._usage(payload, latency)
            calls.append(usage)
            content = payload["choices"][0]["message"].get("content") or ""
            try:
                data = extract_json(content)
                return Reply(data=data, raw_text=content, usage=_sum(calls), calls=calls)
            except ValueError as e:
                last_error = f"{e}; начало ответа: {content[:200]!r}"
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Ответ не разобрался как JSON. Верни только один JSON-объект по заданной схеме, без пояснений."},
                ]
        raise GLMError(f"модель не вернула JSON: {last_error}")

    def _usage(self, payload, latency):
        u = payload.get("usage") or {}
        prompt = u.get("prompt_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        completion = u.get("completion_tokens", 0)
        return Usage(prompt, cached, completion,
                     config.cost_usd(self.model, prompt, cached, completion), latency)


def _sum(calls):
    total = Usage()
    for c in calls:
        total.prompt_tokens += c.prompt_tokens
        total.cached_tokens += c.cached_tokens
        total.completion_tokens += c.completion_tokens
        total.cost_usd += c.cost_usd
        total.latency_ms += c.latency_ms
    return total
