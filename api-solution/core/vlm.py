"""Общий контракт vision-модели и базовый OpenAI-совместимый клиент.

Всё, что выше этого слоя (промпты, чек-листы, оценка этапов, хронология),
знает о модели только одно: у клиента есть ask_json(system, image_url,
prompt, max_tokens) → Reply и три атрибута — provider, model, thinking.
Любой провайдер, выполняющий этот контракт, получает всю логику сервиса
без изменений. Как подключить новый — docs/local-model.md.
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests


class VLMError(RuntimeError):
    """Ошибка модели или транспорта. Сервис ловит её и продолжает со следующим кадром."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class Reply:
    data: dict                      # разобранный JSON-объект из ответа
    raw_text: str                   # ответ модели как есть — для отладки и кассеты
    usage: Usage                    # сумма по всем HTTP-вызовам
    calls: list = field(default_factory=list)  # Usage каждого вызова, включая повтор


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


def cache_id(client) -> str:
    """Идентичность модели для кэша разборов: разные провайдеры и режимы не смешиваются."""
    provider = getattr(client, "provider", "zai")
    return f"{provider}:{client.model}:{'think' if client.thinking else 'plain'}"


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions — z.ai, Ollama, vLLM, LM Studio говорят на этом протоколе.

    Наследник задаёт provider, _headers(), _extra_body() и _cost(). Всё
    остальное — ретраи, порядок частей сообщения, извлечение JSON с одним
    повтором — общее.
    """

    provider = "base"

    def __init__(self, base_url, model, thinking=False, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking = thinking
        self.timeout = timeout

    # --- точки расширения ---

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _extra_body(self, max_tokens):
        """Поля тела запроса, специфичные для провайдера (thinking, response_format…)."""
        return {"max_tokens": max_tokens}

    def _cost(self, prompt, cached, completion):
        return 0.0

    def _check_ready(self):
        """Бросить VLMError до сети, если клиент не настроен (нет ключа, нет адреса)."""

    def _post_kwargs(self):
        """Дополнительные аргументы requests.post (например, proxies для локального адреса)."""
        return {}

    # --- общее ---

    def _post(self, messages, max_tokens):
        self._check_ready()
        body = {"model": self.model, "messages": messages, "temperature": 0.1, **self._extra_body(max_tokens)}
        delay = 2.0
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(),
                                  timeout=self.timeout, **self._post_kwargs())
            except requests.RequestException as e:
                if attempt == 3:
                    raise VLMError(f"сеть: {e}") from e
                time.sleep(delay)
                delay *= 2
                continue
            latency = int((time.monotonic() - t0) * 1000)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 3:
                    raise VLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                # 401 — неверный ключ, 400/1113 — нет баланса: повторять бессмысленно.
                raise VLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json(), latency
        raise VLMError("не удалось получить ответ")

    def ask_json(self, system: str, image_url: str, prompt, max_tokens: int) -> Reply:
        """prompt — строка или список строк: неизменная часть первой, изменяемая (контекст) — после."""
        # Текст до картинки: z.ai кэширует только текстовый префикс (замер: при
        # картинке первой в кэш попадал лишь системный промпт). Неизменные
        # инструкции разведки со второго кадра идут по цене кэша.
        texts = [prompt] if isinstance(prompt, str) else [p for p in prompt if p]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": t} for t in texts]
             + [{"type": "image_url", "image_url": {"url": image_url}}]},
        ]
        calls, last_error = [], None
        for _ in range(2):
            payload, latency = self._post(messages, max_tokens)
            calls.append(self._usage(payload, latency))
            content = payload["choices"][0]["message"].get("content") or ""
            try:
                return Reply(data=extract_json(content), raw_text=content, usage=_sum(calls), calls=calls)
            except ValueError as e:
                last_error = f"{e}; начало ответа: {content[:200]!r}"
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Ответ не разобрался как JSON. Верни только один JSON-объект по заданной схеме, без пояснений."},
                ]
        raise VLMError(f"модель не вернула JSON: {last_error}")

    def _usage(self, payload, latency):
        u = payload.get("usage") or {}
        prompt = u.get("prompt_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        completion = u.get("completion_tokens", 0)
        return Usage(prompt, cached, completion, self._cost(prompt, cached, completion), latency)


def _sum(calls):
    total = Usage()
    for c in calls:
        total.prompt_tokens += c.prompt_tokens
        total.cached_tokens += c.cached_tokens
        total.completion_tokens += c.completion_tokens
        total.cost_usd += c.cost_usd
        total.latency_ms += c.latency_ms
    return total
