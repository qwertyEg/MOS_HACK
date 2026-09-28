"""Клиент vision-LLM по OpenAI-совместимому протоколу: z.ai (GLM-4.6V) и локальные Ollama/vLLM.

Порт `api-solution/core/vlm.py` + `providers.py` + `glm.py` + `local.py` (Никита) с
находками Дениса из `app/pipeline/model_b.py` и `app/netutil.py`.

Всё, что выше этого слоя (модель Б, детектор на GLM), знает о модели одно:
у клиента есть `ask_json(system, image_url, prompt, max_tokens) -> Reply`,
`ready() -> (bool, причина)` и `model_id`. Любой сервер, говорящий на
`POST {base_url}/chat/completions`, подключается без правок логики.

Что исправлено против наследия (см. разбор веток, баги A7, A14, D5, D6):

- **Любая** ошибка разбора ответа (не JSON в теле 200, нет `choices`, пустой
  `content`, битый JSON дважды) становится `VLMError`. Раньше AttributeError /
  KeyError / JSONDecodeError пролетали мимо `except VLMError` и обрывали весь
  прогон разбора.
- **Худший случай ограничен.** `timeout` — общий бюджет одного `ask_json`,
  включая ретраи и повтор «верни JSON». Раньше было 180 с × 4 попытки × 2
  повтора ≈ 12 минут на один вызов. Теперь не дольше `timeout` плюс секунды.
- Пустой ответ рассуждающей модели (`finish_reason == "length"`, весь бюджет
  токенов ушёл в рассуждение) — явная ошибка, а не молчаливое «не видно»:
  иначе тихо портится вся статистика (находка Дениса).
- Для локального адреса системный прокси обходится (`trust_env=False`): на
  машине разработки HTTP_PROXY уводил запросы к 127.0.0.1 в Squid (Денис).
- Порядок частей сообщения зависит от провайдера: z.ai кэширует только
  текстовый префикс — текст идёт до картинки (замер Никиты); Ollama
  переиспользует KV-кэш картинки между вызовами — картинка идёт первой, и
  дорогой префилл изображения делается один раз на серию вопросов (замер Дениса).

Для тестов и офлайн-демо — `CassetteClient`: запись и воспроизведение ответов
(порт `tests/cassette.py` Никиты, формат кассеты тот же).
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_DEFAULT_MODEL = "glm-4.6v"
# Ollama по умолчанию слушает 11434. Модель по умолчанию — та, что влезает в
# ноутбук на 8 ГБ; на GPU-хосте команды задаётся VLM_MODEL (qwen3-vl:30b-a3b-instruct).
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_DEFAULT_MODEL = "qwen2.5vl:3b"


class VLMError(RuntimeError):
    """Ошибка модели или транспорта. Вызывающий ловит её и переходит к следующему кадру.

    `usage` — сколько уже потрачено до сбоя (оплаченные вызовы не должны теряться
    из учёта расходов, даже если следующий шаг упал).
    """

    def __init__(self, message: str, usage: "Usage | None" = None):
        super().__init__(message)
        self.usage = usage


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class Reply:
    data: dict                    # разобранный JSON-объект из ответа
    text: str                     # ответ модели как есть — для отладки и кассеты
    usage: Usage                  # сумма по всем HTTP-вызовам (включая повтор за JSON)
    latency_ms: float = 0.0
    calls: list[Usage] = field(default_factory=list)  # Usage каждого вызова

    @property
    def raw_text(self) -> str:
        """Старое имя поля (api-solution) — на него ссылаются кассеты и инструменты."""
        return self.text


@dataclass(frozen=True)
class Price:
    """Доллары за 1M токенов (docs.z.ai/guides/overview/pricing, сентябрь 2026)."""
    input: float
    cached_input: float
    output: float


ZAI_PRICES = {
    "glm-4.6v": Price(0.30, 0.05, 0.90),
    "glm-4.6v-flashx": Price(0.04, 0.004, 0.40),
    "glm-4.6v-flash": Price(0.0, 0.0, 0.0),
    "glm-4.5v": Price(0.60, 0.11, 1.80),
    "glm-5v-turbo": Price(1.20, 0.24, 4.00),
}


# --------------------------------------------------------------------------
# разбор ответа
# --------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Первый JSON-объект из ответа модели: без <think>, без ```-ограждений, без box-токенов GLM.

    Бросает ValueError, если объекта нет: решение «переспросить или сдаться»
    принимает клиент, а не парсер.
    """
    if not isinstance(text, str):
        raise ValueError(f"ответ модели не строка: {type(text).__name__}")
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
        try:
            obj, _ = json.JSONDecoder().raw_decode(_repair(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON не разобрался: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("JSON в ответе — не объект")
    return obj


def _repair(text: str) -> str:
    """Частые поломки JSON у VLM — чиним до того, как платить за повторный запрос.

    Заглушки из схемы, скопированные как есть (`"floors_built": <int или null>`),
    и висячие запятые перед закрывающей скобкой.
    """
    text = re.sub(r":\s*<[^<>\n]*>", ": null", text)
    return re.sub(r",\s*([}\]])", r"\1", text)


def image_to_data_url(image_bgr: np.ndarray, max_side: int = 1280, quality: int = 85) -> str:
    """Кадр OpenCV (BGR/серый/BGRA) → `data:image/jpeg;base64,…` с ужатой длинной стороной.

    1280 — компромисс Никиты: токены картинки растут с разрешением, а техника и
    конструкции на обзорном кадре различимы и на 1280.
    """
    import cv2  # лениво: модуль импортируют и там, где картинок нет

    if image_bgr is None or not hasattr(image_bgr, "shape") or image_bgr.size == 0:
        raise ValueError("пустой кадр")
    img = image_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("не удалось закодировать кадр в JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def is_local_url(url: str) -> bool:
    """Локальный ли адрес (порт `app/netutil.is_local` Дениса): для него прокси не нужен."""
    host = (urlparse(url).hostname or "").lower()
    if not host or host in ("localhost", "host.docker.internal"):
        return True
    if host.endswith(".local") or "." not in host:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_unspecified


def _sum(calls: list[Usage]) -> Usage:
    total = Usage()
    for c in calls:
        total.prompt_tokens += c.prompt_tokens
        total.cached_tokens += c.cached_tokens
        total.completion_tokens += c.completion_tokens
        total.cost_usd += c.cost_usd
        total.latency_ms += c.latency_ms
    return total


# --------------------------------------------------------------------------
# клиент
# --------------------------------------------------------------------------

_JSON_REMINDER = ("Ответ не разобрался как JSON. Верни только один JSON-объект по заданной схеме, "
                  "без пояснений.")


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions — z.ai, Ollama, vLLM, LM Studio говорят на этом протоколе.

    `timeout` — общий бюджет одного `ask_json` в секундах (все попытки и повтор
    за JSON вместе). Наследники задают провайдер, тело запроса и цену.
    """

    provider = "openai"
    image_first = False          # картинка до текста (выгодно для KV-кэша Ollama)
    supports_schema = False      # умеет ли сервер response_format=json_schema
    supports_thinking = False
    requires_key = False
    max_retries = 2              # повторов на 429/5xx/сеть (внутри общего бюджета)
    connect_timeout = 10.0
    min_attempt_s = 3.0          # меньше этого остатка бюджета новую попытку не начинаем
    temperature = 0.1

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 thinking: bool = False, timeout: float = 180):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.thinking = bool(thinking) and self.supports_thinking
        self.timeout = float(timeout)
        self.session = requests.Session()
        # Локальному серверу прокси не нужен никогда, а HTTP_PROXY без исключения
        # для localhost уводил запросы в чужой Squid (находка Дениса).
        self.session.trust_env = not is_local_url(self.base_url)
        # Подменяются в тестах: время и сон, чтобы проверять бюджет без ожидания.
        self._clock = time.monotonic
        self._sleep = time.sleep

    # --- точки расширения ---

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _extra_body(self, max_tokens: int) -> dict:
        return {"max_tokens": max_tokens}

    def _cost(self, prompt: int, cached: int, completion: int) -> float:
        return 0.0

    def _check_ready(self) -> None:
        """Бросить VLMError до сети, если клиент не настроен."""
        if not self.base_url:
            raise VLMError("не задан адрес сервера модели")
        if self.requires_key and not self.api_key:
            raise VLMError(self._missing_key_reason())

    def _missing_key_reason(self) -> str:
        return "не задан API-ключ"

    # --- публичное ---

    @property
    def model_id(self) -> str:
        """Идентичность модели для кэша разборов и записи в БД: провайдеры и режимы не смешиваются."""
        return f"{self.provider}:{self.model}:{'think' if self.thinking else 'plain'}"

    def ready(self) -> tuple[bool, str]:
        """(готов ли, человекочитаемая причина) — для страницы настроек. Сеть не трогает."""
        try:
            self._check_ready()
        except VLMError as e:
            return False, str(e)
        return True, ""

    def ask_json(self, system: str, image_url: str, prompt: str | list[str], max_tokens: int,
                 schema: dict | None = None) -> Reply:
        """Один вопрос по картинке → разобранный JSON.

        prompt — строка или список строк: неизменная часть первой (её кэширует z.ai),
        изменяемая (контекст стройки) — после. schema — JSON-схема ответа для
        серверов с ограничением декодирования (Ollama); остальные её игнорируют.
        """
        calls: list[Usage] = []
        try:
            self._check_ready()
            deadline = self._clock() + self.timeout
            texts = [prompt] if isinstance(prompt, str) else [p for p in prompt if p]
            text_parts = [{"type": "text", "text": t} for t in texts]
            image_part = [{"type": "image_url", "image_url": {"url": image_url}}] if image_url else []
            content = image_part + text_parts if self.image_first else text_parts + image_part
            messages: list[dict] = [{"role": "system", "content": system},
                                    {"role": "user", "content": content}]
            last_error = ""
            for _ in range(2):
                payload, latency = self._post(messages, max_tokens, schema, deadline)
                calls.append(self._usage(payload, latency))
                text, finish = self._content(payload)
                if not text.strip():
                    if finish == "length":
                        raise VLMError("пустой ответ: бюджет max_tokens целиком ушёл на рассуждение модели "
                                       "(увеличьте max_tokens или выключите thinking)")
                    last_error = "пустой ответ модели"
                else:
                    try:
                        data = extract_json(text)
                        total = _sum(calls)
                        return Reply(data=data, text=text, usage=total,
                                     latency_ms=float(total.latency_ms), calls=calls)
                    except ValueError as e:
                        last_error = f"{e}; начало ответа: {text[:200]!r}"
                if self._clock() >= deadline - self.min_attempt_s:
                    break
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": _JSON_REMINDER}]
            raise VLMError(f"модель не вернула JSON: {last_error}")
        except VLMError as e:
            if e.usage is None:
                e.usage = _sum(calls)
            raise
        except Exception as e:  # noqa: BLE001 — любой сбой разбора — это сбой модели, а не всего прогона
            raise VLMError(f"сбой разбора ответа модели: {type(e).__name__}: {e}", usage=_sum(calls)) from e

    # --- транспорт ---

    def _post(self, messages: list[dict], max_tokens: int, schema: dict | None,
              deadline: float) -> tuple[dict, int]:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": self.temperature, **self._extra_body(max_tokens)}
        if schema is not None and self.supports_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema}}
        url = f"{self.base_url}/chat/completions"
        attempt, delay, last = 0, 1.5, ""
        while True:
            remaining = deadline - self._clock()
            if remaining < self.min_attempt_s:
                raise VLMError(f"истёк бюджет времени {self.timeout:.0f} с на запрос к модели"
                               + (f" (последняя ошибка: {last})" if last else ""))
            t0 = self._clock()
            try:
                r = self.session.post(url, json=body, headers=self._headers(),
                                      timeout=(min(self.connect_timeout, remaining), remaining))
            except requests.Timeout:
                last = f"таймаут ответа сервера {self.base_url}"
            except requests.RequestException as e:
                last = f"сеть: {e}"
            else:
                latency = int((self._clock() - t0) * 1000)
                code = getattr(r, "status_code", 0)
                if code == 200:
                    try:
                        payload = r.json()
                    except ValueError as e:
                        body = str(getattr(r, "text", ""))[:200]
                        raise VLMError(f"сервер вернул 200, но тело не JSON: {body!r}") from e
                    if not isinstance(payload, dict):
                        raise VLMError("сервер вернул не JSON-объект")
                    return payload, latency
                text = str(getattr(r, "text", ""))[:300]
                if code in (408, 409, 425, 429) or code >= 500:
                    last = f"HTTP {code}: {text}"
                else:
                    # 400 (в т.ч. «не умею response_format»), 401 неверный ключ, 402/1113 нет
                    # баланса, 404 нет модели — повторять бессмысленно.
                    raise VLMError(f"HTTP {code}: {text}")
            attempt += 1
            if attempt > self.max_retries:
                raise VLMError(f"не удалось получить ответ за {attempt} попыток: {last}")
            self._sleep(max(0.0, min(delay, deadline - self._clock() - self.min_attempt_s)))
            delay *= 2

    @staticmethod
    def _content(payload: dict) -> tuple[str, str]:
        try:
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            finish = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise VLMError(f"ответ без choices/message: {str(payload)[:200]}") from e
        if content is None:
            content = ""
        if isinstance(content, list):  # некоторые серверы отдают content частями
            content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        if not isinstance(content, str):
            raise VLMError(f"content не строка: {type(content).__name__}")
        return content, finish

    def _usage(self, payload: dict, latency: int) -> Usage:
        u = payload.get("usage") or {}
        if not isinstance(u, dict):
            u = {}
        prompt = int(u.get("prompt_tokens") or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        completion = int(u.get("completion_tokens") or 0)
        return Usage(prompt, cached, completion, self._cost(prompt, cached, completion), latency)


class ZaiClient(OpenAICompatibleClient):
    """z.ai (GLM). У vision-моделей GLM нет принудительного JSON (response_format только
    text) — JSON вырезается из текста. Рассуждение включается полем тела запроса."""

    provider = "zai"
    supports_thinking = True
    requires_key = True

    def _missing_key_reason(self) -> str:
        return "нет ZAI_API_KEY в окружении (.env) — внешний API недоступен"

    def _extra_body(self, max_tokens: int) -> dict:
        return {
            # Рассуждение добавляет к ответу тысячи выходных токенов — самых дорогих.
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "max_tokens": max_tokens + (6000 if self.thinking else 0),
        }

    def _cost(self, prompt: int, cached: int, completion: int) -> float:
        price = ZAI_PRICES.get(self.model)
        if price is None:
            return 0.0
        fresh = max(prompt - cached, 0)
        return (fresh * price.input + cached * price.cached_input + completion * price.output) / 1_000_000


class LocalVLMClient(OpenAICompatibleClient):
    """Локальная VLM (Ollama / vLLM / LM Studio) по тому же протоколу.

    Картинка первой: Ollama переиспользует KV-кэш изображения между вызовами,
    и серия вопросов по одному кадру платит за префилл картинки один раз.
    Ollama поддерживает ограничение декодирования JSON-схемой (замер Дениса:
    схема не даёт модели пропустить вопрос или ответить не из списка).
    """

    provider = "local"
    image_first = True
    supports_schema = True
    temperature = 0.0

    def ready(self) -> tuple[bool, str]:
        """Сервер отвечает и модель на нём есть — короткий GET /models (3 с)."""
        try:
            self._check_ready()
        except VLMError as e:
            return False, str(e)
        try:
            r = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=3)
        except requests.RequestException as e:
            return False, f"локальная VLM не отвечает по адресу {self.base_url}: {type(e).__name__}"
        if getattr(r, "status_code", 0) != 200:
            return False, f"локальная VLM ответила HTTP {getattr(r, 'status_code', '?')} на {self.base_url}/models"
        try:
            ids = [str(m.get("id", "")) for m in (r.json().get("data") or []) if isinstance(m, dict)]
        except (ValueError, AttributeError):
            return True, ""  # сервер жив, но список моделей не отдаёт — проверим боем
        if ids and not _model_listed(self.model, ids):
            return False, f"модель {self.model} не загружена в локальный сервер; доступны: {', '.join(ids[:8])}"
        return True, ""


def _model_listed(model: str, ids: list[str]) -> bool:
    def norm(x: str) -> str:
        return x[:-7] if x.endswith(":latest") else x
    return norm(model) in {norm(i) for i in ids}


def make_client(provider: str, model: str | None = None, thinking: bool = False,
                timeout: float | None = None) -> OpenAICompatibleClient:
    """Клиент по имени провайдера. Настройки — из окружения, чтобы подставить свой сервер без правки кода.

    zai:   ZAI_API_KEY, ZAI_BASE_URL (по умолчанию https://api.z.ai/api/paas/v4),
           GLM_MODEL (glm-4.6v), ZAI_TIMEOUT (90 с на вызов).
    local: VLM_BASE_URL (http://localhost:11434/v1 — Ollama), VLM_MODEL, VLM_API_KEY,
           VLM_TIMEOUT (180 с — CPU-инференс медленный).
    """
    if provider in ("zai", "glm", "external"):
        return ZaiClient(
            base_url=os.getenv("ZAI_BASE_URL", ZAI_BASE_URL),
            model=model or os.getenv("GLM_MODEL", ZAI_DEFAULT_MODEL),
            api_key=os.getenv("ZAI_API_KEY", "") or None,
            thinking=thinking,
            timeout=timeout or float(os.getenv("ZAI_TIMEOUT", "90")),
        )
    if provider in ("local", "local_vlm", "ollama"):
        return LocalVLMClient(
            base_url=os.getenv("VLM_BASE_URL", LOCAL_BASE_URL),
            model=model or os.getenv("VLM_MODEL", LOCAL_DEFAULT_MODEL),
            api_key=os.getenv("VLM_API_KEY", "") or None,
            thinking=False,
            timeout=timeout or float(os.getenv("VLM_TIMEOUT", "180")),
        )
    raise ValueError(f"неизвестный провайдер VLM: {provider!r} (ожидается 'zai' или 'local')")


# --------------------------------------------------------------------------
# кассета: запись и воспроизведение ответов
# --------------------------------------------------------------------------


class CassetteMiss(VLMError):
    """В кассете нет ответа на такой запрос, а живого клиента нет (или запись запрещена)."""


class CassetteClient:
    """Запись/воспроизведение ответов VLM — порт `tests/cassette.py` Никиты, формат тот же.

    Ключ записи — хэш (модель, thinking, системный промпт, текст запроса, картинка).
    Повторный прогон с теми же промптами не тратит токены; сырой текст ответа
    хранится как есть и при воспроизведении заново проходит extract_json — так
    тестируется и парсер. Годится и для офлайн-демо без ключа z.ai.
    """

    def __init__(self, path: str | Path, real: OpenAICompatibleClient | None = None,
                 model: str = ZAI_DEFAULT_MODEL, thinking: bool = False, provider: str = "zai",
                 record: bool = True):
        self.path = Path(path)
        self.real = real
        self.provider = real.provider if real else provider
        self.model = real.model if real else model
        self.thinking = real.thinking if real else thinking
        self.record = record
        self.tape: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.recorded = 0
        self.replayed = 0

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}:{'think' if self.thinking else 'plain'}"

    def ready(self) -> tuple[bool, str]:
        if self.real is not None:
            return self.real.ready()
        return (True, "") if self.tape else (False, "кассета пуста и живого клиента нет")

    def _key(self, system: str, image_url: str, text: str) -> str:
        img = hashlib.sha1(image_url.encode()).hexdigest()
        # Прежние записи (только z.ai) — без провайдера в ключе, чтобы не пропали.
        head = [self.model] if self.provider == "zai" else [self.provider, self.model]
        raw = "\x1f".join(head + [str(self.thinking), system, text, img])
        return hashlib.sha1(raw.encode()).hexdigest()

    def ask_json(self, system: str, image_url: str, prompt: str | list[str], max_tokens: int,
                 schema: dict | None = None) -> Reply:
        text = prompt if isinstance(prompt, str) else "\n\n".join(t for t in prompt if t)
        key = self._key(system, image_url, text)
        item = self.tape.get(key)
        if item is not None:
            calls = [Usage(**u) for u in item["calls"]]
            total = _sum(calls)
            self.replayed += 1
            try:
                data = extract_json(item["raw_text"])
            except ValueError as e:
                raise VLMError(f"в кассете ответ без JSON: {e}", usage=total) from e
            return Reply(data=data, text=item["raw_text"], usage=total,
                         latency_ms=float(total.latency_ms), calls=calls)
        if self.real is None or not self.record:
            raise CassetteMiss("нет записи ответа в кассете и нет живого клиента")
        reply = self.real.ask_json(system, image_url, prompt, max_tokens, schema=schema)
        self.tape[key] = {"step": text.split("\n", 1)[0], "raw_text": reply.text,
                          "calls": [asdict(c) for c in reply.calls]}
        # Пишем сразу: если прогон упадёт на середине, оплаченное не потеряется.
        self.path.write_text(json.dumps(self.tape, ensure_ascii=False, indent=1), encoding="utf-8")
        self.recorded += 1
        return reply
