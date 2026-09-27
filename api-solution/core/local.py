"""Провайдер «локальная модель» — СЛОТ, реализацию делает Денис.

Инструкция: docs/local-model.md. Коротко: локальный сервер (Ollama, vLLM,
LM Studio) отвечает по тому же OpenAI-совместимому протоколу, что и z.ai,
поэтому класс уже наследует весь общий транспорт. Осталось:

  1. проверить и при необходимости поправить тело запроса (_extra_body);
  2. обход HTTP_PROXY для локального адреса (_post_kwargs);
  3. прогнать контрактный и live-тесты, выставить IMPLEMENTED = True.

До этого выбор «Локальная модель» в интерфейсе показывает, что она не
подключена, а разбор падает с понятной ошибкой ещё до сети.
"""

from . import config
from .vlm import OpenAICompatibleClient, VLMError


class LocalVLMClient(OpenAICompatibleClient):
    provider = "local"

    # TODO(Денис): True после прогона tests/test_provider_contract.py и live-теста.
    IMPLEMENTED = False

    def __init__(self, base_url=None, model=None, thinking=False, timeout=300):
        super().__init__(base_url or config.LOCAL_BASE_URL, model or config.LOCAL_MODEL, thinking, timeout)
        self.api_key = config.LOCAL_API_KEY

    @classmethod
    def status(cls):
        """(готов ли, почему нет) — для интерфейса."""
        if not cls.IMPLEMENTED:
            return False, "локальная модель ещё не подключена — см. docs/local-model.md"
        if not config.LOCAL_BASE_URL:
            return False, "не задан LOCAL_VLM_BASE_URL в .env"
        return True, ""

    def _check_ready(self):
        ok, reason = self.status()
        if not ok:
            raise VLMError(reason)

    def _headers(self):
        # Ollama ключ не проверяет, но OpenAI-совместимые серверы часто требуют заголовок.
        return {"Authorization": f"Bearer {self.api_key or 'local'}", "Content-Type": "application/json"}

    def _extra_body(self, max_tokens):
        # TODO(Денис): проверить на своей модели.
        #  - Поле "thinking" z.ai сюда не передаём. Для qwen3-vl рассуждение
        #    выключает instruct-тег модели (qwen3-vl:30b-a3b-instruct).
        #  - Ollama поддерживает response_format={"type": "json_schema", ...} —
        #    в dev-lamonifi это уже работает (app/pipeline/model_b.py::ask_batch).
        #    Без схемы ответ всё равно разбирается extract_json, так что это
        #    улучшение надёжности, а не обязательный шаг.
        return {"max_tokens": max_tokens}

    def _post_kwargs(self):
        # TODO(Денис): при HTTP_PROXY в окружении requests уведёт в прокси даже
        # 127.0.0.1. Нужны все три ключа — см. dev-lamonifi/app/netutil.py.
        return {"proxies": {"http": None, "https": None, "all": None}}
