"""Провайдер z.ai (GLM).

У vision-моделей GLM нет режима принудительного JSON (response_format только
text) — JSON вырезается из текста базовым клиентом. Рассуждение (thinking)
включается параметром тела запроса.
"""

from . import config
from .vlm import OpenAICompatibleClient, Reply, Usage, VLMError, extract_json  # noqa: F401  (реэкспорт)

GLMError = VLMError  # старое имя, на него ссылаются тесты и инструменты


class GLMClient(OpenAICompatibleClient):
    provider = "zai"

    def __init__(self, api_key=None, base_url=None, model=None, thinking=False, timeout=180):
        super().__init__(base_url or config.BASE_URL, model or config.DEFAULT_MODEL, thinking, timeout)
        self.api_key = api_key if api_key is not None else config.API_KEY

    def _check_ready(self):
        if not self.api_key:
            raise VLMError("Не задан ZAI_API_KEY (файл .env)")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _extra_body(self, max_tokens):
        return {
            # Рассуждение добавляет к ответу тысячи выходных токенов — самых дорогих.
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "max_tokens": max_tokens + (6000 if self.thinking else 0),
        }

    def _cost(self, prompt, cached, completion):
        return config.cost_usd(self.model, prompt, cached, completion)
