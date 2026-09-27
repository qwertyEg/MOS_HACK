"""Реестр провайдеров vision-моделей: что показывать в интерфейсе и как создать клиент.

Новый провайдер = класс с контрактом из core/vlm.py + строка здесь.
"""

from . import config
from .glm import GLMClient
from .local import LocalVLMClient


def _zai_status():
    return (True, "") if config.API_KEY else (False, "нет ZAI_API_KEY в .env")


PROVIDERS = {
    "zai": {
        "label": "z.ai (облако, GLM)",
        "client": GLMClient,
        "models": list(config.PRICES),
        "default_model": config.DEFAULT_MODEL,
        "supports_thinking": True,
        "status": _zai_status,
    },
    "local": {
        "label": "Локальная модель",
        "client": LocalVLMClient,
        "models": [config.LOCAL_MODEL],
        "default_model": config.LOCAL_MODEL,
        "supports_thinking": False,
        "status": LocalVLMClient.status,
    },
}


def make_client(provider, model=None, thinking=False):
    spec = PROVIDERS[provider]
    return spec["client"](model=model or spec["default_model"],
                          thinking=thinking and spec["supports_thinking"])
