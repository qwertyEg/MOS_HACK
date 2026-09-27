"""Настройки сервиса: ключ, адрес API, модели и их цены."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# SERVICE_DATA_DIR — чтобы тесты интерфейса и эксперименты не трогали рабочую базу.
DATA_DIR = Path(os.getenv("SERVICE_DATA_DIR") or ROOT / "data")
DB_PATH = DATA_DIR / "service.sqlite"
IMAGES_DIR = DATA_DIR / "images"
CHECKLIST_PATH = ROOT / "reference" / "checklist.json"

API_KEY = os.getenv("ZAI_API_KEY", "")
BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4").rstrip("/")
DEFAULT_MODEL = os.getenv("GLM_MODEL", "glm-4.6v")

# Локальная модель (docs/local-model.md). Умолчания — как у модели Б в dev-lamonifi.
LOCAL_BASE_URL = os.getenv("LOCAL_VLM_BASE_URL", "http://localhost:11435/v1").rstrip("/")
LOCAL_MODEL = os.getenv("LOCAL_VLM_MODEL", "qwen3-vl:30b-a3b-instruct")
LOCAL_API_KEY = os.getenv("LOCAL_VLM_API_KEY", "")

# Длинная сторона кадра перед отправкой. Токены картинки растут с разрешением,
# а техника и конструкции на обзорном кадре различимы и на 1280.
IMAGE_MAX_SIDE = 1280
JPEG_QUALITY = 85

# Меняется вручную при правке промптов — старые ответы из кэша перестают
# подходить. Правка checklist.json инвалидирует кэш сама (по хэшу файла).
PROMPT_VERSION = "p3"


@dataclass(frozen=True)
class Price:
    """Доллары за 1M токенов (docs.z.ai/guides/overview/pricing, сентябрь 2026)."""
    input: float
    cached_input: float
    output: float


PRICES = {
    "glm-4.6v": Price(0.30, 0.05, 0.90),
    "glm-4.6v-flashx": Price(0.04, 0.004, 0.40),
    "glm-4.6v-flash": Price(0.0, 0.0, 0.0),
    "glm-4.5v": Price(0.60, 0.11, 1.80),
    "glm-5v-turbo": Price(1.20, 0.24, 4.00),
}


def cost_usd(model, prompt_tokens, cached_tokens, completion_tokens):
    price = PRICES.get(model)
    if price is None:
        return 0.0
    fresh = max(prompt_tokens - cached_tokens, 0)
    return (fresh * price.input + cached_tokens * price.cached_input
            + completion_tokens * price.output) / 1_000_000
