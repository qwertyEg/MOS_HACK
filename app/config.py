"""Конфигурация сервиса: всё внешнее — переменными окружения или `.env`.

Значения по умолчанию подобраны так, чтобы `python -m app` на ноутбуке
поднимался без единой настройки: SQLite в `var/app.db`, файлы в `var/storage`,
локальные модели. PostgreSQL и S3 включаются переменными (см. `.env.example`).

Относительные пути считаются от корня репозитория, а не от текущего каталога:
у Дениса тесты и сервис падали, если их запускали не из корня.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # --- запуск ---
    host: str = "0.0.0.0"
    port: int = 8000
    public_base_url: str = "http://localhost:8000"   # адрес приёмника, который сообщаем камере-потоку

    # --- база ---
    database_url: str = "sqlite:///var/app.db"

    # --- хранилище кадров ---
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: str = "var/storage"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "frames"
    s3_region: str = "us-east-1"

    # --- внешний API (GLM-4.6V, z.ai) ---
    zai_api_key: str = ""
    zai_base_url: str = "https://api.z.ai/api/paas/v4"
    zai_model: str = "glm-4.6v"

    # --- локальная VLM по OpenAI-совместимому API (Ollama / vLLM) ---
    # Пусто — берутся умолчания core.vlm_client.make_client("local").
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""

    # --- локальные модели ---
    # YOLO модели А: файл весов, каталог с ними или equipment_classes.json (см. models/README.md)
    equipment_weights: str = "models/equipment_yolo.pt"
    # SigLIP модели Б (id HF или путь). Та же модель по умолчанию уточняет подтип
    # грузовиков у модели А — в памяти одна копия (providers.shared_embedder).
    stage_clip_model: str = "google/siglip2-base-patch16-224"
    equipment_threads: int = 0     # потоков torch на CPU для YOLO/SigLIP; 0 — как решит torch
    warm_models: bool = True       # при старте загрузить модели текущего режима (первый кадр без ~50 с ожидания)

    # --- веб ---
    secret_key: str = "dev-secret-change-me"
    admin_login: str = "admin"
    admin_password: str = "admin"
    session_max_age_h: int = 72
    session_https_only: bool = False      # true за TLS-прокси: кука сессии с флагом Secure
    login_max_failures: int = 5           # неверных паролей подряд с одного адреса на логин…
    login_lockout_s: int = 60             # …и потом пауза: 429 без проверки пароля
    max_body_mb: float = 8.0              # тело JSON (план, маска кистью…); загрузки — свои лимиты ниже
    camera_allow_loopback: bool = True    # камера-поток на 127.0.0.1 (simcam на той же машине); в сети — false
    max_frame_mb: float = 64.0            # один кадр: /api/ingest, /api/analyze, /api/detect, импорт плана

    # --- режим по умолчанию (пока в таблице settings ничего не сохранено) ---
    default_mode: Literal["local", "external", "hybrid"] = "local"

    # --- конвейер ---
    workers_enabled: bool = True          # False — кадры только сохраняются (утилиты, отладка)
    recompute_debounce_s: float = 3.0     # пересчёт площадки не чаще, чем раз в N секунд
    queue_limit: int = 500                # кадров в очереди камеры, дальше /api/ingest отвечает 503
    postponed_retry_s: float = 60.0       # как часто пробовать отложенные кадры (провайдер ожил?)
    recover_scan_s: float = 30.0          # как часто подбирать «pending»-кадры, записанные мимо очереди
    provider_status_ttl_s: float = 30.0   # кэш готовности провайдеров (ready() локальной VLM ходит в сеть)
    max_upload_mb: int = 1024
    max_image_mp: float = 50.0            # больше — JPEG уменьшается при чтении, прочее отклоняется (бомбы PNG)
    zip_max_unpacked_mb: int = 4096       # распакованный объём zip (zip-бомба)
    preview_width: int = 640
    video_max_frames: int = 3000
    zip_max_members: int = 20000
    default_interval_min: int = 20
    demo_dir: str = "datasets/demo"
    tmp_dir: str = "var/tmp"              # временные файлы загрузок (удаляются после задания)

    def path(self, value: str) -> Path:
        """Путь из настройки: относительный — от корня репозитория."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else BASE_DIR / p

    def export_env(self) -> None:
        """Прокинуть ключи моделей в окружение процесса.

        pydantic читает `.env`, но в `os.environ` его не кладёт, а
        core.vlm_client и реализации моделей читают именно окружение. Явно
        заданное окружение не перетираем.
        """
        pairs = {
            "ZAI_API_KEY": self.zai_api_key,
            "ZAI_BASE_URL": self.zai_base_url,
            "ZAI_MODEL": self.zai_model,
            "GLM_MODEL": self.zai_model,          # так имя модели читает core.vlm_client
            "VLM_BASE_URL": self.vlm_base_url,
            "VLM_MODEL": self.vlm_model,
            "VLM_API_KEY": self.vlm_api_key,
            "EQUIPMENT_WEIGHTS": str(self.path(self.equipment_weights)) if self.equipment_weights else "",
            "STAGE_CLIP_MODEL": self.stage_clip_model,
            "EQUIPMENT_THREADS": str(self.equipment_threads) if self.equipment_threads > 0 else "",
        }
        for key, value in pairs.items():
            if value and not os.environ.get(key):
                os.environ[key] = value


settings = Settings()

# Страховка OpenCV от «бомб» в форматах, чей заголовок не читает PIL: свой лимит пикселей
# OpenCV читает из окружения при первом декодировании (по умолчанию ~1 Гп = 3 ГБ памяти).
# Вдвое выше нашего лимита — JPEG, уменьшенный при чтении, проходит.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(int(settings.max_image_mp * 2_000_000)))
