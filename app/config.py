"""Конфигурация сервиса. Всё внешнее подключается переменными окружения,
чтобы организатор мог подставить свои эндпоинты, ничего не правя в коде.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- база ---
    database_url: str = "postgresql+psycopg://stroyka:stroyka@localhost:5432/stroyka"

    # --- хранилище кадров ---
    # Кадры намеренно вынесены из БД и из контейнера приложения: приложение
    # можно пересобрать, снимки останутся. s3 → MinIO или любой S3-совместимый.
    storage_backend: str = "s3"          # s3 | local
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "frames"
    s3_public_endpoint: str = ""         # для presigned-ссылок в браузер
    local_storage_dir: str = "./var/frames"

    # --- модель Б (VLM), внешний сервис ---
    # 11435 — Bearer-прокси проекта; сам Ollama слушает только на loopback:11436.
    vlm_base_url: str = "http://localhost:11435/v1"
    vlm_api_key: str = ""
    vlm_model: str = "qwen3-vl:30b-a3b-instruct"
    vlm_timeout: int = 120
    vlm_temperature: float = 0.0
    # Бюджет большой не потому, что нужен длинный ответ — ответ односложный.
    # Qwen3-VL рассуждает перед ответом всегда: ни `think: false`, ни
    # `enable_thinking: false`, ни `reasoning_effort: none` рассуждение не
    # выключают, оно просто уходит в отдельное поле. Если бюджета не хватит,
    # он весь уйдёт на рассуждение, а сам ответ окажется пустым.
    vlm_max_tokens: int = 512

    # --- модель А (детектор), пока заглушка ---
    model_a_url: str = ""                # пусто → используется заглушка

    # --- конвейер ---
    frame_interval_min: int = 20         # подтверждено организаторами
    vlm_calls_per_hour: int = 1          # модель Б не на каждом кадре
    mask_window_days: int = 10           # окно медианы для динамической маски
    mask_lock_windows: int = 3           # окон подряд до фиксации точки
    mask_max_growth_pct: float = 8.0     # потолок прироста компоненты за окно
    parked_after_days: int = 2           # техника → PARKED

    # --- приём кадров с камер ---
    # Адрес, по которому камера достучится до этого сервиса. Камера живёт
    # отдельным процессом (а в жизни — на другой машине), и «localhost» для
    # неё не обязательно мы. Этот адрес мы сообщаем ей при подключении.
    public_base_url: str = "http://localhost:8000"
    # Разбирать ли приходящие кадры моделью Б. Выключается, когда нужно
    # посмотреть только на работу маски и не ждать VLM.
    live_model_b: bool = True
    # Сколько времени съёмки должно пройти между двумя вызовами модели Б.
    # Считается по метке кадра, а не по часам: при ускоренном прогоне сутки
    # съёмки проходят за минуту, и по настенным часам вызов был бы один.
    live_model_b_hours: int = 1

    # --- веб ---
    # Корень, глубже которого не пускает выбор папки с кадрами. Пусто →
    # домашний каталог пользователя, под которым запущен сервис.
    fs_browse_root: str = ""
    secret_key: str = "dev-secret-change-me"
    admin_login: str = "admin"
    admin_password: str = "admin"


settings = Settings()
