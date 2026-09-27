"""Подключение к БД. Схема создаётся из моделей — отдельного файла миграций нет."""

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# Колонки, добавленные к уже существующим таблицам. `create_all` их не
# заметит: он создаёт недостающие таблицы целиком, а таблицу, которая уже
# есть, не трогает вовсе. Отдельный инструмент миграций на хакатон — лишний
# сервис ради десяти строк, а вот молча работать со старой схемой нельзя:
# симптомом будет «колонка не найдена» посреди приёма кадров.
DRIFT = [
    "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS ingest_key varchar(64) DEFAULT ''",
    "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS last_seen_at timestamptz",
    "ALTER TABLE frames ADD COLUMN IF NOT EXISTS meta jsonb DEFAULT '{}'::jsonb",
    "ALTER TABLE macro_stages ADD COLUMN IF NOT EXISTS object_types varchar[] DEFAULT '{}'",
    "ALTER TABLE site_stages ADD COLUMN IF NOT EXISTS questions jsonb DEFAULT '[]'::jsonb",
]


def ensure_columns() -> None:
    with engine.begin() as conn:
        for stmt in DRIFT:
            conn.execute(text(stmt))


def init_db() -> None:
    Base.metadata.create_all(engine)
    ensure_columns()


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
