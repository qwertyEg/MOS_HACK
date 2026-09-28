"""Подключение к БД: SQLAlchemy 2, SQLite по умолчанию, PostgreSQL по DATABASE_URL.

Движок создаётся лениво и может быть переконфигурирован (`configure`) —
тесты подменяют базу на временный файл, утилиты — на свою. Фоновые потоки
берут сессию через `session()` в момент работы, а не держат ссылку на фабрику
с импорта, иначе переконфигурация до них бы не доходила.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import BASE_DIR, settings
from app.models import Base

_engine: Engine | None = None
_factory: sessionmaker | None = None


def _normalize(url: str) -> str:
    """Относительный путь SQLite — от корня репозитория; каталог создаём."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url in ("sqlite://", "sqlite:///:memory:"):
        return url
    raw = url[len(prefix):]
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return prefix + str(path)


def configure(url: str | None = None) -> Engine:
    global _engine, _factory
    if _engine is not None:
        _engine.dispose()
    url = _normalize(url or settings.database_url)
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        # Очередь пишет из фоновых потоков, API читает из пула запросов:
        # SQLite это выдерживает при WAL и таймауте ожидания блокировки.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):
        in_memory = ":memory:" in url or url == "sqlite://"

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    _engine = engine
    _factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine


def engine() -> Engine:
    if _engine is None:
        configure()
    return _engine  # type: ignore[return-value]


def session() -> Session:
    if _factory is None:
        configure()
    return _factory()  # type: ignore[misc]


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    s = session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    Base.metadata.create_all(engine())


def get_session() -> Iterator[Session]:
    """Зависимость FastAPI."""
    s = session()
    try:
        yield s
    finally:
        s.close()
