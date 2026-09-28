"""Хранилище файлов кадров: локальная папка или S3-совместимое (MinIO).

Кадры вынесены из БД намеренно (решение Дениса): приложение можно пересобрать,
снимки останутся. В БД — только ключи.

Изменения против ветки Дениса:
- проверка выхода за корень — `Path.resolve().is_relative_to`, а не
  `str.startswith`: префиксное сравнение пропускало соседний каталог
  `var/storage-evil/...`;
- браузер всегда получает `/media/{key}` — отдача идёт через приложение под
  авторизацией и для S3 тоже. Presigned-ссылки на MinIO ломались в Docker
  (адрес `localhost:9000` из контейнера и снаружи — разные машины) и открывали
  кадры без входа;
- хранилище создаётся лениво и переконфигурируется (`configure`) — для тестов.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import quote

from app.config import settings

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "minio", "host.docker.internal")


def check_key(key: str) -> str:
    """Ключ — относительный путь без `..`; иначе это попытка выйти за корень."""
    if not key or key.startswith(("/", "\\")) or "\\" in key or "\x00" in key:
        raise ValueError(f"недопустимый ключ хранилища: {key!r}")
    if any(part in ("..", "") for part in key.split("/")):
        raise ValueError(f"недопустимый ключ хранилища: {key!r}")
    return key


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def url(self, key: str) -> str:
        return "/media/" + quote(key)

    def ping(self) -> bool:
        return True


class LocalStorage(Storage):
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        p = (self._root / check_key(key)).resolve()
        if not p.is_relative_to(self._root):
            raise ValueError(f"ключ выходит за пределы хранилища: {key!r}")
        return p

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)          # атомарно: читатель не увидит полузаписанный кадр
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except ValueError:
            pass

    def ping(self) -> bool:
        return self._root.is_dir()


def _s3_config(endpoint: str):
    """Обход системного прокси для локального MinIO (грабли Дениса: 503 от прокси)."""
    from urllib.parse import urlparse

    from botocore.config import Config

    host = urlparse(endpoint).hostname or ""
    kwargs = {"signature_version": "s3v4", "retries": {"max_attempts": 2, "mode": "standard"}}
    if host in _LOCAL_HOSTS:
        kwargs["proxies"] = {}
    return Config(**kwargs)


class S3Storage(Storage):
    def __init__(self) -> None:
        import boto3

        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint or None,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            region_name=settings.s3_region,
            config=_s3_config(settings.s3_endpoint),
        )
        self._bucket_checked = False

    def _ensure_bucket(self) -> None:
        if self._bucket_checked:
            return
        from botocore.exceptions import ClientError
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)
        self._bucket_checked = True

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        self._ensure_bucket()
        self._client.upload_fileobj(io.BytesIO(data), self._bucket, check_key(key),
                                    ExtraArgs={"ContentType": content_type})
        return key

    def get(self, key: str) -> bytes:
        buf = io.BytesIO()
        self._client.download_fileobj(self._bucket, check_key(key), buf)
        return buf.getvalue()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(Bucket=self._bucket, Key=check_key(key))
            return True
        except (ClientError, ValueError):
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=check_key(key))

    def ping(self) -> bool:
        try:
            self._ensure_bucket()
            return True
        except Exception:  # noqa: BLE001 — health-проверка не должна падать
            return False


_impl: Storage | None = None


def configure(backend: str | None = None, root: str | Path | None = None) -> Storage:
    global _impl
    backend = backend or settings.storage_backend
    if backend == "s3":
        _impl = S3Storage()
    else:
        _impl = LocalStorage(Path(root) if root else settings.path(settings.local_storage_dir))
    return _impl


def get() -> Storage:
    """Хранилище создаётся при первом обращении: недоступный S3 не должен ронять импорт."""
    if _impl is None:
        configure()
    return _impl  # type: ignore[return-value]
