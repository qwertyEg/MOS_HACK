"""Хранилище кадров.

Кадры вынесены из БД и из контейнера приложения намеренно: приложение можно
пересобрать, снимки останутся. Основной бэкенд — S3-совместимый (MinIO),
чтобы организатор мог подставить свой S3 сменой одной переменной окружения.

Локальный бэкенд оставлен страховкой на случай, если MinIO не поднимется
в момент демонстрации: переключается флагом, код вызова не меняется.
"""

from __future__ import annotations

import io
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import settings


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def url(self, key: str, expires: int = 3600) -> str:
        """Ссылка для браузера. У S3 — presigned, отдаётся минуя бэкенд."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                "minio", "host.docker.internal")


def _s3_config(endpoint: str):
    """Конфиг клиента с обходом системного прокси для локальных адресов.

    Иначе запросы к MinIO на localhost уходят в HTTP_PROXY и возвращают 503.
    На машине разработки прокси прописан без исключения для localhost, и
    симптом выглядит как «хранилище недоступно», хотя оно работает.
    """
    from urllib.parse import urlparse

    from botocore.config import Config

    host = urlparse(endpoint).hostname or ""
    kwargs = {"signature_version": "s3v4",
              "retries": {"max_attempts": 2, "mode": "standard"}}
    if host in _LOCAL_HOSTS:
        kwargs["proxies"] = {}
    return Config(**kwargs)


class S3Storage(Storage):
    def __init__(self) -> None:
        import boto3

        def client(endpoint: str):
            return boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
                config=_s3_config(endpoint),
            )

        self._client = client(settings.s3_endpoint)
        self._bucket = settings.s3_bucket
        self._ensure_bucket()

        # Presigned-ссылки должны указывать на адрес, доступный браузеру,
        # а он может отличаться от внутреннего адреса контейнера.
        public = settings.s3_public_endpoint or settings.s3_endpoint
        self._public_client = (client(public) if public != settings.s3_endpoint
                               else self._client)

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        self._client.upload_fileobj(
            io.BytesIO(data), self._bucket, key,
            ExtraArgs={"ContentType": content_type},
        )
        return key

    def get(self, key: str) -> bytes:
        buf = io.BytesIO()
        self._client.download_fileobj(self._bucket, key, buf)
        return buf.getvalue()

    def url(self, key: str, expires: int = 3600) -> str:
        return self._public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires,
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False


class LocalStorage(Storage):
    """Фолбэк. Отдаёт файлы через собственный роут приложения."""

    def __init__(self) -> None:
        self._root = Path(settings.local_storage_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self._root / key).resolve()
        if not str(p).startswith(str(self._root.resolve())):
            raise ValueError(f"ключ выходит за пределы хранилища: {key}")
        return p

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def put_file(self, key: str, src: Path) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def url(self, key: str, expires: int = 3600) -> str:
        return f"/media/{key}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


def make_storage() -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage()
    return S3Storage()


class _LazyStorage:
    """Хранилище создаётся при первом обращении, а не при импорте.

    Иначе недоступный MinIO валит импорт любого модуля, который его
    упоминает, — включая запуск тестов и утилит, которым хранилище
    вообще не нужно.
    """

    _impl: Storage | None = None

    def _get(self) -> Storage:
        if self._impl is None:
            self._impl = make_storage()
        return self._impl

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


storage: Storage = _LazyStorage()  # type: ignore[assignment]
