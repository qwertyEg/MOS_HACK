"""Мелочь про сеть, из-за которой всё ломается молча.

На машине разработки в окружении прописан HTTP_PROXY без исключения для
локальных адресов. `requests` его слушается, и запрос к камере на
127.0.0.1:9101 уходит в прокси, который про неё ничего не знает. Симптом —
«камера не отвечает» при живой и работающей камере.

Тот же капкан уже сработал на MinIO, см. `storage._s3_config`. Камеры и
приёмник кадров всегда рядом — в той же машине или в той же сети, — и через
прокси им ходить незачем ни при каких настройках.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

# Ключей три, и третий обязателен. `requests` сначала дописывает в словарь
# прокси из окружения через setdefault, и только потом выбрасывает ключи со
# значением None. Без ключа "all" из ALL_PROXY туда попадает адрес прокси,
# переживает чистку и уводит запрос в прокси — хотя http и https отключены.
# Проверено: с двумя ключами 503 от Squid, с тремя — 200 от камеры.
NO_PROXY = {"http": None, "https": None, "all": None}


def is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host or host in ("localhost", "host.docker.internal"):
        return True
    if host.endswith(".local") or "." not in host:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_unspecified


def proxies_for(url: str) -> dict | None:
    """`proxies=` для requests: пусто для локальных адресов, иначе как в окружении.

    Каждый раз новый словарь: requests правит переданный ему на месте, и
    общая константа после первого же запроса перестала бы быть константой.
    """
    return dict(NO_PROXY) if is_local(url) else None
