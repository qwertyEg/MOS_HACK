"""`python -m app` — запуск сервиса (uvicorn, один процесс: очередь живёт в нём)."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="СтройВзор — веб-сервис мониторинга стройплощадки")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Один воркер намеренно: очередь кадров и кэш моделей живут в памяти процесса.
    # server_header=False: «server: uvicorn» лишний раз рассказывает, чем атаковать.
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level=args.log_level, workers=1,
                server_header=False)


if __name__ == "__main__":
    main()
