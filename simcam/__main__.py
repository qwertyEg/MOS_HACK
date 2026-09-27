"""Запуск имитатора камеры.

    .venv/bin/python -m simcam --folder ~/data/for_test --port 9101 \
        --name "Камера 1 — юг" --interval 30

Несколько камер — несколько процессов на разных портах. Смотреть они могут
на одну и ту же площадку с разных сторон: в основном сервисе это две камеры
одного объекта, и каждая ведёт свою историю чек-листов.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import uvicorn

from simcam.camera import Config, app, configure


def parse_meta(pairs: list[str]) -> dict:
    """`--meta погода=снег` → {"погода": "снег"}. Значение, похожее на число,
    приводится к числу: иначе `высота=42` пришло бы строкой."""
    out = {}
    for item in pairs:
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        try:
            out[key] = int(value) if value.isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Имитатор камеры")
    ap.add_argument("--folder", required=True, type=Path,
                    help="папка с кадрами; метка времени берётся из имени файла")
    ap.add_argument("--name", default="Камера", help="как камера себя называет")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--interval", type=float, default=30.0,
                    help="пауза между кадрами, секунд (в жизни 1200)")
    ap.add_argument("--loop", action="store_true",
                    help="после последнего кадра начинать серию сначала")
    ap.add_argument("--start-date", default="",
                    help="сдвинуть серию так, чтобы первый кадр попал на эту "
                         "дату (ГГГГ-ММ-ДД); интервалы сохраняются")
    ap.add_argument("--meta", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                    help="метаданные, уходящие с каждым кадром; можно повторять")
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"папка не найдена: {folder}")
        return 1

    configure(Config(
        name=args.name, folder=folder, interval=args.interval, loop=args.loop,
        start_date=dt.date.fromisoformat(args.start_date) if args.start_date else None,
        meta=parse_meta(args.meta),
    ))

    from simcam.camera import _frames
    print(f"«{args.name}»: кадров {len(_frames)}, пауза {args.interval:g} с, "
          f"адрес http://{args.host}:{args.port}")
    if not _frames:
        print("внимание: в папке нет файлов с меткой времени вида "
              "doric_2006_11_23_12_30_21.jpg — слать будет нечего")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
