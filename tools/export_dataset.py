#!/usr/bin/env python3
"""Выгрузить размеченную технику как датасет YOLO для дообучения детектора модели А.

    python tools/export_dataset.py --site 9 --out var/datasets/rus_frame.zip
    python tools/export_dataset.py --site 6 --site 9 --scope all --dir var/datasets/ru
    python tools/export_dataset.py --all-sites --scope reviewed --val 0.2 --out ds.zip

Кадры — оригиналы из хранилища, рамки — с учётом ручных правок оператора
(класс рамки / машины, удалённые и дорисованные рамки). data.yaml — по 21 классу
словаря (reference/checklist.json), manifest.json — откуда кадр и какие рамки
поставил человек. scope=reviewed — только кадры, которые оператор правил или
подтвердил; all — все разобранные кадры (рамки модели как псевдоразметка).
То же из интерфейса: вкладка «Техника» → «Датасет», или GET /api/sites/{id}/dataset.zip.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app import db  # noqa: E402
from app.models import Site  # noqa: E402
from app.services import annotations  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--site", type=int, action="append", default=[], help="номер объекта (можно несколько)")
    p.add_argument("--all-sites", action="store_true", help="все объекты")
    p.add_argument("--scope", choices=annotations.EXPORT_SCOPES, default="reviewed")
    p.add_argument("--val", type=float, default=0.2, help="доля валидации (по суткам камеры)")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", help="куда записать zip")
    out.add_argument("--dir", help="распаковать в каталог (для yolo train data=<каталог>/data.yaml)")
    args = p.parse_args(argv)

    db.init_db()
    with db.session() as s:
        ids = list(s.scalars(select(Site.id))) if args.all_sites else args.site
    if not ids:
        print("укажите --site N или --all-sites", file=sys.stderr)
        return 2

    if args.out:
        target = Path(args.out)
    else:
        fd, name = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        target = Path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with db.session() as s, zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
            stats = annotations.export_dataset(s, ids, zf, scope=args.scope, val_ratio=args.val)
    except (LookupError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    if args.dir:
        Path(args.dir).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as zf:
            zf.extractall(args.dir)
        target.unlink()
        where = args.dir
    else:
        where = str(target)
    print(f"кадров {stats['frames']} (обучение {stats['train']}, валидация {stats['val']}), "
          f"рамок {stats['boxes']}, правленых вручную {stats['manual_boxes']}, "
          f"проверенных кадров {stats['reviewed_frames']} → {where}")
    for cls, n in sorted(stats["by_class"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
