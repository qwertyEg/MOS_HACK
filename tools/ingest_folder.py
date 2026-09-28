#!/usr/bin/env python3
"""Загрузить папку снимков (и видео/zip в ней) в камеру и обработать.

    python tools/ingest_folder.py --camera 1 --folder data/doric
    python tools/ingest_folder.py --site 1 --camera-name "Камера 2" --folder data/ionic --recursive
    python tools/ingest_folder.py --camera 1 --folder photos --start-at 2026-05-01T08:00 --interval-min 30
    python tools/ingest_folder.py --camera 1 --folder data/doric --no-process   # обработает сервис

Метка времени кадра — из имени файла (`*_YYYY_MM_DD_HH_MM_SS` и похожие), EXIF,
иначе start_at + i·interval (см. app/services/ingest.py). Повторный запуск по
той же папке ничего не дублирует (sha256).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import auth, db  # noqa: E402
from app.models import Camera, Site  # noqa: E402
from app.services import adapters, cli, ingest  # noqa: E402


def params_from_args(args, site: Site) -> ingest.UploadParams:
    start = ingest.parse_datetime_input(args.start_at)
    return ingest.UploadParams(interval_min=args.interval_min,
                               start_at=adapters.to_utc(start, adapters.site_tz(site)) if start else None,
                               video_mode=args.video_mode, step_s=args.step_s, every_n=args.every_n)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--interval-min", type=int, default=20, help="шаг съёмки для файлов без даты / timelapse")
    p.add_argument("--start-at", help="время первого кадра без даты (ISO, время площадки)")
    p.add_argument("--video-mode", choices=ingest.VIDEO_MODES, default="realtime")
    p.add_argument("--step-s", type=float, help="realtime-видео: шаг выборки, секунд")
    p.add_argument("--every-n", type=int, default=1, help="timelapse-видео: брать каждый N-й кадр")
    p.add_argument("--no-process", action="store_true", help="только сохранить кадры, анализ — в сервисе")
    p.add_argument("--timeout", type=float, help="ждать обработку не дольше, секунд")


def resolve_camera(args) -> tuple[Camera, Site]:
    with db.session() as s:
        if args.camera:
            cam = s.get(Camera, args.camera)
            if cam is None:
                raise SystemExit(f"камера {args.camera} не найдена")
        else:
            site = s.get(Site, args.site)
            if site is None:
                raise SystemExit(f"объект {args.site} не найден")
            cam = Camera(site_id=site.id, name=args.camera_name, kind="folder", interval_min=args.interval_min,
                         ingest_key=auth.new_ingest_key(), source_uri=str(Path(args.folder).resolve()))
            s.add(cam)
            s.commit()
            print(f"создана камера {cam.id} «{cam.name}»")
        return cam, s.get(Site, cam.site_id)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--camera", type=int, help="id камеры")
    who.add_argument("--site", type=int, help="id объекта (камера создаётся, нужен --camera-name)")
    p.add_argument("--camera-name", default="Камера из папки")
    p.add_argument("--folder", required=True)
    p.add_argument("--recursive", action="store_true")
    add_common(p)
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"папка не найдена: {folder}")
    files = ingest.list_media(folder, recursive=args.recursive)
    if not files:
        raise SystemExit(f"в {folder} нет изображений, видео или zip")
    cli.bootstrap()
    cam, site = resolve_camera(args)
    print(f"{len(files)} файл(ов) → камера {cam.id} «{cam.name}»")
    job_id = cli.ingest_files(cam.id, files, params_from_args(args, site))
    jobs = cli.jobs_state([job_id]) if args.no_process else cli.process([job_id], timeout=args.timeout)
    cli.print_jobs(jobs)
    return 0 if jobs and jobs[0]["state"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
