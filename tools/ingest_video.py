#!/usr/bin/env python3
"""Нарезать видео на кадры (OpenCV) и загрузить в камеру.

    # обычное видео: кадр раз в 10 с ролика, метки = начало + позиция в ролике
    python tools/ingest_video.py --camera 1 --video clip.mp4 --step-s 10 --start-at 2026-05-01T08:00

    # таймлапс-архив (каждый кадр ролика — отдельный снимок раз в 30 мин)
    python tools/ingest_video.py --camera 1 --video corinthian_raw_images.avi \\
        --video-mode timelapse --interval-min 30 --start-at 2006-01-01T12:00

Без --start-at берётся дата из имени файла, иначе время загрузки (честно
помечается в meta.ts_source кадра).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import cli, ingest  # noqa: E402
from tools.ingest_folder import add_common, params_from_args, resolve_camera  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--camera", type=int, help="id камеры")
    who.add_argument("--site", type=int, help="id объекта (камера создаётся, нужен --camera-name)")
    p.add_argument("--camera-name", default="Камера из видео")
    p.add_argument("--video", required=True)
    add_common(p)
    args = p.parse_args()

    video = Path(args.video)
    if not video.is_file() or ingest.classify(video.name) != "video":
        raise SystemExit(f"не видеофайл: {video} (поддерживаются {', '.join(sorted(ingest.VIDEO_EXT))})")
    args.folder = str(video.parent)
    cli.bootstrap()
    cam, site = resolve_camera(args)
    print(f"{video.name} → камера {cam.id} «{cam.name}», режим {args.video_mode}")
    job_id = cli.ingest_files(cam.id, [video], params_from_args(args, site))
    jobs = cli.jobs_state([job_id]) if args.no_process else cli.process([job_id], timeout=args.timeout)
    cli.print_jobs(jobs)
    return 0 if jobs and jobs[0]["state"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
