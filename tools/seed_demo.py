#!/usr/bin/env python3
"""Засеять демо-объекты из datasets/demo/sites/<объект>/ и обработать кадры.

    python tools/seed_demo.py                    # все объекты каталога
    python tools/seed_demo.py --only pit --replace
    python tools/seed_demo.py --no-process       # только БД и кадры, анализ — в сервисе

Формат каталога (site.json, камеры, калибровка, зоны, план, парк) описан в
app/services/demo.py. План без явных дат — демо-план под диапазон дат кадров
(core.plan.importer.demo_plan), парк — по нормам этапов (core.plan.norms).
То же самое делает кнопка «Засеять демо» (POST /api/demo/seed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.models import Site  # noqa: E402
from app.services import cli, demo, ingest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=settings.demo_dir, help="каталог демо-данных (в нём sites/)")
    p.add_argument("--only", action="append", help="имя каталога или объекта (можно несколько раз)")
    p.add_argument("--replace", action="store_true", help="пересоздать объект, если он уже есть")
    p.add_argument("--no-process", action="store_true")
    p.add_argument("--timeout", type=float)
    args = p.parse_args()

    cli.bootstrap()
    with db.session() as s:
        try:
            result = demo.seed(s, root=settings.path(args.root), only=args.only, replace=args.replace,
                               start_jobs=False)
        except demo.DemoError as exc:
            raise SystemExit(str(exc)) from None
    for w in result["warnings"]:
        print("·", w)
    job_ids = []
    for job in result["jobs"]:
        print(f"камера {job['camera_id']}: {len(job['files'])} файл(ов)")
        ingest.run_job(job["job_id"], job["camera_id"], job["files"], job["params"], submit=False)
        job_ids.append(job["job_id"])
    if not job_ids:
        print("нечего засевать")
        return 0
    jobs = cli.jobs_state(job_ids) if args.no_process else cli.process(job_ids, timeout=args.timeout)
    cli.print_jobs(jobs)
    with db.session() as s:
        for site_out in result["sites"]:
            site = s.get(Site, site_out["id"])
            rep = site.report or {}
            print(f"объект {site.id} «{site.name}»: вердикт {rep.get('verdict', '—')}, "
                  f"этап {rep.get('current_stage') or '—'}, отставание {rep.get('lag_days')} дн.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
