#!/usr/bin/env python3
"""Повторно разобрать все папки датасета для одного объекта.

Пример:
    .venv/bin/python tools/reprocess_site.py --site test_1 \
        --dataset data/raw_noon_images --replace-existing

Без `--replace-existing` скрипт только сверяет папки, камеры и число кадров.
Перезаписываются только кадры и чек-листы выбранного объекта; план, камеры и
другие объекты остаются нетронутыми.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.db import SessionLocal, init_db
from app.models import Camera, Frame, Site, StageTemplate
from app.pipeline import ingest
from app.pipeline.model_b import ModelB, active_profile
from app.seed import seed_templates


def _camera_folder(camera: Camera, dataset: Path) -> Path:
    # Имена test_1 содержат служебный префикс вида «cam_2 - doric».
    slug = camera.name.rsplit("-", 1)[-1].strip().casefold()
    return dataset / slug


def _frames_count(session, camera_id: int) -> int:
    return session.scalar(select(func.count(Frame.id)).where(
        Frame.camera_id == camera_id)) or 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="точное имя объекта")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="корневая папка с подпапкой на каждую камеру")
    parser.add_argument("--replace-existing", action="store_true",
                        help="удалить прежние кадры объекта и запустить новый разбор")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        parser.error(f"папка датасета не найдена: {dataset}")

    init_db()
    with SessionLocal() as session:
        site = session.scalar(select(Site).where(Site.name == args.site))
        if site is None:
            parser.error(f"объект не найден: {args.site}")
        cameras = list(site.cameras)
        if not cameras:
            parser.error(f"у объекта «{site.name}» нет камер")

        pairs = []
        for camera in cameras:
            folder = _camera_folder(camera, dataset)
            files = ingest.list_frames(folder) if folder.is_dir() else []
            if not files:
                parser.error(f"для камеры «{camera.name}» нет датированных кадров в {folder}")
            if camera.use_mask and (camera.state is None or not camera.state.mask_approved):
                parser.error(f"у камеры «{camera.name}» включена маска, но она не подтверждена")
            pairs.append((camera.id, camera.name, folder, len(files),
                          _frames_count(session, camera.id)))

        if len(site.stages) != 7:
            parser.error(f"ожидалось 7 этапов, в плане объекта {len(site.stages)}")
        for camera_id, name, folder, files, old in pairs:
            print(f"{name}: {files} файлов в {folder}; старых кадров в БД: {old}")

        if not args.replace_existing:
            print("Проверка завершена. Для замены старых результатов добавьте --replace-existing.")
            return 0

        profile = active_profile()
        if not ModelB(profile=profile).health():
            parser.error(f"модель Б недоступна по активному профилю ({profile.model or 'без имени'})")

        # Шаблоны читаются из CSV, а снимки вопросов обновляются только у выбранного объекта.
        seed_templates(session)
        for stage in site.stages:
            template = session.scalar(select(StageTemplate).where(
                StageTemplate.macro_stage_id == stage.macro_stage_id))
            if template is None:
                parser.error(f"нет шаблона чек-листа для этапа «{stage.title}»")
            stage.questions = template.questions or []
        site.sim_today = None

        camera_ids = [camera.id for camera in cameras]
        for camera_id in camera_ids:
            session.query(Frame).filter(Frame.camera_id == camera_id).delete(
                synchronize_session=False)
        session.commit()
        print(f"Удалены старые результаты только объекта «{site.name}». Начинаю полный разбор.")

    for camera_id, name, folder, total, _old in pairs:
        started = time.monotonic()
        last_report = [0.0]
        print(f"\nКамера «{name}»: {total} кадров")
        with SessionLocal() as session:
            camera = session.get(Camera, camera_id)

            def report(progress: ingest.Progress) -> None:
                now = time.monotonic()
                if progress.done != progress.total and now - last_report[0] < 10:
                    return
                last_report[0] = now
                print(f"  {progress.done}/{progress.total} ({progress.done / max(1, progress.total):.0%}), "
                      f"пропущено {progress.skipped}: {progress.message}", flush=True)

            result = ingest.run(session, camera, folder, model_b=True,
                                on_progress=report)
        print(f"Готово: {result.done} кадров, пропущено {result.skipped}, "
              f"{(time.monotonic() - started) / 60:.1f} мин.")

    print("\nПолный разбор объекта завершён.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
