"""Засев демо-объектов из `datasets/demo/sites/<объект>/`.

Формат каталога объекта (всё, кроме снимков, необязательно):

    datasets/demo/sites/<slug>/
      site.json
      <камера>/*.jpg|*.png|*.mp4|*.avi   — кадры (метка времени в имени файла)

    site.json:
    {
      "name": "ЖК «Пример», корпус 1", "address": "...", "object_type": "Жильё",
      "floors_total": 17, "timezone": "Europe/Moscow", "shift_hours": 10,
      "plan": {"start": "2026-04-01", "end": "2026-10-31", "stage_ids": [1, 2, 3]}
              | [{"stage_id": 3, "planned_start": "...", "planned_end": "...", "equipment": {...}}],
      "fleet": {"excavator": 2, "dump_truck": 3},
      "cameras": [{
        "name": "Камера 1 — север", "dir": "cam1", "interval_min": 30,
        "start_at": "2026-05-01T08:00:00",            # для файлов/видео без даты
        "video_mode": "timelapse", "step_s": 10,
        "calibration": {"image_points": [[x, y], ...], "site_points": [[X, Y], ...]},
        "homography": [[...], [...], [...]],           # вместо calibration
        "zones": [{"name": "Котлован", "kind": "work", "polygon": [[x, y], ...]}]
      }]
    }

Без `cameras` камерой считается каждый подкаталог со снимками. Без `plan` —
демо-план под диапазон дат кадров (core.plan.importer.demo_plan), с явной
пометкой source="demo". Без `fleet` — парк по нормам этапов (core.plan.norms).
План создаётся до постановки кадров в очередь: моточасы списываются на этап,
идущий по плану в день кадра.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import auth
from app.config import settings
from app.models import Camera, PlanItem, Site, SiteFleet, Zone
from app.services import adapters, ingest, providers
from app.services.pipeline import compute_plan_hours
from app.services.sites import delete_site


class DemoError(ValueError):
    pass


def discover(root: Path) -> list[Path]:
    base = root / "sites"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def _camera_specs(site_dir: Path, cfg: dict) -> list[dict]:
    if cfg.get("cameras"):
        return list(cfg["cameras"])
    specs = []
    for sub in sorted(p for p in site_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if ingest.list_media(sub):
            specs.append({"name": sub.name, "dir": sub.name})
    if not specs and ingest.list_media(site_dir):
        specs.append({"name": "Камера 1", "dir": "."})
    return specs


def _frame_range(files: list[Path]) -> tuple[dt.date, dt.date] | None:
    stamps = [ts for ts, _ in filter(None, (ingest.parse_timestamp(p.name) for p in files))]
    if not stamps:
        return None
    return min(stamps).date(), max(stamps).date()


def _plan(s: Session, site: Site, cfg: dict, span: tuple[dt.date, dt.date] | None, warnings: list[str]) -> None:
    spec = cfg.get("plan")
    items = []
    source = "demo"
    if isinstance(spec, list):
        from core.contracts import PlanItem as PlanItemC
        for raw in spec:
            items.append(PlanItemC(stage_id=int(raw["stage_id"]),
                                   planned_start=adapters._date(raw.get("planned_start")),
                                   planned_end=adapters._date(raw.get("planned_end")),
                                   name=raw.get("name", ""), work_codes=raw.get("work_codes", []),
                                   equipment=raw.get("equipment", {}), planned_hours=raw.get("planned_hours", {}),
                                   hours_manual=bool(raw.get("hours_manual"))))
        source = "manual"
    else:
        spec = spec or {}
        start = adapters._date(spec.get("start")) or (span[0] if span else dt.date.today())
        end = adapters._date(spec.get("end")) or (span[1] if span else None)
        importer = providers.optional_module("core.plan.importer")
        if importer is None:
            warnings.append(f"{site.name}: план не создан — модуль core.plan не подключён")
            return
        items = importer.demo_plan(start, end, stage_ids=spec.get("stage_ids"))
        warnings.append(f"{site.name}: демо-план {start}…{end or '—'} (сгенерирован под даты кадров)")
    for i, item in enumerate(items):
        s.add(adapters.plan_row_from_item(item, site.id, source, i))
    s.flush()


def _fleet(s: Session, site: Site, cfg: dict) -> None:
    fleet: dict[str, int] = {}
    if isinstance(cfg.get("fleet"), dict):
        fleet = {k: int(v) for k, v in cfg["fleet"].items()}
    else:
        norms = providers.optional_module("core.plan.norms")
        if norms is not None:
            for p in s.scalars(select(PlanItem).where(PlanItem.site_id == site.id)):
                for cls, n in (norms.default_equipment(p.stage_id) or {}).items():
                    fleet[cls] = max(fleet.get(cls, 0), int(n))
    for cls, n in fleet.items():
        s.add(SiteFleet(site_id=site.id, cls=cls, count=n))


def _calibrate(cam: Camera, spec: dict, warnings: list[str]) -> None:
    if spec.get("homography"):
        cam.homography = spec["homography"]
        return
    calib = spec.get("calibration")
    if not calib:
        return
    fusion = providers.optional_module("core.equipment.fusion")
    if fusion is None:
        warnings.append(f"{cam.name}: калибровка пропущена — модуль core.equipment не подключён")
        return
    H, err = fusion.homography_from_points(calib["image_points"], calib["site_points"])
    cam.homography = H
    cam.calib_points = {"image_points": calib["image_points"], "site_points": calib["site_points"],
                        "reproj_error": err}


def seed(s: Session, root: Path | None = None, only: list[str] | None = None, replace: bool = False,
         start_jobs: bool = True) -> dict[str, Any]:
    root = root or settings.path(settings.demo_dir)
    dirs = discover(root)
    if not dirs:
        raise DemoError(f"нет демо-данных: ожидается каталог {root}/sites/<объект>/ со снимками и site.json")
    result: dict[str, Any] = {"sites": [], "jobs": [], "warnings": []}
    warnings = result["warnings"]
    for site_dir in dirs:
        cfg_path = site_dir / "site.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        except ValueError as exc:
            warnings.append(f"{site_dir.name}: site.json не читается ({exc}) — пропущен")
            continue
        name = cfg.get("name") or site_dir.name
        if only and site_dir.name not in only and name not in only:
            continue
        existing = s.scalar(select(Site).where(Site.name == name))
        if existing is not None:
            if not replace:
                warnings.append(f"{name}: уже есть (id={existing.id}) — пропущен; replace=true пересоздаст")
                continue
            delete_site(s, existing.id)

        site = Site(name=name, address=cfg.get("address", ""), object_type=cfg.get("object_type", "Жильё"),
                    floors_total=cfg.get("floors_total"), timezone=cfg.get("timezone", "Europe/Moscow"),
                    shift_hours=float(cfg.get("shift_hours", 10.0)))
        s.add(site)
        s.flush()

        specs = _camera_specs(site_dir, cfg)
        cam_files: list[tuple[Camera, dict, list[Path]]] = []
        all_files: list[Path] = []
        for spec in specs:
            folder = (site_dir / spec.get("dir", spec.get("name", ""))).resolve()
            if not folder.is_relative_to(site_dir.resolve()) or not folder.is_dir():
                warnings.append(f"{name}: каталог камеры {spec.get('dir')} не найден")
                continue
            files = ingest.list_media(folder)
            if spec.get("video"):
                video = folder / spec["video"]
                files = [video] if video.exists() else files
            cam = Camera(site_id=site.id, name=spec.get("name") or folder.name, kind="folder",
                         interval_min=int(spec.get("interval_min", settings.default_interval_min)),
                         ingest_key=auth.new_ingest_key(), source_uri=str(folder))
            _calibrate(cam, spec, warnings)
            s.add(cam)
            s.flush()
            for z in spec.get("zones") or []:
                s.add(Zone(site_id=site.id, camera_id=cam.id, name=z.get("name", ""),
                           kind=z.get("kind", "work"), polygon=z.get("polygon", [])))
            cam_files.append((cam, spec, files))
            all_files.extend(files)

        _plan(s, site, cfg, _frame_range(all_files), warnings)
        _fleet(s, site, cfg)
        s.flush()
        warnings.extend(compute_plan_hours(s, site))
        s.commit()

        site_out = {"id": site.id, "name": site.name, "cameras": []}
        tz = adapters.site_tz(site)
        for cam, spec, files in cam_files:
            start_at = ingest.parse_datetime_input(spec.get("start_at"))
            params = ingest.UploadParams(
                interval_min=cam.interval_min,
                start_at=adapters.to_utc(start_at, tz) if start_at else None,
                video_mode=spec.get("video_mode", "realtime"), step_s=spec.get("step_s"),
                every_n=int(spec.get("every_n", 1)))
            job = ingest.new_job(s, "seed", cam, message=f"демо: {len(files)} файлов из {site_dir.name}")
            if start_jobs:
                ingest.start_job(job.id, cam.id, files, params)
            site_out["cameras"].append({"id": cam.id, "name": cam.name, "files": len(files), "job_id": job.id})
            result["jobs"].append({"job_id": job.id, "camera_id": cam.id, "params": params, "files": files})
        result["sites"].append(site_out)
    return result
