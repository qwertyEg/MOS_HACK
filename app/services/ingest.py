"""Приём кадров: изображения, zip, видео, поток камеры → строки `frames` + файлы.

Главное правило (§4): кадр сохраняется **всегда**, независимо от готовности
моделей; анализ — потом, в очереди. Поэтому здесь нет ни одного импорта из
`core.*`: только декодирование, метка времени, дедупликация, превью.

Метка времени кадра, по убыванию надёжности:
1. имя файла с датой и временем (`doric_2006_03_08_12_30_22.jpg`,
   `IMG_20240314_103000.jpg`, `2024-03-14T10-30-00.png` …);
2. EXIF DateTimeOriginal (во вложенном Exif IFD — грабли, найденные Никитой);
3. имя файла только с датой → полдень этого дня;
4. `start_at + i·interval_min` для файлов без даты (порядок — по имени);
5. без `start_at` — время загрузки (последний файл = «сейчас»), источник
   честно записывается в `meta.ts_source`, а не выдаётся за дату съёмки
   (у Никиты фото без даты молча получали now()).
Наивное время трактуется как время площадки (`sites.timezone`) и хранится в UTC.

Дубликаты: одинаковый sha256 в той же камере — пропуск. Разные снимки с одной
меткой (например, два файла с датой без времени) — при загрузке метка
сдвигается на секунды с пометкой `meta.ts_adjusted_s`; в потоке камеры повтор
метки считается повтором кадра (камера переслала после сбоя сети).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import re
import shutil
import threading
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import db, storage
from app.config import settings
from app.models import Camera, Frame, Job, Site, utcnow
from app.services.adapters import site_tz, to_utc

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
ZIP_EXT = {".zip"}
SUPPORTED_EXT = IMAGE_EXT | VIDEO_EXT | ZIP_EXT
VIDEO_MODES = ("realtime", "timelapse")

_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                  ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}

# Порядок важен: сначала форматы со временем, иначе «2024-03-14_10-30-00»
# распознался бы как одна дата.
_WITH_TIME = [
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[ T_-](\d{2})[-_.:h](\d{2})[-_.:m](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[ T_-]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[ T_-](\d{2})[-_.:h](\d{2})(?!\d)"),
]
_DATE_ONLY = [
    (re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)"), "ymd"),
]
_EXIF_DATE_TAGS = (36867, 36868, 306)   # DateTimeOriginal, DateTimeDigitized, DateTime


def _plausible(value: dt.datetime) -> bool:
    return 1990 <= value.year <= 2100


def parse_timestamp(name: str) -> tuple[dt.datetime, bool] | None:
    """Метка из имени файла: (наивное время, есть ли время суток)."""
    stem = Path(name).name
    for rx in _WITH_TIME:
        for mt in rx.finditer(stem):
            g = [int(x) for x in mt.groups()]
            try:
                value = dt.datetime(*g)
            except ValueError:
                continue
            if _plausible(value):
                return value, True
    for rx, kind in _DATE_ONLY:
        for mt in rx.finditer(stem):
            g = [int(x) for x in mt.groups()]
            try:
                value = dt.datetime(g[0], g[1], g[2], 12) if kind == "ymd" else dt.datetime(g[2], g[1], g[0], 12)
            except ValueError:
                continue
            if _plausible(value):
                return value, False
    return None


def exif_timestamp(data: bytes) -> dt.datetime | None:
    try:
        from PIL import ExifTags, Image
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:  # noqa: BLE001 — не картинка или битый EXIF: даты просто нет
        return None
    sources = [exif]
    try:
        sources.append(exif.get_ifd(ExifTags.IFD.Exif))
    except Exception:  # noqa: BLE001
        pass
    for tag in _EXIF_DATE_TAGS:
        for src in sources:
            value = src.get(tag)
            if not value:
                continue
            try:
                parsed = dt.datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
            if _plausible(parsed):
                return parsed
    return None


def parse_datetime_input(value: str | None) -> dt.datetime | None:
    """Время из формы/API (ISO 8601). ValueError — невалидный ввод."""
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"не удалось разобрать время «{value}» — нужен ISO 8601, например 2026-05-01T08:00") from exc


# --------------------------------------------------------------------------
# кадр → хранилище + БД
# --------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("не удалось закодировать JPEG")
    return buf.tobytes()


def make_preview(img: np.ndarray, width: int) -> bytes:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return encode_jpeg(img, 82)


@dataclass
class SaveResult:
    status: str                      # saved | duplicate | error
    frame: Frame | None = None
    reason: str = ""


def save_frame(s: Session, cam: Camera, data: bytes, captured_at: dt.datetime, meta: dict,
               ext: str = ".jpg", job_id: str | None = None, on_collision: str = "bump",
               image: np.ndarray | None = None) -> SaveResult:
    """Сохранить один кадр. `captured_at` — aware. `on_collision`: bump | skip."""
    if not data:
        return SaveResult("error", reason="пустой файл")
    sha = hashlib.sha256(data).hexdigest()
    dup = s.scalar(select(Frame).where(Frame.camera_id == cam.id, Frame.sha256 == sha).limit(1))
    if dup is not None:
        return SaveResult("duplicate", dup, "такой снимок уже загружен в эту камеру")
    img = image if image is not None else decode_image(data)
    if img is None:
        return SaveResult("error", reason="не удалось прочитать изображение (формат не поддерживается или файл повреждён)")

    when = captured_at.astimezone(dt.UTC).replace(microsecond=0)
    shift = 0
    while s.scalar(select(Frame.id).where(Frame.camera_id == cam.id, Frame.captured_at == when)) is not None:
        if on_collision == "skip":
            existing = s.scalar(select(Frame).where(Frame.camera_id == cam.id, Frame.captured_at == when))
            return SaveResult("duplicate", existing, "кадр с такой меткой времени уже принят")
        when += dt.timedelta(seconds=1)
        shift += 1
        if shift > 3600:
            return SaveResult("error", reason="не удалось подобрать свободную метку времени")
    meta = dict(meta or {})
    if shift:
        meta["ts_adjusted_s"] = shift

    h, w = img.shape[:2]
    ext = ext.lower() if ext.lower() in _CONTENT_TYPES else ".jpg"
    base = f"frames/{cam.id}/{when:%Y/%m/%d/%H%M%S}_{sha[:10]}"
    st = storage.get()
    key = st.put(base + ext, data, _CONTENT_TYPES.get(ext, "image/jpeg"))
    preview_key = st.put(base + "_p.jpg", make_preview(img, settings.preview_width))

    fr = Frame(camera_id=cam.id, captured_at=when, key=key, preview_key=preview_key, sha256=sha,
               width=w, height=h, meta=meta, status="pending", job_id=job_id)
    s.add(fr)
    if cam.last_frame_at is None or when > cam.last_frame_at:
        cam.last_frame_at = when
    if not cam.image_w or not cam.image_h:
        cam.image_w, cam.image_h = w, h
    try:
        s.commit()
    except IntegrityError:
        # Та же метка успела прийти параллельно (поток камеры + загрузка).
        s.rollback()
        if on_collision == "skip":
            return SaveResult("duplicate", None, "кадр с такой меткой времени уже принят")
        return save_frame(s, s.get(Camera, cam.id), data, when + dt.timedelta(seconds=1), meta, ext,
                          job_id, on_collision, img)
    return SaveResult("saved", fr)


def _submit(fr: Frame) -> None:
    from app.services.queue import frame_queue
    frame_queue.submit(fr.camera_id, fr.id, fr.captured_at)


# --------------------------------------------------------------------------
# поток камеры (/api/ingest)
# --------------------------------------------------------------------------

def ingest_stream_frame(s: Session, cam: Camera, data: bytes, captured_at: dt.datetime | None,
                        meta: dict) -> SaveResult:
    site = s.get(Site, cam.site_id)
    when = to_utc(captured_at, site_tz(site)) if captured_at else dt.datetime.now(dt.UTC)
    meta = {**(meta or {}), "ts_source": "камера" if captured_at else "время приёма"}
    res = save_frame(s, cam, data, when, meta, on_collision="skip")
    if res.status == "saved":
        _submit(res.frame)
    return res


# --------------------------------------------------------------------------
# задания загрузки: файлы, zip, видео
# --------------------------------------------------------------------------

@dataclass
class UploadParams:
    interval_min: int
    start_at: dt.datetime | None = None      # aware UTC; None — см. правила меток в шапке
    video_mode: str = "realtime"             # realtime: шаг по времени видео; timelapse: кадр = снимок
    step_s: float | None = None              # realtime: шаг выборки, с (по умолчанию interval_min·60)
    every_n: int = 1                         # timelapse: брать каждый N-й кадр видео


@dataclass
class _Image:
    name: str
    read: Callable[[], bytes]
    ts: dt.datetime | None = None            # aware UTC
    source: str = ""


@dataclass
class _Video:
    name: str
    path: Path


@dataclass
class _Plan:
    images: list[_Image] = field(default_factory=list)
    videos: list[_Video] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def classify(name: str) -> str | None:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in ZIP_EXT:
        return "zip"
    return None


def _expand(paths: list[Path], tmp_dir: Path, closers: list) -> _Plan:
    plan = _Plan()
    for path in paths:
        kind = classify(path.name)
        if kind == "image":
            plan.images.append(_Image(path.name, path.read_bytes))
        elif kind == "video":
            plan.videos.append(_Video(path.name, path))
        elif kind == "zip":
            try:
                zf = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                plan.errors.append(f"{path.name}: повреждённый zip")
                continue
            closers.append(zf)
            members = [i for i in zf.infolist() if not i.is_dir()
                       and not Path(i.filename).name.startswith(".") and "__MACOSX" not in i.filename]
            if len(members) > settings.zip_max_members:
                plan.errors.append(f"{path.name}: слишком много файлов ({len(members)} > {settings.zip_max_members})")
                continue
            for info in sorted(members, key=lambda i: i.filename):
                sub = classify(info.filename)
                if sub == "image":
                    plan.images.append(_Image(Path(info.filename).name,
                                              lambda zf=zf, n=info.filename: zf.read(n)))
                elif sub == "video":
                    target = tmp_dir / f"{uuid.uuid4().hex}{Path(info.filename).suffix.lower()}"
                    with zf.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    plan.videos.append(_Video(Path(info.filename).name, target))
                else:
                    plan.errors.append(f"{path.name}/{info.filename}: формат не поддерживается, пропущен")
        else:
            plan.errors.append(f"{path.name}: формат не поддерживается, пропущен")
    return plan


def _assign_image_times(images: list[_Image], params: UploadParams, tz: dt.tzinfo) -> None:
    """Метки изображениям по правилам из шапки; без даты — start_at + i·interval."""
    undated: list[_Image] = []
    for img in images:
        parsed = parse_timestamp(img.name)
        if parsed and parsed[1]:
            img.ts, img.source = to_utc(parsed[0], tz), "имя файла"
            continue
        exif = exif_timestamp(img.read())
        if exif:
            img.ts, img.source = to_utc(exif, tz), "exif"
        elif parsed:
            img.ts, img.source = to_utc(parsed[0], tz), "имя файла (только дата, время 12:00)"
        else:
            undated.append(img)
    step = dt.timedelta(minutes=params.interval_min)
    if params.start_at is not None:
        for i, img in enumerate(undated):
            img.ts, img.source = params.start_at + i * step, "start_at + i·interval"
    else:
        now = dt.datetime.now(dt.UTC).replace(microsecond=0)
        for i, img in enumerate(undated):
            img.ts = now - (len(undated) - 1 - i) * step
            img.source = "время загрузки (дата не найдена)"


def iter_video(path: Path, params: UploadParams, base: dt.datetime | None
               ) -> Iterator[tuple[np.ndarray, dt.datetime, dict]]:
    """Кадры видео с метками. realtime — шаг по времени ролика, метка = начало +
    позиция; timelapse — каждый (N-й) кадр ролика = отдельный снимок, метка =
    начало + i·interval (таймлапсы архивов вроде corinthian_raw_images.avi)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("не удалось открыть видео (кодек не поддерживается OpenCV?)")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        timelapse = params.video_mode == "timelapse"
        step_frames = max(1, int(params.every_n)) if timelapse else \
            max(1, round((params.step_s or params.interval_min * 60) * fps))
        interval = dt.timedelta(minutes=params.interval_min)
        if base is None:
            expected = max(1, count // step_frames) if count else 1
            span = (expected - 1) * interval if timelapse else dt.timedelta(seconds=count / fps)
            base = dt.datetime.now(dt.UTC).replace(microsecond=0) - span
        idx = taken = 0
        while taken < settings.video_max_frames:
            if idx % step_frames == 0:
                ok, img = cap.read()
                if not ok or img is None:
                    break
                ts = base + taken * interval if timelapse else base + dt.timedelta(seconds=idx / fps)
                yield img, ts, {"video_frame": idx, "video_pos_s": round(idx / fps, 3)}
                taken += 1
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def new_job(s: Session, kind: str, camera: Camera | None = None, site_id: int | None = None,
            message: str = "") -> Job:
    job = Job(id=uuid.uuid4().hex, kind=kind, camera_id=camera.id if camera else None,
              site_id=camera.site_id if camera else site_id, state="ingesting", message=message)
    s.add(job)
    s.commit()
    return job


def _job_error(job: Job, text: str) -> None:
    errors = list(job.errors or [])
    if len(errors) < 200:
        errors.append(text)
    job.errors = errors


def run_job(job_id: str, camera_id: int, paths: list[Path], params: UploadParams,
            cleanup_dir: Path | None = None, submit: bool = True) -> None:
    """Выполнить задание загрузки (в потоке задания или синхронно из утилит)."""
    closers: list = []
    tmp_dir = settings.path(settings.tmp_dir) / f"job-{job_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    s = db.session()
    try:
        job = s.get(Job, job_id)
        cam = s.get(Camera, camera_id)
        if job is None or cam is None:
            return
        site = s.get(Site, cam.site_id)
        tz = site_tz(site)
        plan = _expand(paths, tmp_dir, closers)
        for err in plan.errors:
            _job_error(job, err)
            job.skipped += 1
        s.commit()

        def store(data: bytes, when: dt.datetime, meta: dict, ext: str, image=None) -> None:
            res = save_frame(s, cam, data, when, meta, ext=ext, job_id=job_id, image=image)
            if res.status == "saved":
                job.total += 1
                if submit:
                    _submit(res.frame)
            elif res.status == "duplicate":
                job.duplicates += 1
            else:
                job.skipped += 1
                _job_error(job, f"{meta.get('source_file', '?')}: {res.reason}")
            s.commit()

        _assign_image_times(plan.images, params, tz)
        # Порядок сохранения = порядок съёмки: очередь камеры обрабатывает кадры
        # по времени, и трекер модели А сравнивает кадр с предыдущим.
        for img in sorted(plan.images, key=lambda i: (i.ts, i.name)):
            try:
                data = img.read()
            except Exception as exc:  # noqa: BLE001 — один битый файл не рвёт задание
                job.skipped += 1
                _job_error(job, f"{img.name}: не прочитан ({exc})")
                continue
            store(data, img.ts, {"source_file": img.name, "ts_source": img.source}, Path(img.name).suffix)

        for vid in plan.videos:
            parsed = parse_timestamp(vid.name)
            if params.start_at is not None:
                base, source = params.start_at, "видео: start_at + позиция"
            elif parsed:
                base, source = to_utc(parsed[0], tz), "видео: дата из имени + позиция"
            else:
                base, source = None, "видео: время загрузки + позиция"
            try:
                for frame_img, when, extra in iter_video(vid.path, params, base):
                    meta = {"source_file": vid.name, "ts_source": source, "video_mode": params.video_mode, **extra}
                    store(encode_jpeg(frame_img), when, meta, ".jpg", image=frame_img)
            except ValueError as exc:
                job.skipped += 1
                _job_error(job, f"{vid.name}: {exc}")
                s.commit()

        job.state = "queued"
        job.finished_at = utcnow()
        if not job.total and not job.duplicates:
            job.message = job.message or "ни одного кадра не сохранено — см. errors"
        s.commit()
    except Exception as exc:  # noqa: BLE001 — задание падает целиком, но сервис живёт
        log.exception("задание %s упало", job_id)
        s.rollback()
        job = s.get(Job, job_id)
        if job is not None:
            job.state = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.finished_at = utcnow()
            s.commit()
    finally:
        s.close()
        for zf in closers:
            zf.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def start_job(job_id: str, camera_id: int, paths: list[Path], params: UploadParams,
              cleanup_dir: Path | None = None) -> threading.Thread:
    t = threading.Thread(target=run_job, args=(job_id, camera_id, paths, params, cleanup_dir),
                         name=f"job-{job_id[:8]}", daemon=True)
    _jobs_running.add(t)
    t.start()
    return t


_jobs_running: set[threading.Thread] = set()


def jobs_busy() -> bool:
    for t in list(_jobs_running):
        if not t.is_alive():
            _jobs_running.discard(t)
    return bool(_jobs_running)


def list_media(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and not p.name.startswith(".") and classify(p.name))
