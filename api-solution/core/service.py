"""Сценарии сервиса без интерфейса: приём фото, разбор, план, отчёт по объекту.

UI (app.py) только вызывает эти функции — всё, что здесь, покрыто тестами.

Кадры одного объекта разбираются строго по дате съёмки: каждый получает
контекст по уже разобранным более ранним фото (core/site.py). Ключ кэша
включает этот контекст, поэтому фото, загруженное задним числом, делает
разборы следующих за ним кадров устаревшими — они попадают в «ожидают
разбора» и переразбираются с учётом новой истории.
"""

from datetime import datetime

from . import timeline
from .images import data_url, detect_date, prepare, sha256
from .plan import plan_between
from .site import ContextBuilder
from .vlm import VLMError


def ingest(storage, object_id, filename, data, taken_at=None, source="вручную"):
    """Сохраняет фото в объект. Дата: явная → EXIF → имя файла; без даты кадр не принимаем."""
    if taken_at is None:
        taken_at, source = detect_date(data, filename)
        if taken_at is None:
            raise ValueError(f"{filename}: дата съёмки не найдена ни в EXIF, ни в имени файла — укажите вручную")
    sha = sha256(data)
    storage.add_frame(object_id, sha, filename, taken_at.isoformat(timespec="seconds"), source, prepare(data))
    refresh_auto_plan(storage, object_id)
    return sha


def delete_frame(storage, object_id, frame_id):
    storage.delete_frame(frame_id)
    refresh_auto_plan(storage, object_id)


def refresh_auto_plan(storage, object_id):
    """Автоплан: период от первой до последней даты фото, этапы по типовым пропорциям.

    Пересчитывается при каждом изменении набора фото, пока план не задан
    вручную или файлом. Нужны фото минимум за два разных дня.
    """
    if storage.get_plan_source(object_id) != "auto":
        return
    days = sorted({datetime.fromisoformat(f["taken_at"]).date() for f in storage.list_frames(object_id)})
    storage.save_plan(object_id, plan_between(days[0], days[-1]) if len(days) >= 2 else {}, "auto")


def _walk(storage, analyzer, obj, run=False, on_progress=None):
    """Проход по кадрам объекта по дате с накоплением контекста.

    run=False — только собрать, что есть; run=True — разобрать недостающее.
    Разбор при текущих настройках модели; если его нет — последний любой,
    чтобы смена модели в UI не прятала результаты и не рвала контекст.
    """
    frames = storage.list_frames(obj["id"])
    ctx = ContextBuilder(analyzer.checklist, obj.get("floors_total"))
    out, spent, errors = [], 0.0, []
    # Кадр без разбора в середине серии: когда его разберут, история всех
    # следующих изменится — считаем их тоже ожидающими, чтобы счётчик в UI
    # не занижал предстоящие расходы.
    gap = False
    for i, f in enumerate(frames):
        a = storage.get_analysis(analyzer.cache_key(f["sha256"], ctx))
        current = a is not None and not gap
        if not current and run:
            if on_progress:
                on_progress(i, len(frames), f)
            path = f["image_path"]
            try:
                a, cached = analyzer.analyze(f["sha256"], lambda: data_url(open(path, "rb").read()), ctx)
                current = True
                if not cached:
                    spent += a["usage"]["cost_usd"]
            except VLMError as e:
                errors.append(f"{f['filename']}: {e}")
        if a is None:
            gap = gap or not run
            a = storage.latest_analysis(f["sha256"])
        out.append({**f, "analysis": a, "current": current})
        if a:
            ctx.add(a, datetime.fromisoformat(f["taken_at"]).date())
    return out, spent, errors


def frames_with_analysis(storage, analyzer, obj):
    return _walk(storage, analyzer, obj)[0]


def pending(storage, analyzer, obj):
    """Кадры без разбора при текущих настройках и текущей истории стройки."""
    return [f for f in frames_with_analysis(storage, analyzer, obj) if not f["current"]]


def analyze_frames(storage, analyzer, obj, on_progress=None):
    """Разбирает ожидающие кадры по порядку дат; сбой одного кадра не останавливает остальные.

    → (потрачено $, ошибки)
    """
    _, spent, errors = _walk(storage, analyzer, obj, run=True, on_progress=on_progress)
    return spent, errors


def report(checklist, storage, analyzer, obj):
    frames = frames_with_analysis(storage, analyzer, obj)
    plan = storage.get_plan(obj["id"])
    return frames, plan, timeline.build(checklist, frames, plan, obj["floors_total"])
