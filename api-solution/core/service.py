"""Сценарии сервиса без интерфейса: приём фото, разбор, отчёт по объекту.

UI (app.py) только вызывает эти функции — всё, что здесь, покрыто тестами.
"""

from . import timeline
from .glm import GLMError
from .images import data_url, detect_date, prepare, sha256


def ingest(storage, object_id, filename, data, taken_at=None, source="вручную"):
    """Сохраняет фото в объект. Дата: явная → EXIF → имя файла; без даты кадр не принимаем."""
    if taken_at is None:
        taken_at, source = detect_date(data, filename)
        if taken_at is None:
            raise ValueError(f"{filename}: дата съёмки не найдена ни в EXIF, ни в имени файла — укажите вручную")
    sha = sha256(data)
    storage.add_frame(object_id, sha, filename, taken_at.isoformat(timespec="seconds"), source, prepare(data))
    return sha


def frames_with_analysis(storage, analyzer, object_id):
    """Кадры объекта с разбором при текущих настройках, а если его нет — с последним любым."""
    out = []
    for f in storage.list_frames(object_id):
        a = storage.get_analysis(analyzer.cache_key(f["sha256"])) or storage.latest_analysis(f["sha256"])
        out.append({**f, "analysis": a})
    return out


def pending(storage, analyzer, object_id):
    return [f for f in storage.list_frames(object_id) if not storage.get_analysis(analyzer.cache_key(f["sha256"]))]


def analyze_frames(analyzer, frames, on_progress=None):
    """Разбирает кадры по одному; ошибка одного кадра не останавливает остальные. → (потрачено $, ошибки)."""
    spent, errors = 0.0, []
    for i, f in enumerate(frames):
        if on_progress:
            on_progress(i, len(frames), f)
        path = f["image_path"]
        try:
            result, cached = analyzer.analyze(f["sha256"], lambda: data_url(open(path, "rb").read()))
            if not cached:
                spent += result["usage"]["cost_usd"]
        except GLMError as e:
            errors.append(f"{f['filename']}: {e}")
    return spent, errors


def report(checklist, storage, analyzer, obj):
    frames = frames_with_analysis(storage, analyzer, obj["id"])
    plan = storage.get_plan(obj["id"])
    return frames, plan, timeline.build(checklist, frames, plan, obj["floors_total"])
