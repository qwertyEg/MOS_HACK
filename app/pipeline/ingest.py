"""Приём кадров: папка со снимками → обработанные кадры в БД и хранилище.

Модель Б здесь не вызывается. Задача этого шага — прогнать историю камеры
через маску и сложить результат так, чтобы качество работы алгоритма можно
было оценить глазами, кадр за кадром.

По каждому кадру сохраняется три картинки:
    original  — как пришло с камеры
    masked    — фон погашен, ровно то, что ушло бы в модель Б
    overlay   — маска залита красным, чтобы видеть её границы
"""

from __future__ import annotations

import datetime as dt
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Camera, CameraState, Frame
from app.pipeline import mask as M
from app.storage import storage

# doric_2006_11_23_12_30_21.jpg → 2006-11-23 12:30:21
STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PREVIEW_WIDTH = 900       # ширина сохраняемых превью
NIGHT_SATURATION = 12.0   # ИК-режим даёт почти монохром


@dataclass(slots=True)
class Progress:
    total: int
    done: int = 0
    skipped: int = 0
    message: str = ""


def parse_stamp(name: str) -> dt.datetime | None:
    m = STAMP.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.UTC)
    except ValueError:
        return None


def list_frames(folder: Path) -> list[tuple[Path, dt.datetime]]:
    """Кадры папки, отсортированные по времени съёмки."""
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        when = parse_stamp(p.name)
        if when is None:
            continue
        out.append((p, when))
    out.sort(key=lambda t: t[1])
    return out


def assess_quality(img: np.ndarray) -> tuple[bool, bool, str]:
    """Годен ли кадр и ночной ли он. Возвращает (годен, ночной, причина)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < 18:
        return False, True, "темно"
    if mean > 243:
        return False, False, "засвет"
    if float(gray.std()) < 8:
        return False, False, "нет контраста"

    # Ночной режим камеры почти монохромен: насыщенность около нуля.
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    night = sat < NIGHT_SATURATION or mean < 55
    return True, night, ""


def _encode(img: np.ndarray, width: int = PREVIEW_WIDTH) -> bytes:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _mask_to_png(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    return buf.tobytes() if ok else b""


def _png_to_mask(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr > 127


def save_state(session: Session, cam_state: CameraState, st: M.MaskState,
               initial: bool = False) -> None:
    """Пишет состояние маски в хранилище и обновляет строку в БД."""
    prefix = f"cam/{cam_state.camera_id}"
    cam_state.background_key = storage.put(
        f"{prefix}/background.png", _mask_to_png(st.background), "image/png")
    if initial:
        cam_state.initial_mask_key = storage.put(
            f"{prefix}/initial.png", _mask_to_png(st.background), "image/png")

    # Счётчики лежат одним архивом. hot_count терять нельзя: по нему
    # выбираются опорные клетки освещения, и без него первые окна после
    # перезапуска мерили бы засветку по чему попало.
    buf = io.BytesIO()
    np.savez_compressed(buf, evidence=st.evidence, hot_count=st.hot_count)
    cam_state.evidence_key = storage.put(
        f"{prefix}/counters.npz", buf.getvalue(), "application/octet-stream")

    cam_state.work_h, cam_state.work_w = st.shape
    cam_state.windows_accumulated = st.windows
    cam_state.masked_ratio = st.masked_ratio
    cam_state.retained = st.retained
    cam_state.mask_top_edge_px = st.top_edge()
    session.flush()


def initial_state(cam_state: CameraState) -> M.MaskState | None:
    """Маска ровно такой, какой её нарисовал оператор, со сброшенными счётчиками."""
    if not cam_state.initial_mask_key or not cam_state.work_w:
        return None
    bg = _png_to_mask(storage.get(cam_state.initial_mask_key))
    return M.MaskState(shape=(cam_state.work_h, cam_state.work_w), background=bg)


def work_shape(img: np.ndarray) -> tuple[int, int]:
    return M.prepare(img).shape


def run(
    session: Session,
    camera: Camera,
    folder: Path,
    limit: int = 0,
    threshold: float = M.CHANGE_THRESHOLD,
    lock_windows: int = M.LOCK_WINDOWS,
    window_days: int = M.WINDOW_DAYS,
    on_progress: Callable[[Progress], None] | None = None,
) -> Progress:
    """Прогон истории камеры через маску.

    Маска должна быть уже нарисована оператором и подтверждена — без неё
    сжимать нечего, а начинать с пустой значит вернуться к автоматическому
    варианту, который на этих данных не работает.
    """
    cam_state = camera.state
    if cam_state is None or not cam_state.mask_approved:
        raise ValueError("маска не нарисована или не подтверждена")

    # Прогон считает папку целиком и стирает прежние кадры, поэтому и маска
    # начинается заново — с той, что нарисовал оператор. Продолжить от текущей
    # значило бы сжимать уже сжатое: второй прогон по тем же кадрам съедал бы
    # маску вдвое, третий втрое, и результат зависел бы от числа запусков.
    st = initial_state(cam_state)
    if st is None:
        raise ValueError("исходная маска не найдена — нарисуйте её заново")

    items = list_frames(folder)
    if limit:
        items = items[:limit]
    prog = Progress(total=len(items))

    # Старые кадры этой камеры убираем: прогон повторяемый.
    session.execute(delete(Frame).where(Frame.camera_id == camera.id))
    session.flush()

    ring: list[np.ndarray] = []
    prev_change = 0.0

    for path, when in items:
        frame = cv2.imread(str(path))
        if frame is None:
            prog.skipped += 1
            continue

        ok, night, reason = assess_quality(frame)
        h, w = frame.shape[:2]

        row = Frame(camera_id=camera.id, captured_at=when, object_key="",
                    width=w, height=h, is_night=night,
                    quality_ok=ok, reject_reason=reason)

        # Ночные кадры и брак в расчёт маски не идут: ИК-режим ломает
        # сравнение яркостей. Но сам кадр сохраняем — он нужен модели А.
        if ok and not night:
            ring.append(M.daily_median([frame]))
            if len(ring) > window_days:
                ring.pop(0)
            if len(ring) >= window_days:
                ch = M.change_map(ring)
                prev_change = float((ch > threshold).mean())
                M.update(st, ring, threshold=threshold, lock_windows=lock_windows)

        key = f"cam/{camera.id}/frames/{when:%Y%m%d_%H%M%S}"
        row.object_key = storage.put(f"{key}_orig.jpg", _encode(frame))
        row.masked_key = storage.put(f"{key}_masked.jpg",
                                     _encode(M.render_masked(frame, st)))
        row.overlay_key = storage.put(f"{key}_overlay.jpg",
                                      _encode(M.render_overlay(frame, st)))
        row.masked_ratio = st.masked_ratio
        row.retained = st.retained
        row.top_edge_px = st.top_edge()
        row.change_pct = prev_change

        session.add(row)
        prog.done += 1
        # Сброс в базу реже, чем доклад о прогрессе: flush стоит заметно
        # дороже, а полоса должна двигаться плавно.
        if prog.done % 20 == 0:
            session.flush()
        if on_progress and prog.done % 5 == 0:
            prog.message = (f"{when:%Y-%m-%d}  маска {st.masked_ratio:.1%}  "
                            f"цела {st.retained:.0%}")
            on_progress(prog)

    save_state(session, cam_state, st)
    session.commit()
    if on_progress:
        prog.message = "готово"
        on_progress(prog)
    return prog


def iter_folders(root: Path) -> Iterator[Path]:
    """Подпапки, похожие на набор кадров одной камеры."""
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXT for f in p.iterdir()):
            yield p
