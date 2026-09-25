"""Приём кадров с камеры в реальном времени.

Отличие от `ingest.run` не в алгоритме, а в том, кто владеет временем.
Прогон папки знает всю историю заранее и идёт по ней сам; здесь кадр
приходит, когда его прислали, и обработать его надо до следующего.

Поэтому три вещи устроены иначе.

**Состояние живёт между кадрами.** Прогон папки начинает с маски оператора и
проходит историю заново — так он остаётся повторяемым. Поток начать заново не
может: вчерашние кадры больше не придут. Маска, счётчики и окно дневных
медиан поднимаются из хранилища при первом кадре и дописываются дальше.

**Окно считается по календарным суткам, а не по кадрам.** Прогон папки
складывал в окно каждый кадр, и на датасете с одним кадром в день это то же
самое. В жизни кадр приходит раз в 20 минут, и окно «в десять кадров» было бы
окном в три часа: за это время не меняется ничего, а разность полуокон
измеряла бы только движение солнца. Кадры суток сворачиваются медианой, в
окно уходит она.

**Приём и разбор разведены.** HTTP-запрос камеры только кладёт кадр в очередь
и сразу отвечает: разбор занимает секунды (модель Б — почти десяток), а
камера не должна их ждать, иначе она начнёт отставать от собственного
расписания. Очередь на камеру одна и разбирается одним потоком — порядок
кадров важен, маска зависит от предыдущих.
"""

from __future__ import annotations

import datetime as dt
import io
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Camera, CameraState, Frame
from app.pipeline import ingest
from app.pipeline import mask as M
from app.storage import storage

QUEUE_LIMIT = 200          # кадров в очереди, дальше камере отвечаем «занято»
IDLE_OFFLINE_SEC = 180     # столько без кадров — камера считается замолчавшей


@dataclass
class LiveStatus:
    """Что происходит с камерой прямо сейчас. Живёт в памяти процесса."""
    camera_id: int
    received: int = 0        # принято запросов
    processed: int = 0       # разобрано кадров
    skipped: int = 0         # брак и нечитаемые
    duplicates: int = 0      # кадр с такой меткой уже был
    checklists: int = 0      # заполнено чек-листов моделью Б
    last_capture: dt.datetime | None = None
    last_touch: float = field(default_factory=time.monotonic)
    last_error: str = ""
    message: str = ""

    @property
    def pending(self) -> int:
        w = _workers.get(self.camera_id)
        return w.q.qsize() if w else 0

    @property
    def silent_for(self) -> float:
        return time.monotonic() - self.last_touch

    @property
    def online(self) -> bool:
        return self.received > 0 and self.silent_for < IDLE_OFFLINE_SEC


_workers: dict[int, "_Worker"] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# состояние маски между кадрами
# ---------------------------------------------------------------------------

def resume_state(cam_state: CameraState) -> M.MaskState | None:
    """Маска в том виде, в каком её оставил прошлый кадр.

    Именно текущая, а не исходная: поток непрерывен, и начать с нарисованной
    оператором значило бы каждый раз возвращать зданию уже отвоёванное.
    """
    if not cam_state.background_key or not cam_state.work_w:
        return None
    bg = ingest._png_to_mask(storage.get(cam_state.background_key))
    st = M.MaskState(shape=(cam_state.work_h, cam_state.work_w), background=bg)
    st.windows = cam_state.windows_accumulated

    # Исходная площадь нужна для «сколько маски цело»: без неё отсчёт пошёл
    # бы от текущей, и доля всегда была бы 100%.
    if cam_state.initial_mask_key:
        st.initial_area = int(ingest._png_to_mask(
            storage.get(cam_state.initial_mask_key)).sum())

    if cam_state.evidence_key:
        try:
            with np.load(io.BytesIO(storage.get(cam_state.evidence_key))) as z:
                st.evidence = z["evidence"]
                st.hot_count = z["hot_count"]
        except Exception:
            pass          # счётчики восстановимы наблюдением, маска — нет
    return st


def _ring_key(camera_id: int) -> str:
    return f"cam/{camera_id}/ring.npz"


def load_ring(camera_id: int) -> list[np.ndarray]:
    """Окно дневных медиан. Без него после перезапуска маска встала бы на
    десять суток — ровно столько копится полное окно."""
    try:
        with np.load(io.BytesIO(storage.get(_ring_key(camera_id)))) as z:
            days = z["days"]
            return [days[i] for i in range(len(days))]
    except Exception:
        return []


def save_ring(camera_id: int, ring: list[np.ndarray]) -> None:
    if not ring:
        return
    buf = io.BytesIO()
    np.savez_compressed(buf, days=np.stack(ring).astype(np.float32))
    storage.put(_ring_key(camera_id), buf.getvalue(), "application/octet-stream")


# ---------------------------------------------------------------------------
# рабочий поток камеры
# ---------------------------------------------------------------------------

class _Worker(threading.Thread):
    def __init__(self, camera_id: int) -> None:
        super().__init__(name=f"live-cam-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_LIMIT)
        self.status = LiveStatus(camera_id=camera_id)

        self._st: M.MaskState | None = None
        self._ring: list[np.ndarray] = []
        self._day: dt.date | None = None
        self._buf: list[np.ndarray] = []      # подготовленные кадры текущих суток
        self._last_vlm: dt.datetime | None = None

    # --- приём -------------------------------------------------------------

    def submit(self, data: bytes, when: dt.datetime, meta: dict) -> bool:
        try:
            self.q.put_nowait((data, when, meta))
        except queue.Full:
            return False
        self.status.received += 1
        self.status.last_touch = time.monotonic()
        return True

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                with SessionLocal() as session:
                    self._handle(session, *item)
            except Exception as exc:                        # noqa: BLE001
                # Кадр не должен уносить с собой поток: следующий может быть
                # в порядке, а камера о нашей беде не узнает.
                self.status.last_error = f"{exc.__class__.__name__}: {exc}"
            finally:
                self.q.task_done()

    # --- разбор одного кадра ----------------------------------------------

    def _handle(self, session: Session, data: bytes, when: dt.datetime,
                meta: dict) -> None:
        cam = session.get(Camera, self.camera_id)
        if cam is None:
            return
        cam_state = cam.state
        if cam_state is None or not cam_state.mask_approved:
            self.status.last_error = "маска не задана — кадр принят, но не разобран"
            return

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.status.skipped += 1
            return

        exists = session.scalar(
            select(Frame.id).where(Frame.camera_id == cam.id,
                                   Frame.captured_at == when))
        if exists:
            self.status.duplicates += 1
            return

        if self._st is None:
            self._st = resume_state(cam_state)
            if self._st is None:
                self.status.last_error = "состояние маски не найдено"
                return
            self._ring = load_ring(cam.id)

        st = self._st
        ok, night, reason = ingest.assess_quality(frame)
        h, w = frame.shape[:2]

        change_pct = self._advance_mask(cam.id, st, frame, when, ok and not night)

        row = Frame(camera_id=cam.id, captured_at=when, object_key="",
                    width=w, height=h, is_night=night, quality_ok=ok,
                    reject_reason=reason, meta=meta or {})

        key = f"cam/{cam.id}/frames/{when:%Y%m%d_%H%M%S}"
        row.object_key = storage.put(f"{key}_orig.jpg", ingest._encode(frame))
        row.masked_key = storage.put(f"{key}_masked.jpg",
                                     ingest._encode(M.render_masked(frame, st)))
        row.overlay_key = storage.put(f"{key}_overlay.jpg",
                                      ingest._encode(M.render_overlay(frame, st)))
        row.masked_ratio = st.masked_ratio
        row.retained = st.retained
        row.top_edge_px = st.top_edge()
        row.change_pct = change_pct
        session.add(row)
        session.flush()

        if settings.live_model_b and ok and not night:
            self._ask_model_b(session, cam, row, frame, st, when)

        ingest.save_state(session, cam_state, st)
        cam.last_seen_at = when
        session.commit()

        self.status.processed += 1
        self.status.last_capture = when
        self.status.last_touch = time.monotonic()
        self.status.message = (f"{when:%d.%m.%Y %H:%M}  маска {st.masked_ratio:.1%}  "
                               f"цела {st.retained:.0%}")

    def _advance_mask(self, camera_id: int, st: M.MaskState, frame: np.ndarray,
                      when: dt.datetime, usable: bool) -> float:
        """Копит сутки, на смене суток двигает окно и сжимает маску.

        Возвращает долю изменившихся клеток в последнем посчитанном окне —
        то же число, что показывает прогон папки.
        """
        day = when.date()
        change_pct = 0.0

        if self._day is not None and day != self._day and self._buf:
            median = np.median(np.stack(self._buf), axis=0).astype(np.float32)
            self._buf = []
            self._ring.append(median)
            if len(self._ring) > M.WINDOW_DAYS:
                self._ring.pop(0)
            if len(self._ring) >= M.WINDOW_DAYS:
                change_pct = float((M.change_map(self._ring) > M.CHANGE_THRESHOLD).mean())
                M.update(st, self._ring)
            save_ring(camera_id, self._ring)

        if self._day is None or day != self._day:
            self._day = day
        if usable:
            self._buf.append(M.prepare(frame))
        return change_pct

    def _ask_model_b(self, session: Session, cam: Camera, row: Frame,
                     frame: np.ndarray, st: M.MaskState, when: dt.datetime) -> None:
        """Чек-листы по кадру — не чаще, чем раз в `live_model_b_hours` съёмки."""
        gap = settings.live_model_b_hours * 3600
        if self._last_vlm is not None and (when - self._last_vlm).total_seconds() < gap:
            return
        questions = ingest.site_questions(cam)
        if not questions:
            return
        try:
            h, w = frame.shape[:2]
            visible = M.to_full_res(M.visible_mask(st), w, h) if st.useful else None
            answered = ingest.ask_model_b(questions, frame, visible)
            self.status.checklists += ingest.save_checklists(session, cam, row,
                                                             answered)
            self._last_vlm = when
        except Exception as exc:                            # noqa: BLE001
            self.status.last_error = f"модель Б: {exc}"

    # --- сброс -------------------------------------------------------------

    def forget_state(self) -> None:
        self._st = None
        self._ring = []
        self._buf = []
        self._day = None
        self._last_vlm = None


# ---------------------------------------------------------------------------
# точки входа
# ---------------------------------------------------------------------------

def worker(camera_id: int) -> _Worker:
    with _lock:
        w = _workers.get(camera_id)
        if w is None or not w.is_alive():
            w = _Worker(camera_id)
            _workers[camera_id] = w
            w.start()
        return w


def submit(camera_id: int, data: bytes, when: dt.datetime, meta: dict) -> bool:
    """Кадр в очередь камеры. False — очередь переполнена."""
    return worker(camera_id).submit(data, when, meta)


def status(camera_id: int) -> LiveStatus | None:
    w = _workers.get(camera_id)
    return w.status if w else None


def reset(session: Session, camera: Camera) -> None:
    """Забыть всё, что камера наблюдала, и вернуть маску к нарисованной.

    Нужно ровно для повторного показа: имитатор шлёт ту же папку заново, и
    без сброса маска сжималась бы поверх уже сжатой — второй прогон съедал
    бы её вдвое, третий втрое, и результат зависел бы от числа показов.
    """
    from sqlalchemy import delete

    session.execute(delete(Frame).where(Frame.camera_id == camera.id))
    st = ingest.initial_state(camera.state)
    if st is not None:
        ingest.save_state(session, camera.state, st)
    session.commit()

    w = _workers.get(camera.id)
    if w is not None:
        w.forget_state()
        w.status = LiveStatus(camera_id=camera.id)
    try:
        storage.put(_ring_key(camera.id), b"", "application/octet-stream")
    except Exception:
        pass
