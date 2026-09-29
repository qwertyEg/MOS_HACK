"""Фоновая очередь обработки кадров — в процессе, поток на камеру.

Почему так (решение Дениса, сохранено): порядок кадров камеры важен — трекер
модели А сравнивает кадр с предыдущим, маска копит историю. Поэтому у каждой
камеры свой поток и очередь с приоритетом по времени съёмки: пачка из zip в
любом порядке всё равно разбирается хронологически.

Отличия от `app/pipeline/live.py`:
- в очереди — только id кадра; сам кадр уже лежит в БД и хранилище, так что
  ответ 202 камере означает «сохранено», а не «в памяти до первого рестарта»;
- после рестарта незавершённые кадры подбираются по статусу/флагам
  processed_* (`recover`), отложенные — когда провайдер оживёт;
- ошибка кадра помечает кадр, а не роняет поток;
- пересчёт площадки — с дебаунсом (`Recomputer`), а не на каждом кадре.
"""
from __future__ import annotations

import datetime as dt
import itertools
import logging
import queue as _queue
import threading
import time

from sqlalchemy import select

from app import db
from app.config import settings
from app.models import Camera, Frame
from app.services import pipeline
from app.services import settings as settings_svc
from app.services.providers import registry

log = logging.getLogger(__name__)

_STOP = object()
_seq = itertools.count()


class _CameraWorker(threading.Thread):
    def __init__(self, camera_id: int) -> None:
        super().__init__(name=f"cam-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.q: _queue.PriorityQueue = _queue.PriorityQueue()
        self.queued: set[int] = set()
        self.lock = threading.Lock()
        self.busy = False
        self.processed = 0
        self.errors = 0
        self.last_error = ""
        self.stopping = False

    def submit(self, frame_id: int, when: dt.datetime | None) -> bool:
        with self.lock:
            if frame_id in self.queued:
                return False
            self.queued.add(frame_id)
        ts = when.timestamp() if when else time.time()
        self.q.put((ts, next(_seq), frame_id))
        return True

    def pending(self) -> int:
        with self.lock:
            return len(self.queued)

    def cancel(self) -> None:
        """Выбросить всё, что ещё не начато (переанализ)."""
        with self.lock:
            self.queued.clear()
        while True:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except _queue.Empty:
                break

    def run(self) -> None:
        while True:
            _ts, _n, item = self.q.get()
            try:
                if item is _STOP:
                    return
                with self.lock:
                    if item not in self.queued:
                        continue          # отменён
                    self.queued.discard(item)
                    self.busy = True
                self._process(item)
            finally:
                self.busy = False
                self.q.task_done()

    def _process(self, frame_id: int) -> None:
        if not pipeline.claim(frame_id):
            return                        # уже обработан или занят другим процессом
        try:
            pipeline.process_frame(frame_id)
            self.processed += 1
        except Exception as exc:  # noqa: BLE001 — кадр не должен уносить поток
            self.errors += 1
            self.last_error = f"кадр {frame_id}: {type(exc).__name__}: {exc}"
            log.exception("кадр %s не обработан", frame_id)
            _mark_error(frame_id, exc)


def _mark_error(frame_id: int, exc: Exception) -> None:
    try:
        with db.session() as s:
            fr = s.get(Frame, frame_id)
            if fr is not None:
                fr.status = "error"
                fr.note = f"{type(exc).__name__}: {exc}"[:2000]
                s.commit()
    except Exception:  # noqa: BLE001
        log.exception("не удалось пометить кадр %s ошибкой", frame_id)


class Recomputer:
    """Пересчёт площадки с дебаунсом: пачка кадров → один пересчёт."""

    def __init__(self) -> None:
        self._timers: dict[int, threading.Timer] = {}
        self._running: set[int] = set()
        self._lock = threading.Lock()

    def request(self, site_id: int) -> None:
        delay = settings.recompute_debounce_s
        if delay <= 0:
            self.run(site_id)
            return
        with self._lock:
            if site_id in self._timers:
                return
            t = threading.Timer(delay, self._fire, args=(site_id,))
            t.daemon = True
            self._timers[site_id] = t
            t.start()

    def _fire(self, site_id: int) -> None:
        with self._lock:
            self._timers.pop(site_id, None)
        self.run(site_id)

    def run(self, site_id: int) -> dict | None:
        with self._lock:
            self._running.add(site_id)
        try:
            return pipeline.recompute_site(site_id)
        except Exception:  # noqa: BLE001
            log.exception("пересчёт площадки %s упал", site_id)
            return None
        finally:
            with self._lock:
                self._running.discard(site_id)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._timers or self._running)

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()


class Replayer:
    """Перепрогон модели А по сохранённым рамкам после ручной правки (pipeline.replay_model_a).

    Правки на кадре идут пачкой (сменил класс, удалил рамку, дорисовал) — перепрогон
    один, через паузу после последней; правка во время перепрогона — ещё один после
    него. Склейку машин на небольшом объекте API перепрогоняет сразу (`run`), чтобы
    оператор увидел итог в ответе. После перепрогона — пересчёт аналитики площадки.
    """

    def __init__(self) -> None:
        self._timers: dict[int, threading.Timer] = {}
        self._running: dict[int, dict] = {}
        self._again: set[int] = set()
        self._last: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._site_locks: dict[int, threading.Lock] = {}

    def _site_lock(self, site_id: int) -> threading.Lock:
        with self._lock:
            return self._site_locks.setdefault(site_id, threading.Lock())

    def request(self, site_id: int) -> None:
        delay = settings.recompute_debounce_s
        if delay <= 0:
            self.run(site_id)
            return
        with self._lock:
            if site_id in self._running:
                self._again.add(site_id)
                return
            old = self._timers.pop(site_id, None)
            if old is not None:
                old.cancel()
            t = threading.Timer(min(delay, 1.5), self._fire, args=(site_id,))
            t.daemon = True
            self._timers[site_id] = t
            t.start()

    def _fire(self, site_id: int) -> None:
        with self._lock:
            self._timers.pop(site_id, None)
        self.run(site_id)

    def run(self, site_id: int) -> dict | None:
        with self._site_lock(site_id):
            with self._lock:
                t = self._timers.pop(site_id, None)
                if t is not None:
                    t.cancel()
                self._running[site_id] = {"done": 0, "total": 0,
                                          "started_at": dt.datetime.now(dt.UTC).isoformat()}

            def progress(done: int, total: int) -> None:
                with self._lock:
                    self._running[site_id] = {**self._running.get(site_id, {}), "done": done, "total": total}

            res = None
            try:
                res = pipeline.replay_model_a(site_id, progress=progress)
                self._last[site_id] = {**res, "error": None, "finished_at": dt.datetime.now(dt.UTC).isoformat()}
            except Exception as exc:  # noqa: BLE001 — перепрогон не должен ронять поток API
                log.exception("перепрогон модели А площадки %s упал", site_id)
                self._last[site_id] = {"error": f"{type(exc).__name__}: {exc}",
                                       "finished_at": dt.datetime.now(dt.UTC).isoformat()}
            finally:
                with self._lock:
                    self._running.pop(site_id, None)
                    again = site_id in self._again
                    self._again.discard(site_id)
        if again:
            self.request(site_id)
        else:
            recomputer.request(site_id)
        return res

    def state(self, site_id: int) -> dict:
        with self._lock:
            if site_id in self._running:
                return {"state": "running", **self._running[site_id], "last": self._last.get(site_id)}
            if site_id in self._timers:
                return {"state": "queued", "last": self._last.get(site_id)}
            return {"state": "idle", "last": self._last.get(site_id)}

    def busy(self) -> bool:
        with self._lock:
            return bool(self._timers or self._running)

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()


class FrameQueue:
    def __init__(self) -> None:
        self._workers: dict[int, _CameraWorker] = {}
        self._lock = threading.Lock()
        self._supervisor: threading.Thread | None = None
        self._stop = threading.Event()
        self.started = False

    # --- приём -----------------------------------------------------------

    def _worker(self, camera_id: int) -> _CameraWorker:
        with self._lock:
            w = self._workers.get(camera_id)
            if w is None or not w.is_alive():
                w = _CameraWorker(camera_id)
                self._workers[camera_id] = w
                w.start()
            return w

    def submit(self, camera_id: int, frame_id: int, when: dt.datetime | None = None) -> bool:
        """Поставить кадр в очередь камеры. Без запущенной очереди кадр остаётся
        `pending` в БД и будет подобран `recover` при старте."""
        if not self.started:
            return False
        return self._worker(camera_id).submit(frame_id, when)

    def pending(self, camera_id: int | None = None) -> int:
        with self._lock:
            workers = list(self._workers.values()) if camera_id is None else \
                [w for cid, w in self._workers.items() if cid == camera_id]
        return sum(w.pending() + (1 if w.busy else 0) for w in workers)

    def status(self) -> dict:
        with self._lock:
            workers = dict(self._workers)
        return {
            "running": self.started,
            "pending": sum(w.pending() for w in workers.values()),
            "busy": sum(1 for w in workers.values() if w.busy),
            "recompute_pending": recomputer.busy(),
            "cameras": {cid: {"pending": w.pending(), "busy": w.busy, "processed": w.processed,
                              "errors": w.errors, "last_error": w.last_error}
                        for cid, w in workers.items()},
        }

    def cancel(self, camera_ids: list[int]) -> None:
        with self._lock:
            workers = [self._workers[c] for c in camera_ids if c in self._workers]
        for w in workers:
            w.cancel()

    def wait_cameras(self, camera_ids: list[int], timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self.pending(cid) == 0 for cid in camera_ids):
                return True
            time.sleep(0.05)
        return False

    # --- восстановление --------------------------------------------------

    def recover(self, include_postponed: bool = False) -> int:
        """Поставить в очередь незавершённые кадры из БД (рестарт, запись мимо
        очереди утилитой, ожившие провайдеры для отложенных)."""
        if not self.started:
            return 0
        statuses = ["pending"]
        need_a = need_b = True
        if include_postponed:
            with db.session() as s:
                state = settings_svc.get_state(s)
            ready_a = registry.ready("detector", state["model_a"])[0]
            ready_b = registry.ready("classifier", state["model_b"])[0]
            if ready_a or ready_b:
                statuses.append("postponed")
                need_a, need_b = ready_a, ready_b
        count = 0
        with db.session() as s:
            rows = s.execute(select(Frame.id, Frame.camera_id, Frame.captured_at, Frame.status,
                                    Frame.processed_a, Frame.processed_b)
                             .where(Frame.status.in_(statuses)).order_by(Frame.captured_at)).all()
        for fid, cam_id, when, status, pa, pb in rows:
            if status == "postponed" and not ((not pa and need_a) or (not pb and need_b)):
                continue
            if self.submit(cam_id, fid, when):
                count += 1
        return count

    def reset_stale(self) -> int:
        """Кадры, застрявшие в `processing` после падения процесса, — снова в работу."""
        with db.session() as s:
            res = s.execute(Frame.__table__.update().where(Frame.status == "processing")
                            .values(status="pending"))
            s.commit()
            return res.rowcount or 0

    def _supervise(self) -> None:
        # Отложенные кадры проверяем сразу, но в этом потоке: проверка готовности
        # может грузить веса или ходить в сеть — старт сервиса её не ждёт.
        try:
            self.recover(include_postponed=True)
        except Exception:  # noqa: BLE001
            log.exception("восстановление отложенных кадров")
        last_scan = last_retry = time.monotonic()
        while not self._stop.wait(1.0):
            now = time.monotonic()
            try:
                if now - last_retry >= settings.postponed_retry_s:
                    last_retry = last_scan = now
                    self.recover(include_postponed=True)
                elif now - last_scan >= settings.recover_scan_s:
                    last_scan = now
                    self.recover()
            except Exception:  # noqa: BLE001
                log.exception("надзор очереди")

    # --- жизненный цикл --------------------------------------------------

    def start(self, supervise: bool = True) -> None:
        if self.started:
            return
        self.started = True
        self._stop.clear()
        self.reset_stale()
        self.recover()
        if supervise:
            self._supervisor = threading.Thread(target=self._supervise, name="queue-supervisor", daemon=True)
            self._supervisor.start()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Дождаться, пока очередь, задания загрузки и пересчёты затихнут (тесты, утилиты)."""
        from app.services.ingest import jobs_busy

        deadline = time.monotonic() + timeout
        quiet_since = None
        while time.monotonic() < deadline:
            idle = self.pending() == 0 and not jobs_busy() and not recomputer.busy() and not replayer.busy()
            if idle:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since > 0.15:
                    return True
            else:
                quiet_since = None
            time.sleep(0.03)
        return False

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.cancel()
            w.q.put((float("-inf"), next(_seq), _STOP))
        for w in workers:
            w.join(timeout)
        recomputer.cancel_all()
        replayer.cancel_all()
        self.started = False


def camera_ids_for_site(site_id: int) -> list[int]:
    with db.session() as s:
        return list(s.scalars(select(Camera.id).where(Camera.site_id == site_id)))


recomputer = Recomputer()
replayer = Replayer()
frame_queue = FrameQueue()
