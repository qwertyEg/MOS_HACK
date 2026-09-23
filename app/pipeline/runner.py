"""Запуск прогона камеры в фоне, с состоянием для прогресс-бара.

Прогон идёт минуты: 300+ кадров, на каждый — чтение, маска и три картинки в
хранилище. Держать на нём HTTP-запрос нельзя, поэтому работа уходит в поток,
а страница опрашивает состояние.

Поток, а не отдельный процесс и не очередь задач: работа упирается в диск и
в OpenCV, который отпускает GIL, а очередь задач — это ещё один сервис
разворачивать ради одной фоновой операции. Состояние живёт в памяти и умирает
вместе с ним — это осознанно: прогон повторяемый, потерянный прогресс
восстанавливается повторным запуском, а не хранением.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.db import SessionLocal
from app.models import Camera
from app.pipeline import ingest


@dataclass
class RunStatus:
    camera_id: int
    total: int = 0
    done: int = 0
    skipped: int = 0
    message: str = ""
    error: str = ""
    finished: bool = False
    started_at: float = field(default_factory=time.monotonic)

    @property
    def percent(self) -> int:
        if not self.total:
            return 0
        return min(100, int(self.done * 100 / self.total))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def eta(self) -> float | None:
        """Сколько осталось, по средней скорости. None, пока не на чем считать."""
        if self.done < 5 or self.finished:
            return None
        rate = self.done / max(self.elapsed, 1e-6)
        return max(0.0, (self.total - self.done) / rate) if rate else None


_runs: dict[int, RunStatus] = {}
_lock = threading.Lock()


def status(camera_id: int) -> RunStatus | None:
    return _runs.get(camera_id)


def active(camera_id: int) -> bool:
    st = _runs.get(camera_id)
    return st is not None and not st.finished


def forget(camera_id: int) -> None:
    _runs.pop(camera_id, None)


def start(camera_id: int, limit: int = 0) -> RunStatus:
    """Ставит прогон в работу. Повторный запуск поверх идущего игнорируется."""
    with _lock:
        running = _runs.get(camera_id)
        if running is not None and not running.finished:
            return running
        st = RunStatus(camera_id=camera_id)
        _runs[camera_id] = st

    def worker() -> None:
        try:
            with SessionLocal() as session:
                cam = session.get(Camera, camera_id)
                if cam is None:
                    raise ValueError("камера не найдена")
                folder = Path(cam.source_uri or "")
                if not folder.is_dir():
                    raise ValueError(f"папка с кадрами недоступна: {folder}")

                def report(p: ingest.Progress) -> None:
                    st.total = p.total
                    st.done = p.done
                    st.skipped = p.skipped
                    st.message = p.message

                ingest.run(session, cam, folder, limit=limit, on_progress=report)
        except Exception as exc:                      # noqa: BLE001
            # Причина нужна на странице: «ничего не произошло» без объяснения
            # хуже, чем текст ошибки.
            st.error = str(exc) or exc.__class__.__name__
        finally:
            st.finished = True

    threading.Thread(target=worker, name=f"ingest-cam-{camera_id}",
                     daemon=True).start()
    return st
