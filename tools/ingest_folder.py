#!/usr/bin/env python3
"""Прогон истории камеры через маску.

Модель Б не вызывается: на этом шаге проверяется только качество маски.
Результат складывается в БД и хранилище, смотреть в веб-интерфейсе:
    /cameras/<id>/frames

    python tools/ingest_folder.py --camera 1
    python tools/ingest_folder.py --camera 1 --limit 60 --threshold 30

Маска должна быть нарисована и подтверждена в интерфейсе до запуска.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import Camera
from app.pipeline import ingest, mask as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="id камеры")
    ap.add_argument("--folder", type=Path,
                    help="папка с кадрами; по умолчанию берётся из камеры")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=M.CHANGE_THRESHOLD,
                    help=f"порог изменения клетки (по умолчанию {M.CHANGE_THRESHOLD})")
    ap.add_argument("--lock", type=int, default=M.LOCK_WINDOWS,
                    help=f"окон подряд до стирания (по умолчанию {M.LOCK_WINDOWS})")
    ap.add_argument("--window", type=int, default=M.WINDOW_DAYS,
                    help=f"размер окна в днях (по умолчанию {M.WINDOW_DAYS})")
    ap.add_argument("--model-b", action="store_true",
                    help="разбирать кадры моделью Б и писать чек-листы")
    ap.add_argument("--list", action="store_true", help="показать камеры и выйти")
    args = ap.parse_args()

    init_db()
    with SessionLocal() as s:
        if args.list or not args.camera:
            cams = s.scalars(select(Camera).order_by(Camera.id)).all()
            if not cams:
                print("камер нет — заведите объект и камеру в интерфейсе")
                return 1
            print(f"{'id':>3}  {'камера':<22} {'маска':<12} источник")
            for c in cams:
                mark = ("подтверждена" if c.state and c.state.mask_approved
                        else "НЕ ЗАДАНА")
                print(f"{c.id:>3}  {c.name[:22]:<22} {mark:<12} {c.source_uri}")
            return 0 if args.list else 1

        cam = s.get(Camera, args.camera)
        if cam is None:
            print(f"камеры {args.camera} нет", file=sys.stderr)
            return 1

        folder = args.folder or (Path(cam.source_uri) if cam.source_uri else None)
        if not folder or not folder.is_dir():
            print(f"папка с кадрами не найдена: {folder}", file=sys.stderr)
            return 1

        if cam.state is None or not cam.state.mask_approved:
            print("маска не нарисована. Откройте /cameras/"
                  f"{cam.id} и закрасьте фон.", file=sys.stderr)
            return 1

        started = time.time()
        last = [0.0]

        def report(p: ingest.Progress) -> None:
            now = time.time()
            if now - last[0] < 1.0 and p.message != "готово":
                return
            last[0] = now
            pct = p.done / p.total if p.total else 0
            rate = p.done / max(1e-6, now - started)
            eta = (p.total - p.done) / rate if rate else 0
            print(f"\r  {p.done:>5}/{p.total}  {pct:5.0%}  "
                  f"{rate:4.1f} кадр/с  осталось {eta/60:4.1f} мин  "
                  f"{p.message}", end="", flush=True)

        print(f"камера {cam.id} «{cam.name}»")
        print(f"папка  {folder}")
        print(f"порог {args.threshold}, окно {args.window} дн., "
              f"стирание после {args.lock} окон\n")

        prog = ingest.run(s, cam, folder, limit=args.limit, model_b=args.model_b,
                          threshold=args.threshold, lock_windows=args.lock,
                          window_days=args.window, on_progress=report)

        st = cam.state
        print(f"\n\n{'─' * 56}")
        print(f"обработано кадров : {prog.done}  (пропущено {prog.skipped})")
        print(f"скрыто кадра      : {st.masked_ratio:.1%}")
        print(f"маски цело        : {st.retained:.0%} от нарисованной")
        print(f"окон обработано   : {st.windows_accumulated}")
        print(f"верхняя граница   : {st.mask_top_edge_px}")
        print(f"время             : {(time.time() - started) / 60:.1f} мин")
        print(f"\nсмотреть: http://localhost:8000/cameras/{cam.id}/frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
