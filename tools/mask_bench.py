#!/usr/bin/env python3
"""Замер качества динамической маски на реальном прогоне камеры.

Глазами видно только «маска съелась» или «не съелась», а нужно знать, ЧТО
именно съелось: растущее здание или фон. Этот стенд отвечает на такой вопрос
числом и картинкой, не требуя ручной разметки.

**Откуда берётся эталон.** Из необратимости, на которой стоит весь метод.
Стройка необратима: клетка, на месте которой встала стена, к прежней яркости
уже не вернётся. Фон обратим: снег тает, асфальт сохнет, листва отрастает,
контейнер увозят. Значит «здесь действительно построили» = уровень клетки на
конце ряда устойчиво ушёл от исходного, с поправкой на общее освещение.

Эталон приблизительный, и это надо помнить: застройка того же тона, что и
скрытый ею фон, в него не попадёт. Но он проверяем — на прогоне, который
оператор признал удачным, он обязан дать заметно меньше ошибок, чем на
признанном неудачным. На данных Эдинбурга так и выходит: F1 0.80 против 0.42.

Запуск:
    python tools/mask_bench.py --camera 2
    python tools/mask_bench.py --camera 2 --panel var/mask_bench.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session                                   # noqa: E402
from app.models import Camera                                    # noqa: E402
from app.pipeline import mask as M                               # noqa: E402
from app.pipeline.ingest import assess_quality, list_frames      # noqa: E402
from app.storage import storage                                  # noqa: E402

TAIL_DAYS = 25      # хвост ряда, по которому берётся конечный уровень
DEPART = 30.0       # уровней яркости: уход от исходного, считаемый застройкой


def cells_of(frame: np.ndarray) -> np.ndarray:
    gh, gw = frame.shape[0] // M.CELL, frame.shape[1] // M.CELL
    return cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)


def load_days(folder: Path, limit: int = 0):
    """Дневные медианы и даты. Ночь и брак в расчёт не идут."""
    items = list_frames(folder)
    if limit:
        items = items[:limit]
    days, dates, paths = [], [], []
    for path, when in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        ok, night, _ = assess_quality(img)
        if not ok or night:
            continue
        days.append(M.daily_median([img]))
        dates.append(when)
        paths.append(path)
    return days, dates, paths


def replay(days, init_bg):
    """Прогон через настоящий update(). Возвращает момент стирания каждой клетки."""
    h, w = days[0].shape
    st = M.init_from_bitmap((init_bg * 255).astype(np.uint8), (h, w))
    gh, gw = st.grid
    erased_at = np.full((gh, gw), -1, np.int32)
    ring: list[np.ndarray] = []
    grew = False
    prev = int(st.background.sum())

    for i, day in enumerate(days):
        ring.append(day)
        if len(ring) > M.WINDOW_DAYS:
            ring.pop(0)
        M.update(st, ring)
        now_cells = cv2.resize(st.background.astype(np.uint8), (gw, gh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        erased_at[~now_cells & (erased_at < 0)] = i
        cur = int(st.background.sum())
        grew = grew or cur > prev
        prev = cur
    return st, erased_at, grew


def truth(cells, baseline, init_cells):
    """Клетки исходной маски, где к концу ряда что-то построили."""
    tail = np.median(cells[-TAIL_DAYS:], axis=0)
    shift = float(np.median((tail - baseline)[init_cells])) if init_cells.any() else 0.0
    return (np.abs(tail - shift - baseline) > DEPART) & init_cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="взять первые N кадров")
    ap.add_argument("--panel", default="", help="куда положить картинку сравнения")
    args = ap.parse_args()

    session = next(get_session())
    cam = session.get(Camera, args.camera)
    if cam is None:
        print(f"камеры {args.camera} нет", file=sys.stderr)
        return 1
    if cam.state is None or not cam.state.initial_mask_key:
        print("у камеры нет исходной маски — нечего замерять", file=sys.stderr)
        return 1

    folder = Path(cam.source_uri)
    if not folder.is_dir():
        print(f"папка не найдена: {folder}", file=sys.stderr)
        return 1

    raw = storage.get(cam.state.initial_mask_key)
    init = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE) > 127

    print(f"камера {cam.id} «{cam.name}», папка {folder}")
    days, dates, paths = load_days(folder, args.limit)
    if len(days) < M.WINDOW_DAYS * 2:
        print(f"кадров слишком мало: {len(days)}", file=sys.stderr)
        return 1

    h, w = days[0].shape
    if init.shape != (h, w):
        init = cv2.resize(init.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0

    st, erased_at, grew = replay(days, init)

    cells = np.stack([cells_of(d) for d in days])
    baseline = np.median(cells[:M.WINDOW_DAYS], axis=0)
    gh, gw = cells.shape[1:]
    init_cells = cv2.resize(init.astype(np.uint8), (gw, gh),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    built = truth(cells, baseline, init_cells)
    erased = (erased_at >= 0) & init_cells

    tp = int((built & erased).sum())
    fp = int((~built & erased).sum())
    fn = int((built & ~erased).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"дней в расчёте: {len(days)}   "
          f"{dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}")
    print(f"маска оператора: {init.mean():.1%} кадра, "
          f"{int(init_cells.sum())} клеток")
    print(f"из них застроено по эталону: {int(built.sum())}")
    print()
    print(f"  стёрто верно (там стройка):     {tp}")
    print(f"  стёрто зря (там остался фон):   {fp}")
    print(f"  не стёрто, хотя застроено:      {fn}")
    print(f"  точность {prec:.2f}   полнота {rec:.2f}   F1 {f1:.2f}")
    print()
    print(f"на конец ряда скрыто {st.masked_ratio:.1%} кадра, "
          f"от исходной маски цело {st.retained:.0%}")
    if grew:
        print("ВНИМАНИЕ: маска выросла — нарушена монотонность, это ошибка")

    if args.panel:
        out = Path(args.panel)
        out.parent.mkdir(parents=True, exist_ok=True)
        picks = np.linspace(M.WINDOW_DAYS, len(days) - 1, 6).astype(int)
        tiles = []
        for t in picks:
            frame = cv2.resize(cv2.imread(str(paths[t])), (w, h))
            alive = (erased_at < 0) | (erased_at > t)

            def up(m):
                return cv2.resize(m.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

            tile = frame.copy()
            for sel, colour in ((alive & init_cells, (210, 210, 210)),
                                (~alive & init_cells & built, (80, 200, 80)),
                                (~alive & init_cells & ~built, (60, 60, 230))):
                m = up(sel)
                tile[m] = (tile[m] * .5 + np.array(colour) * .5).astype(np.uint8)
            cv2.putText(tile, dates[t].strftime("%Y-%m-%d"), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(tile)
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nкартинка: {out}")
        print("серое — маска цела, зелёное — стёрто верно, красное — стёрто зря")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
