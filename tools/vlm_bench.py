#!/usr/bin/env python3
"""Замер модели Б на вопросах чек-листа.

Гоняет тот же код, что и сервис (app.pipeline.model_b), а не свою копию —
иначе замер проверял бы не то, что поедет в продакшене.

Меряет три вещи, которые решают, годится модель:
  1. латентность на вопрос и в пересчёте на сутки работы камеры;
  2. стабильность ответа при повторном запросе (при temperature 0 ответ
     обязан совпадать, и если нет — на такой модели нельзя строить вердикт);
  3. долю «не уверена» и обрывов бюджета.

    python tools/vlm_bench.py data/Строительная_техника --limit 10

Сравнение моделей — сменой VLM_MODEL, результаты пишутся в отдельные файлы.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.config import settings
from app.pipeline.model_b import Answer, MaskMode, ModelB, apply_mask, encode
from app.seed import GLOBAL_QUESTIONS, TEMPLATES

# Вопросы берём из настоящих шаблонов чек-листов, а не выдуманные: замер
# должен показывать качество на той формулировке, которая пойдёт в работу.
# По одному характерному вопросу на этап, чтобы прогон не был бесконечным.
PROBE_KEYS = {
    1: "fence", 2: "rig", 3: "pit", 4: "formwork",
    5: "crane", 6: "roof_cover", 7: "cladding", 8: "paving",
}


def build_questions() -> list[dict]:
    out = []
    for stage_id, key in sorted(PROBE_KEYS.items()):
        for k, polarity, text in TEMPLATES[stage_id]["questions"]:
            if k == key:
                out.append({"key": f"{stage_id}:{k}", "text": text,
                            "polarity": polarity})
                break
    for k, polarity, text in GLOBAL_QUESTIONS:
        out.append({"key": k, "text": text, "polarity": polarity})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=Path)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=2,
                    help="сколько раз задать тот же вопрос — проверка стабильности")
    ap.add_argument("--mask-mode", default="none",
                    choices=[m.value for m in MaskMode],
                    help="способ гашения фона, см. PLAN.md §3.4.7")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:args.limit]
    if not files:
        print(f"нет изображений в {args.images_dir}", file=sys.stderr)
        return 1

    client = ModelB()
    if not client.health():
        print(f"модель недоступна: {client.base_url}", file=sys.stderr)
        return 1

    questions = build_questions()
    print(f"модель:    {client.model}")
    print(f"эндпоинт:  {client.base_url}")
    print(f"кадров {len(files)}, вопросов {len(questions)}, повторов {args.repeat}, "
          f"маска {args.mask_mode}\n")

    latencies: list[float] = []
    answers_count: Counter[str] = Counter()
    unstable = truncated = total = 0
    records = []

    for path in files:
        img = Image.open(path)
        img = apply_mask(img, None, MaskMode(args.mask_mode))
        uri = encode(img)

        row: dict = {"file": path.name, "answers": {}}
        line = []
        for q in questions:
            got = []
            for _ in range(args.repeat):
                try:
                    reply = client.ask(uri, q["text"])
                except Exception as exc:
                    got.append(("error", str(exc)[:60], 0))
                    continue
                latencies.append(reply.latency_ms / 1000)
                if reply.raw.startswith("<обрыв"):
                    truncated += 1
                got.append((reply.answer.value, reply.raw, reply.latency_ms))

            vals = [g[0] for g in got]
            total += 1
            stable = len(set(vals)) == 1
            unstable += not stable
            answers_count[vals[0]] += 1

            row["answers"][q["key"]] = {
                "answer": vals[0], "stable": stable, "raw": got[0][1],
                "latency_ms": got[0][2],
            }
            mark = vals[0][0] if stable else "?"
            line.append(f"{q['key'].split(':')[-1][:8]}={mark}")

        records.append(row)
        print(f"{path.name:<24} " + " ".join(line))

    print("\n" + "─" * 66)
    if latencies:
        lat = sorted(latencies)
        med = statistics.median(lat)
        print(f"латентность на вопрос : медиана {med:.2f} c, "
              f"p90 {lat[int(len(lat) * 0.9) - 1]:.2f} c")
        per_frame = med * len(questions)
        print(f"на кадр ({len(questions)} вопросов)   : {per_frame:.1f} c")
        # Модель Б вызывается не на каждом кадре, а раз в час — §4.2 плана.
        print(f"сутки одной камеры    : {per_frame * 24 / 60:.1f} мин "
              f"(24 вызова при кадре раз в {settings.frame_interval_min} мин)")
    if total:
        print(f"нестабильных ответов  : {unstable}/{total} ({unstable / total:.0%})"
              + ("  ← при temperature 0 должно быть 0" if unstable else ""))
        print(f"обрывов по бюджету    : {truncated}")
        dist = ", ".join(f"{k}={v}" for k, v in answers_count.most_common())
        print(f"распределение ответов : {dist}")
        unsure = answers_count.get(Answer.UNSURE.value, 0)
        print(f"доля «не уверена»     : {unsure / total:.0%}")

    out = args.out or Path(f"vlm_bench_{client.model.replace(':', '_')}"
                           f"_{args.mask_mode}.json")
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"подробности           : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
