"""Проверка ключа и модели на одном фото, без базы и без кэша.

    .venv/bin/python tools/smoke.py photo.jpg [--provider local] [--model glm-4.6v-flash] [--thinking] [--strategy per_stage]

Печатает разведку, ответы чек-листа, оценку кадра и расход.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import STRATEGIES, Analyzer  # noqa: E402
from core.checklist import Checklist  # noqa: E402
from core.providers import PROVIDERS, make_client  # noqa: E402
from core.images import data_url, prepare, sha256  # noqa: E402
from core.scoring import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--provider", default="zai", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--strategy", default="two_step", choices=list(STRATEGIES))
    args = ap.parse_args()

    raw = Path(args.photo).read_bytes()
    checklist = Checklist()
    analyzer = Analyzer(checklist, None, make_client(args.provider, args.model, args.thinking), args.strategy)
    result, _ = analyzer.analyze(sha256(raw), lambda: data_url(prepare(raw)))
    score = evaluate(checklist, result)

    print(json.dumps(result["triage"], ensure_ascii=False, indent=2))
    print("кандидаты:", result["candidates"])
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
    front = score["front"]
    print(f"этап: {front}. {checklist.stage_by_id[front]['name']}" if front else "этап не определён",
          f"| готовность по кадру {score['overall_pct']}% | идут: {', '.join(score['active_substages']) or '—'}")
    for call in result["calls"]:
        print(f"  {call['step']:<16} вход {call['prompt_tokens']:>6} (кэш {call['cached_tokens']:>5}) "
              f"выход {call['completion_tokens']:>5}  ${call['cost_usd']:.5f}  {call['latency_ms'] / 1000:.1f} с")
    print(f"итого ${result['usage']['cost_usd']:.5f}")


if __name__ == "__main__":
    main()
