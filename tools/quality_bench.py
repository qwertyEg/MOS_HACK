#!/usr/bin/env python3
"""Замер «помех на камере»: core.stage.quality на размеченных кадрах → таблица для docs/limitations.md.

Наборы:
  --testset DIR      testset/: conditions/<условие>/*.jpg (метка = условие: night, rain, fog, dusk,
                     winter) и equipment/<класс>/*.jpg (метка day — дневные фото для ложных отбраковок)
  --photos DIR:МЕТКА все *.jpg каталога с одной меткой (например, московские фото — day)
  --site DIR         демо-объект (site.json + камеры): кадры по порядку съёмки, время — из имени
                     файла в поясе объекта; метка по солнцу (day / twilight / night), если её нет
                     в --labels
  --labels CSV       ручная разметка «имя_файла,метка» (drops, glare, fog, occluded, day, …) —
                     важнее метки по солнцу

Режимы: --norm — кадры каждой камеры объекта идут через норму камеры (как в конвейере);
--clip — погода по SigLIP (нужны torch и transformers; только для кадров, которые пошли бы
в модель Б, и для фото без часов — как в конвейере и «Проверить снимок»).

    python tools/quality_bench.py --testset /data/testset --site /data/sites/ru_cityzen_pit \\
        --labels reference/quality_labels.csv --norm --clip --out /tmp/qb

Что считается «верно» по метке — в SCORING. Выход: markdown-таблица в stdout,
<out>.jsonl (по кадру) и <out>.json (сводка).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from core.stage import quality as Q  # noqa: E402
from core.stage.camera_norm import CameraNorm  # noqa: E402

TS = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
DETECT = {
    "night": lambda r: r["night"],
    "fog": lambda r: bool({"fog", "haze", "low_visibility"} & set(r["flags"])),
    "rain": lambda r: bool({"rain", "drops"} & set(r["flags"])),
    "drops": lambda r: bool({"drops", "occluded", "low_visibility", "rain"} & set(r["flags"])) or not r["ok"],
    "glare": lambda r: bool({"glare", "glare_block"} & set(r["flags"])),
    "winter": lambda r: bool({"snow_cover", "snowfall"} & set(r["flags"])),
    "snowfall": lambda r: "snowfall" in r["flags"],
    "occluded": lambda r: not r["usable"],
    "day": lambda r: r["usable"],
}
# что должно быть с моделью Б: True — кадр должен в неё пойти, False — не должен, None — всё равно
STAGE_EXPECTED = {"day": True, "night": False, "drops": False, "occluded": False, "fog": False,
                  "glare": None, "rain": None, "winter": None, "dusk": None, "twilight": None, "snowfall": False}


def _when(path: Path, tz: ZoneInfo | None) -> dt.datetime | None:
    m = TS.search(path.name)
    if not m or tz is None:
        return None
    return dt.datetime(*map(int, m.groups()), tzinfo=tz).astimezone(dt.timezone.utc)


def collect(args) -> list[dict]:
    items: list[dict] = []
    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if row and not row[0].startswith("#") and len(row) >= 2:
                    labels[row[0].strip()] = row[1].strip()
    if args.testset:
        t = Path(args.testset)
        for p in sorted((t / "conditions").glob("*/*.jpg")):
            items.append({"path": p, "set": "testset", "label": labels.get(p.name, p.parent.name), "when": None})
        for p in sorted((t / "equipment").glob("*/*.jpg")):
            items.append({"path": p, "set": "testset_day", "label": labels.get(p.name, "day"), "when": None})
    for spec in args.photos or []:
        d, _, lab = spec.rpartition(":")
        for p in sorted(Path(d).rglob("*.jpg")):
            items.append({"path": p, "set": Path(d).name, "label": labels.get(p.name, lab), "when": None})
    for d in args.site or []:
        d = Path(d)
        meta = json.loads((d / "site.json").read_text(encoding="utf-8")) if (d / "site.json").exists() else {}
        tz = ZoneInfo(meta.get("timezone") or "Europe/Moscow")
        for cam in sorted(x for x in d.iterdir() if x.is_dir()):
            frames = sorted(cam.glob("*.jpg"), key=lambda p: _when(p, tz) or dt.datetime.min.replace(
                tzinfo=dt.timezone.utc))
            for p in frames:
                when = _when(p, tz)
                lab = labels.get(p.name)
                if lab is None and when is not None:
                    cfg = Q.config_for_timezone(str(tz))
                    sun = Q.sun_elevation_deg(when, cfg.latitude, cfg.longitude)
                    lab = "night" if sun < -6 else "twilight" if sun < 3 else "day"
                items.append({"path": p, "set": d.name, "camera": cam.name, "label": lab or "day",
                              "when": when, "tz": str(tz)})
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset")
    ap.add_argument("--photos", action="append")
    ap.add_argument("--site", action="append")
    ap.add_argument("--labels")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--out", default="quality_bench")
    args = ap.parse_args(argv)

    items = collect(args)
    weather = None
    if args.clip:
        from core.stage.checklist_clip import ClipConfig, get_embedder
        from core.stage.weather_clip import WeatherClip
        cc = ClipConfig()
        weather = WeatherClip(get_embedder(cc.backend, cc.model_name, cc.device))
    norms: dict[tuple, CameraNorm] = {}
    rows = []
    t0 = time.perf_counter()
    clip_ms = []
    for it in items:
        img = cv2.imread(str(it["path"]))
        if img is None:
            continue
        cfg = Q.config_for_timezone(it.get("tz")) if it.get("tz") else Q.QualityConfig()
        norm = None
        if args.norm and it.get("camera"):
            norm = norms.setdefault((it["set"], it["camera"]), CameraNorm())
        a = time.perf_counter()
        # как в конвейере: без SigLIP → если кадр годен для модели Б (или часов нет), спросить погоду
        rep = Q.assess(img, it["when"], cfg, norm=norm)
        q_ms = (time.perf_counter() - a) * 1000
        if weather is not None and (rep.usable_for_stage or (it["when"] is None and rep.quality_ok)):
            b = time.perf_counter()
            probs = weather.probs(img)
            clip_ms.append((time.perf_counter() - b) * 1000)
            rep = Q.add_weather(img, it["when"], cfg, rep, probs)
        rows.append({"path": str(it["path"]), "set": it["set"], "label": it["label"], "usable": rep.usable_for_stage,
                     "ok": rep.quality_ok, "night": rep.is_night, "weather": rep.weather.value, "flags": rep.flags,
                     "reason": rep.reject_reason, "ms": round(q_ms, 1), "details": rep.details})
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary["timing"] = {"frames": len(rows), "seconds": round(elapsed, 1),
                         "quality_ms_median": _median([r["ms"] for r in rows]),
                         "clip_ms_median": _median(clip_ms) if clip_ms else None}
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(render(summary))
    return 0


def _median(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 1) if xs else None


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["label"])].append(r)
    table = []
    for (s, lab), rs in sorted(groups.items()):
        det = DETECT.get(lab)
        exp = STAGE_EXPECTED.get(lab)
        n = len(rs)
        entry = {"set": s, "label": lab, "n": n,
                 "detected": sum(1 for r in rs if det(r)) if det else None,
                 "to_stage": sum(1 for r in rs if r["usable"]),
                 "defect": sum(1 for r in rs if not r["ok"]),
                 "stage_right": (sum(1 for r in rs if r["usable"] == exp) if exp is not None else None),
                 "flags": dict(Counter(f for r in rs for f in r["flags"]).most_common(6))}
        if lab == "day":
            entry["false_reject"] = [f"{Path(r['path']).name}: {r['reason']}" for r in rs if not r["usable"]][:15]
        table.append(entry)
    return {"table": table}


def render(summary: dict) -> str:
    lines = ["| набор | метка | кадров | помечено верно | пошло в модель Б | брак | флаги |",
             "|---|---|---:|---:|---:|---:|---|"]
    for e in summary["table"]:
        det = f"{e['detected']} из {e['n']}" if e["detected"] is not None else "—"
        flags = ", ".join(f"{k} {v}" for k, v in e["flags"].items())
        lines.append(f"| {e['set']} | {e['label']} | {e['n']} | {det} | {e['to_stage']} | {e['defect']} | {flags} |")
    t = summary.get("timing") or {}
    if t:
        lines.append(f"\nкадров {t['frames']}, {t['seconds']} с; оценка качества — медиана {t['quality_ms_median']} мс"
                     + (f", SigLIP-погода — {t['clip_ms_median']} мс" if t.get("clip_ms_median") else ""))
    for e in summary["table"]:
        if e.get("false_reject"):
            lines.append(f"\nложные отбраковки {e['set']}: " + "; ".join(e["false_reject"]))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
