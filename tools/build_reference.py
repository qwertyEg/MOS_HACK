"""Сборка справочника этапов из checklist.json и исходного xlsx.

Источник правды — checklist.json (этапы, признаки, техника, метрики и привязка
кодов справочника к подэтапам). Скрипт:

  1. раскладывает каждую строку xlsx по подэтапу, «невидимым» работам или
     «вне охвата» (только дороги) → work_map.csv;
  2. проверяет, что ни одна строка по зданиям не потерялась и что в JSON нет
     ссылок на несуществующие строки, признаки или технику;
  3. генерирует checklist.md — читаемую версию для команды и защиты.

Запуск: python build.py [путь к xlsx]
"""

import csv
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
DEFAULT_XLSX = HERE.parent.parent / "Сводный_перечень_строительных_работ_ЛТЦ.xlsx"

# Колонки C..K листа: восемь типов зданий и «Дороги».
OBJECT_COLUMNS = ["Жильё", "Образование", "Здравоохранение", "Спорт", "Культура",
                  "Административные здания", "ДОУ", "Офисно-деловой центр", "Дороги"]


def norm_code(value):
    # Excel превратил коды вида «10.1.» в даты 10 января — возвращаем обратно.
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        return f"{value.day}.{value.month}."
    return str(value).strip()


def norm_name(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def read_rows(xlsx_path):
    ws = openpyxl.load_workbook(xlsx_path).active
    rows, parent = [], ""
    for r in ws.iter_rows(min_row=4):
        if r[1].value is None:
            continue
        code = norm_code(r[0].value)
        if code:
            parent = code
        types = [t for t, c in zip(OBJECT_COLUMNS, r[2:11]) if c.value]
        rows.append({
            "row": r[0].row,
            "code": code,
            "parent_code": "" if code else parent,
            "name": norm_name(r[1].value),
            "object_types": types,
        })
    return rows


def row_key(row):
    return row["code"] or f"{row['parent_code']}/{row['name']}"


def ancestors(code):
    # «10.13.1.» → «10.13.», «10.»; дочерние строки без кода наследуют родителя.
    parts = [p for p in code.rstrip(".").split(".") if p]
    for i in range(len(parts) - 1, 0, -1):
        yield ".".join(parts[:i]) + "."


def explicit_targets(checklist):
    targets = {}
    for stage in checklist["stages"]:
        for sub in stage["substages"]:
            for key in sub["xlsx"]:
                targets[key] = ("substage", stage, sub, "")
    stages = {s["id"]: s for s in checklist["stages"]}
    for item in checklist["unobservable"]:
        # Невидимая работа всё равно стоит в календарном плане — этап нужен,
        # чтобы её даты не выпали из сравнения плана с фактом.
        stage = stages.get(item.get("stage"))
        for key in item["xlsx"]:
            targets[key] = ("unobservable", stage, None, item["reason"])
    for key in checklist["group_headers"]:
        targets[key] = ("header", None, None, "заголовок группы")
    return targets


def resolve(rows, targets):
    used, out, errors = set(), [], []
    for row in rows:
        key = row_key(row)
        rec = dict(row, status="", stage_id="", substage_id="", rule="", reason="")
        buildings = [t for t in row["object_types"] if t != "Дороги"]
        if not buildings:
            rec.update(status="out_of_scope", reason="только для дорог")
            out.append(rec)
            continue

        hit, via = None, ""
        if key in targets:
            hit, via = targets[key], "явно"
            used.add(key)
        else:
            chain = [row["parent_code"]] if row["parent_code"] else []
            chain += list(ancestors(row["code"] or row["parent_code"]))
            for anc in chain:
                if anc in targets and targets[anc][0] != "header":
                    hit, via = targets[anc], f"наследует {anc}"
                    used.add(anc)
                    break
        if hit is None:
            errors.append(f"строка {row['row']} не разложена: {key}")
            out.append(rec)
            continue

        kind, stage, sub, reason = hit
        rec.update(status=kind, rule=via, reason=reason)
        if stage:
            rec["stage_id"] = stage["id"]
        if sub:
            rec["substage_id"] = sub["id"]
        out.append(rec)

    for key in targets:
        if key not in used and not any(row_key(r) == key for r in rows):
            errors.append(f"в JSON есть ключ, которого нет в xlsx: {key}")
    return out, errors


def check_references(checklist):
    errors = []
    signs = {s["key"] for s in checklist["signs"]}
    equipment = {e["key"] for e in checklist["equipment"]}
    if sum(s["weight"] for s in checklist["stages"]) != 100:
        errors.append("веса этапов не дают в сумме 100")
    for stage in checklist["stages"]:
        for field in ("must_have", "must_not_have"):
            errors += [f"этап {stage['id']}: нет признака {k}" for k in stage[field] if k not in signs]
        for field in ("equipment_expected", "equipment_optional", "equipment_forbidden"):
            errors += [f"этап {stage['id']}: нет техники {k}" for k in stage[field] if k not in equipment]
        clash = set(stage["equipment_expected"] + stage["equipment_optional"]) & set(stage["equipment_forbidden"])
        if clash:
            errors.append(f"этап {stage['id']}: техника и ожидается, и запрещена: {clash}")
        if sum(s["weight"] for s in stage["substages"]) != 100:
            errors.append(f"этап {stage['id']}: веса подэтапов не дают в сумме 100")
        for sub in stage["substages"]:
            for field in ("active_when", "done_when"):
                errors += [f"{sub['id']}: нет признака {k}" for k in sub[field] if k not in signs]
            errors += [f"{sub['id']}: нет техники {k}" for k in sub["equipment"] if k not in equipment]
    return errors


def warn_shared(checklist):
    """Подэтап без отличительных признаков не может сам открыть свой этап — только через разведку."""
    earlier, out = set(), []
    for stage in sorted(checklist["stages"], key=lambda s: s["id"]):
        positive = set(stage["must_have"])
        for sub in stage["substages"]:
            positive |= set(sub["active_when"]) | set(sub["done_when"])
        for sub in stage["substages"]:
            keys = set(sub["active_when"]) | set(sub["done_when"])
            if keys and not keys - earlier:
                out.append(f"предупреждение: {sub['id']} — все признаки есть у более ранних этапов: {sorted(keys)}")
        earlier |= positive
    return out


def write_work_map(records, path):
    fields = ["row", "code", "parent_code", "name", "status", "stage_id", "substage_id",
              "rule", "reason", "object_types"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: "|".join(r[k]) if k == "object_types" else r[k] for k in fields})


def render_md(checklist, records):
    eq = {e["key"]: e["name"] for e in checklist["equipment"]}
    by_sub = defaultdict(list)
    for r in records:
        if r["status"] == "substage":
            by_sub[r["substage_id"]].append(r)

    def sign_list(keys):
        return ", ".join(f"`{k}`" for k in keys) or "—"

    def eq_list(keys):
        return ", ".join(eq[k] for k in keys) or "—"

    L = []
    L.append("# Чек-лист этапов строительства для VLM (GLM)\n")
    L.append("> Файл сгенерирован `build.py` из `checklist.json` — правки вносить в JSON.\n")
    L.append(f"Источник: `{checklist['source']}`. Охват: здания ({', '.join(checklist['scope']['object_types'])}); "
             "дороги не входят.\n")
    L.append("## Как устроено\n")
    L.append("- **8 макроэтапов** (совпадают с веткой `dev-lamonifi`) → **подэтапы**, к которым привязана каждая "
             "строка справочника. Полная раскладка по строкам — `work_map.csv`.")
    L.append("- **Признаки** — общий пул вопросов к модели с ответом `yes / no / unsure`. Этап ссылается на признаки: "
             "`must_have` — результат этапа виден, `must_not_have` — противоречит этапу. У подэтапа: "
             "`active_when` — работы идут сейчас, `done_when` — результат есть.")
    L.append("- **latching** — признак после подтверждения не пропадает (котлован вырыт — значит был вырыт, "
             "даже если сейчас его закрыл кран).")
    L.append("- **Техника**: `ожидается` — без неё этап не движется; `возможна` — не аномалия; "
             "`не по этапу` — если работает, поднимаем отклонение. Припаркованная техника в сопоставлении не "
             "участвует (PLAN.md §3.9).")
    L.append("- **Прогресс-бар**: общий % = Σ вес этапа × прогресс этапа; прогресс этапа = Σ вес подэтапа × "
             "(1 — готов, доля по метрике или 0.5 — идёт, 0 — не начат). Учитываются только работы, видимые "
             "снаружи; внутренние идут параллельно фасаду и в % не входят.\n")

    L.append("## Этапы и веса\n")
    L.append("| # | Этап | Вес, % | Ракурс | Подэтапы |")
    L.append("|---|---|---|---|---|")
    for s in checklist["stages"]:
        subs = "; ".join(f"{x['id']} {x['name']}" for x in s["substages"])
        L.append(f"| {s['id']} | {s['name']} | {s['weight']} | {s['view']} | {subs} |")
    L.append("")

    for s in checklist["stages"]:
        L.append(f"## {s['id']}. {s['name']} — вес {s['weight']}%\n")
        L.append(f"{s['summary']}\n")
        L.append(f"- **Должно быть видно:** {sign_list(s['must_have'])}")
        L.append(f"- **Не должно быть:** {sign_list(s['must_not_have'])}")
        L.append(f"- **Техника ожидается:** {eq_list(s['equipment_expected'])}")
        L.append(f"- **Техника возможна:** {eq_list(s['equipment_optional'])}")
        L.append(f"- **Техника не по этапу:** {eq_list(s['equipment_forbidden'])}")
        L.append(f"- **Метрика прогресса:** {s['progress_metric']}\n")

        for sub in s["substages"]:
            L.append(f"### {sub['id']}. {sub['name']} — {sub['weight']}% этапа\n")
            L.append(f"{sub['visual']}\n")
            L.append(f"- Работы идут: {sign_list(sub['active_when'])}; результат есть: {sign_list(sub['done_when'])}")
            L.append(f"- Характерная техника: {eq_list(sub['equipment'])}")
            rows = by_sub.get(sub["id"], [])
            types = sorted({t for r in rows for t in r["object_types"] if t != "Дороги"},
                           key=OBJECT_COLUMNS.index)
            if types and len(types) < 8:
                L.append(f"- Типы объектов: {', '.join(types)}")
            if sub.get("note"):
                L.append(f"- *Примечание:* {sub['note']}")
            names = [f"{r['code'] or '·'} {r['name']}" for r in rows]
            L.append(f"- Строки справочника ({len(rows)}): " + "; ".join(names))
            L.append("")

        L.append("**Метрики этапа**\n")
        L.append("| Метрика | Ед. | Как считаем |")
        L.append("|---|---|---|")
        for m in s["metrics"]:
            L.append(f"| {m['name']} | {m['unit']} | {m['method']} |")
        L.append("")
        L.append("**Отклонения**\n")
        L.append("| Условие | Предупреждение | Уровень |")
        L.append("|---|---|---|")
        for d in s["deviation_rules"]:
            L.append(f"| {d['if']} | {d['then']} | {d['severity']} |")
        L.append("")

    L.append("## Признаки (вопросы к модели)\n")
    L.append("| Ключ | Вопрос | Как выглядит | latching |")
    L.append("|---|---|---|---|")
    for x in checklist["signs"]:
        L.append(f"| `{x['key']}` | {x['question']} | {x['hint']} | {'да' if x['latching'] else ''} |")
    L.append("")

    L.append("## Измерения с кадра\n")
    L.append("| Ключ | Тип | Вопрос |")
    L.append("|---|---|---|")
    for m in checklist["frame_measurements"]:
        L.append(f"| `{m['key']}` | {m['type']} | {m['question']} |")
    L.append("")

    L.append("## Техника\n")
    L.append("| Ключ | Название | Из ТЗ | Как выглядит |")
    L.append("|---|---|---|---|")
    for e in checklist["equipment"]:
        L.append(f"| `{e['key']}` | {e['name']} | {'да' if e['tz'] else ''} | {e['look']} |")
    L.append("")

    L.append("## Общие метрики площадки\n")
    L.append("| Метрика | Ед. | Как считаем |")
    L.append("|---|---|---|")
    for m in checklist["site_metrics"]:
        L.append(f"| {m['name']} | {m['unit']} | {m['method']} |")
    L.append("")

    L.append("## Не наблюдаемо с камеры\n")
    unobs = [r for r in records if r["status"] == "unobservable"]
    for item in checklist["unobservable"]:
        L.append(f"- **{', '.join(item['xlsx'])}** — {item['reason']}")
    L.append(f"\nВсего строк справочника, отнесённых к невидимым: {len(unobs)}.\n")
    return "\n".join(L)


def main():
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    checklist = json.loads((HERE / "checklist.json").read_text(encoding="utf-8"))
    rows = read_rows(xlsx)
    records, errors = resolve(rows, explicit_targets(checklist))
    errors += check_references(checklist)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    for w in warn_shared(checklist):
        print(w)
    write_work_map(records, HERE / "work_map.csv")
    (HERE / "checklist.md").write_text(render_md(checklist, records), encoding="utf-8")

    counts = defaultdict(int)
    for r in records:
        counts[r["status"]] += 1
    print(f"строк: {len(records)}; " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
