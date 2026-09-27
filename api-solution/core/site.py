"""Единая база знаний по стройке: всё, что известно по всем её фото вместе.

Два назначения:

1. Контекст для модели. Кадры разбираются по порядку дат, и перед каждым
   модель получает короткую сводку того, что уже подтверждено по более
   ранним фото (этап, выполненные работы, техника). Так разборы не
   противоречат друг другу: стройка не идёт назад, а снимок, который всё же
   противоречит истории, модель отмечает явно (context_conflict).

2. Сводка по стройке и выгрузка в CSV: техника за весь период, подтверждённые
   факты, этапы, противоречия, таблица по каждому фото.

Сводка собирается из разборов, сохранённых в SQLite, а не хранится отдельно —
поэтому всегда согласована с фото: удалили кадр или разобрали другой моделью —
сводка сразу верная.
"""

import csv
import hashlib
import io
from collections import defaultdict

from .scoring import evaluate


class ContextBuilder:
    """Накопитель контекста: подаются разборы кадров по возрастанию даты."""

    def __init__(self, checklist, floors_total=None):
        self.checklist = checklist
        self.floors_total = floors_total
        self.latching = {k for k, s in checklist.signs.items() if s["latching"]}
        self.latched = {}          # признак → дата первого подтверждения
        self.front = None
        self.front_since = None
        self.equipment = {}        # тип → максимум единиц одновременно
        self.frames = 0
        self.last_date = None

    def add(self, analysis, day):
        answers = dict(analysis["answers"])
        for k in self.latched:
            answers[k] = "yes"
        for k, v in analysis["answers"].items():
            if v == "yes" and k in self.latching:
                self.latched.setdefault(k, day)
        front = evaluate(self.checklist, analysis, answers, self.floors_total)["front"]
        if front is not None and (self.front is None or front > self.front):
            self.front, self.front_since = front, day
        for e in analysis["triage"]["equipment"]:
            self.equipment[e["type"]] = max(self.equipment.get(e["type"], 0), e["total"])
        self.frames += 1
        self.last_date = day

    def text(self):
        """Контекст для промпта; пустая строка, если предыдущих кадров нет."""
        if not self.frames:
            return ""
        c = self.checklist
        lines = [f"Контекст этой стройки по {self.frames} предыдущим снимкам (последний — {self.last_date:%d.%m.%Y}):"]
        if self.front:
            done = f"; этапы 1–{self.front - 1} выполнены" if self.front > 1 else ""
            lines.append(f"- достигнутый этап: {self.front} «{c.stage_by_id[self.front]['name']}» "
                         f"(с {self.front_since:%d.%m.%Y}){done}.")
        if self.latched:
            facts = "; ".join(f"{_short(c.signs[k]['question'])} ({d:%d.%m.%Y})"
                              for k, d in sorted(self.latched.items(), key=lambda kv: kv[1]))
            lines.append(f"- уже подтверждено: {facts}.")
        if self.equipment:
            eq = ", ".join(f"{c.equipment_name(k)} (до {n})" for k, n in self.equipment.items() if k != "other")
            if eq:
                lines.append(f"- техника, которую уже видели на площадке: {eq}.")
        lines.append("Стройка не идёт назад. Используй контекст, чтобы не противоречить истории, "
                     "но отвечай только по тому, что видно на этом снимке. Если снимок явно противоречит "
                     "контексту, коротко опиши это в поле context_conflict.")
        return "\n".join(lines)

    def digest(self):
        """Хэш контекста — часть ключа кэша: изменилась история → кадр разбирается заново."""
        return hashlib.sha1(self.text().encode()).hexdigest()[:12] if self.frames else ""


def _short(question):
    """«Виден ли открытый котлован — выемка…?» → «открытый котлован»: короче для промпта."""
    q = question.rstrip("?")
    for prefix in ("Виден ли ", "Видна ли ", "Видно ли ", "Видны ли ", "Установлены ли ", "Установлено ли ",
                   "Уложено ли ", "Закрыта ли ", "Закрыт ли ", "Засыпаны ли ", "Перекрыт ли ", "Оформлены ли ",
                   "Возвышается ли ", "Расчищена ли ", "Выровнено ли "):
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    return q.split(" — ")[0].split(", ")[0]


# --- сводка по стройке ---

def knowledge(checklist, tl):
    """Сводка по всей стройке из хронологии timeline.build."""
    frames = tl["frames"]
    eq = defaultdict(lambda: {"photos": 0, "days": set(), "max": 0, "total": 0, "working": 0,
                              "first": None, "last": None, "stages": set()})
    for f in frames:
        for e in f["analysis"]["triage"]["equipment"]:
            r = eq[e["type"]]
            r["photos"] += 1
            r["days"].add(f["date"])
            r["max"] = max(r["max"], e["total"])
            r["total"] += e["total"]
            r["working"] += e["working"]
            r["first"] = r["first"] or f["date"]
            r["last"] = f["date"]
            if f["score"]["front"]:
                r["stages"].add(f["score"]["front"])
    equipment = [{
        "type": k, "name": checklist.equipment_name(k), "photos": r["photos"], "days": len(r["days"]),
        "max_at_once": r["max"], "working_share": round(r["working"] / r["total"], 2) if r["total"] else 0,
        "first_seen": r["first"], "last_seen": r["last"], "stages": sorted(r["stages"]),
    } for k, r in sorted(eq.items(), key=lambda kv: -kv[1]["photos"])]

    frame_by_date = {}
    for f in frames:
        frame_by_date.setdefault(f["date"], f["filename"])
    facts = [{"key": k, "fact": checklist.signs[k]["question"], "since": d, "photo": frame_by_date.get(d)}
             for k, d in sorted(tl["latched"].items(), key=lambda kv: kv[1])]

    sch = {s["stage"]: s for s in tl["schedule"]["stages"]} if tl["schedule"] else {}
    stages = [{
        "stage": s["id"], "name": s["name"], "progress_pct": round(100 * tl["stage_progress"][s["id"]]),
        "first_seen": tl["stage_first_seen"].get(s["id"]), "done_at": tl["stage_done_at"].get(s["id"]),
        "plan_start": sch.get(s["id"], {}).get("plan_start"), "plan_end": sch.get(s["id"], {}).get("plan_end"),
    } for s in checklist.stages]

    conflicts = [{"date": f["date"], "photo": f["filename"], "note": f["analysis"]["triage"].get("context_conflict")}
                 for f in frames if f["analysis"]["triage"].get("context_conflict")]

    return {
        "period": (frames[0]["date"], frames[-1]["date"]) if frames else None,
        "photos": len(frames), "days": len(tl["days"]),
        "front": tl["front"], "overall_pct": tl["overall_pct"],
        "equipment": equipment, "facts": facts, "stages": stages, "conflicts": conflicts,
    }


def frames_table(checklist, tl):
    """Строка на фото — главная таблица для выгрузки и анализа."""
    types = sorted({e["type"] for f in tl["frames"] for e in f["analysis"]["triage"]["equipment"]})
    progress = dict(tl["series"])
    rows = []
    for f in tl["frames"]:
        t, s = f["analysis"]["triage"], f["score"]
        row = {
            "дата": f["taken_at"], "файл": f["filename"], "ракурс": t["view"], "качество": t["quality"],
            "этап_кадра": s["front"], "этап_название": checklist.stage_by_id[s["front"]]["name"] if s["front"] else "",
            "готовность_кадра_%": s["overall_pct"], "готовность_накопленная_%": progress.get(f["taken_at"]),
            "идут_подэтапы": " ".join(s["active_substages"]),
            "рабочих": t["workers_count"], "этажей": t["floors_built"], "остеклено_этажей": t["floors_glazed"],
            "облицовано_%": t["facade_clad_pct"], "котлован_%": t["pit_area_pct"],
        }
        by_type = {e["type"]: e for e in t["equipment"]}
        for k in types:
            name = checklist.equipment_name(k)
            row[f"{name}: всего"] = by_type[k]["total"] if k in by_type else 0
            row[f"{name}: в работе"] = by_type[k]["working"] if k in by_type else 0
        row["отклонения"] = "; ".join(d["title"] for d in f["deviations"])
        row["противоречие_истории"] = t.get("context_conflict") or ""
        row["описание"] = t["description"]
        row["модель"] = f["analysis"].get("provider", "zai") + ":" + f["analysis"]["model"]
        row["стоимость_$"] = round(f["analysis"]["usage"]["cost_usd"], 5)
        rows.append(row)
    return rows


def to_csv(rows):
    """CSV для Excel: разделитель «;», UTF-8 с BOM — иначе кириллица и числа ломаются при открытии."""
    if not rows:
        return b""
    buf = io.StringIO()
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    w = csv.DictWriter(buf, fieldnames=fields, delimiter=";")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(v) for k, v in r.items()})
    return buf.getvalue().encode("utf-8-sig")


def _cell(v):
    if isinstance(v, (list, tuple, set)):
        return " ".join(map(str, v))
    return "" if v is None else v
