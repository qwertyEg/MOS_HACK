"""Серия кадров объекта → хронология, простои, сравнение с планом, прогноз.

Кадры идут по дате съёмки. Признаки с latching=true, однажды подтверждённые,
держатся и дальше (котлован был вырыт, даже если теперь его не видно), а
прогресс этапов берётся как накопленный максимум — кадр с неудачного ракурса
не откатывает объект назад.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import median

from . import rules
from .scoring import evaluate

# Отставание или опережение в пределах допуска считается «в срок».
ON_TIME_TOLERANCE_DAYS = 7
IDLE_STREAK_WARN = 3


def _day(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def expected_pct(checklist, plan, day):
    """Сколько процентов объекта должно быть готово к дате по плану (линейно внутри этапа)."""
    total = 0.0
    for stage in checklist.stages:
        start, end = plan.get(stage["id"], (None, None))
        if not start or not end:
            continue
        s, e = _day(start), _day(end)
        span = max((e - s).days, 1)
        frac = min(max((day - s).days / span, 0.0), 1.0)
        total += stage["weight"] * frac
    return total


def plan_weight(checklist, plan):
    return sum(s["weight"] for s in checklist.stages if all(plan.get(s["id"], (None, None))))


def build(checklist, frames, plan=None, floors_total=None):
    """frames: [{id, taken_at, filename, analysis}] — analysis может быть None (ещё не разобран)."""
    plan = plan or {}
    frames = sorted((f for f in frames if f.get("analysis")), key=lambda f: f["taken_at"])
    latching = {k for k, s in checklist.signs.items() if s["latching"]}

    latched = {}
    stage_progress = {s["id"]: 0.0 for s in checklist.stages}
    stage_first_seen, stage_done_at = {}, {}
    max_front = None
    out_frames, series, deviations = [], [], []

    for f in frames:
        a = f["analysis"]
        d = _day(f["taken_at"])
        answers = dict(a["answers"])
        for k in latched:
            answers[k] = "yes"
        for k, v in a["answers"].items():
            if v == "yes" and k in latching:
                latched.setdefault(k, d)

        score = evaluate(checklist, a, answers, floors_total)
        if score["front"] is not None:
            max_front = score["front"] if max_front is None else max(max_front, score["front"])
        for sid, st in score["stages"].items():
            stage_progress[sid] = max(stage_progress[sid], st["progress"])
            if st["progress"] > 0 and sid not in stage_first_seen:
                stage_first_seen[sid] = d
            if stage_progress[sid] >= 0.999 and sid not in stage_done_at:
                stage_done_at[sid] = d
        overall = sum(s["weight"] * stage_progress[s["id"]] for s in checklist.stages)
        series.append((f["taken_at"], round(overall, 1)))

        devs = rules.frame_deviations(checklist, a, score)
        conflict = a["triage"].get("context_conflict")
        if conflict:
            devs.append({"rule": "history_conflict", "severity": rules.INFO, "stage": score["front"],
                         "title": "Снимок противоречит истории стройки", "detail": conflict})
        for dev in devs:
            deviations.append({**dev, "date": d, "frame_ids": [f["id"]]})
        out_frames.append({"id": f["id"], "date": d, "taken_at": f["taken_at"], "filename": f.get("filename"),
                           "analysis": a, "score": score, "deviations": devs})

    days = _daily(checklist, out_frames)
    for day in days:
        miss = day.pop("_missing")
        if miss:
            deviations.append({**miss, "date": day["date"], "frame_ids": day["frame_ids"]})

    # Серия — только по соседним календарным дням: два простоя с разрывом
    # в месяц — не «простой два дня подряд».
    streak = 0
    for i in range(len(days) - 1, -1, -1):
        if not days[i]["idle"]:
            break
        if streak and (days[i + 1]["date"] - days[i]["date"]).days != 1:
            break
        streak += 1
    if streak >= IDLE_STREAK_WARN:
        deviations.append({"rule": "idle_streak", "severity": rules.WARNING, "stage": max_front,
                           "title": f"Простой {streak} дн. подряд",
                           "detail": "На кадрах этих дней нет ни работающей техники, ни рабочих.",
                           "date": days[-1]["date"], "frame_ids": days[-1]["frame_ids"]})

    last_day = _day(frames[-1]["taken_at"]) if frames else None
    overall = series[-1][1] if series else 0.0
    schedule = _schedule(checklist, plan, last_day, overall, stage_progress, stage_first_seen, deviations)
    forecast = _forecast(checklist, series, plan, schedule)

    latest_day = [f for f in out_frames if f["date"] == last_day]
    active_now = sorted({s for f in latest_day for s in f["score"]["active_substages"]})

    return {
        "frames": out_frames,
        "series": series,
        "overall_pct": overall,
        "stage_progress": stage_progress,
        "front": max_front,
        "active_substages": active_now,
        "stage_first_seen": stage_first_seen,
        "stage_done_at": stage_done_at,
        "days": days,
        "idle_streak": streak,
        "latched": latched,
        "schedule": schedule,
        "forecast": forecast,
        "deviations": sorted(_merge(deviations), key=lambda d: (d["date"], -_sev_rank(d["severity"]))),
        "metrics": _metrics(out_frames, days, max_front),
    }


def _sev_rank(sev):
    return {rules.CRITICAL: 0, rules.WARNING: 1, rules.INFO: 2}.get(sev, 3)


def _daily(checklist, frames):
    by_day = defaultdict(list)
    for f in frames:
        by_day[f["date"]].append(f)
    days = []
    for d in sorted(by_day):
        fs = by_day[d]
        present, working = defaultdict(int), defaultdict(int)
        workers = []
        for f in fs:
            per_frame_p, per_frame_w = defaultdict(int), defaultdict(int)
            for e in f["analysis"]["triage"]["equipment"]:
                per_frame_p[e["type"]] += e["total"]
                per_frame_w[e["type"]] += e["working"]
            # Максимум по кадрам дня: одна машина на двух кадрах — всё ещё одна машина.
            for k, v in per_frame_p.items():
                present[k] = max(present[k], v)
            for k, v in per_frame_w.items():
                working[k] = max(working[k], v)
            wc = f["analysis"]["triage"].get("workers_count")
            if wc is not None:
                workers.append(wc)
        usable = [f for f in fs if f["analysis"]["triage"]["quality"] not in ("blurred", "obstructed")]
        active = any(working.values()) or any(w > 0 for w in workers)
        # Обзорный кадр видит площадку целиком; на виде фасада сбоку технику и
        # людей внизу просто не видно, и их отсутствие ничего не доказывает.
        overview = any(f["analysis"]["triage"]["view"] == "top" for f in usable)
        # Простой — только при доказательстве: обзорный кадр или явный ноль рабочих.
        idle = bool(usable) and not active and (overview or 0 in workers)
        fronts = [f["score"]["front"] for f in fs if f["score"]["front"] is not None]
        front = max(fronts) if fronts else None
        # В день простоя отсутствие техники — не отдельное отклонение, его покрывает простой.
        missing = None
        if front is not None and overview and active:
            missing = rules.missing_equipment(checklist, front, [k for k, v in present.items() if v])
        days.append({
            "date": d, "frames": len(fs), "frame_ids": [f["id"] for f in fs],
            "front": front, "equipment": dict(present), "working": dict(working),
            "workers": max(workers) if workers else None,
            "active": active, "idle": idle, "_missing": missing,
        })
    return days


def _schedule(checklist, plan, last_day, overall, stage_progress, first_seen, deviations):
    if not last_day or not plan_weight(checklist, plan):
        return None
    expected = expected_pct(checklist, plan, last_day)

    # Отставание в днях: когда по плану должна была быть достигнута фактическая готовность.
    starts = [_day(s) for s, e in plan.values() if s]
    ends = [_day(e) for s, e in plan.values() if e]
    day, lag = min(starts), None
    while day <= max(ends):
        if expected_pct(checklist, plan, day) >= overall - 1e-6:
            lag = (last_day - day).days
            break
        day += timedelta(days=1)
    if lag is None:
        lag = (last_day - max(ends)).days  # готовность выше плана на конец — всё равно считаем от конца

    if lag > ON_TIME_TOLERANCE_DAYS:
        verdict = "отставание"
    elif lag < -ON_TIME_TOLERANCE_DAYS:
        verdict = "опережение"
    else:
        verdict = "в срок"

    stages = []
    for stage in checklist.stages:
        sid = stage["id"]
        start, end = plan.get(sid, (None, None))
        if not start or not end:
            continue
        s, e = _day(start), _day(end)
        p = stage_progress[sid]
        planned = "не начат" if last_day < s else "идёт" if last_day <= e else "завершён"
        actual = "не начат" if p <= 0 else "завершён" if p >= 0.999 else "идёт"
        stages.append({"stage": sid, "name": stage["name"], "plan_start": s, "plan_end": e,
                       "planned": planned, "actual": actual, "progress_pct": round(100 * p),
                       "actual_start": first_seen.get(sid)})
        if planned == "завершён" and actual != "завершён":
            overdue = (last_day - e).days
            deviations.append({"rule": "stage_overdue", "stage": sid, "date": last_day, "frame_ids": [],
                               "severity": rules.CRITICAL if overdue > 14 else rules.WARNING,
                               "title": f"Этап «{stage['name']}» не завершён в срок",
                               "detail": f"По плану окончание {e:%d.%m.%Y}, просрочка {overdue} дн., готовность {round(100 * p)}%."})
        elif planned != "не начат" and actual == "не начат" and (last_day - s).days > ON_TIME_TOLERANCE_DAYS:
            deviations.append({"rule": "stage_not_started", "stage": sid, "date": last_day, "frame_ids": [],
                               "severity": rules.WARNING,
                               "title": f"Этап «{stage['name']}» не начат",
                               "detail": f"По плану начало {s:%d.%m.%Y}, признаков работ на кадрах нет."})
        elif planned == "не начат" and actual != "не начат":
            deviations.append({"rule": "stage_early", "stage": sid, "date": last_day, "frame_ids": [],
                               "severity": rules.INFO,
                               "title": f"Этап «{stage['name']}» начат раньше плана",
                               "detail": f"По плану начало {s:%d.%m.%Y}."})

    return {"date": last_day, "expected_pct": round(expected, 1), "actual_pct": overall,
            "lag_days": lag, "verdict": verdict, "stages": stages}


def _forecast(checklist, series, plan, schedule):
    """Прогноз окончания объекта.

    С планом — по темпу относительно плана: во сколько раз фактический прирост
    готовности быстрее или медленнее планового за тот же отрезок. Линейная
    экстраполяция процентов здесь врёт: ранние этапы лёгкие по весу и быстрые,
    их темп переносить на каркас нельзя. Без плана — линейно, с оговоркой.
    """
    by_day = {}
    for t, p in series:
        by_day[_day(t)] = max(by_day.get(_day(t), 0), p)
    if len(by_day) < 2:
        return None
    d0, d1 = min(by_day), max(by_day)
    span = (d1 - d0).days
    gain = by_day[d1] - by_day[d0]
    short = "на отрезке меньше месяца прогноз грубый" if span < 30 else ""
    if span <= 0 or gain <= 0:
        return {"finish": None, "span_days": span,
                "note": "прогресса между первым и последним кадром не видно — прогноз невозможен"}

    plan_end = max(_day(e) for s, e in plan.values() if e) if schedule else None
    if schedule:
        plan_gain = expected_pct(checklist, plan, d1) - expected_pct(checklist, plan, d0)
        # Дата, когда по плану достигается фактическая готовность, — от неё план «догоняется».
        equivalent = d1 - timedelta(days=schedule["lag_days"])
        remaining = max((plan_end - equivalent).days, 0)
        ratio = gain / plan_gain if plan_gain > 0 else None
        if ratio:
            ratio = min(max(ratio, 0.2), 5.0)
            finish = d1 + timedelta(days=round(remaining / ratio))
            return {"finish": finish, "plan_end": plan_end, "delay_days": (finish - plan_end).days,
                    "pace_vs_plan": round(ratio, 2), "span_days": span, "note": short}

    # Плана нет или отрезок целиком вне плановых дат (плановый прирост нулевой).
    pace = gain / span
    finish = d1 + timedelta(days=round((100 - by_day[d1]) / pace))
    return {"finish": finish, "plan_end": plan_end,
            "delay_days": (finish - plan_end).days if plan_end else None, "pace_vs_plan": None,
            "span_days": span, "note": ("линейно по темпу серии; " + short).strip("; ")}


def _merge(deviations, gap_days=3):
    """Одно и то же отклонение в соседние дни — одна запись с диапазоном дат."""
    merged = []
    for d in sorted(deviations, key=lambda d: d["date"]):
        prev = next((m for m in reversed(merged)
                     if (m["rule"], m["stage"], m["title"]) == (d["rule"], d["stage"], d["title"])), None)
        if prev and (d["date"] - prev["date"]).days <= gap_days:
            prev["date"] = d["date"]
            prev["frame_ids"] = list(dict.fromkeys(prev["frame_ids"] + d["frame_ids"]))
            prev["count"] += 1
        else:
            merged.append({**d, "date_from": d["date"], "count": 1})
    return merged


def _metrics(frames, days, front):
    """Метрики, которые можно снять с загруженной серии. Остальные из checklist.json требуют потока с камеры."""
    out = []
    if not days:
        return out
    last = days[-1]
    out.append(("Активных дней", sum(d["active"] for d in days), f"из {len(days)} дн. с кадрами"))
    out.append(("Дней простоя", sum(d["idle"] for d in days), "дни с кадрами без работ"))
    total = sum(last["equipment"].values())
    work = sum(last["working"].values())
    out.append(("Техники в последний день", total, f"в работе {work}"))
    if total:
        out.append(("Доля работающей техники", round(100 * work / total), "%"))
    workers = [d["workers"] for d in days if d["workers"] is not None]
    if workers:
        out.append(("Рабочих (медиана по дням)", round(median(workers)), "чел."))

    def latest(key):
        vals = [f["analysis"]["triage"].get(key) for f in frames if f["analysis"]["triage"].get(key) is not None]
        return vals[-1] if vals else None

    if front == 3:
        ex, tr = last["equipment"].get("excavator", 0), last["equipment"].get("dump_truck", 0)
        if ex:
            out.append(("Самосвалов на экскаватор", round(tr / ex, 1), "норма 3–5"))
        if latest("pit_area_pct") is not None:
            out.append(("Котлован в кадре", latest("pit_area_pct"), "% площадки"))
    if front in (4, 5):
        mixers = [d["equipment"].get("concrete_mixer", 0) for d in days]
        out.append(("Бетоносмесителей (макс. за день)", max(mixers), "шт."))
    if front in (5, 6, 7):
        fb = [(f["date"], f["analysis"]["triage"]["floors_built"]) for f in frames
              if f["analysis"]["triage"].get("floors_built")]
        if fb:
            out.append(("Этажей каркаса", fb[-1][1], "по оценке модели"))
            first, lastfb = fb[0], fb[-1]
            if lastfb[1] > first[1]:
                active = sum(d["active"] for d in days if first[0] <= d["date"] <= lastfb[0])
                out.append(("Цикл этажа", round(max(active, 1) / (lastfb[1] - first[1]), 1), "акт. дней/этаж"))
    if front in (7, 8):
        if latest("floors_glazed") is not None:
            out.append(("Остеклено этажей", latest("floors_glazed"), "по оценке модели"))
        if latest("facade_clad_pct") is not None:
            out.append(("Облицовано фасада", latest("facade_clad_pct"), "% видимого"))
    return out
