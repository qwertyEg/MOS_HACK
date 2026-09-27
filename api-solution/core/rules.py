"""Отклонения «этап → техника» по одному кадру.

Правила из checklist.json (equipment_expected / optional / forbidden) проверяются
обобщённо, плюс несколько парных правил, которые общим списком не выразить:
экскаватор без самосвалов, бетононасос без миксеров и т.п. Отклонения по
графику и простоям считаются в timeline.py — им нужна серия кадров.
"""

CRITICAL, WARNING, INFO = "critical", "warning", "info"

# Этапы, где экскаватор грузит грунт или мусор в самосвалы.
_LOADING_STAGES = {1, 3, 4, 8}


def _counts(triage):
    present, working = {}, {}
    for e in triage.get("equipment", []):
        present[e["type"]] = present.get(e["type"], 0) + e["total"]
        working[e["type"]] = working.get(e["type"], 0) + e["working"]
    return present, working


def _dev(rule, severity, title, detail, stage=None):
    return {"rule": rule, "severity": severity, "title": title, "detail": detail, "stage": stage}


def frame_deviations(checklist, analysis, score):
    triage = analysis["triage"]
    if triage["quality"] in ("blurred", "obstructed"):
        return []
    front = score["front"]
    if front is None:
        return []
    present, working = _counts(triage)
    name = checklist.equipment_name
    stage = checklist.stage_by_id[front]
    out = []

    # Этапы, идущие на кадре: фронт плюс те, где есть активные подэтапы.
    active_stages = {front} | {sid for sid, st in score["stages"].items()
                               if any(v == "active" for v in st["substages"].values())}
    allowed = set()
    for sid in active_stages:
        s = checklist.stage_by_id[sid]
        allowed |= set(s["equipment_expected"]) | set(s["equipment_optional"])

    for eq, n in working.items():
        if n and eq in stage["equipment_forbidden"] and eq not in allowed:
            out.append(_dev("forbidden_equipment", WARNING,
                            f"Техника не по этапу: {name(eq)}",
                            f"{name(eq)} работает ({n} шт.), а на этапе «{stage['name']}» она не ожидается.",
                            front))

    if working.get("excavator") and not present.get("dump_truck") and front in _LOADING_STAGES:
        out.append(_dev("excavator_no_trucks", WARNING,
                        "Экскаватор работает без самосвалов",
                        "Вывоз грунта не обеспечен — возможное снижение темпа работ.", front))

    loaders = sum(working.get(k, 0) for k in ("excavator", "wheel_loader", "backhoe_loader"))
    if present.get("dump_truck", 0) >= 2 and not loaders and front in _LOADING_STAGES:
        out.append(_dev("trucks_waiting", INFO,
                        "Самосвалы без погрузки",
                        f"На площадке {present['dump_truck']} самосвала, но экскаватор или погрузчик не работает.",
                        front))

    # Только работающий насос: стоящий часто просто ждёт вывоза (PLAN.md §3.9).
    if working.get("concrete_pump") and not present.get("concrete_mixer"):
        out.append(_dev("pump_no_mixer", WARNING,
                        "Бетононасос без бетоносмесителей",
                        "Бетон не подвозится — бетонирование может прерваться (риск холодного шва).", front))

    if working.get("asphalt_paver") and not present.get("roller"):
        out.append(_dev("paver_no_roller", WARNING,
                        "Асфальтоукладчик без катка",
                        "Уложенный асфальт не уплотняется — риск брака покрытия.", front))

    if working.get("drilling_rig") and front == 2 and not present.get("concrete_mixer"):
        out.append(_dev("rig_no_mixer", INFO,
                        "Буровая без бетоносмесителя",
                        "Если бурят буронабивные сваи, скважины нужно бетонировать сразу.", front))

    # Башенный кран стоит весь цикл и на одиночном кадре почти всегда «не работает».
    total_present = sum(v for k, v in present.items() if k != "tower_crane")
    if total_present >= 2 and not any(working.values()) and triage["quality"] == "good":
        out.append(_dev("all_idle", INFO,
                        "Техника на площадке не работает",
                        f"Видно {total_present} ед. техники, ни одна не в работе. Если так несколько дней — простой.",
                        front))
    return out


def missing_equipment(checklist, stage_id, present_types):
    """Нет ни одной единицы из обязательной техники этапа — проверяется по дню, не по кадру."""
    stage = checklist.stage_by_id[stage_id]
    expected = stage["equipment_expected"]
    if expected and not (set(expected) & set(present_types)):
        names = ", ".join(checklist.equipment_name(k) for k in expected)
        return _dev("missing_equipment", WARNING,
                    f"Нет техники этапа «{stage['name']}»",
                    f"За день не замечено ни одной единицы из ожидаемой техники: {names}.", stage_id)
    return None
