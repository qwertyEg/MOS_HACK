"""Ответы модели по кадру → статусы подэтапов, текущий этап и процент готовности.

Логика опирается на то, что этапы здания идут по порядку: если видна работа
этапа N, то этапы до N в основном выполнены. «Фронт» — самый поздний этап с
достаточными признаками; всё до фронта считается сделанным, кроме подэтапов,
которые на кадре явно ещё идут (каркас верхних этажей при начатом фасаде).
"""

DONE, ACTIVE, NOT_STARTED, UNKNOWN = "done", "active", "not_started", "unknown"

# Этап считается увиденным при таком балле. Балл — доля подтверждённых must_have
# из чек-листа, а если этап чек-листом не проверяли — вероятность из разведки.
# Среднее двух не годится: на прогоне test_photos разведка дала этапу 4 ноль,
# а чек-лист подтвердил половину его признаков — и армирование плиты уходило в котлован.
FRONT_THRESHOLD = 0.5
ACTIVE_CREDIT = 0.5  # вклад идущего подэтапа, если нет числовой метрики
# Разведка назвала этап самым поздним видимым — это сильнее, чем вероятность
# «работы идут сейчас», которая для готового фасада близка к нулю.
LATEST_PRIOR = 0.7
# Мин. вероятность из разведки, при которой этап выше latest_stage можно открыть
# одним отличительным признаком; иначе нужен балл чек-листа ≥ FRONT_THRESHOLD.
SIGN_OPENS_MIN_LIKELIHOOD = 0.3


def _likelihood(triage, sid):
    lk = triage.get("stage_likelihood", {})
    value = float(lk.get(str(sid), lk.get(sid, 0.0)) or 0.0)
    return max(value, LATEST_PRIOR) if triage.get("latest_stage") == sid else value


def substage_status(sub, answers):
    active = any(answers.get(k) == "yes" for k in sub["active_when"])
    done_keys = sub["done_when"]
    done = bool(done_keys) and all(answers.get(k) == "yes" for k in done_keys)
    if active:
        return ACTIVE
    if done:
        return DONE
    asked = [k for k in sub["active_when"] + done_keys if k in answers]
    if not asked or all(answers[k] == "unsure" for k in asked):
        return UNKNOWN
    return NOT_STARTED


def stage_evidence(stage, answers):
    """Доля подтверждённых must_have за вычетом противоречий; None, если этап не спрашивали."""
    asked = [k for k in stage["must_have"] + stage["must_not_have"] if k in answers]
    if not asked:
        return None
    have = stage["must_have"]
    score = sum(answers.get(k) == "yes" for k in have) / len(have) if have else 0.0
    contra = stage["must_not_have"]
    if contra:
        score -= 0.5 * sum(answers.get(k) == "yes" for k in contra) / len(contra)
    return max(score, 0.0)


def metric_fraction(stage_id, triage, floors_total):
    """Числовая доля готовности этапа, если её можно снять с кадра."""
    fb, fg, clad = triage.get("floors_built"), triage.get("floors_glazed"), triage.get("facade_clad_pct")
    if stage_id == 5 and fb and floors_total:
        return min(fb / floors_total, 1.0)
    if stage_id == 7:
        parts = []
        if fg is not None and floors_total:
            parts.append(min(fg / floors_total, 1.0))
        if clad is not None:
            parts.append(clad / 100)
        if parts:
            return sum(parts) / len(parts)
    return None


def evaluate(checklist, analysis, answers=None, floors_total=None):
    """answers — ответы с учётом «защёлкнутых» признаков из хронологии; по умолчанию — ответы кадра."""
    triage = analysis["triage"]
    answers = analysis["answers"] if answers is None else answers
    # Признаки общие для нескольких этапов (опалубка — и подземная часть, и каркас),
    # поэтому ответам доверяем только по этапам, которые разведка выбрала кандидатами.
    # Остальные этапы оцениваются лишь по вероятности из разведки.
    candidates = {int(c) for c in analysis.get("candidates", [])}

    stages = {}
    for stage in checklist.stages:
        sid = stage["id"]
        if sid in candidates:
            subs = {sub["id"]: substage_status(sub, answers) for sub in stage["substages"]}
            evidence = stage_evidence(stage, answers)
        else:
            subs = {sub["id"]: UNKNOWN for sub in stage["substages"]}
            evidence = None
        lk = _likelihood(triage, sid)
        score = lk if evidence is None else evidence
        distinct = checklist.distinctive[sid]
        by_sign = any(status in (ACTIVE, DONE) and any(answers.get(k) == "yes" for k in
                                                      set(sub["active_when"] + sub["done_when"]) & distinct)
                      for sub, status in zip(stage["substages"], subs.values()))
        # Одним признаком подэтапа этап открывается, только если разведка его допускает.
        # Замер: на армировании фундаментной плиты «опалубка наверху» = да (модель
        # приняла опалубку стен подвала) открывала каркас, хотя разведка дала ему 0.
        latest = triage.get("latest_stage")
        seen = by_sign and ((latest is not None and sid <= latest) or lk >= SIGN_OPENS_MIN_LIKELIHOOD)
        stages[sid] = {"score": round(score, 3), "evidence": evidence, "likelihood": lk,
                       "substages": subs, "seen": seen or score >= FRONT_THRESHOLD}

    front = max((sid for sid, s in stages.items() if s["seen"]), default=None)

    total = 0.0
    for stage in checklist.stages:
        sid = stage["id"]
        st = stages[sid]
        metric = metric_fraction(sid, triage, floors_total)
        subs = stage["substages"]
        progress = 0.0
        if front is not None and sid <= front:
            for i, sub in enumerate(subs):
                status = st["substages"][sub["id"]]
                later_seen = any(st["substages"][s["id"]] in (ACTIVE, DONE) for s in subs[i + 1:])
                if status == DONE:
                    credit = 1.0
                elif status == ACTIVE:
                    credit = metric if metric is not None else ACTIVE_CREDIT
                elif sid < front or later_seen:
                    # Работа более позднего этапа или подэтапа уже видна — этот выполнен.
                    credit = 1.0
                else:
                    credit = 0.0
                progress += sub["weight"] / 100 * credit
        st["progress"] = round(progress, 3)
        st["status"] = (DONE if progress >= 0.999 else ACTIVE if progress > 0
                        else UNKNOWN if not st["seen"] and front is None else NOT_STARTED)
        total += stage["weight"] * progress

    active_subs = [sub_id for st in stages.values() for sub_id, v in st["substages"].items() if v == ACTIVE]
    return {"front": front, "overall_pct": round(total, 1), "stages": stages, "active_substages": active_subs}
