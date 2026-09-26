"""Разбор одного кадра моделью: разведка → чек-листы вероятных этапов.

Стратегия определяет только то, как этапы-кандидаты раскладываются по запросам
чек-листа. two_step — все кандидаты одним запросом (дёшево), per_stage — по
запросу на этап (точнее на длинных списках, дороже). Разведка общая: без неё
нет техники, счётчиков и выбора кандидатов.
"""

import hashlib
import json
from dataclasses import asdict

from . import config, prompts
from .glm import GLMClient

QUALITY = {"good", "night", "fog_rain", "obstructed", "blurred"}
VIEWS = {"top", "side", "ground", "unknown"}
YES = {"yes", "да", "true", "1", "y"}
NO = {"no", "нет", "false", "0", "n"}

# Кандидаты для чек-листа: последний видимый этап и следующий за ним (разведка
# склонна отставать: на прогоне test_photos залитую плиту этапа 4 не проверили,
# потому что разведка дала этапу 4 лишь 0.2), затем этапы с вероятностью от порога.
# Всего не больше MAX_CANDIDATES; если уверенных нет — два лучших по вероятности.
CANDIDATE_THRESHOLD = 0.3
MAX_CANDIDATES = 3
MIN_CANDIDATES = 2


def two_step_groups(candidates):
    return [candidates]


def per_stage_groups(candidates):
    return [[c] for c in candidates]


STRATEGIES = {
    "two_step": two_step_groups,
    "per_stage": per_stage_groups,
}


def _int_or_none(value, lo=0, hi=None):
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    v = max(v, lo)
    return min(v, hi) if hi is not None else v


def normalize_triage(data, checklist):
    equipment = []
    for item in data.get("equipment") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("type", "other")).strip().lower()
        if key not in checklist.equipment:
            key = "other"
        total = _int_or_none(item.get("total")) or 1
        working = min(_int_or_none(item.get("working")) or 0, total)
        equipment.append({"type": key, "total": total, "working": working,
                          "evidence": str(item.get("evidence", ""))[:200]})

    likelihood = {}
    raw = data.get("stage_likelihood") or {}
    for s in checklist.stages:
        try:
            v = float(raw.get(str(s["id"]), raw.get(s["id"], 0)) or 0)
        except (TypeError, ValueError):
            v = 0.0
        likelihood[s["id"]] = min(max(v, 0.0), 1.0)

    latest = _int_or_none(data.get("latest_stage"))
    quality = data.get("quality")
    view = data.get("view")
    return {
        "quality": quality if quality in QUALITY else "good",
        "view": view if view in VIEWS else "unknown",
        "description": str(data.get("description", ""))[:500],
        "equipment": equipment,
        "workers_count": _int_or_none(data.get("workers_count")),
        "floors_built": _int_or_none(data.get("floors_built")),
        "floors_glazed": _int_or_none(data.get("floors_glazed")),
        "facade_clad_pct": _int_or_none(data.get("facade_clad_pct"), 0, 100),
        "pit_area_pct": _int_or_none(data.get("pit_area_pct"), 0, 100),
        "latest_stage": latest if latest in likelihood else None,
        "stage_likelihood": likelihood,
    }


def normalize_answers(data, keys):
    raw = data.get("answers") if isinstance(data.get("answers"), dict) else data
    out = {}
    for k in keys:
        v = str(raw.get(k, "")).strip().lower()
        out[k] = "yes" if v in YES else "no" if v in NO else "unsure"
    return out


def pick_candidates(likelihood, latest=None):
    ranked = sorted(likelihood, key=lambda s: -likelihood[s])
    picked = []
    if latest:
        picked = [latest] + ([latest + 1] if latest + 1 in likelihood else [])
    picked += [s for s in ranked if likelihood[s] >= CANDIDATE_THRESHOLD and s not in picked]
    picked = picked[:MAX_CANDIDATES]
    for s in ranked:
        if len(picked) >= MIN_CANDIDATES:
            break
        if s not in picked:
            picked.append(s)
    return sorted(picked)


class Analyzer:
    def __init__(self, checklist, storage=None, client=None, strategy="two_step"):
        self.checklist = checklist
        self.storage = storage
        self.client = client or GLMClient()
        self.strategy = strategy

    def cache_key(self, sha):
        parts = [sha, self.client.model, str(self.client.thinking), self.strategy,
                 self.checklist.digest, config.PROMPT_VERSION]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()

    def analyze(self, sha, image_url_fn):
        """image_url_fn — ленивый data-URL: при попадании в кэш картинку не кодируем."""
        key = self.cache_key(sha)
        if self.storage:
            cached = self.storage.get_analysis(key)
            if cached:
                return cached, True

        image_url = image_url_fn()
        calls = []

        def ask(step, text, max_tokens):
            reply = self.client.ask_json(prompts.SYSTEM, image_url, text, max_tokens=max_tokens)
            # Пишем расход сразу: если следующий шаг упадёт, потраченное не потеряется.
            if self.storage:
                for u in reply.calls:
                    self.storage.log_call(key, step, self.client.model, u)
            calls.append((step, reply))
            return reply

        reply = ask("triage", prompts.triage(self.checklist), 1200)
        triage = normalize_triage(reply.data, self.checklist)

        candidates = pick_candidates(triage["stage_likelihood"], triage["latest_stage"])
        answers, comments = {}, []
        for group in STRATEGIES[self.strategy](candidates):
            keys = prompts.sign_keys_for(self.checklist, group)
            reply = ask("checklist:" + ",".join(map(str, group)),
                        prompts.checklist_step(self.checklist, group), 60 + 12 * len(keys))
            answers.update(normalize_answers(reply.data, keys))
            if reply.data.get("comment"):
                comments.append(str(reply.data["comment"])[:300])

        result = {
            "model": self.client.model,
            "thinking": self.client.thinking,
            "strategy": self.strategy,
            "triage": triage,
            "candidates": candidates,
            "answers": answers,
            "comments": comments,
            "usage": {
                "prompt_tokens": sum(r.usage.prompt_tokens for _, r in calls),
                "cached_tokens": sum(r.usage.cached_tokens for _, r in calls),
                "completion_tokens": sum(r.usage.completion_tokens for _, r in calls),
                "cost_usd": sum(r.usage.cost_usd for _, r in calls),
                "latency_ms": sum(r.usage.latency_ms for _, r in calls),
            },
            "calls": [{"step": step, **asdict(r.usage)} for step, r in calls],
            "raw": {step: r.raw_text for step, r in calls},
        }
        # Через JSON, чтобы свежий результат не отличался от кэшированного
        # (ключи stage_likelihood становятся строками в обоих случаях).
        result = json.loads(json.dumps(result, ensure_ascii=False))
        if self.storage:
            self.storage.save_analysis(key, sha, self.client.model, self.client.thinking, self.strategy, result)
        return result, False
