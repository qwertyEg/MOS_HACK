"""Модель Б на vision-LLM: EXTERNAL GLM-4.6V (двухшаговый разбор Никиты) и LOCAL VLM (батчи Дениса).

**GlmChecklistClassifier** — порт `api-solution/core/analyzer.py`: разведка
(качество, ракурс, техника total/working, счётчики, latest_stage, stage_likelihood)
→ чек-лист признаков 2–3 этапов-кандидатов. Кандидаты: последний видимый этап и
следующий за ним (разведка склонна отставать), этап, достигнутый по истории
(`context["front"]`), затем уверенные по вероятности; всего 2..3. Исправлено:

- любые ошибки (не только VLMError: кривой stage_likelihood списком, KeyError,
  битый JSON) превращаются в VLMError — вызывающий пропускает кадр, а не падает
  весь прогон (A7); потраченное до сбоя видно в `VLMError.usage`;
- `total=0` больше не превращается в одну фантомную машину (A15);
- `not_visible` / пропущенный ключ → UNSURE, который не голосует (A6);
- вероятности разведки этап не открывают — только ответы (см. scoring).

**LocalVlmClassifier** — порт `ModelB.ask_batch` Дениса: вопросы группами по
4–6 (батч из 26 вопросов «входит в ритм» и схлопывает «не видно» в «нет» — замер
Дениса), группа — чек-лист одного этапа, ответ ограничен JSON-схемой с
перечислением «да / нет / не видно». Исправлен баг D5: при разборе свободного
текста «Не видно», «не вижу», «кран не виден» давали NO («не» было в словаре
отрицаний), теперь — UNSURE. Сбой группы не превращается молча во «все не видно»
(баг D6): сначала повтор без схемы, потом VLMError.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from core import taxonomy
from core.contracts import Answer, ChecklistResult, FrameInfo, Provider
from core.stage import prompts
from core.stage.mask import masked_for_model
from core.vlm_client import OpenAICompatibleClient, Reply, Usage, VLMError, image_to_data_url, make_client

QUALITY = {"good", "night", "fog_rain", "obstructed", "blurred"}
VIEWS = {"top", "side", "ground", "unknown"}
_YES = {"yes", "да", "true", "1", "y"}
_NO = {"no", "нет", "false", "0", "n"}

# Кандидаты для чек-листа (замер Никиты на test_photos: разведка дала залитой плите
# этапа 4 лишь 0.2, поэтому «следующий за последним видимым» проверяется всегда).
CANDIDATE_THRESHOLD = 0.3
MAX_CANDIDATES = 3
MIN_CANDIDATES = 2


def _int_or_none(value: Any, lo: int = 0, hi: int | None = None) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    v = max(v, lo)
    return min(v, hi) if hi is not None else v


def normalize_triage(data: Any) -> dict:
    """Ответ разведки → нормализованный словарь. Никогда не падает на кривом типе поля."""
    data = data if isinstance(data, dict) else {}
    equipment_keys = set(taxonomy.equipment())
    equipment = []
    items = data.get("equipment")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("type", "other")).strip().lower()
        if key not in equipment_keys:
            key = "other"
        total = _int_or_none(item.get("total"))
        working = _int_or_none(item.get("working")) or 0
        if total is None:
            total = max(1, working)   # тип назван, число забыто — хотя бы одна единица
        if total == 0:
            continue                  # «0 штук» — это отсутствие, а не одна фантомная машина
        equipment.append({"type": key, "total": total, "working": min(working, total),
                          "evidence": str(item.get("evidence", ""))[:200]})

    raw = data.get("stage_likelihood")
    raw = raw if isinstance(raw, dict) else {}
    likelihood = {}
    for sid in sorted(taxonomy.stages()):
        v = raw.get(str(sid), raw.get(sid, 0))
        try:
            v = float(v or 0)
        except (TypeError, ValueError):
            v = 0.0
        likelihood[sid] = min(max(v, 0.0), 1.0) if v == v else 0.0   # NaN → 0

    latest = _int_or_none(data.get("latest_stage"))
    conflict = str(data.get("context_conflict") or "").strip()
    if conflict.lower() in ("null", "none", "нет", "-", "no"):
        conflict = ""
    quality, view = data.get("quality"), data.get("view")
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
        "context_conflict": conflict[:300] or None,
        "stage_likelihood": likelihood,
    }


def normalize_answers(data: Any, keys: list[str]) -> dict[str, Answer]:
    """{"answers": {...}} или плоский словарь → Answer по каждому ключу; нет ключа / not_visible → UNSURE."""
    data = data if isinstance(data, dict) else {}
    raw = data.get("answers") if isinstance(data.get("answers"), dict) else data
    out = {}
    for k in keys:
        v = str(raw.get(k, "")).strip().lower()
        out[k] = Answer.YES if v in _YES else Answer.NO if v in _NO else Answer.UNSURE
    return out


def pick_candidates(likelihood: dict[int, float], latest: int | None = None,
                    prev_front: int | None = None) -> list[int]:
    """prev_front — этап, достигнутый по истории: проверяем всегда, чтобы кадр сверился с ней."""
    ranked = sorted(likelihood, key=lambda s: -likelihood[s])
    picked: list[int] = []
    if latest:
        picked = [latest] + ([latest + 1] if latest + 1 in likelihood else [])
    if prev_front and prev_front in likelihood and prev_front not in picked:
        picked.append(prev_front)
    picked += [s for s in ranked if likelihood[s] >= CANDIDATE_THRESHOLD and s not in picked]
    picked = picked[:MAX_CANDIDATES]
    for s in ranked:
        if len(picked) >= MIN_CANDIDATES:
            break
        if s not in picked:
            picked.append(s)
    return sorted(picked)


def _sum(usages: list[Usage]) -> Usage:
    total = Usage()
    for u in usages:
        total.prompt_tokens += u.prompt_tokens
        total.cached_tokens += u.cached_tokens
        total.completion_tokens += u.completion_tokens
        total.cost_usd += u.cost_usd
        total.latency_ms += u.latency_ms
    return total


class TwoStepAnalyzer:
    """Разведка → чек-лист кандидатов по готовому data-URL кадра (кэш и кассета работают по URL)."""

    def __init__(self, client, strategy: str = "two_step"):
        if strategy not in ("two_step", "per_stage"):
            raise ValueError(f"неизвестная стратегия {strategy!r}: two_step | per_stage")
        self.client = client
        self.strategy = strategy

    def analyze(self, image_url: str, context: dict | None = None, keys: list[str] | None = None,
                masked: bool = False) -> dict:
        context = context or {}
        ctx_text = str(context.get("text") or "")
        extra = [ctx_text, prompts.MASK_NOTE if masked else ""]
        calls: list[tuple[str, Reply]] = []

        def ask(step: str, text: str, max_tokens: int) -> Reply:
            try:
                reply = self.client.ask_json(prompts.SYSTEM, image_url, [text, *extra], max_tokens=max_tokens)
            except VLMError as e:
                spent = [r.usage for _, r in calls] + ([e.usage] if e.usage else [])
                raise VLMError(f"{step}: {e}", usage=_sum(spent)) from e
            calls.append((step, reply))
            return reply

        try:
            reply = ask("triage", prompts.triage(), 1200)
            triage = normalize_triage(reply.data)
            if keys:
                groups = [list(keys)]
                candidates = sorted({sid for sid in taxonomy.stages() for k in keys
                                     if k in prompts.stage_sign_keys(sid)})
            else:
                candidates = pick_candidates(triage["stage_likelihood"], triage["latest_stage"],
                                             _int_or_none(context.get("front")))
                stage_groups = [candidates] if self.strategy == "two_step" else [[c] for c in candidates]
                groups = [prompts.sign_keys_for(g) for g in stage_groups]
            answers: dict[str, Answer] = {}
            comments = []
            for i, group_keys in enumerate(groups):
                if not keys and self.strategy == "two_step":
                    text = prompts.checklist_step(candidates)   # побайтно как у Никиты — ключ кассеты
                else:
                    text = prompts.checklist_keys_step(group_keys)
                reply = ask(f"checklist:{i}", text, 60 + 12 * len(group_keys))
                answers.update(normalize_answers(reply.data, group_keys))
                comment = reply.data.get("comment") if isinstance(reply.data, dict) else None
                if comment:
                    comments.append(str(comment)[:300])
        except VLMError:
            raise
        except Exception as e:  # noqa: BLE001 — что угодно в разборе ответа — сбой кадра, а не прогона
            raise VLMError(f"сбой разбора ответа GLM: {type(e).__name__}: {e}",
                           usage=_sum([r.usage for _, r in calls])) from e

        total = _sum([r.usage for _, r in calls])
        return {
            "triage": triage,
            "candidates": candidates,
            "answers": answers,
            "comments": comments,
            "usage": total,
            "calls": [{"step": s, "prompt_tokens": r.usage.prompt_tokens, "cached_tokens": r.usage.cached_tokens,
                       "completion_tokens": r.usage.completion_tokens, "cost_usd": r.usage.cost_usd,
                       "latency_ms": r.usage.latency_ms} for s, r in calls],
            "raw_text": {s: r.text for s, r in calls},
        }


class GlmChecklistClassifier:
    """StageClassifier (EXTERNAL): GLM-4.6V, двухшаговый разбор."""

    name = "glm"
    provider = Provider.EXTERNAL

    def __init__(self, client: OpenAICompatibleClient | None = None, strategy: str = "two_step",
                 model: str | None = None, image_max_side: int = 1280, mask_mode: str = "darken"):
        self.client = client if client is not None else make_client("zai", model)
        self.analyzer = TwoStepAnalyzer(self.client, strategy)
        self.image_max_side = image_max_side
        self.mask_mode = mask_mode

    def ready(self) -> tuple[bool, str]:
        return self.client.ready()

    def assess(self, image_bgr: np.ndarray, frame: FrameInfo, keys: list[str] | None = None,
               context: dict[str, Any] | None = None) -> ChecklistResult:
        """keys=None — признаки кандидатных этапов по разведке (спрашивать все 60 разом — терять
        качество ответа); context: front (этап по истории), text (контекст стройки), mask."""
        ctx = dict(context or {})
        ctx.setdefault("mask_mode", self.mask_mode)
        t0 = time.perf_counter()
        try:
            image, masked = masked_for_model(image_bgr, ctx)
            url = image_to_data_url(image, self.image_max_side)
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"кадр не подготовлен для GLM: {e}") from e
        return self.result_from_analysis(self.analyzer.analyze(url, ctx, keys, masked),
                                         latency_ms=(time.perf_counter() - t0) * 1000)

    def result_from_analysis(self, a: dict, latency_ms: float | None = None) -> ChecklistResult:
        answers = a["answers"]
        triage = a["triage"]
        equipment_hint: dict[str, int] = {}
        for e in triage["equipment"]:
            equipment_hint[e["type"]] = equipment_hint.get(e["type"], 0) + int(e["total"])
        usage: Usage = a["usage"]
        return ChecklistResult(
            answers=answers,
            scores={k: 1.0 if v is Answer.YES else 0.0 if v is Answer.NO else 0.5 for k, v in answers.items()},
            stage_likelihood=dict(triage["stage_likelihood"]),
            model=self.client.model, provider=Provider.EXTERNAL,
            latency_ms=round(latency_ms if latency_ms is not None else usage.latency_ms, 1),
            cost_usd=round(usage.cost_usd, 6),
            equipment_hint=equipment_hint,
            raw={"triage": triage, "candidates": a["candidates"], "comments": a["comments"],
                 "calls": a["calls"], "raw_text": a["raw_text"], "strategy": self.analyzer.strategy,
                 "prompt_version": prompts.PROMPT_VERSION},
        )


# --------------------------------------------------------------------------
# LOCAL VLM (Ollama / vLLM) — порт app/pipeline/model_b.py Дениса
# --------------------------------------------------------------------------

# Маркеры «не видно / не уверен» проверяются ДО «нет»: у Дениса «не» стояло в словаре
# отрицаний, и «Не видно» (ровно то, что просит системный промпт) давало NO.
_UNSURE_MARKERS = ("не видно", "не вижу", "не виден", "не видна", "не видны", "невидно", "не различ",
                   "не увер", "неуверен", "не могу", "не ясно", "неясно", "непонятно", "затрудня",
                   "сложно сказать", "нет данных", "not visible", "not_visible", "unsure", "unclear",
                   "can't tell", "cannot tell")
_TRIM = " \t\n.,!:;—-*_#«»\"'()"


def parse_answer(raw: Any) -> Answer:
    """Свободный ответ локальной модели → Answer. Сомнение — всегда UNSURE, а не NO."""
    low = str(raw or "").strip().lower().replace("ё", "е")
    if not low or any(m in low for m in _UNSURE_MARKERS):
        return Answer.UNSURE
    words = [w.strip(_TRIM) for w in low.split()]
    words = [w for w in words if w]
    if not words:
        return Answer.UNSURE
    if words[0] in ("да", "yes", "true"):
        return Answer.YES
    if words[0] in ("нет", "no", "false"):
        return Answer.NO
    # Развёрнутый ответ вопреки инструкции («здание не строится») угадывать не будем:
    # ошибочное «да» или «нет» дороже честного «не уверен».
    return Answer.UNSURE


class LocalVlmClassifier:
    """StageClassifier (LOCAL): локальная VLM по OpenAI-API, батчи по этапам с JSON-схемой."""

    name = "local_vlm"
    provider = Provider.LOCAL

    def __init__(self, client: OpenAICompatibleClient | None = None, group_size: int = 5,
                 max_tokens: int = 512, image_max_side: int = 1024, mask_mode: str = "darken",
                 model: str | None = None):
        # 512 токенов не потому, что ответ длинный: Qwen3-VL рассуждает перед ответом,
        # и при малом бюджете content приходит пустым (замер Дениса).
        self.client = client if client is not None else make_client("local", model)
        self.group_size = max(1, min(int(group_size), 8))
        self.max_tokens = max_tokens
        self.image_max_side = image_max_side   # выше 1024 на грубых вопросах качества не прибавляет
        self.mask_mode = mask_mode

    def ready(self) -> tuple[bool, str]:
        return self.client.ready()

    def select_keys(self, keys: list[str] | None, context: dict | None) -> list[str]:
        """Какие признаки спрашивать. Без явного списка — этапы вокруг известного фронта
        (front−1 … front+2), а без истории — признаки всех этапов."""
        if keys:
            return list(dict.fromkeys(keys))
        stages = sorted(taxonomy.stages())
        front = _int_or_none((context or {}).get("front"))
        if front in stages:
            stages = [s for s in stages if front - 1 <= s <= front + 2]
        return prompts.sign_keys_for(stages)

    def groups(self, keys: list[str]) -> list[list[str]]:
        """Группа — признаки одного этапа (естественная смысловая группа, локализует сбой разбора),
        порезанные на куски ≤ group_size."""
        remaining = list(dict.fromkeys(keys))
        out: list[list[str]] = []
        for sid in sorted(taxonomy.stages()):
            stage_keys = [k for k in prompts.stage_sign_keys(sid) if k in remaining]
            for k in stage_keys:
                remaining.remove(k)
            out += [stage_keys[i:i + self.group_size] for i in range(0, len(stage_keys), self.group_size)]
        out += [remaining[i:i + self.group_size] for i in range(0, len(remaining), self.group_size)]
        return [g for g in out if g]

    def _ask_group(self, url: str, group: list[str], system: str, spent: list[Usage]) -> tuple[dict[str, Answer], str]:
        text = prompts.local_questions(group)
        try:
            reply = self.client.ask_json(system, url, text, self.max_tokens, schema=prompts.local_schema(group))
        except VLMError as first:
            if first.usage:
                spent.append(first.usage)
            # Эндпоинт без ограничения декодирования (400 на response_format) или кривой ответ:
            # один повтор без схемы, JSON вырезается из текста.
            try:
                reply = self.client.ask_json(system, url, text, self.max_tokens)
            except VLMError as e:
                if e.usage:
                    spent.append(e.usage)
                raise VLMError(f"локальная VLM не ответила на группу {group}: {e} (со схемой: {first})",
                               usage=_sum(spent)) from e
        spent.append(reply.usage)
        data = reply.data if isinstance(reply.data, dict) else {}
        data = data.get("answers") if isinstance(data.get("answers"), dict) else data
        return {k: parse_answer(data.get(k)) for k in group}, reply.text

    def assess(self, image_bgr: np.ndarray, frame: FrameInfo, keys: list[str] | None = None,
               context: dict[str, Any] | None = None) -> ChecklistResult:
        ctx = dict(context or {})
        ctx.setdefault("mask_mode", self.mask_mode)
        t0 = time.perf_counter()
        try:
            image, masked = masked_for_model(image_bgr, ctx)
            url = image_to_data_url(image, self.image_max_side)
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"кадр не подготовлен для локальной VLM: {e}") from e
        system = prompts.LOCAL_SYSTEM + ("\n" + prompts.MASK_NOTE if masked else "")
        answers: dict[str, Answer] = {}
        raw: dict[str, str] = {}
        spent: list[Usage] = []
        try:
            for group in self.groups(self.select_keys(keys, ctx)):
                got, text = self._ask_group(url, group, system, spent)
                answers.update(got)
                raw[",".join(group)] = text[:500]
        except VLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"сбой разбора ответа локальной VLM: {type(e).__name__}: {e}", usage=_sum(spent)) from e
        return ChecklistResult(
            answers=answers,
            scores={k: 1.0 if v is Answer.YES else 0.0 if v is Answer.NO else 0.5 for k, v in answers.items()},
            model=self.client.model, provider=Provider.LOCAL,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1), cost_usd=0.0,
            raw={"groups": raw, "masked": masked},
        )
