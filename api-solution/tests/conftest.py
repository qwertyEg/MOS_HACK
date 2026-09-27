import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.checklist import Checklist  # noqa: E402
from core.glm import Reply, Usage  # noqa: E402


@pytest.fixture(scope="session")
def checklist():
    return Checklist()


class FakeClient:
    """Подменяет GLM: разведка возвращает заданный triage, чек-лист — «yes» на заданные признаки."""

    model = "glm-4.6v"
    thinking = False

    def __init__(self, triage, yes=(), no=()):
        self.triage = triage
        self.yes, self.no = set(yes), set(no)
        self.requests = []

    def ask_json(self, system, image_url, prompt, max_tokens):
        text = prompt if isinstance(prompt, str) else "\n\n".join(p for p in prompt if p)
        self.requests.append(text)
        usage = Usage(prompt_tokens=3000, cached_tokens=1500, completion_tokens=300, cost_usd=0.001)
        if text.startswith("Шаг 1"):
            return Reply(self.triage, "{}", usage, [usage])
        keys = [line[2:].split(":")[0] for line in text.splitlines() if line.startswith("- ")]
        answers = {k: "yes" if k in self.yes else "no" if k in self.no else "unsure" for k in keys}
        return Reply({"answers": answers, "comment": "тест"}, "{}", usage, [usage])


def triage(likelihood, equipment=(), **measure):
    return {
        "quality": "good", "view": measure.get("view", "side"), "description": "тест",
        "equipment": [{"type": t, "total": n, "working": w, "evidence": ""} for t, n, w in equipment],
        "workers_count": measure.get("workers_count"),
        "floors_built": measure.get("floors_built"),
        "floors_glazed": measure.get("floors_glazed"),
        "facade_clad_pct": measure.get("facade_clad_pct"),
        "pit_area_pct": measure.get("pit_area_pct"),
        "latest_stage": measure.get("latest_stage"),
        "stage_likelihood": {str(k): v for k, v in likelihood.items()},
    }
