"""LOCAL модель Б (SigLIP) на фейковом эмбеддере — без torch и без весов.

Фейк раскладывает каждую формулировку в свой базисный вектор, поэтому тест
точно задаёт s⁺ и s⁻ по любому признаку и проверяет арифметику порогов,
кэш текстов, маску и распределение по этапам.
"""
import datetime as dt
import importlib.util
import re

import numpy as np
import pytest

from core import taxonomy
from core.contracts import Answer, FrameInfo, Provider
from core.stage import get_classifier
from core.stage.checklist_clip import ClipChecklistClassifier, ClipConfig, load_prompts

FRAME = FrameInfo(frame_id=1, camera_id=1, site_id=1, captured_at=dt.datetime(2026, 6, 1, 9, tzinfo=dt.timezone.utc),
                  width=640, height=480)
PROMPTS = load_prompts()
TEMPLATE = PROMPTS["template"]
# Арифметика порогов проверяется в исходной постановке (k модели, 0.62/0.38);
# откалиброванные умолчания — отдельным тестом ниже.
LEGACY = dict(scale=None, yes_threshold=0.62, no_threshold=0.38)


class BasisEmbedder:
    name = "fake-siglip"
    logit_scale = 100.0

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index: dict[str, int] = {}
        self.text_calls = 0
        self.images_seen: list[np.ndarray] = []
        self.weights: dict[str, float] = {}

    def embed_texts(self, texts):
        self.text_calls += 1
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            j = self.index.setdefault(t, len(self.index))
            out[i, j] = 1.0
        return out

    def embed_images(self, images):
        self.images_seen.append(images[0])
        v = np.zeros(self.dim, np.float32)
        for text, w in self.weights.items():
            v[self.index[text]] = w
        v[-1] = np.sqrt(max(0.0, 1.0 - float((v ** 2).sum())))   # «остальная сцена» — держит норму 1
        return np.stack([v] * len(images))

    def lean(self, sign: str, x: float):
        """Сдвинуть кадр к позитивным (x > 0) или негативным (x < 0) формулировкам признака."""
        side = "pos" if x > 0 else "neg"
        for t in PROMPTS["signs"][sign][side]:
            self.weights[TEMPLATE.format(t)] = abs(x)

    def lean_stage(self, sid: int, x: float):
        for t in PROMPTS["stages"][str(sid)]:
            self.weights[TEMPLATE.format(t)] = x


@pytest.fixture
def fake():
    return BasisEmbedder()


def image(h=480, w=640, value=150):
    return np.full((h, w, 3), value, np.uint8)


def test_thresholds_give_yes_no_unsure(fake):
    clf = ClipChecklistClassifier(embedder=fake, config=ClipConfig(**LEGACY))
    clf.assess(image(), FRAME, keys=["pit"])            # первый вызов строит словарь формулировок
    fake.lean("pit", 0.05)          # diff = √2·0.05 ≈ 0.07 → p = σ(7) ≈ 1
    fake.lean("cladding", -0.05)    # в сторону «нет»
    fake.lean("slab", 0.003)        # diff ≈ 0.004 → p ≈ 0.6 — между порогами
    r = clf.assess(image(), FRAME, keys=["pit", "cladding", "slab", "rebar"])
    assert r.answers == {"pit": Answer.YES, "cladding": Answer.NO, "slab": Answer.UNSURE, "rebar": Answer.UNSURE}
    assert r.scores["pit"] > 0.99 and r.scores["cladding"] < 0.01 and 0.5 < r.scores["slab"] < 0.62
    assert r.scores["rebar"] == pytest.approx(0.5)
    assert r.provider is Provider.LOCAL and r.model == "fake-siglip" and r.cost_usd == 0
    assert r.unsure_ratio == pytest.approx(0.5)


def test_thresholds_are_configurable_globally_and_per_sign(fake):
    clf = ClipChecklistClassifier(embedder=fake, config=ClipConfig(**LEGACY))
    clf.assess(image(), FRAME, keys=["slab"])
    fake.lean("slab", 0.003)
    assert clf.assess(image(), FRAME, keys=["slab"]).answers["slab"] is Answer.UNSURE
    per_sign = ClipChecklistClassifier(embedder=fake, config=ClipConfig(**LEGACY), per_sign={"slab": (0.55, 0.3)})
    assert per_sign.assess(image(), FRAME, keys=["slab"]).answers["slab"] is Answer.YES
    loose = get_classifier("siglip", embedder=fake, scale=None, yes_threshold=0.55)
    assert loose.config.yes_threshold == 0.55
    assert loose.assess(image(), FRAME, keys=["slab"]).answers["slab"] is Answer.YES


def test_scale_none_is_taken_from_the_model(fake):
    clf = ClipChecklistClassifier(embedder=fake, config=ClipConfig(**LEGACY))
    clf.assess(image(), FRAME, keys=["slab"])
    fake.lean("slab", 0.003)
    p100 = clf.assess(image(), FRAME, keys=["slab"]).scores["slab"]
    fake.logit_scale = 400.0
    assert clf.assess(image(), FRAME, keys=["slab"]).scores["slab"] > p100


def test_text_embeddings_are_computed_once(fake):
    clf = ClipChecklistClassifier(embedder=fake)
    clf.assess(image(), FRAME)
    calls = fake.text_calls
    for _ in range(3):
        r = clf.assess(image(), FRAME)
    assert fake.text_calls == calls
    assert set(r.answers) == set(taxonomy.signs())      # keys=None — все 60 признаков


def test_mask_is_applied_before_embedding(fake):
    clf = ClipChecklistClassifier(embedder=fake)
    visible = np.ones((480, 640), bool)
    visible[:200] = False                               # верх — соседние дома
    r = clf.assess(image(value=200), FRAME, keys=["pit"], context={"mask": visible})
    seen = fake.images_seen[-1]
    assert seen[:200].mean() < 100 and seen[300:].mean() == pytest.approx(200)   # darken ×0.3
    assert r.raw["masked"] is True
    clf.assess(image(value=200), FRAME, keys=["pit"], context={"mask": visible, "mask_mode": "black"})
    assert fake.images_seen[-1][:200].max() == 0


def test_stage_likelihood_is_a_distribution_over_8_stages(fake):
    clf = ClipChecklistClassifier(embedder=fake)
    clf.assess(image(), FRAME, keys=["pit"])
    fake.lean_stage(3, 0.03)
    r = clf.assess(image(), FRAME, keys=["pit"])
    assert sorted(r.stage_likelihood) == list(range(1, 9))
    assert sum(r.stage_likelihood.values()) == pytest.approx(1.0, abs=1e-3)
    assert max(r.stage_likelihood, key=r.stage_likelihood.get) == 3


def test_prompt_file_covers_every_sign_and_stage_in_english():
    signs = PROMPTS["signs"]
    assert set(signs) == set(taxonomy.signs())
    cyrillic = re.compile("[а-яА-ЯёЁ]")
    for key, spec in signs.items():
        assert spec["pos"] and spec["neg"], key
        for t in spec["pos"] + spec["neg"]:
            assert len(t) > 15 and not cyrillic.search(t), (key, t)
        # отрицание CLIP понимает плохо: негатив не должен быть «no <объект>»
        assert not any(t.lower().startswith(("no ", "without ")) for t in spec["neg"]), key
    assert sorted(int(s) for s in PROMPTS["stages"]) == sorted(taxonomy.stages())


def test_ready_reports_missing_ml_packages(monkeypatch, fake):
    assert ClipChecklistClassifier(embedder=fake).ready() == (True, "")
    real_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("torch", "transformers") else real_find(name, *a))
    ok, reason = ClipChecklistClassifier().ready()
    assert not ok and "torch" in reason


def test_unknown_sign_key_is_an_error_not_silent_unsure(fake):
    clf = ClipChecklistClassifier(embedder=fake)
    with pytest.raises(KeyError):
        clf.assess(image(), FRAME, keys=["no_such_sign"])


def test_heavy_ml_packages_are_imported_lazily():
    import subprocess
    import sys

    code = ("import sys, numpy as np, datetime as dt\n"
            "from core.stage import get_classifier, quality, mask, scoring, sequence\n"
            "from core.stage import checklist_clip, checklist_vlm\n"
            "clf = get_classifier('siglip'); clf.ready()\n"
            "print(','.join(m for m in ('torch', 'transformers', 'open_clip') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""                       # веб-слой и тесты не тянут torch


def test_features_accept_transformers5_output():
    """transformers 5: get_*_features возвращает объект с pooler_output, а не тензор (падало на сервере)."""
    from core.stage.checklist_clip import _features

    class Tensor:
        def float(self):
            return self

    t = Tensor()
    assert _features(t) is t                      # transformers 4.x — уже тензор

    class Output:
        pooler_output = t

    assert _features(Output()) is t               # transformers 5.x — BaseModelOutputWithPooling


def test_calibrated_defaults_need_a_clear_margin_for_yes(fake):
    """Умолчания (k = 35, 0.8/0.46): «да» — только при разности сходства ≥ 0.04; слабый перевес,
    на котором SigLIP с k модели уже отвечал «да», теперь «не уверен» (калибровка по демо-объектам)."""
    clf = ClipChecklistClassifier(embedder=fake)
    assert (clf.config.scale, clf.config.yes_threshold, clf.config.no_threshold) == (35.0, 0.8, 0.46)
    clf.assess(image(), FRAME, keys=["pit"])
    fake.lean("pit", 0.035)         # diff ≈ 0.049 → p = σ(1.7) ≈ 0.85 → «да»
    fake.lean("slab", 0.01)         # diff ≈ 0.014 → p ≈ 0.62 — раньше «да», теперь «не уверен»
    fake.lean("cladding", -0.005)   # diff ≈ −0.007 → p ≈ 0.44 → «нет»
    r = clf.assess(image(), FRAME, keys=["pit", "slab", "cladding"])
    assert r.answers == {"pit": Answer.YES, "slab": Answer.UNSURE, "cladding": Answer.NO}
