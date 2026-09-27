"""Доступ к справочнику этапов (reference/checklist.json)."""

import hashlib
import json
from functools import cached_property

from .config import CHECKLIST_PATH


class Checklist:
    def __init__(self, path=CHECKLIST_PATH):
        raw = path.read_bytes()
        self.data = json.loads(raw)
        self.digest = hashlib.sha1(raw).hexdigest()[:10]

    @cached_property
    def stages(self):
        return self.data["stages"]

    @cached_property
    def stage_by_id(self):
        return {s["id"]: s for s in self.stages}

    @cached_property
    def signs(self):
        return {s["key"]: s for s in self.data["signs"]}

    @cached_property
    def equipment(self):
        return {e["key"]: e for e in self.data["equipment"]}

    def stage_sign_keys(self, stage_id):
        """Все признаки, нужные для разбора этапа: этапные и подэтапные, без повторов."""
        stage = self.stage_by_id[stage_id]
        keys = stage["must_have"] + stage["must_not_have"]
        for sub in stage["substages"]:
            keys += sub["active_when"] + sub["done_when"]
        return list(dict.fromkeys(keys))

    def positive_signs(self, stage_id):
        """Признаки, «да» на которые говорит в пользу этапа (без must_not_have)."""
        stage = self.stage_by_id[stage_id]
        keys = set(stage["must_have"])
        for sub in stage["substages"]:
            keys |= set(sub["active_when"]) | set(sub["done_when"])
        return keys

    @cached_property
    def distinctive(self):
        """Признаки этапа, которых нет среди положительных у более ранних этапов.

        Только они могут сдвинуть фронт работ вперёд: «идёт разработка грунта»
        есть и у котлована (3), и у засыпки пазух (4.6), и без этого правила
        котлован на прогоне test_photos превращался в подземный монолит.
        """
        out, earlier = {}, set()
        for s in sorted(self.stages, key=lambda s: s["id"]):
            out[s["id"]] = self.positive_signs(s["id"]) - earlier
            earlier |= self.positive_signs(s["id"])
        return out

    def equipment_name(self, key):
        eq = self.equipment.get(key)
        return eq["name"] if eq else key
