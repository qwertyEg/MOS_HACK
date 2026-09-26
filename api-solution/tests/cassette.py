"""Запись и воспроизведение ответов GLM.

Ключ записи — хэш (модель, thinking, системный промпт, текст запроса, картинка).
Повторный прогон с теми же промптами не тратит токены вовсе; после правки
промпта платно только то, что изменилось. Сырой текст ответа хранится как есть
и при воспроизведении заново проходит extract_json — парсер тоже тестируется.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from core.glm import Reply, Usage, extract_json


class CassetteMiss(Exception):
    pass


class CassetteClient:
    def __init__(self, path: Path, real=None, model="glm-4.6v", thinking=False):
        self.path = path
        self.real = real
        self.model = real.model if real else model
        self.thinking = real.thinking if real else thinking
        self.tape = json.loads(path.read_text()) if path.exists() else {}
        self.recorded = 0
        self.replayed = 0

    def _key(self, system, image_url, text):
        img = hashlib.sha1(image_url.encode()).hexdigest()
        raw = "\x1f".join([self.model, str(self.thinking), system, text, img])
        return hashlib.sha1(raw.encode()).hexdigest()

    def ask_json(self, system, image_url, text, max_tokens):
        key = self._key(system, image_url, text)
        if key in self.tape:
            item = self.tape[key]
            calls = [Usage(**u) for u in item["calls"]]
            total = Usage(*(sum(getattr(c, f) for c in calls) for f in
                            ("prompt_tokens", "cached_tokens", "completion_tokens", "cost_usd", "latency_ms")))
            self.replayed += 1
            return Reply(extract_json(item["raw_text"]), item["raw_text"], total, calls)
        if self.real is None:
            raise CassetteMiss("нет записи ответа и нет ключа API")
        reply = self.real.ask_json(system, image_url, text, max_tokens)
        self.tape[key] = {"step": text.split("\n", 1)[0], "raw_text": reply.raw_text,
                          "calls": [asdict(c) for c in reply.calls]}
        # Пишем сразу: если прогон упадёт на середине, оплаченное не потеряется.
        self.path.write_text(json.dumps(self.tape, ensure_ascii=False, indent=1))
        self.recorded += 1
        return reply
