"""Turni JSONL e ricerca semantica in RAM, separati dall'evidenza documentale."""
from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from collections import deque
from typing import Sequence

import numpy as np

from ragmens_core.config import Config
from ragmens_core.embeddings import encode_texts
from ragmens_core.retrieval import source_scope


class ConversationMemory:
    def __init__(self, config: Config, embedder: object, corpus_signature: str):
        self.config = config
        self.embedder = embedder
        self.corpus_signature = corpus_signature
        session_hash = hashlib.sha256(config.session_id.encode("utf-8")).hexdigest()
        corpus_hash = hashlib.sha256(corpus_signature.encode("utf-8")).hexdigest()
        self.path = config.artifact_dir / "memory" / corpus_hash / f"{session_hash}.jsonl"
        self._recent: deque[dict] = deque(maxlen=config.memory_recent_turns)
        self._items: list[dict] = []
        if config.use_memory and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        self._validate(item)
                        if (item["session_id"] == config.session_id
                                and item["corpus_signature"] == corpus_signature):
                            self._items.append(item)
                    except (ValueError, TypeError, KeyError):
                        warnings.warn(f"Record memoria non valido ignorato: {self.path}, riga {number}.",
                                      stacklevel=2)

            self._recent.extend(self._items[-config.memory_recent_turns:])

    @staticmethod
    def _validate(item: dict) -> None:
        if not isinstance(item, dict) or item.get("version") != 1:
            raise ValueError("Schema memoria non valido.")
        for key in ("id", "session_id", "corpus_signature", "question", "answer"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError("Campo memoria non valido.")
        if not isinstance(item["sources"], list):
            raise ValueError("Fonti della memoria non valide.")
        used = source_scope(item["sources"])
        if item["scope"] is not None and not isinstance(item["scope"], list):
            raise ValueError("Ambito della memoria non valido.")
        scope = source_scope(item["scope"])
        if scope is not None and not set(used).issubset(scope):
            raise ValueError("Fonti esterne all'ambito della memoria.")

    def add_turn(self, question: str, answer: str, *, sources: Sequence[str] = (),
                 scope: Sequence[str] | None = None) -> None:
        if not self.config.use_memory:
            return
        normalized = source_scope(scope)
        used = source_scope(sources)
        if used is None:
            raise ValueError("Specificare le fonti effettivamente incluse, anche come lista vuota.")
        item = {
            "version": 1, "id": str(uuid.uuid4()), "session_id": self.config.session_id,
            "corpus_signature": self.corpus_signature,
            "scope": list(normalized) if normalized is not None else None,
            "sources": list(used), "question": question, "answer": answer,
        }
        self._validate(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._items.append(item)
        self._recent.append(item)

    def recall(self, query: str, sources: Sequence[str] | None = None) -> dict[str, list[dict]]:
        if not self.config.use_memory:
            return {"recent": [], "relevant": []}
        scope = source_scope(sources)

        def allowed(item: dict) -> bool:
            return (item["session_id"] == self.config.session_id
                    and item["corpus_signature"] == self.corpus_signature
                    and source_scope(item["scope"]) == scope
                    and (scope is None or set(item["sources"]).issubset(scope)))

        recent = [item for item in self._recent if allowed(item)]
        recent_ids = {item["id"] for item in recent}
        older = [item for item in self._items if allowed(item) and item["id"] not in recent_ids]
        relevant = []
        if older and query.strip():
            texts = [item["question"] for item in older]
            vectors = encode_texts(self.embedder, texts)
            query_vector = encode_texts(self.embedder, [query])[0]
            scores = vectors @ query_vector
            ids = np.argsort(scores)[::-1][:self.config.memory_retrieval_top_k]
            relevant = [{**older[int(i)], "score": float(scores[i])} for i in ids]
        return {"recent": recent, "relevant": relevant}

    def context(self, query: str, sources: Sequence[str] | None = None) -> str:
        recalled = self.recall(query, sources)
        parts = []
        length = 0
        # Priorità ai turni più recenti, poi ai ricordi semanticamente rilevanti.
        ordered = [("recente", item) for item in reversed(recalled["recent"])]
        ordered += [("recuperato", item) for item in recalled["relevant"]]
        for kind, item in ordered:
            line = json.dumps({"tipo": kind, "domanda": item["question"],
                               "risposta_precedente_non_verificata": item["answer"]}, ensure_ascii=False)
            addition = len(line) + (1 if parts else 0)
            if length + addition <= self.config.memory_max_chars:
                parts.append(line)
                length += addition
        return "\n".join(parts)
