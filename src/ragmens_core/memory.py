from __future__ import annotations

import json
import uuid
from collections import deque
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ShortTermMemory:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def text(self, max_chars: int = 1200) -> str:
        parts: list[str] = []
        total = 0
        for message in list(self._messages)[-self.max_messages :]:
            line = f"{message['role'].upper()}: {message['content']}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line) + 1
        return "\n".join(parts).strip()

    def clear(self) -> None:
        self._messages.clear()


class VectorMemory:
    """Memoria lunga conversazionale salvata su FAISS separato."""

    def __init__(
        self,
        embedder: SentenceTransformer,
        mem_dir: Path,
        corpus_signature: str | None = None,
        jsonl_name: str = "memory.jsonl",
        faiss_name: str = "memory.faiss",
    ):
        self.embedder = embedder
        self.mem_dir = Path(mem_dir)
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_signature = corpus_signature
        self.jsonl_path = self.mem_dir / jsonl_name
        self.faiss_path = self.mem_dir / faiss_name
        self.items: list[dict] = []
        self.index: faiss.Index | None = None

        if self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))

        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))

    def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        text_for_embedding = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        vec = self.embedder.encode([text_for_embedding], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(vec.shape[1])

        item = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "corpus_signature": self.corpus_signature,
            "user": user_text,
            "assistant": assistant_text,
            "embed_text": text_for_embedding,
        }

        self.index.add(vec)
        self.items.append(item)

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        faiss.write_index(self.index, str(self.faiss_path))

    def search(self, session_id: str, query: str, top_k: int = 4) -> list[dict]:
        if self.index is None or not self.items:
            return []

        qv = self.embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")

        k = min(max(top_k * 5, top_k), len(self.items))
        scores, ids = self.index.search(qv, k)

        out: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = self.items[int(idx)]
            if item["session_id"] != session_id:
                continue
            if self.corpus_signature and item.get("corpus_signature") != self.corpus_signature:
                continue
            out.append({**item, "score": float(score)})
            if len(out) >= top_k:
                break
        return out


def ltm_text(mem_hits: list[dict], max_chars: int = 1200) -> str:
    parts: list[str] = []
    total = 0
    for hit in mem_hits:
        line = f"[MEM score {hit['score']:.3f}] {hit['embed_text']}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts).strip()

