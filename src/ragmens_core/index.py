from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragmens_core.config import RagConfig
from ragmens_core.context import build_context
from ragmens_core.documents import ingest_pdfs


def load_chunks(chunks_path: Path, min_chars: int = 50) -> list[dict]:
    if not chunks_path.exists():
        return []

    loaded: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            text = (chunk.get("text") or "").strip()
            if len(text) >= min_chars:
                loaded.append(chunk)
    return loaded


def pdf_manifest(pdf_dir: Path) -> list[dict]:
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    return [
        {
            "path": str(pdf_path.relative_to(pdf_dir)),
            "size": pdf_path.stat().st_size,
            "mtime_ns": pdf_path.stat().st_mtime_ns,
        }
        for pdf_path in pdf_files
    ]


def corpus_signature(manifest: list[dict]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_index_meta(index_meta_path: Path) -> dict:
    if not index_meta_path.exists():
        return {}
    try:
        with index_meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def chunk_belongs_to_documents(chunk: dict, document_ids: list[str] | None) -> bool:
    if not document_ids:
        return True
    source = str(chunk.get("source") or "")
    chunk_id = str(chunk.get("id") or "")
    return any(source == doc_id or chunk_id.startswith(f"{doc_id}::") for doc_id in document_ids)


@dataclass
class RagIndex:
    config: RagConfig
    chunks: list[dict]
    embedder: SentenceTransformer
    index: faiss.Index
    manifest: list[dict]
    corpus_signature: str

    @classmethod
    def prepare(cls, config: RagConfig | None = None, force_rebuild: bool | None = None) -> "RagIndex":
        cfg = config or RagConfig.from_env()
        cfg.ensure_dirs()

        embedder = SentenceTransformer(cfg.embedding_model)
        manifest = pdf_manifest(cfg.pdf_dir)
        if not manifest:
            raise FileNotFoundError(f"Nessun PDF trovato in {cfg.pdf_dir}")

        signature = corpus_signature(manifest)
        reason = cls._index_rebuild_reason(cfg, manifest, signature, force_rebuild)

        if reason:
            chunks = ingest_pdfs(
                pdf_dir=cfg.pdf_dir,
                chunks_path=cfg.chunks_path,
                output_debug_dir=cfg.output_debug_dir,
                save_debug_text=True,
                chunk_size=cfg.chunk_size,
                overlap=cfg.chunk_overlap,
                min_chars=cfg.min_chunk_chars,
            )
            index = cls._build_faiss_index(cfg, chunks, embedder, manifest, signature)
        else:
            chunks = load_chunks(cfg.chunks_path, min_chars=cfg.min_chunk_chars)
            if not chunks:
                chunks = ingest_pdfs(
                    pdf_dir=cfg.pdf_dir,
                    chunks_path=cfg.chunks_path,
                    output_debug_dir=cfg.output_debug_dir,
                    save_debug_text=True,
                    chunk_size=cfg.chunk_size,
                    overlap=cfg.chunk_overlap,
                    min_chars=cfg.min_chunk_chars,
                )
                index = cls._build_faiss_index(cfg, chunks, embedder, manifest, signature)
            else:
                index = faiss.read_index(str(cfg.faiss_path))

        if index.ntotal != len(chunks):
            raise RuntimeError(f"Indice/chunks non coerenti: index.ntotal={index.ntotal}, chunks={len(chunks)}")

        return cls(
            config=cfg,
            chunks=chunks,
            embedder=embedder,
            index=index,
            manifest=manifest,
            corpus_signature=signature,
        )

    @classmethod
    def load(cls, index_dir: str | Path, pdf_dir: str | Path | None = None, embedding_model: str | None = None) -> "RagIndex":
        base_dir = Path.cwd().resolve()
        cfg = RagConfig(
            base_dir=base_dir,
            pdf_dir=Path(pdf_dir).resolve() if pdf_dir else base_dir / "pdf_immagini",
            index_dir=Path(index_dir).resolve(),
            mem_dir=base_dir / "memory",
            output_debug_dir=base_dir / "outputs_debug",
            embedding_model=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
        )
        chunks = load_chunks(cfg.chunks_path, min_chars=cfg.min_chunk_chars)
        if not chunks:
            raise FileNotFoundError(f"Nessun chunk trovato in {cfg.chunks_path}")
        if not cfg.faiss_path.exists():
            raise FileNotFoundError(f"Indice FAISS non trovato in {cfg.faiss_path}")

        meta = load_index_meta(cfg.index_meta_path)
        model = embedding_model or meta.get("model") or cfg.embedding_model
        cfg = RagConfig(
            base_dir=cfg.base_dir,
            pdf_dir=Path(meta.get("pdf_dir", cfg.pdf_dir)).resolve(),
            index_dir=cfg.index_dir,
            mem_dir=cfg.mem_dir,
            output_debug_dir=cfg.output_debug_dir,
            embedding_model=model,
        )
        manifest = meta.get("pdf_manifest") or pdf_manifest(cfg.pdf_dir)
        signature = meta.get("corpus_signature") or corpus_signature(manifest)

        return cls(
            config=cfg,
            chunks=chunks,
            embedder=SentenceTransformer(cfg.embedding_model),
            index=faiss.read_index(str(cfg.faiss_path)),
            manifest=manifest,
            corpus_signature=signature,
        )

    @staticmethod
    def _index_rebuild_reason(
        cfg: RagConfig,
        manifest: list[dict],
        signature: str,
        force_rebuild: bool | None,
    ) -> str | None:
        should_force_rebuild = cfg.force_rebuild_index if force_rebuild is None else force_rebuild
        if should_force_rebuild:
            return "FORCE_REBUILD_INDEX=1"
        if not cfg.chunks_path.exists():
            return f"file mancante: {cfg.chunks_path}"
        if not cfg.faiss_path.exists():
            return f"file mancante: {cfg.faiss_path}"
        if not cfg.index_meta_path.exists():
            return f"file mancante: {cfg.index_meta_path}"

        meta = load_index_meta(cfg.index_meta_path)
        if meta.get("pdf_dir") != str(cfg.pdf_dir):
            return "PDF_DIR cambiata"
        if meta.get("model") != cfg.embedding_model:
            return "modello embedding cambiato"
        if meta.get("pdf_manifest") != manifest:
            return "PDF aggiunti, rimossi o modificati"
        if meta.get("corpus_signature") != signature:
            return "firma corpus non coerente"
        return None

    @staticmethod
    def _build_faiss_index(
        cfg: RagConfig,
        chunks: list[dict],
        embedder: SentenceTransformer,
        manifest: list[dict],
        signature: str,
    ) -> faiss.Index:
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            raise RuntimeError("Nessun testo valido per embedding.")

        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise RuntimeError(f"Embeddings non validi: shape={embeddings.shape}")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(cfg.faiss_path))

        meta = {
            "n_chunks": len(chunks),
            "model": cfg.embedding_model,
            "dim": int(embeddings.shape[1]),
            "pdf_dir": str(cfg.pdf_dir),
            "pdf_manifest": manifest,
            "corpus_signature": signature,
        }
        with cfg.index_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        return index

    def retrieve(self, query: str, top_k: int | None = None, document_ids: list[str] | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []

        limit = top_k or self.config.top_k
        qv = self.embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32").reshape(1, -1)

        if qv.shape[1] != self.index.d:
            raise ValueError(
                f"Dimension mismatch: qv dim={qv.shape[1]} vs index.d={self.index.d}. "
                "Indice FAISS e/o modello embedding non coerenti."
            )

        filtered_ids = [doc_id for doc_id in (document_ids or []) if doc_id]
        search_k = len(self.chunks) if filtered_ids else min(limit, len(self.chunks))
        scores, ids = self.index.search(qv, search_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[int(idx)])
            if not chunk_belongs_to_documents(item, filtered_ids):
                continue
            item["score"] = float(score)
            results.append(item)
            if len(results) >= limit:
                break
        return results

    def build_context(self, results: list[dict], max_chars: int | None = None) -> str:
        return build_context(results, max_chars=max_chars or self.config.context_max_chars)
