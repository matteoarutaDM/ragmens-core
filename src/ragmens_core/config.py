from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RagConfig:
    base_dir: Path
    pdf_dir: Path
    index_dir: Path
    mem_dir: Path
    output_debug_dir: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 12
    context_max_chars: int = 4500
    force_rebuild_index: bool = False
    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_chunk_chars: int = 50

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def index_meta_path(self) -> Path:
        return self.index_dir / "index_meta.json"

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None, env_file: str | Path | None = None) -> "RagConfig":
        resolved_base = Path(base_dir or Path.cwd()).resolve()
        load_dotenv(Path(env_file).resolve() if env_file else resolved_base / ".env")

        def path_from_env(name: str, default: Path) -> Path:
            return Path(os.getenv(name, str(default))).resolve()

        return cls(
            base_dir=resolved_base,
            pdf_dir=path_from_env("PDF_DIR", resolved_base / "pdf_immagini"),
            index_dir=path_from_env("INDEX_DIR", resolved_base / "index"),
            mem_dir=path_from_env("MEM_DIR", resolved_base / "memory"),
            output_debug_dir=path_from_env("OUTPUT_DEBUG_DIR", resolved_base / "outputs_debug"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=int(os.getenv("TOP_K", "12")),
            context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "4500")),
            force_rebuild_index=os.getenv("FORCE_REBUILD_INDEX", "0") == "1",
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            min_chunk_chars=int(os.getenv("MIN_CHUNK_CHARS", "50")),
        )

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.output_debug_dir.mkdir(parents=True, exist_ok=True)

