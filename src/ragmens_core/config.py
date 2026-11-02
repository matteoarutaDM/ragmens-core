"""Configurazione isolata per istanza."""
from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from dotenv import dotenv_values

@dataclass(frozen=True)
class Config:
    knowledge_dir: Path = Path('knowledge')
    artifact_dir: Path = Path('rag_artifacts')
    project_id: str = 'default'
    force_rebuild_index: bool = False
    docling_ocr: bool = False
    allow_partial: bool = False
    save_raw_json: bool = True
    save_clean_json: bool = True
    docling_tables: bool = True
    chunker: str = 'hybrid'
    max_chunk_tokens: int = 256
    chunk_merge_peers: bool = True
    chunk_repeat_table_header: bool = True
    chunk_omit_header_on_overflow: bool = False
    chunk_always_emit_headings: bool = False
    embedding_model: str = 'Qwen/Qwen3-Embedding-4B'
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 512
    embedding_device: str = 'auto'
    vector_db: str = 'qdrant'
    qdrant_mode: str = 'local'
    qdrant_path: Path = Path('rag_artifacts/qdrant')
    qdrant_collection: str = 'documents'
    qdrant_url: str = ''
    qdrant_api_key: str = field(default='', repr=False)
    top_k: int = 5
    query_instruction: str = 'Given a question, retrieve relevant passages from documents that answer the question.'
    llm_provider: str = 'ollama'
    ollama_model: str = 'qwen3:14b'
    ollama_base_url: str = 'http://localhost:11434'
    llm_temperature: float = 0.15
    llm_num_ctx: int = 16384
    llm_num_predict: int = 3500
    llm_repeat_penalty: float = 1.08
    llm_timeout: float = 120.0
    use_memory: bool = False
    session_id: str = 'default'
    memory_recent_turns: int = 6
    memory_retrieval_top_k: int = 3
    memory_max_chars: int = 4000
    context_max_tokens: int = 8000
    context_max_chars: int = 30000
    system_reserve_tokens: int = 1500
    memory_reserve_tokens: int = 1000
    response_reserve_tokens: int = 3500

    def __post_init__(self):
        for name in ('knowledge_dir','artifact_dir','qdrant_path'):
            object.__setattr__(self, name, Path(getattr(self,name)).expanduser().resolve())
        if self.artifact_dir.is_relative_to(self.knowledge_dir) or self.qdrant_path.is_relative_to(self.knowledge_dir):
            raise ValueError('Artefatti e Qdrant devono essere esterni a KNOWLEDGE_DIR')
        for name in ('max_chunk_tokens','embedding_batch_size','embedding_max_seq_length','top_k','llm_num_ctx','llm_num_predict','memory_recent_turns','memory_retrieval_top_k','memory_max_chars','context_max_tokens','context_max_chars'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1: raise ValueError(f'{name} deve essere positivo')
        if self.max_chunk_tokens > self.embedding_max_seq_length: raise ValueError('MAX_CHUNK_TOKENS supera EMBEDDING_MAX_SEQ_LENGTH')
        if min(self.system_reserve_tokens,self.memory_reserve_tokens,self.response_reserve_tokens) < 0: raise ValueError('Riserve token negative')
        if self.system_reserve_tokens+self.memory_reserve_tokens+max(self.response_reserve_tokens,self.llm_num_predict) >= self.llm_num_ctx: raise ValueError('Riserve superiori alla finestra LLM')
        if self.vector_db != 'qdrant' or self.chunker != 'hybrid' or self.llm_provider != 'ollama': raise ValueError('Adapter non supportato')
        if self.qdrant_mode not in ('local','remote') or (self.qdrant_mode == 'remote' and not self.qdrant_url): raise ValueError('Configurazione Qdrant non valida')
        if self.embedding_device not in ('auto','cpu','cuda','mps'): raise ValueError('Device non valido')
        if not all(getattr(self,n).strip() for n in ('project_id','session_id','embedding_model','qdrant_collection','ollama_model','ollama_base_url','query_instruction')): raise ValueError('Configurazione testuale vuota')
        if not 0 <= self.llm_temperature <= 2 or self.llm_timeout <= 0 or self.llm_repeat_penalty <= 0: raise ValueError('Parametri LLM non validi')

    @classmethod
    def from_env(cls, base_dir=None, env_file=None, **overrides):
        base = Path(base_dir or Path.cwd()).resolve()
        envpath = Path(env_file or '.env')
        if not envpath.is_absolute(): envpath = base / envpath
        env = {**dotenv_values(envpath, interpolate=False), **os.environ}
        aliases = {'artifact_dir':'RAG_ARTIFACT_DIR','force_rebuild_index':'FORCE_REBUILD_INDEX'}
        defaults = cls()
        values = {}
        for f in fields(cls):
            if f.name in overrides: continue
            raw = env.get(aliases.get(f.name, f.name.upper()))
            if raw is None: continue
            default = getattr(defaults,f.name)
            if isinstance(default,bool):
                if raw.lower() not in ('0','1','true','false'): raise ValueError(f'{f.name}: booleano non valido')
                value = raw.lower() in ('1','true')
            elif isinstance(default,int): value = int(raw)
            elif isinstance(default,float): value = float(raw)
            elif isinstance(default,Path): value = Path(raw)
            else: value = raw
            values[f.name] = value
        values.update(overrides)
        for name in ('knowledge_dir','artifact_dir','qdrant_path'):
            path = Path(values.get(name, {'knowledge_dir':'knowledge','artifact_dir':'rag_artifacts','qdrant_path':'rag_artifacts/qdrant'}[name])).expanduser()
            values[name] = path if path.is_absolute() else base / path
        return cls(**values)

RagConfig = Config
