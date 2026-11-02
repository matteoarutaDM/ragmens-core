"""Firma deterministica e checksum degli artefatti."""
import hashlib, json
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
PIPELINE_VERSION='3.0.1'

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def manifest(config,files):
    versions={}
    for name in ('docling','docling-core','sentence-transformers','transformers','qdrant-client','ftfy'):
        try: versions[name]=version(name)
        except PackageNotFoundError: versions[name]='missing'
    names=('project_id','embedding_model','embedding_max_seq_length','docling_ocr','docling_tables','allow_partial','save_raw_json','save_clean_json','max_chunk_tokens','chunk_merge_peers','chunk_repeat_table_header','chunk_omit_header_on_overflow','chunk_always_emit_headings')
    return {'pipeline':PIPELINE_VERSION,'root':str(config.knowledge_dir),'files':[{'source':p.relative_to(config.knowledge_dir).as_posix(),'sha256':file_sha256(p)} for p in files], 'config':{n:getattr(config,n) for n in names},'versions':versions}

def signature(data): return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
