"""Préparation transactionnelle des artefacts et d'une collection Qdrant isolée."""
from __future__ import annotations
import json, logging, os, shutil, tempfile, uuid
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from ragmens_core.cache import manifest, signature, file_sha256
from ragmens_core.cleaning import clean_document
from ragmens_core.documents import source_files, convert_documents
from ragmens_core.chunking import chunk_documents
from ragmens_core.embeddings import Embedder, encode_texts
from ragmens_core.models import Chunk
logger=logging.getLogger(__name__)

@dataclass
class RAGState:
    config: object
    chunks: list
    embedder: object
    client: object
    collection: str
    corpus_signature: str
    cache_dir: Path
    cache_hit: bool
    def close(self): self.client.close()

def open_client(config):
    from qdrant_client import QdrantClient
    if config.qdrant_mode == 'remote': return QdrantClient(url=config.qdrant_url,api_key=config.qdrant_api_key or None,timeout=config.llm_timeout)
    config.qdrant_path.mkdir(parents=True,exist_ok=True)
    return QdrantClient(path=str(config.qdrant_path))

def validate_cache(folder,client,manifest_data,signature_value,dimension):
    from qdrant_client import models
    meta=json.loads((folder/'index_meta.json').read_text())
    if meta['signature'] != signature_value or meta['manifest'] != manifest_data or meta['dimension'] != dimension: raise ValueError('Manifest incompatibile')
    if not {'chunks.jsonl','embedding_metadata.jsonl','embeddings.npy','manifest.json'} <= set(meta['checksums']): raise ValueError('Artefatti mancanti')
    for name, checksum in meta['checksums'].items():
        if file_sha256(folder/name) != checksum: raise ValueError(f'Checksum non valido: {name}')
    chunks=[Chunk(**json.loads(line)) for line in (folder/'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line]
    records=[json.loads(line) for line in (folder/'embedding_metadata.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if records != [{'embedding_index':i,**asdict(c)} for i,c in enumerate(chunks)]: raise ValueError('Metadati embedding incoerenti')
    vectors=np.load(folder/'embeddings.npy',allow_pickle=False)
    ids=[c.chunk_id for c in chunks]
    if not chunks or len(set(ids)) != len(ids) or vectors.shape != (len(chunks),dimension) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4): raise ValueError('Embedding o chunk incoerenti')
    collection=meta['collection']
    if not client.collection_exists(collection): raise ValueError('Collection assente')
    info=client.get_collection(collection)
    params=info.config.params.vectors
    if params.size != dimension or params.distance != models.Distance.COSINE or client.count(collection,exact=True).count != len(chunks): raise ValueError('Collection incompatibile')
    points,offset=client.scroll(collection,limit=len(chunks)+1,with_payload=True,with_vectors=True)
    found={p.payload.get('chunk_id') for p in points}
    expected={c.chunk_id:asdict(c) for c in chunks}
    if offset is not None or found != set(ids) or any(p.payload != {**expected.get(p.payload.get('chunk_id'),{}),'corpus_signature':signature_value} for p in points): raise ValueError('Payload Qdrant incoerenti')
    positions={c.chunk_id:i for i,c in enumerate(chunks)}
    if any(p.vector is None or not np.allclose(np.asarray(p.vector,dtype=np.float32),vectors[positions[p.payload['chunk_id']]],atol=1e-4) for p in points): raise ValueError('Vettori Qdrant incoerenti')
    return chunks,collection

def prepare_rag(config=None, *, embedder=None, force_rebuild=None, converter=None, client=None):
    from ragmens_core.config import Config
    from qdrant_client import models
    from docling_core.types.doc.document import DoclingDocument
    cfg=config or Config.from_env()
    files=source_files(cfg.knowledge_dir)
    m=manifest(cfg,files); sig=signature(m)
    root=cfg.artifact_dir/'indexes'/sig
    folder=root
    encoder=embedder or Embedder(cfg.embedding_model,cfg.embedding_device,cfg.embedding_batch_size,cfg.embedding_max_seq_length,cfg.query_instruction)
    if encoder.model_name != cfg.embedding_model: raise ValueError('Modello embedding incompatibile')
    q=client or open_client(cfg)
    try:
        if not (cfg.force_rebuild_index if force_rebuild is None else force_rebuild):
            try:
                pointer=json.loads((root/'active.json').read_text(encoding='utf-8'))
                generation=pointer['generation']
                if not isinstance(generation,str) or not generation.isalnum(): raise ValueError('Puntatore cache non valido')
                folder=root/'generations'/generation
                chunks,collection=validate_cache(folder,q,m,sig,encoder.dimension)
                return RAGState(cfg,chunks,encoder,q,collection,sig,folder,True)
            except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError) as exc: logger.info('Cache non disponibile: %s',exc)
        cfg.artifact_dir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.build-',dir=cfg.artifact_dir) as temp:
            stage=Path(temp); prepared=[]
            for source,doc in convert_documents(files,cfg,converter):
                raw=doc.export_to_dict()
                clean=clean_document(raw)
                name=source+'.json'
                if cfg.save_raw_json:
                    path=stage/'raw'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(raw,ensure_ascii=False),encoding='utf-8')
                if cfg.save_clean_json:
                    path=stage/'clean'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(clean,ensure_ascii=False),encoding='utf-8')
                prepared.append((source,name,DoclingDocument.model_validate(clean)))
            chunks=chunk_documents(prepared,cfg,encoder)
            vectors=encode_texts(encoder,[c.text for c in chunks])
            if manifest(cfg,source_files(cfg.knowledge_dir)) != m: raise RuntimeError('Corpus modificato durante indicizzazione')
            (stage/'chunks.jsonl').write_text(''.join(json.dumps(asdict(c),ensure_ascii=False)+'\n' for c in chunks),encoding='utf-8')
            (stage/'embedding_metadata.jsonl').write_text(''.join(json.dumps({'embedding_index':i,**asdict(c)},ensure_ascii=False)+'\n' for i,c in enumerate(chunks)),encoding='utf-8')
            np.save(stage/'embeddings.npy',vectors)
            (stage/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
            collection=f'{cfg.qdrant_collection}_{sig[:20]}_{uuid.uuid4().hex[:12]}'
            q.create_collection(collection_name=collection,vectors_config=models.VectorParams(size=encoder.dimension,distance=models.Distance.COSINE))
            q.upload_points(collection_name=collection,points=[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,c.chunk_id)),vector=vectors[i].tolist(),payload={**asdict(c),'corpus_signature':sig}) for i,c in enumerate(chunks)],wait=True)
            checksums={p.relative_to(stage).as_posix():file_sha256(p) for p in stage.rglob('*') if p.is_file()}
            meta={'signature':sig,'manifest':m,'collection':collection,'dimension':encoder.dimension,'checksums':checksums}
            (stage/'index_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            validate_cache(stage,q,m,sig,encoder.dimension)
            generation=uuid.uuid4().hex
            folder=root/'generations'/generation
            folder.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(stage,folder)
            validate_cache(folder,q,m,sig,encoder.dimension)
            # Il puntatore viene sostituito solo dopo la verifica dell'intera generazione.
            tmp=root/f'.active-{generation}.json'
            tmp.write_text(json.dumps({'generation':generation}),encoding='utf-8')
            os.replace(tmp,root/'active.json')
            return RAGState(cfg,chunks,encoder,q,collection,sig,folder,False)
    except BaseException:
        if client is None: q.close()
        raise

RagIndex=RAGState
