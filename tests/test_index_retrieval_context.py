from dataclasses import replace
from ragmens_core import prepare_rag,retrieve,build_context

def test_qdrant_cache_filters_and_reopen(config,embedder):
    state=prepare_rag(config,embedder=embedder)
    assert not state.cache_hit and len(state.chunks)==2
    assert retrieve(state,'pompa',top_k=1)[0]['source']=='pompa.txt'
    assert retrieve(state,'pompa',top_k=1,sources=['valvola.md'])[0]['source']=='valvola.md'
    assert retrieve(state,'pompa',sources=[])==[]
    first=state.corpus_signature
    state.close()
    state=prepare_rag(config,embedder=embedder)
    assert state.cache_hit
    state.close()
    (config.knowledge_dir/'pompa.txt').write_text('pompa nuova')
    state=prepare_rag(config,embedder=embedder)
    assert state.corpus_signature != first and not state.cache_hit
    state.close()

def test_corrupt_rebuild_and_force(config,embedder):
    a=prepare_rag(config,embedder=embedder); folder=a.cache_dir; collection=a.collection; a.close()
    (folder/'embeddings.npy').write_bytes(b'broken')
    b=prepare_rag(config,embedder=embedder)
    assert not b.cache_hit and b.collection!=collection
    old=b.cache_dir; b.close()
    c=prepare_rag(config,embedder=embedder,force_rebuild=True)
    assert not c.cache_hit and old.exists(); c.close()

def test_context_budget():
    hits=[{'source':'a','pages':[2],'text':'testo molto lungo'}]
    text,used,excluded,truncated=build_context(hits,150,65)
    assert len(text)<=65 and len(used)==1 and used[0]['citation']=='FONTE 1' and truncated
    assert used[0]['included_text'] in text
    assert build_context(hits,0,0)[2]==hits

def test_cache_missing_points_and_failed_build_preserves_active(config,embedder,monkeypatch):
    import ragmens_core.index as index
    state=prepare_rag(config,embedder=embedder)
    pointer=(config.artifact_dir/'indexes'/state.corpus_signature/'active.json')
    before=pointer.read_text(); state.close()
    original=index.chunk_documents
    def fail(*args): raise RuntimeError('interrupted')
    monkeypatch.setattr(index,'chunk_documents',fail)
    import pytest
    with pytest.raises(RuntimeError): prepare_rag(config,embedder=embedder,force_rebuild=True)
    assert pointer.read_text()==before
    monkeypatch.setattr(index,'chunk_documents',original)
    state=prepare_rag(config,embedder=embedder)
    assert state.cache_hit
    state.client.delete_collection(state.collection); state.close()
    state=prepare_rag(config,embedder=embedder)
    assert not state.cache_hit; state.close()

def test_chunk_limit_and_provenance(config,embedder):
    from ragmens_core.chunking import chunk_documents
    from ragmens_core.documents import convert_documents,source_files
    from dataclasses import replace
    cfg=replace(config,max_chunk_tokens=8)
    docs=convert_documents(source_files(cfg.knowledge_dir),cfg)
    chunks=chunk_documents([(s,s+'.json',d) for s,d in docs],cfg,embedder)
    assert all(c.token_count<=8 and c.doc_item_refs and c.source for c in chunks)
    assert len({c.chunk_id for c in chunks})==len(chunks)
