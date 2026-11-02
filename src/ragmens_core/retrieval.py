"""Ricerca Qdrant e filtri eseguiti sul database."""
from ragmens_core.embeddings import validate_vectors

def source_scope(sources):
    if sources is None: return None
    if isinstance(sources,str) or any(not isinstance(s,str) or not s.strip() for s in sources): raise ValueError('sources non valido')
    return tuple(sorted(set(sources)))

def retrieve(state,query,top_k=None,sources=None):
    from qdrant_client import models
    if not query.strip(): raise ValueError('Domanda vuota')
    k=state.config.top_k if top_k is None else top_k
    if type(k) is not int or k<1: raise ValueError('top_k non valido')
    scope=source_scope(sources)
    if scope == (): return []
    vector=validate_vectors(state.embedder.encode_query(query),1,state.embedder.dimension)[0]
    filt=None if scope is None else models.Filter(must=[models.FieldCondition(key='source',match=models.MatchAny(any=list(scope)))])
    result=state.client.query_points(collection_name=state.collection,query=vector.tolist(),query_filter=filt,limit=k,with_payload=True)
    return [{**point.payload,'score':float(point.score)} for point in result.points]
