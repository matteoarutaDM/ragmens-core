"""Metriche di retrieval su esempi esplicitamente annotati dall'utente."""
from time import perf_counter

def evaluate_retrieval(pipeline, cases, top_k=5):
    """Ogni caso contiene question e expected_sources, percorsi relativi non vuoti."""
    if not cases or top_k < 1: raise ValueError('Casi e top_k devono essere non vuoti')
    rows=[]
    for case in cases:
        expected=set(case['expected_sources'])
        if not expected: raise ValueError('Fonti attese mancanti')
        start=perf_counter()
        hits=pipeline.preview(case['question'],top_k=top_k)['hits']
        elapsed=perf_counter()-start
        found=[hit['source'] for hit in hits]
        matched=expected.intersection(found)
        rank=next((i for i,source in enumerate(found,1) if source in expected),None)
        rows.append({'question':case['question'],'recall':len(matched)/len(expected),'hit':bool(matched),'reciprocal_rank':1/rank if rank else 0.0,'seconds':elapsed})
    n=len(rows)
    return {'recall_at_k':sum(x['recall'] for x in rows)/n,'hit_rate_at_k':sum(x['hit'] for x in rows)/n,'mrr':sum(x['reciprocal_rank'] for x in rows)/n,'mean_seconds':sum(x['seconds'] for x in rows)/n,'cases':rows}
