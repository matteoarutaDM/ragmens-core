"""Contesto con budget stimato conservativamente e fonti esplicite."""
def estimate_tokens(text):
    # Un byte UTF-8 per token è una stima prudente, non un tokenizer Ollama.
    return len(text.encode('utf-8'))

def build_context(hits, max_tokens=8000, max_chars=30000):
    if max_tokens < 0 or max_chars < 0: raise ValueError('Budget negativo')
    parts=[]; included=[]; excluded=[]; truncated=False
    for hit in hits:
        label=f'FONTE {len(included)+1}'
        header=f'[{label}]\nDocumento: {hit["source"]}\nPagine: {hit.get("pages", [])} | Posizioni: {hit.get("locations", [])}\nTesto:\n'
        sep='\n\n' if parts else ''
        prefix=''.join(parts)+sep+header
        available=min(max_chars-len(prefix), max_tokens-estimate_tokens(prefix))
        if available <= 0:
            excluded.append(hit); continue
        text=hit['text']; excerpt=text[:available]
        while excerpt and estimate_tokens(prefix+excerpt)>max_tokens: excerpt=excerpt[:-1]
        if not excerpt:
            excluded.append(hit); continue
        is_truncated=len(excerpt)<len(text)
        truncated |= is_truncated
        parts.extend([sep,header,excerpt])
        included.append({**hit,'citation':label,'included_text':excerpt,'truncated':is_truncated})
    return ''.join(parts),included,excluded,truncated
