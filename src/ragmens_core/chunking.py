"""HybridChunker con tokenizer dell'embedder e controllo dopo contestualizzazione."""
import hashlib, json
from ragmens_core.models import Chunk

def chunk_documents(documents, config, embedder):
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    tokenizer = HuggingFaceTokenizer(tokenizer=embedder.tokenizer,max_tokens=config.max_chunk_tokens)
    chunker = HybridChunker(tokenizer=tokenizer,merge_peers=config.chunk_merge_peers,repeat_table_header=config.chunk_repeat_table_header,omit_header_on_overflow=config.chunk_omit_header_on_overflow,always_emit_headings=config.chunk_always_emit_headings)
    output=[]
    for source, source_json, doc in documents:
        document_id = hashlib.sha256(source.encode()).hexdigest()
        for n, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            text = chunker.contextualize(chunk=chunk)
            limit = min(config.max_chunk_tokens,config.embedding_max_seq_length)
            # Dividiamo il testo contestualizzato a confini di caratteri senza scartarlo.
            # La ricerca binaria verifica ogni segmento con il tokenizer reale, special token inclusi.
            pieces = []
            remaining = text
            while remaining:
                if len(embedder.tokenizer.encode(remaining,add_special_tokens=True)) <= limit:
                    pieces.append(remaining); break
                lo, hi = 1, len(remaining)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(embedder.tokenizer.encode(remaining[:mid],add_special_tokens=True)) <= limit: lo = mid
                    else: hi = mid - 1
                if len(embedder.tokenizer.encode(remaining[:lo],add_special_tokens=True)) > limit:
                    raise ValueError(f'Limite token insufficiente per {source}')
                pieces.append(remaining[:lo]); remaining = remaining[lo:]
            pages=sorted({p.page_no for item in chunk.meta.doc_items for p in (item.prov or []) if p.page_no is not None})
            refs=[str(item.self_ref) for item in chunk.meta.doc_items]
            for part, piece in enumerate(pieces):
                identity=json.dumps([source,n,part,piece],ensure_ascii=False)
                output.append(Chunk(hashlib.sha256(identity.encode()).hexdigest(),document_id,source,source_json,doc.name,piece,chunk.text,list(chunk.meta.headings or []),pages,refs,[item.orig for item in chunk.meta.doc_items if not source.lower().endswith('.pdf') and getattr(item,'orig',None)],len(embedder.tokenizer.encode(piece,add_special_tokens=True))))
    if not output: raise ValueError('Nessun chunk prodotto')
    return output
