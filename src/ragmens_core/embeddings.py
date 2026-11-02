"""Embedding Qwen normalizzati, con caricamento su richiesta."""
from __future__ import annotations
import numpy as np

def validate_vectors(vectors, count, dimension=None):
    a = np.asarray(vectors,dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != count or not count or not a.shape[1] or (dimension is not None and a.shape[1] != dimension) or not np.isfinite(a).all(): raise ValueError('Matrice embedding non valida')
    norms = np.linalg.norm(a,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0): raise ValueError('Embedding nulli o non validi')
    return np.ascontiguousarray(a/norms,dtype=np.float32)

class Embedder:
    def __init__(self, model_name, device='auto', batch_size=8, max_seq_length=512, instruction='Given a question, retrieve relevant passages from documents that answer the question.'):
        self.model_name, self.device, self.batch_size, self.max_seq_length, self.instruction = model_name,device,batch_size,max_seq_length,instruction
        self._model = None
    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = self.device
            if device == 'auto': device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if device in ('cuda','mps') else torch.float32)
            self._model = SentenceTransformer(self.model_name,device=device,model_kwargs={'torch_dtype':dtype,'low_cpu_mem_usage':True},tokenizer_kwargs={'padding_side':'left'})
            self._model.max_seq_length = self.max_seq_length
        return self._model
    @property
    def tokenizer(self): return self._load().tokenizer
    @property
    def dimension(self):
        dim = self._load().get_sentence_embedding_dimension()
        if not isinstance(dim,int) or dim < 1: raise ValueError('Dimensione embedding non valida')
        return dim
    def encode(self,texts):
        if any(len(self.tokenizer.encode(t,add_special_tokens=True)) > self.max_seq_length for t in texts): raise ValueError('Testo oltre il limite del tokenizer embedding')
        return validate_vectors(self._load().encode(list(texts),batch_size=self.batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False),len(texts),self.dimension)
    def encode_query(self,query):
        return self.encode([f'Instruct: {self.instruction}\nQuery: {query}'])

def encode_texts(embedder,texts): return validate_vectors(embedder.encode(texts),len(texts),embedder.dimension)
