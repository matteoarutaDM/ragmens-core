from dataclasses import replace
import numpy as np
import pytest
from ragmens_core import Config, clean_document
from ragmens_core.embeddings import validate_vectors

def test_config_defaults_and_validation(config):
    assert Config().embedding_model == 'Qwen/Qwen3-Embedding-4B'
    assert Config().ollama_model == 'qwen3:14b'
    with pytest.raises(ValueError): replace(config,max_chunk_tokens=1000)
    with pytest.raises(ValueError): replace(config,qdrant_mode='remote')

def test_env(tmp_path,monkeypatch):
    monkeypatch.setenv('TOP_K','9')
    assert Config.from_env(base_dir=tmp_path).top_k == 9
    assert Config.from_env(base_dir=tmp_path).knowledge_dir == tmp_path/'knowledge'

def test_cleaning_preserves_structural_fields():
    original={'texts':[{'text':'  CafÃ©\u00ad  x\u200b ','self_ref':'#/texts/0','prov':[{'page_no':1}]}], 'tables':[{'data':{'table_cells':[{'text':' A\xa0 B '}]}}]}
    result=clean_document(original)
    assert result['texts'][0]['text']=='Café x'
    assert result['texts'][0]['prov']==original['texts'][0]['prov']
    assert result['tables'][0]['data']['table_cells'][0]['text']=='A B'
    assert clean_document(result)==result and original['texts'][0]['text'] != result['texts'][0]['text']

def test_vectors():
    a=validate_vectors([[3,4]],1,2)
    assert a.dtype==np.float32 and np.allclose(a,[[.6,.8]])
    for v in ([[0,0]],[[float('nan'),1]]):
        with pytest.raises(ValueError): validate_vectors(v,1)

def test_embedding_query_instruction_and_normalization(monkeypatch):
    import sys, types
    from ragmens_core.embeddings import Embedder
    class Tokenizer:
        def encode(self,text,add_special_tokens=True): return list(text)
    class Model:
        tokenizer=Tokenizer()
        def __init__(self,*args,**kwargs): self.last=None
        def get_sentence_embedding_dimension(self): return 2
        def encode(self,texts,**kwargs): self.last=texts; return [[3,4]]
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=Model))
    e=Embedder('fake',device='cpu',max_seq_length=256,instruction='istruzione')
    result=e.encode_query('domanda')
    assert np.allclose(result,[[.6,.8]])
    assert e._model.last == ['Instruct: istruzione\nQuery: domanda']
    e.encode(['documento'])
    assert e._model.last == ['documento']
