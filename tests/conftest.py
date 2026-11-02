import numpy as np
import pytest
from transformers import BertTokenizerFast
from ragmens_core import Config

class FakeEmbedder:
    model_name='fake'
    dimension=4
    def __init__(self, tokenizer): self.tokenizer=tokenizer; self.calls=[]
    def encode(self,texts):
        self.calls.append(list(texts))
        return np.asarray([[t.lower().count('pompa')+0.1,t.lower().count('valvola')+0.1,t.lower().count('motore')+0.1,0.1] for t in texts],dtype='float32')
    def encode_query(self,query): return self.encode([query])

@pytest.fixture
def config(tmp_path):
    root=tmp_path/'knowledge'; root.mkdir()
    (root/'pompa.txt').write_text('La pompa richiede controllo pressione.',encoding='utf-8')
    (root/'valvola.md').write_text('La valvola regola la portata.',encoding='utf-8')
    return Config(knowledge_dir=root,artifact_dir=tmp_path/'artifacts',qdrant_path=tmp_path/'qdrant',embedding_model='fake')

@pytest.fixture
def embedder(tmp_path):
    p=tmp_path/'vocab.txt'; p.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nla\npompa\nvalvola\nrichiede\ncontrollo\npressione\nregola\nportata\n.\n')
    return FakeEmbedder(BertTokenizerFast(vocab_file=str(p)))
