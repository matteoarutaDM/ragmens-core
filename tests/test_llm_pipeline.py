from dataclasses import replace
import pytest
from ragmens_core import RAGPipeline, LLMError
from ragmens_core.llm import OllamaClient

class FakeLLM:
    def __call__(self,messages): return 'risposta [FONTE 1]'
    def stream(self,messages): yield 'ris'; yield 'posta'

def test_pipeline_and_stream(config,embedder):
    rag=RAGPipeline(replace(config,use_memory=True),embedder=embedder,llm=FakeLLM())
    assert rag.preview('pompa')['sources'][0]['source']=='pompa.txt'
    assert rag.ask('pompa')['answer']=='risposta [FONTE 1]'
    assert ''.join(rag.ask_stream('pompa'))=='risposta'
    assert len(rag.memory._items)==2
    stream=rag.ask_stream('pompa'); next(stream); stream.close()
    assert len(rag.memory._items)==2
    rag.close()

def test_ollama_adapter(config):
    class Client:
        def chat(self,**kwargs):
            assert kwargs['model']=='qwen3:14b' and kwargs['options']['num_ctx']==16384
            return {'message':{'content':' ok '}}
    llm=OllamaClient(config,Client())
    assert llm([])=='ok'
    class Empty:
        def chat(self,**kwargs): return {'message':{'content':''}}
    with pytest.raises(LLMError): OllamaClient(config,Empty())([])

def test_ollama_stream_requires_completion(config):
    class Incomplete:
        def chat(self,**kwargs): return iter([{'message':{'content':'abc'},'done':False}])
    with pytest.raises(LLMError): list(OllamaClient(config,Incomplete()).stream([]))
    class Complete:
        def chat(self,**kwargs): return iter([{'message':{'content':'abc'},'done':False},{'message':{'content':''},'done':True}])
    assert ''.join(OllamaClient(config,Complete()).stream([]))=='abc'

def test_api_validation_and_close():
    from fastapi.testclient import TestClient
    from ragmens_core.api import create_app
    class Pipeline:
        closed=False
        def ask(self,q): return {'answer':'ok'}
        def ask_stream(self,q): return iter(['ok'])
        def close(self): self.closed=True
    p=Pipeline()
    with TestClient(create_app(p)) as client:
        assert client.get('/health').status_code==200
        assert client.post('/chat',json={'question':' '}).status_code==400
        assert client.post('/chat',json={'question':'ciao','stream':True}).text=='ok'
    assert p.closed

def test_retrieval_evaluation():
    from ragmens_core.evaluation import evaluate_retrieval
    class Pipeline:
        def preview(self,q,top_k): return {'hits':[{'source':'a'}, {'source':'b'}]}
    m=evaluate_retrieval(Pipeline(),[{'question':'q','expected_sources':['b']}],2)
    assert m['recall_at_k']==m['hit_rate_at_k']==1 and m['mrr']==.5
