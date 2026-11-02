import os
import pytest

@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RAG_RUN_INTEGRATION') != '1',reason='Richiede modelli e Ollama reali')
def test_real_pipeline():
    from ragmens_core import RAGPipeline
    with RAGPipeline.prepare() as rag:
        assert rag.preview('Quali sono le conclusioni?')['hits']
        assert rag.ask('Quali sono le conclusioni?')['answer']
