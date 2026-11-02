from ragmens_core import RagConfig, RAGPipeline

if __name__ == '__main__':
    with RAGPipeline.prepare(RagConfig.from_env()) as rag:
        result = rag.ask('Quali sono le conclusioni principali?')
        print(result['answer'])
        print(result['sources'])
