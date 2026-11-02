from ragmens_core import RagConfig, RAGPipeline

if __name__ == '__main__':
    with RAGPipeline.prepare(RagConfig.from_env()) as rag:
        result = rag.preview('Quali sono le conclusioni principali?')
        print(result['context'])
