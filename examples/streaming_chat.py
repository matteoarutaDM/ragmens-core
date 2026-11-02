from ragmens_core import RAGPipeline

if __name__ == '__main__':
    with RAGPipeline.prepare() as rag:
        for text in rag.ask_stream('Quali sono le conclusioni principali?'):
            print(text, end='', flush=True)
        print()
