from __future__ import annotations

from ragmens_core import RagConfig, RagIndex


def main() -> None:
    config = RagConfig.from_env()

    print("PDF_DIR:", config.pdf_dir)
    print("INDEX_DIR:", config.index_dir)
    print("EMBEDDING_MODEL:", config.embedding_model)

    rag = RagIndex.prepare(config)

    domanda = "come verifico il corretto funzionamento della pompa idraulica?"
    hits = rag.retrieve(domanda, top_k=5)
    context = rag.build_context(hits, max_chars=3000)

    print("\nDOMANDA")
    print(domanda)

    print("\nFONTI TROVATE")
    for hit in hits:
        print(
            f"- {hit['source']} | pagina {hit['page']} | "
            f"chunk {hit['chunk_index']} | score {hit['score']:.3f}"
        )

    print("\nCONTESTO DA PASSARE ALL'LLM")
    print(context)


if __name__ == "__main__":
    main()

