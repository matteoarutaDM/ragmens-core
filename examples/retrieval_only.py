from ragmens_core import RagConfig, RagIndex


def main() -> None:
    config = RagConfig.from_env()
    rag = RagIndex.prepare(config)

    question = "come verifico il corretto funzionamento della pompa idraulica?"
    hits = rag.retrieve(question, top_k=config.top_k)

    print("TOP HITS")
    for hit in hits:
        print(f"- {hit['source']} | pagina {hit['page']} | chunk {hit['chunk_index']} | score {hit['score']:.3f}")

    print("\nCONTESTO\n")
    print(rag.build_context(hits))


if __name__ == "__main__":
    main()

