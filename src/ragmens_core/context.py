from __future__ import annotations


def build_context(results: list[dict], max_chars: int = 4500) -> str:
    parts: list[str] = []
    total = 0

    for result in results:
        header = (
            f"[Fonte: {result['source']} | Pagina: {result['page']} | "
            f"Chunk: {result['chunk_index']} | score: {result['score']:.3f}]"
        )
        snippet = header + "\n" + str(result["text"]).strip()
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet) + 2

    return "\n\n---\n\n".join(parts)

