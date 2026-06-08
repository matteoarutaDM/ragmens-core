from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180, min_chars: int = 50) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
    chunks_out: list[str] = []
    current = ""

    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks_out.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{block}".strip()
        else:
            start = 0
            while start < len(block):
                end = min(len(block), start + chunk_size)
                piece = block[start:end].strip()
                if piece:
                    chunks_out.append(piece)
                if end >= len(block):
                    current = ""
                    break
                start = max(end - overlap, start + 1)

        while len(current) > chunk_size:
            chunks_out.append(current[:chunk_size].strip())
            current = current[max(chunk_size - overlap, 1) :].strip()

    if current:
        chunks_out.append(current)

    return [chunk for chunk in chunks_out if len(chunk.strip()) >= min_chars]

