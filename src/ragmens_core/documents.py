from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm.auto import tqdm

from ragmens_core.chunking import chunk_text


def clean_pdf_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [clean_pdf_text(page.extract_text() or "") for page in reader.pages]


def ingest_pdfs(
    pdf_dir: Path,
    chunks_path: Path,
    output_debug_dir: Path | None = None,
    save_debug_text: bool = True,
    chunk_size: int = 1200,
    overlap: int = 180,
    min_chars: int = 50,
) -> list[dict]:
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Nessun PDF trovato in {pdf_dir}")

    all_chunks: list[dict] = []
    if save_debug_text and output_debug_dir:
        output_debug_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in tqdm(pdf_files, desc="Ingestione PDF"):
        source_name = str(pdf_path.relative_to(pdf_dir))
        pages = read_pdf_pages(pdf_path)

        if save_debug_text and output_debug_dir:
            debug_name = source_name.replace("/", "__")
            debug_path = output_debug_dir / f"{Path(debug_name).stem}_all_pages.txt"
            with debug_path.open("w", encoding="utf-8") as out:
                for page_num, page_text in enumerate(pages, start=1):
                    out.write(page_text)
                    out.write("\n\n" + "-" * 40 + f" [FINE PAGINA {page_num}] " + "-" * 40 + "\n\n")

        for page_num, page_text in enumerate(pages, start=1):
            for chunk_index, chunk in enumerate(
                chunk_text(page_text, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)
            ):
                all_chunks.append(
                    {
                        "id": f"{source_name}::p{page_num}::c{chunk_index}",
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

    if not all_chunks:
        raise RuntimeError("Nessun chunk valido estratto dai PDF.")

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return all_chunks

