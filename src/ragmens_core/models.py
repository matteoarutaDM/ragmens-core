"""Contratti serializzabili per chunk e risultati."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    source_json: str
    document_name: str
    text: str
    raw_text: str
    headings: list[str]
    pages: list[int]
    doc_item_refs: list[str]
    locations: list[str]
    token_count: int
