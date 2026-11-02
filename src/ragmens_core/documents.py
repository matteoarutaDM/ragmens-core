"""Conversione Docling e adapter per formati testuali."""
from __future__ import annotations
import csv, json, logging
from pathlib import Path
from typing import Callable
from ragmens_core.cleaning import clean_text
logger = logging.getLogger(__name__)

def load_text(path): return [(path.read_text(encoding='utf-8-sig'), 'documento')]
def load_json(path): return [(json.dumps(json.loads(path.read_text(encoding='utf-8-sig')),ensure_ascii=False,indent=2),'documento')]
def load_jsonl(path): return [(json.dumps(json.loads(line),ensure_ascii=False),f'record {i}') for i,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1) if line.strip()]
def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return [(json.dumps(row,ensure_ascii=False),f'riga {i}') for i,row in enumerate(csv.DictReader(f),2)]
LOADERS: dict[str,Callable] = {'.txt':load_text,'.md':load_text,'.json':load_json,'.jsonl':load_jsonl,'.csv':load_csv}

def source_files(root: Path) -> list[Path]:
    if not root.is_dir(): raise FileNotFoundError(f'Creare la cartella {root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and (p.suffix.lower() == '.pdf' or p.suffix.lower() in LOADERS))
    if not files: raise ValueError('Nessun documento supportato')
    return files

def convert_documents(files, config, converter=None):
    from docling_core.types.doc import DoclingDocument, DocItemLabel
    pdfs = [p for p in files if p.suffix.lower() == '.pdf']
    if pdfs and converter is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions(do_ocr=config.docling_ocr, do_table_structure=config.docling_tables)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    converted = []
    if pdfs:
        from docling.datamodel.base_models import ConversionStatus
        for result in converter.convert_all(pdfs, raises_on_error=False):
            source = Path(result.input.file).resolve().relative_to(config.knowledge_dir).as_posix()
            if result.status == ConversionStatus.FAILURE or (result.status == ConversionStatus.PARTIAL_SUCCESS and not config.allow_partial):
                logger.warning('Conversione rifiutata %s: %s', source, result.errors)
                continue
            if result.status == ConversionStatus.PARTIAL_SUCCESS: logger.warning('Conversione parziale accettata: %s', source)
            converted.append((source,result.document))
    for path in files:
        if path.suffix.lower() == '.pdf': continue
        source = path.relative_to(config.knowledge_dir).as_posix()
        try:
            doc = DoclingDocument(name=source)
            for text, location in LOADERS[path.suffix.lower()](path):
                if clean_text(text): doc.add_text(label=DocItemLabel.TEXT, text=text, orig=location)
            converted.append((source,doc))
        except (OSError,ValueError,UnicodeError) as exc: logger.warning('Conversione fallita %s: %s',source,exc)
    if not converted: raise ValueError('Nessun documento convertito')
    return converted
