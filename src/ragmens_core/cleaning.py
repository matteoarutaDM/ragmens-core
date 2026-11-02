"""Pulizia mirata dei campi testuali Docling, senza mutare l'originale."""
import copy
import re
import unicodedata
import ftfy

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFC', ftfy.fix_text(text))
    for ch in ('\u00ad','\u200b','\ufeff'): text = text.replace(ch, '')
    text = text.replace('\xa0',' ').replace('\r\n','\n').replace('\r','\n')
    text = re.sub(r'[ \t]+',' ',text)
    text = re.sub(r' *\n *','\n',text)
    return re.sub(r'\n{3,}','\n\n',text).strip()

def clean_document(data: dict) -> dict:
    result = copy.deepcopy(data)
    items = list(result.get('texts', []))
    for table in result.get('tables', []):
        if isinstance(table,dict) and isinstance(table.get('data'),dict):
            items.extend(table['data'].get('table_cells', []))
    for item in items:
        if isinstance(item,dict) and isinstance(item.get('text'),str): item['text'] = clean_text(item['text'])
    return result
