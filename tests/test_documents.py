import json
from ragmens_core.documents import source_files,convert_documents

def test_formats(config):
    root=config.knowledge_dir
    (root/'a.csv').write_text('nome,valore\na,1\n')
    (root/'b.json').write_text('{"x":1}')
    (root/'c.jsonl').write_text('{"x":1}\n{"x":2}\n')
    docs=convert_documents(source_files(root),config)
    assert len(docs)==5
    assert len(next(d for s,d in docs if s=='c.jsonl').texts)==2
    assert next(d for s,d in docs if s=='a.csv').texts[0].orig=='riga 2'

def test_pdf_status_and_partial(config, caplog):
    from dataclasses import replace
    from types import SimpleNamespace
    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc import DoclingDocument, DocItemLabel
    pdf=config.knowledge_dir/'prova.pdf'; pdf.write_bytes(b'%PDF')
    doc=DoclingDocument(name='prova'); doc.add_text(label=DocItemLabel.TEXT,text='testo')
    class Converter:
        def convert_all(self,paths,raises_on_error=False):
            yield SimpleNamespace(input=SimpleNamespace(file=pdf),status=ConversionStatus.PARTIAL_SUCCESS,errors=['partial'],document=doc)
    assert not any(s=='prova.pdf' for s,d in convert_documents(source_files(config.knowledge_dir),config,Converter()))
    assert any(s=='prova.pdf' for s,d in convert_documents(source_files(config.knowledge_dir),replace(config,allow_partial=True),Converter()))
    assert 'parziale' in caplog.text.lower()
