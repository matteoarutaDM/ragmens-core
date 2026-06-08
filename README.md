# ragmens-core

`ragmens-core` e una libreria Python per riusare la parte RAG stabile dei progetti:

- lettura PDF
- pulizia testo
- chunking
- creazione o caricamento indice FAISS
- retrieval semantico
- costruzione del contesto da passare a un LLM
- memoria conversazionale opzionale

La libreria non contiene la parte LLM. Prompt, chiamate OpenAI/Ollama, FastAPI, frontend e logica di chat restano nel progetto applicativo.

## 1. Installazione

### Installazione da GitHub

Nel progetto dove vuoi usare la libreria:

```bash
python3 -m pip install git+https://github.com/matteoarutaDM/ragmens-core.git
```

Poi verifica che l'import funzioni:

```bash
python3 -c "from ragmens_core import RagConfig, RagIndex; print('ragmens-core ok')"
```

Nel codice Python puoi importarla cosi:

```python
from ragmens_core import RagConfig, RagIndex
```

### Installazione locale in sviluppo

Se hai clonato questa repository e vuoi modificarla, entra nella cartella del repo:

```bash
pip install -e .
```

`-e` significa "editable": se modifichi i file dentro `src/ragmens_core/`, Python usa subito la versione aggiornata senza reinstallare.

### Installazione dentro Jupyter o notebook

Se `pip install ...` dal terminale sembra riuscire ma nel notebook ottieni:

```text
ModuleNotFoundError: No module named 'ragmens_core'
```

allora il notebook sta usando un Python diverso da quello del terminale. In questo caso installa la libreria direttamente dal notebook con `sys.executable`:

```python
import sys
!{sys.executable} -m pip install git+https://github.com/matteoarutaDM/ragmens-core.git
```

Poi riavvia il kernel e verifica:

```python
from ragmens_core import RagConfig, RagIndex
print("ragmens-core ok")
```

Per vedere quale Python usa il notebook:

```python
import sys
print(sys.executable)
```

## 2. Struttura della libreria

Il file `pyproject.toml` non e la libreria. Serve solo a dire a Python come installarla.

La libreria vera e qui:

```text
src/ragmens_core/
  __init__.py
  config.py
  chunking.py
  context.py
  documents.py
  index.py
  memory.py
```

Il nome da usare negli import e:

```python
import ragmens_core
```

oppure:

```python
from ragmens_core import RagConfig, RagIndex
```

## 3. Uso minimo consigliato

Questo e il flusso normale:

```python
from ragmens_core import RagConfig, RagIndex

config = RagConfig.from_env()
rag = RagIndex.prepare(config)

domanda = "come verifico il corretto funzionamento della pompa idraulica?"
hits = rag.retrieve(domanda, top_k=5)
context = rag.build_context(hits)

print(context)
```

## 3.1 Uso minimo consigliato con installazione da script

```python
import sys
!{sys.executable} -m pip install git+https://github.com/matteoarutaDM/ragmens-core.git

from ragmens_core import RagConfig, RagIndex

config = RagConfig.from_env()
rag = RagIndex.prepare(config)

domanda = "come verifico il corretto funzionamento della pompa idraulica?"
hits = rag.retrieve(domanda, top_k=5)
context = rag.build_context(hits)

print(context)
```

Cosa succede:

1. `RagConfig.from_env()` legge configurazione e percorsi.
2. `RagIndex.prepare(config)` carica o crea chunks e indice FAISS.
3. `rag.retrieve(...)` cerca i chunk piu rilevanti.
4. `rag.build_context(...)` trasforma i risultati in testo pronto per il prompt LLM.



## 3.2 Uso minimo consigliato con LLM

```python
# ============================================================
# 1) RAG CORE DA LIBRERIA
# ============================================================

from __future__ import annotations

from typing import Dict, List

from ragmens_core import RagConfig, RagIndex, ShortTermMemory, VectorMemory, ltm_text


config = RagConfig.from_env()
rag = RagIndex.prepare(config)
stm = ShortTermMemory(max_messages=10)
ltm = VectorMemory(
    embedder=rag.embedder,
    mem_dir=config.mem_dir,
    corpus_signature=rag.corpus_signature,
)

# ============================================================
# 2) GENERATION LLM - PROMPT COME NEL NOTEBOOK ORIGINALE
# ============================================================

from openai import OpenAI


LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


QA_PROMPT = """Sei un assistente virtuale esperto della conoscenza e conrtesto dei pdf.

Personalita e stile:
- Rispondi sempre in italiano.
- Mantieni un tono amichevole, naturale, professionale e disponibile.
- Se l'utente saluta, ringrazia, chiede come stai, fa conversazione generica o pone una domanda amichevole, rispondi in modo cordiale e umano, senza forzare riferimenti tecnici.
- Non essere freddo o meccanico: usa frasi semplici, dirette e collaborative.
- Evita risposte troppo lunghe quando la domanda e semplice.
- Quando la domanda e tecnica, resta preciso, ordinato e concreto.

Uso del contesto:
- Usa il CONTEXT come fonte principale quando contiene informazioni rilevanti.
- Se il CONTEXT contiene fonti, pagine o chunk utili, integra la risposta con quelle informazioni.
- Se il CONTEXT non contiene abbastanza informazioni per rispondere completamente, non dire in modo esplicito e secco che l'informazione non e presente nel contesto.
- In assenza di informazioni sufficienti nel CONTEXT, costruisci comunque una risposta utile basandoti sulla tua conoscenza generale, mantenendo un tono prudente e professionale.
- Quando usi conoscenza generale non supportata direttamente dal CONTEXT, evita affermazioni troppo assolute e formula la risposta come indicazione generale.
- Non inventare nomi di schermate, menu, percorsi, campi o configurazioni specifiche se non sono nel CONTEXT o non sono ragionevolmente noti.
- Se una risposta richiede dati specifici dell'ambiente del cliente, spiega in modo amichevole cosa verificare o quali informazioni servirebbero.

Qualita della risposta:
- Dai prima la risposta pratica, poi eventuali dettagli.
- Per procedure operative, usa passaggi chiari e numerati.
- Per concetti o definizioni, usa una spiegazione breve seguita da esempi se utili.
- Per domande ambigue, rispondi con l'interpretazione piu probabile e, se serve, chiedi una precisazione alla fine.
- Se l'utente chiede un confronto, struttura la risposta in punti o tabella testuale semplice.
- Se l'utente chiede una configurazione NetSuite, evidenzia prerequisiti, percorso operativo e risultato atteso quando possibile.
- Mantieni risposte concise ma complete: evita testo superfluo, ma non omettere passaggi importanti.

Citazioni e fonti:
- Cita le fonti quando nel CONTEXT sono disponibili riferimenti a Fonte, Pagina o Chunk.
- Non citare fonti inesistenti.
- Se stai rispondendo a una domanda amichevole o generale non tecnica, non serve citare fonti.

MEMORIA BREVE:
{stm}

MEMORIA LUNGA RILEVANTE:
{ltm}

CONTEXT:
{context}

DOMANDA:
{question}

RISPOSTA (in italiano, chiara, strutturata e professionale):"""


def build_prompt(question: str, context: str, memory_hits: List[dict]) -> str:
    return QA_PROMPT.format(
        stm=stm.text(),
        ltm=ltm_text(memory_hits),
        context=context,
        question=question,
    )


def call_llm(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def ask(
    question: str,
    top_k: int | None = None,
    document_ids: List[str] | None = None,
    save_memory: bool = True,
) -> Dict[str, object]:
    hits = rag.retrieve(question, top_k=top_k, document_ids=document_ids)
    context = rag.build_context(hits)
    memory_hits = ltm.search(SESSION_ID, question, top_k=4)
    prompt = build_prompt(question, context, memory_hits)
    answer = call_llm(prompt)

    stm.add("user", question)
    stm.add("assistant", answer)
    if save_memory:
        ltm.add_turn(SESSION_ID, question, answer)

    print("\n=== RISPOSTA ===\n")
    print(answer)
    print("\n=== FONTI RECUPERATE ===")
    for hit in hits:
        print(f"- {hit['source']} | pagina {hit['page']} | chunk {hit['chunk_index']} | score {hit['score']:.3f}")

    return {
        "question": question,
        "answer": answer,
        "hits": hits,
        "context": context,
        "memory_hits": memory_hits,
        "prompt": prompt,
    }


def chat_loop() -> None:
    print("Chat RAG pronta. Scrivi una domanda oppure 'exit' per uscire.")
    while True:
        question = input("\nDomanda> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            ask(question)


print("LLM configurato:", LLM_MODEL, "@", LLM_BASE_URL)

# ============================================================
# 3) DOMANDA DI PROVA
# ============================================================

DOMANDA = """
come verifico il corretto funzionamento della pompa idraulica?
""".strip()

risultato = ask(DOMANDA)
risposta_llm = risultato["answer"]
```



## 4. Variabili ambiente supportate

Puoi creare un file `.env` nella root del progetto.

Esempio:

```env
PDF_DIR=./pdf_immagini
INDEX_DIR=./index
MEM_DIR=./memory
OUTPUT_DEBUG_DIR=./outputs_debug
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=12
CONTEXT_MAX_CHARS=4500
FORCE_REBUILD_INDEX=0
CHUNK_SIZE=1200
CHUNK_OVERLAP=180
MIN_CHUNK_CHARS=50
```

Significato:

| Variabile | Default | A cosa serve |
|---|---:|---|
| `PDF_DIR` | `./pdf_immagini` | Cartella dove la libreria cerca i PDF. |
| `INDEX_DIR` | `./index` | Cartella dove salva `chunks.jsonl`, `faiss.index`, `index_meta.json`. |
| `MEM_DIR` | `./memory` | Cartella per la memoria conversazionale opzionale. |
| `OUTPUT_DEBUG_DIR` | `./outputs_debug` | Cartella dove salva il testo estratto dai PDF per debug. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Modello SentenceTransformers usato per gli embedding. |
| `TOP_K` | `12` | Numero default di risultati recuperati da `retrieve`. |
| `CONTEXT_MAX_CHARS` | `4500` | Numero massimo default di caratteri nel contesto. |
| `FORCE_REBUILD_INDEX` | `0` | Se `1`, ricrea sempre chunks e indice FAISS. |
| `CHUNK_SIZE` | `1200` | Dimensione massima indicativa di ogni chunk. |
| `CHUNK_OVERLAP` | `180` | Caratteri sovrapposti tra chunk consecutivi. |
| `MIN_CHUNK_CHARS` | `50` | Scarta chunk piu corti di questo valore. |

## 5. Import disponibili

Tutti questi oggetti sono importabili direttamente da `ragmens_core`:

```python
from ragmens_core import (
    RagConfig,
    RagIndex,
    ShortTermMemory,
    VectorMemory,
    build_context,
    chunk_text,
    clean_pdf_text,
    ingest_pdfs,
    ltm_text,
    read_pdf_pages,
)
```

### `RagConfig`

Classe di configurazione della pipeline RAG.

Import:

```python
from ragmens_core import RagConfig
```

Uso consigliato:

```python
config = RagConfig.from_env()
```

Parametri principali della classe:

| Parametro | Tipo | A cosa serve |
|---|---|---|
| `base_dir` | `Path` | Directory base del progetto. |
| `pdf_dir` | `Path` | Cartella PDF. |
| `index_dir` | `Path` | Cartella dell'indice FAISS e dei chunk. |
| `mem_dir` | `Path` | Cartella della memoria. |
| `output_debug_dir` | `Path` | Cartella output debug testo PDF. |
| `embedding_model` | `str` | Nome modello SentenceTransformers. |
| `top_k` | `int` | Default per numero risultati retrieval. |
| `context_max_chars` | `int` | Default massimo caratteri context. |
| `force_rebuild_index` | `bool` | Forza ricostruzione indice. |
| `chunk_size` | `int` | Dimensione chunk. |
| `chunk_overlap` | `int` | Sovrapposizione chunk. |
| `min_chunk_chars` | `int` | Lunghezza minima chunk valido. |

Metodi e proprieta:

| Nome | A cosa serve |
|---|---|
| `RagConfig.from_env(base_dir=None, env_file=None)` | Crea config leggendo `.env` e variabili ambiente. |
| `config.ensure_dirs()` | Crea cartelle `index`, `memory`, `outputs_debug` se mancanti. |
| `config.chunks_path` | Percorso `chunks.jsonl`. |
| `config.faiss_path` | Percorso `faiss.index`. |
| `config.index_meta_path` | Percorso `index_meta.json`. |

Esempio con `.env` custom:

```python
config = RagConfig.from_env(env_file="config/.env.rag")
```

Esempio con configurazione manuale:

```python
from pathlib import Path
from ragmens_core import RagConfig

config = RagConfig(
    base_dir=Path.cwd(),
    pdf_dir=Path("pdf_immagini").resolve(),
    index_dir=Path("index").resolve(),
    mem_dir=Path("memory").resolve(),
    output_debug_dir=Path("outputs_debug").resolve(),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=8,
    context_max_chars=5000,
    chunk_size=1200,
    chunk_overlap=180,
    min_chunk_chars=50,
)
```

### `RagIndex`

Classe principale della libreria. Gestisce chunks, modello embedding, indice FAISS e retrieval.

Import:

```python
from ragmens_core import RagIndex
```

Creazione o caricamento automatico:

```python
rag = RagIndex.prepare(config)
```

Parametri di `RagIndex.prepare`:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `config` | `RagConfig | None` | `None` | Configurazione. Se `None`, usa `RagConfig.from_env()`. |
| `force_rebuild` | `bool | None` | `None` | Se `True`, ricrea indice. Se `None`, usa `config.force_rebuild_index`. |

La libreria ricrea l'indice se:

- manca `chunks.jsonl`
- manca `faiss.index`
- manca `index_meta.json`
- cambia `PDF_DIR`
- cambia modello embedding
- vengono aggiunti, rimossi o modificati PDF
- `FORCE_REBUILD_INDEX=1`

Caricamento di un indice gia esistente:

```python
rag = RagIndex.load("index")
```

Parametri di `RagIndex.load`:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `index_dir` | `str | Path` | obbligatorio | Cartella con `chunks.jsonl` e `faiss.index`. |
| `pdf_dir` | `str | Path | None` | `None` | Cartella PDF, se vuoi indicarla esplicitamente. |
| `embedding_model` | `str | None` | `None` | Modello embedding. Se `None`, prova a leggerlo da `index_meta.json`. |

Retrieval:

```python
hits = rag.retrieve("domanda utente", top_k=5)
```

Parametri di `retrieve`:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `query` | `str` | obbligatorio | Domanda o testo da cercare. |
| `top_k` | `int | None` | `None` | Numero risultati. Se `None`, usa `config.top_k`. |
| `document_ids` | `list[str] | None` | `None` | Limita la ricerca a specifici documenti. |

Esempio con filtro documento:

```python
hits = rag.retrieve(
    "pompa idraulica",
    top_k=5,
    document_ids=["manuale.pdf"],
)
```

Ogni hit e un dizionario simile a:

```python
{
    "id": "manuale.pdf::p10::c2",
    "source": "manuale.pdf",
    "page": 10,
    "chunk_index": 2,
    "text": "testo del chunk...",
    "score": 0.72,
}
```

Costruzione context:

```python
context = rag.build_context(hits, max_chars=3000)
```

Parametri di `build_context`:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `results` | `list[dict]` | obbligatorio | Lista di hit restituiti da `retrieve`. |
| `max_chars` | `int | None` | `None` | Limite caratteri. Se `None`, usa `config.context_max_chars`. |

### `chunk_text`

Divide un testo in chunk.

Import:

```python
from ragmens_core import chunk_text
```

Uso:

```python
chunks = chunk_text(
    testo,
    chunk_size=1200,
    overlap=180,
    min_chars=50,
)
```

Parametri:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `text` | `str` | obbligatorio | Testo da dividere. |
| `chunk_size` | `int` | `1200` | Dimensione massima indicativa del chunk. |
| `overlap` | `int` | `180` | Caratteri ripetuti tra chunk consecutivi. |
| `min_chars` | `int` | `50` | Scarta chunk piu corti. |

### `build_context`

Funzione standalone per costruire il contesto da una lista di risultati.

Import:

```python
from ragmens_core import build_context
```

Uso:

```python
context = build_context(hits, max_chars=4500)
```

Parametri:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `results` | `list[dict]` | obbligatorio | Risultati retrieval. |
| `max_chars` | `int` | `4500` | Limite massimo del contesto. |

### `clean_pdf_text`

Pulisce testo estratto da PDF.

Import:

```python
from ragmens_core import clean_pdf_text
```

Uso:

```python
testo_pulito = clean_pdf_text(testo_grezzo)
```

Parametri:

| Parametro | Tipo | A cosa serve |
|---|---|---|
| `text` | `str` | Testo grezzo da normalizzare. |

La funzione normalizza newline, spazi e righe vuote multiple.

### `read_pdf_pages`

Legge un PDF e restituisce una lista di testi, uno per pagina.

Import:

```python
from pathlib import Path
from ragmens_core import read_pdf_pages
```

Uso:

```python
pages = read_pdf_pages(Path("pdf_immagini/manuale.pdf"))
```

Parametri:

| Parametro | Tipo | A cosa serve |
|---|---|---|
| `pdf_path` | `Path` | Percorso del PDF da leggere. |

### `ingest_pdfs`

Funzione piu bassa di livello. Legge PDF, crea chunk e salva `chunks.jsonl`.

Di solito non serve chiamarla direttamente, perche `RagIndex.prepare()` la usa in automatico.

Import:

```python
from pathlib import Path
from ragmens_core import ingest_pdfs
```

Uso:

```python
chunks = ingest_pdfs(
    pdf_dir=Path("pdf_immagini"),
    chunks_path=Path("index/chunks.jsonl"),
    output_debug_dir=Path("outputs_debug"),
    save_debug_text=True,
    chunk_size=1200,
    overlap=180,
    min_chars=50,
)
```

Parametri:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `pdf_dir` | `Path` | obbligatorio | Cartella PDF. |
| `chunks_path` | `Path` | obbligatorio | File JSONL dove salvare i chunk. |
| `output_debug_dir` | `Path | None` | `None` | Cartella per testo estratto dai PDF. |
| `save_debug_text` | `bool` | `True` | Se `True`, salva testo estratto per debug. |
| `chunk_size` | `int` | `1200` | Dimensione chunk. |
| `overlap` | `int` | `180` | Sovrapposizione chunk. |
| `min_chars` | `int` | `50` | Lunghezza minima chunk. |

### `ShortTermMemory`

Memoria breve in RAM. Utile per tenere gli ultimi messaggi nella costruzione del prompt.

Import:

```python
from ragmens_core import ShortTermMemory
```

Uso:

```python
stm = ShortTermMemory(max_messages=10)
stm.add("user", "ciao")
stm.add("assistant", "ciao, dimmi pure")

testo_memoria = stm.text(max_chars=1200)
```

Parametri costruttore:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `max_messages` | `int` | `10` | Numero massimo di messaggi mantenuti. |

Metodi:

| Metodo | A cosa serve |
|---|---|
| `add(role, content)` | Aggiunge un messaggio. |
| `text(max_chars=1200)` | Restituisce la memoria come testo. |
| `clear()` | Svuota la memoria. |

### `VectorMemory`

Memoria lunga conversazionale salvata su FAISS separato.

Import:

```python
from ragmens_core import VectorMemory
```

Uso:

```python
memory = VectorMemory(
    embedder=rag.embedder,
    mem_dir=config.mem_dir,
    corpus_signature=rag.corpus_signature,
)

memory.add_turn(
    session_id="sessione-1",
    user_text="come controllo la pompa?",
    assistant_text="devi verificare...",
)

memory_hits = memory.search(
    session_id="sessione-1",
    query="pompa idraulica",
    top_k=4,
)
```

Parametri costruttore:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `embedder` | `SentenceTransformer` | obbligatorio | Modello embedding gia caricato, di solito `rag.embedder`. |
| `mem_dir` | `Path` | obbligatorio | Cartella dove salvare memoria JSONL e FAISS. |
| `corpus_signature` | `str | None` | `None` | Firma del corpus, utile per evitare memoria di PDF diversi. |
| `jsonl_name` | `str` | `memory.jsonl` | Nome file metadati memoria. |
| `faiss_name` | `str` | `memory.faiss` | Nome file indice memoria. |

Metodi:

| Metodo | A cosa serve |
|---|---|
| `add_turn(session_id, user_text, assistant_text)` | Salva un turno conversazionale. |
| `search(session_id, query, top_k=4)` | Cerca ricordi rilevanti per una sessione. |

### `ltm_text`

Converte risultati di `VectorMemory.search()` in testo pronto per il prompt.

Import:

```python
from ragmens_core import ltm_text
```

Uso:

```python
memory_hits = memory.search("sessione-1", "pompa idraulica")
testo_memoria_lunga = ltm_text(memory_hits, max_chars=1200)
```

Parametri:

| Parametro | Tipo | Default | A cosa serve |
|---|---|---:|---|
| `mem_hits` | `list[dict]` | obbligatorio | Risultati di `VectorMemory.search`. |
| `max_chars` | `int` | `1200` | Limite massimo del testo generato. |

## 6. Script di prova

Ho aggiunto uno script semplice:

```bash
python3 prova_rag.py
```

Lo script:

1. legge la configurazione
2. prepara o carica l'indice
3. esegue una domanda di prova
4. stampa fonti e contesto

Se non ci sono PDF in `pdf_immagini/`, avrai un errore simile:

```text
FileNotFoundError: Nessun PDF trovato in .../pdf_immagini
```

In quel caso metti i PDF nella cartella `pdf_immagini/` oppure imposta `PDF_DIR` nel file `.env`.

## 7. Esempio completo con LLM esterno

La libreria restituisce il `context`; poi lo usi nel tuo prompt.

```python
from ragmens_core import RagConfig, RagIndex

config = RagConfig.from_env()
rag = RagIndex.prepare(config)

question = "come verifico la pompa idraulica?"
hits = rag.retrieve(question, top_k=5)
context = rag.build_context(hits)

prompt = f"""
Usa il seguente contesto per rispondere.

CONTESTO:
{context}

DOMANDA:
{question}
"""

# Qui chiami il tuo LLM: OpenAI, Ollama, FastAPI, ecc.
```

## 8. Quando ricreare l'indice

Normalmente non devi pensarci: `RagIndex.prepare()` decide da solo se ricrearlo.

Per forzare manualmente:

```bash
FORCE_REBUILD_INDEX=1 python3 prova_rag.py
```

Oppure da codice:

```python
rag = RagIndex.prepare(config, force_rebuild=True)
```

## 9. File generati

Dentro `INDEX_DIR`:

```text
index/
  chunks.jsonl
  faiss.index
  index_meta.json
```

Dentro `MEM_DIR`, se usi `VectorMemory`:

```text
memory/
  memory.jsonl
  memory.faiss
```

Dentro `OUTPUT_DEBUG_DIR`:

```text
outputs_debug/
  nome_pdf_all_pages.txt
```

Questi file sono dati generati, non codice sorgente.

# ragmens-core
