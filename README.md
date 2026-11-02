# ragmens-core

`ragmens-core` è una libreria Python modulare per acquisire documenti, cercarne i passaggi pertinenti e ottenere risposte da un LLM esterno. Lo stack predefinito è **Docling → HybridChunker → Qwen3-Embedding-4B → Qdrant → Ollama/qwen3:14b**. Importare il pacchetto non carica modelli e non invia richieste.

## Funzionalità e architettura

```text
OFFLINE  documenti → Docling → JSON → ftfy → HybridChunker → Qwen embedding → Qdrant
                                  └──── cache versionata e checksum ────────────┘
ONLINE   domanda → Qwen query embedding → Qdrant → fonti → contesto + memoria
                                                            → prompt → Ollama → risposta
```

Docling conserva struttura, tabelle e provenienza PDF. I documenti testuali sono adattati a Docling. HybridChunker produce testi contestualizzati e il tokenizer dell'embedder verifica il limite. Qdrant esegue ricerca coseno e filtri sulle fonti. La pipeline include cache verificata, citazioni, memoria opzionale, generazione e streaming.

## Organizzazione

```text
src/ragmens_core/
  __init__.py   API pubblica
  config.py     configurazione e validazione
  models.py     contratto chunk
  documents.py  conversione e loader
  cleaning.py   pulizia strutturata
  chunking.py   HybridChunker
  embeddings.py SentenceTransformers e dispositivo
  cache.py      firme e checksum
  index.py      artefatti e collection Qdrant
  retrieval.py  query e filtri
  context.py    budget e citazioni
  memory.py     memoria conversazionale
  prompts.py    istruzioni e messaggi
  llm.py        adapter Ollama e streaming
  pipeline.py   orchestrazione e chiusura
  api.py        adapter FastAPI opzionale
  evaluation.py metriche retrieval su esempi annotati
examples/       script e notebook dimostrativi
tests/          test locali e integrazione facoltativa
```

## Prerequisiti e installazione

Python 3.10 o superiore compatibile con le versioni delle dipendenze risolte da pip. Qwen3-Embedding-4B e qwen3:14b richiedono molte risorse RAM/VRAM e spazio su disco; su CPU possono essere molto lenti. Docling e SentenceTransformers possono scaricare pesi alla prima preparazione. Per la generazione serve Ollama installato e avviato separatamente:

```bash
ollama pull qwen3:14b
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -c 'from ragmens_core import RagConfig, RAGPipeline; print("import OK")'
cp .env.example .env
```

Per l'API installare `pip install -e '.[api]'`; per notebook usare l'extra `notebook`. OCR può richiedere componenti aggiuntivi del proprio ambiente Docling. La libreria non installa né avvia Ollama o il suo modello.

## Configurazione

`RagConfig.from_env()` legge `.env`, poi le variabili ambiente, poi eventuali parametri Python espliciti. `RagConfig(...)` costruisce una configurazione indipendente. I percorsi relativi di `from_env` sono risolti rispetto alla directory corrente o a `base_dir`. Copiare `.env.example` per vedere tutte le opzioni. Non committare credenziali reali.

| Variabile | Default | Uso e quando modificarla |
|---|---|---|
| PROJECT_ID | default | Identità del progetto nella firma corpus |
| KNOWLEDGE_DIR | knowledge | Cartella documenti |
| RAG_ARTIFACT_DIR | rag_artifacts | Artefatti e memoria, esterna al corpus |
| FORCE_REBUILD_INDEX | 0 | 1 per forzare una nuova generazione |
| DOCLING_OCR | 0 | 1 per PDF scansionati, con OCR disponibile |
| ALLOW_PARTIAL | 0 | 1 per accettare conversioni parziali, segnalate nei log |
| SAVE_RAW_JSON / SAVE_CLEAN_JSON | 1 / 1 | Conservazione dei JSON intermedi |
| DOCLING_TABLES | 1 | Analisi struttura tabelle PDF |
| CHUNKER | hybrid | Adapter chunking supportato |
| MAX_CHUNK_TOKENS | 256 | Limite del tokenizer embedding, special token inclusi |
| CHUNK_MERGE_PEERS | 1 | Unione elementi contigui |
| CHUNK_REPEAT_TABLE_HEADER | 1 | Ripetizione intestazione tabella |
| CHUNK_OMIT_HEADER_ON_OVERFLOW | 0 | Omissione intestazione in overflow |
| CHUNK_ALWAYS_EMIT_HEADINGS | 0 | Emissione autonoma dei titoli |
| EMBEDDING_MODEL | Qwen/Qwen3-Embedding-4B | Modello e tokenizer di indicizzazione |
| EMBEDDING_BATCH_SIZE | 8 | Ridurre se manca memoria |
| EMBEDDING_MAX_SEQ_LENGTH | 512 | Limite sequenza embedding |
| EMBEDDING_DEVICE | auto | auto, cpu, cuda o mps |
| VECTOR_DB | qdrant | Database supportato |
| QDRANT_MODE | local | local o remote |
| QDRANT_PATH | rag_artifacts/qdrant | Database locale |
| QDRANT_COLLECTION | documents | Prefisso collection per firma e generazione |
| QDRANT_URL / QDRANT_API_KEY | vuoto | Endpoint remoto e chiave facoltativa |
| TOP_K | 5 | Numero risultati retrieval |
| QUERY_INSTRUCTION | Given a question, retrieve relevant passages from documents that answer the question. | Istruzione Qwen per query |
| LLM_PROVIDER | ollama | Provider supportato |
| OLLAMA_MODEL | qwen3:14b | Modello generativo già disponibile |
| OLLAMA_BASE_URL | http://localhost:11434 | Endpoint Ollama |
| LLM_TEMPERATURE | 0.15 | Temperatura |
| LLM_NUM_CTX | 16384 | Finestra configurata per Ollama |
| LLM_NUM_PREDICT | 3500 | Numero massimo token generati |
| LLM_REPEAT_PENALTY | 1.08 | Penalità ripetizione |
| LLM_TIMEOUT | 120 | Timeout in secondi |
| USE_MEMORY | 0 | 1 per abilitare la memoria |
| SESSION_ID | default | Identità conversazione |
| MEMORY_RECENT_TURNS | 6 | Numero turni recenti |
| MEMORY_RETRIEVAL_TOP_K | 3 | Numero ricordi semantici |
| MEMORY_MAX_CHARS | 4000 | Limite caratteri memoria |
| CONTEXT_MAX_TOKENS | 8000 | Budget stimato contesto documentale |
| CONTEXT_MAX_CHARS | 30000 | Limite aggiuntivo in caratteri |
| SYSTEM_RESERVE_TOKENS | 1500 | Spazio riservato alle istruzioni |
| MEMORY_RESERVE_TOKENS | 1000 | Spazio riservato alla memoria |
| RESPONSE_RESERVE_TOKENS | 3500 | Spazio riservato alla generazione |

## Documenti e indicizzazione

Inserire PDF, TXT, Markdown, CSV, JSON e JSONL UTF-8 in `knowledge/`. JSON è serializzato interamente, CSV per riga, JSONL per record. I PDF testuali passano a `DocumentConverter`; per PDF scansionati abilitare OCR e installarne i requisiti. Conversioni fallite vengono segnalate e saltate; quelle parziali sono rifiutate salvo `ALLOW_PARTIAL=1`.

La preparazione salva JSON originali e puliti, chunk contestualizzati, embedding NumPy e metadati. La pulizia interessa i campi `text` dei testi e delle celle tabellari, senza modificare riferimenti e coordinate. Il tokenizer Qwen sostituisce quello MiniLM del riferimento. Se dopo la contestualizzazione un chunk supera il limite, viene suddiviso in segmenti consecutivi senza scartare testo; ciascuno conserva i metadati di provenienza del chunk originale.

## Retrieval senza LLM

```python
from ragmens_core import RagConfig, RAGPipeline
with RAGPipeline.prepare(RagConfig.from_env()) as rag:
    result = rag.preview('Quali sono le conclusioni principali?')
    print(result['context'])
    print(result['sources'])
```

`preview()` non chiama Ollama. Il modello embedding è necessario. `sources=['cartella/documento.pdf']` filtra per percorso relativo esatto prima del ranking. Gli score coseno non sono probabilità.

## RAG + LLM e streaming

Con Ollama avviato e il modello già scaricato:

```python
from ragmens_core import RAGPipeline
with RAGPipeline.prepare() as rag:
    result = rag.ask('Quali sono le conclusioni principali?')
    print(result['answer'])
    print(result['sources'])
    for pezzo in rag.ask_stream('Quali limiti presentano?'):
        print(pezzo, end='', flush=True)
```

`ask()` restituisce domanda, risposta, risultati, contesto, fonti incluse, risultati esclusi, troncamento e messaggi. Lo streaming esegue una ricerca e una richiesta; il turno viene salvato solo al completamento dell'iterazione. Le citazioni `[FONTE N]` corrispondono alle fonti effettivamente incluse nel prompt, ma non verificano la veridicità della risposta.

## Memoria

Con `USE_MEMORY=1` i turni vengono salvati in JSONL e quelli recenti mantenuti in RAM; un prodotto scalare NumPy recupera semanticamente i turni precedenti, separatamente dall'indice documentale. Le memorie sono isolate per sessione, firma corpus e filtro fonti. Le risposte precedenti non sono fonti documentali. Il retrieval usa la domanda originale: i follow-up ambigui possono richiedere in futuro un query rewriter.

## Budget, cache e Qdrant

Il budget usa il numero di byte UTF-8 come stima prudente dei token: non è un conteggio esatto Ollama e non include necessariamente tutte le particolarità del template di chat. Sono riservati spazi configurabili per sistema, memoria e risposta; il prompt stimato troppo grande genera un errore. Tarare i limiti sul modello effettivo.

La firma SHA-256 include contenuto e percorso dei documenti, progetto, modello, configurazione Docling/chunking e versioni dipendenze. Sotto `rag_artifacts/indexes/<firma>/generations/` ogni generazione contiene JSON, `chunks.jsonl`, `embeddings.npy`, `embedding_metadata.jsonl`, `manifest.json` e `index_meta.json`. Un puntatore viene aggiornato dopo la verifica. Checksum, vettori, payload, dimensioni e numero punti sono controllati alla riapertura. Una modifica del corpus o parametri cambia firma; una cache corrotta viene ricostruita. `FORCE_REBUILD_INDEX=1` forza una nuova generazione. Le versioni e collection precedenti non vengono cancellate automaticamente.

Qdrant locale richiede accesso esclusivo al percorso: chiudere la pipeline (`close()` o context manager) prima di riaprirlo in un altro processo. Modalità remota: `QDRANT_MODE=remote` e `QDRANT_URL`. Ogni generazione usa una collection dedicata; se esistono collection incompatibili non vengono sovrascritte.

## Personalizzazione e riutilizzo

Installare da un altro progetto con `pip install -e /percorso/ragmens` e importare `ragmens_core` in uno script o notebook. `RagConfig.from_env(embedding_model='...', ollama_model='...', top_k=3)` modifica le impostazioni. `RAGPipeline.prepare(config, system_prompt='...')` sostituisce il prompt. Gli adapter di formati testuali si registrano in `LOADERS` e restituiscono coppie `(testo, posizione)`; dopo cambiamenti alla logica documentale aggiornare `PIPELINE_VERSION` in `cache.py`.

## FastAPI opzionale

L'applicazione ospitante può creare una pipeline una sola volta e passare l'istanza a `ragmens_core.api.create_app(rag)`. Espone `GET /health` e `POST /chat` con JSON `{"question": "...", "stream": true}`; il ciclo di vita chiude la pipeline. Per avviarla creare un modulo applicativo con `app = create_app(RAGPipeline.prepare())`, poi eseguire `uvicorn nome_modulo:app`. Proteggere l'API prima di esporla pubblicamente: autenticazione e gestione concorrente multiutente non sono incluse.

## Migrazione e troubleshooting

Gli indici FAISS/MiniLM precedenti non sono compatibili: mantenere i documenti e ricostruire con la nuova configurazione. `RagConfig` è un alias di `Config`, `RagIndex` di `RAGState`, ma la vecchia costruzione e le vecchie API `ShortTermMemory`, `VectorMemory`, `chunk_text` e `ingest_pdfs` non sono compatibili. Usare `RAGPipeline.prepare`, `preview` e `ask`.

Se l'import fallisce, installare nel Python del kernel corretto. Per modello embedding non disponibile verificare accesso ai pesi e dipendenze; se la GPU esaurisce memoria ridurre batch o scegliere CPU. Per Ollama verificare servizio, URL e `ollama pull qwen3:14b`. Una collection incompatibile o incompleta genera ricostruzione; chiudere altri processi che occupano il database locale. Se non si trovano documenti verificare `KNOWLEDGE_DIR`; PDF senza testo possono richiedere OCR. Una cache corrotta viene ricostruita. Errori di lunghezza tokenizer richiedono riduzione di `MAX_CHUNK_TOKENS`; per prompt troppo grandi ridurre budget, memoria o testo aggiuntivo.

## Test e limiti

```bash
python -m pytest -q
RAG_RUN_INTEGRATION=1 python -m pytest -m integration -q
```

Il secondo comando richiede documenti, pesi e Ollama reali; i test standard usano sostituti leggeri e non dimostrano la qualità dei modelli. Non sono implementati ricerca ibrida, BM25, reranking, GraphRAG, aggiornamento incrementale, gestione concorrente multiutente o verifica automatica della veridicità. Non è disponibile un dataset di benchmark: accuratezza e prestazioni devono essere misurate su dati di riferimento propri. Per valutare il retrieval:

```python
from ragmens_core.evaluation import evaluate_retrieval
cases = [{'question': 'Quali conclusioni?', 'expected_sources': ['documento.pdf']}]
with RAGPipeline.prepare() as rag:
    print(evaluate_retrieval(rag, cases, top_k=5))
```

La funzione restituisce Recall@K, Hit Rate@K, MRR e tempo medio, calcolati sulle fonti attese fornite.
