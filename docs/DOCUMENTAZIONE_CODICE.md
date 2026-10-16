# Documentazione del codice — DVTProject

Descrizione modulo per modulo del codice del progetto. Rispecchia lo stato
attuale: i valori di configurazione citati sono quelli in `config.py`, e le
descrizioni delle funzioni quelle del codice corrente.

Le misure che giustificano molte delle scelte descritte qui stanno in
`docs/RISULTATI_SPERIMENTALI.md`, tenuto separato di proposito: questo documento
dice **cosa fa** il codice, quello dice **cosa è stato misurato**.

## Mappa dei moduli

| modulo | ruolo |
|---|---|
| `config.py` | costanti, prompt hint, regole cross-section |
| `models.py` | schema Pydantic delle 10 sezioni, unica fonte di verità sulle opzioni |
| `schema_fields.py` | lettura di campo, opzioni e titolo di una sezione dallo schema |
| `rag_setup.py` | embedding, vector store, loader, tool di ricerca |
| `agents.py` | i due agenti e il parsing delle risposte |
| `criteria_rules.py` | regole cross-section |
| `agentic_graph.py` | macchina a stati LangGraph della modalità agentica |
| `verifier.py` | Agent 3, il verificatore di ogni risposta finale |
| `pipeline.py` | orchestrazione di un referto |
| `aggregation.py` | serializzazione del form |
| `main.py` | entry point su un singolo referto |
| `run_synthetic_records.py` | esecuzione batch sul corpus |
| `generate_synthetic_records.py` | generazione e audit del corpus sintetico |
| `evaluate_predictions.py` | valutazione contro la ground truth |
| `evaluate_verifier.py` | valutazione del verificatore (Agent 3) contro la ground truth |
| `agreement_runs.py` | concordanza fra run ripetute della stessa configurazione |
| `compare_runs.py` | confronto tra due run, misura del rumore |
| `export_redcap_csv.py` | conversione dei risultati in CSV per REDCap |
| `export_prompts.py` | esportazione di prompt, query e hint in file per la tesi (non versionato) |
| `myo/` | il braccio di trasferimento sul questionario della miocardite |
| `run_myo_t05.sh` | le tre run della miocardite a temperatura 0,5 su mari, con controlli e valutazione |
| `replay_verifier.py` | il verificatore riapplicato a una run già su disco, con un modello a scelta |

---

## 1. `config.py`

Modulo di sole costanti, letto da tutti gli altri. È il pannello di controllo
della pipeline: un esperimento sul corpus TVP è una modifica a questo file, mentre
gli script di `myo/` ne cambiano i valori in memoria.

### Modelli e generazione

Tre ruoli, tre costanti separate, così ognuno si cambia indipendentemente:

- `LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"` — Agent 1 nelle modalità
  `full_text` e `rag`.
- `EVALUATOR_LLM_MODEL_NAME = "qwen3.6:27b"` — Agent 2, in ogni modalità.
- `AGENTIC_LLM_MODEL_NAME = "qwen3.6:27b"` — il solo passo di ricerca in
  modalità `agentic_graph`. È una costante separata perché questo ruolo lega un
  tool e richiede un modello che supporti il tool calling di Ollama: Llama 3 base
  non lo supporta, e Ollama risponde `model does not support tools`. Oggi ha lo
  stesso valore del valutatore, quindi Agent 1 e Agent 2 usano un solo modello
  caricato.

Parametri di generazione:

- `LLM_TEMPERATURE = 0.0` — output deterministico. Vale per Agent 1 e Agent 2;
  il braccio della miocardite la porta a 0,5 (sezione 16), l'Agent 3 ha la sua.
- `LLM_NUM_PREDICT = 2048` — tetto di token. Le due righe di risposta chiudono
  la generazione, quindi un tetto raggiunto prima costa l'intera sezione.
- `LLM_NUM_GPU = 999` — tutti i layer su GPU. La ripartizione automatica di
  Ollama lasciava parte del modello su CPU con VRAM ancora libera, e un layer su
  CPU domina il tempo per token.
- `LLM_REASONING = False` — modalità di ragionamento. Va spenta sui modelli che
  la possiedono: con essa attiva il modello può consumare l'intero tetto di token
  dentro il blocco di ragionamento, che non viaggia nel corpo della risposta,
  restituendo un `content` vuoto. `None` non invia nulla a Ollama e lascia il
  default del modello, che è la scelta corretta per un modello privo di quella
  modalità.
- `LLM_NUM_CTX = 8192` — la finestra di contesto che Ollama alloca per ogni
  richiesta, in token. Deve contenere il prompt più lungo più `LLM_NUM_PREDICT`:
  oltre quel limite Ollama scarta l'inizio del prompt. La configurazione di
  riferimento della TVP usa 1024 e 4096.
- `LLM_REQUEST_TIMEOUT = 180` secondi.

Quattro costanti governano l'Agent 3 (sezione 7):

- `VERIFIER_ENABLED = True` — acceso, dopo le regole cross-section ogni sezione
  riceve nel proprio audit log il verdetto del verificatore e P(YES). Spento, la
  pipeline non fa alcuna richiesta in più.
- `VERIFIER_INPUTS = ("answer", "reasoning")` — le varianti, una richiesta per
  sezione ciascuna: `answer` invia evidenza, opzioni e risposta; `reasoning`
  aggiunge il ragionamento dell'Agent 2; `answer_hint` e `reasoning_hint`,
  disponibili ma non nel default, aggiungono a ciascuna l'hint della sezione.
- `VERIFIER_TEMPERATURE = 0.0` — separata da `LLM_TEMPERATURE`, perché il
  punteggio è una probabilità letta dai logprob e il verdetto deve essere
  ripetibile anche quando gli altri due agenti campionano.
- `VERIFIER_MODEL_NAME = None` — il modello a cui il verificatore invia le
  richieste; con `None` è `EVALUATOR_LLM_MODEL_NAME`, il modello che ha scritto
  le risposte che giudica.
- `VERIFIER_NUM_CTX = 8192` — deve contenere la variante `reasoning`, che
  aggiunge fino a `LLM_NUM_PREDICT` token di ragionamento a un prompt che arriva
  già a circa 3200.

### Embedding e modalità di estrazione

`EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"`: multilingue per
necessità, dato che le query di retrieval sono in inglese e i referti in
italiano, e i referti non vengono tradotti automaticamente per non rischiare che
una traduzione distorca negazioni o terminologia in modo non verificabile.

`EMBEDDING_DEVICE = "cpu"`: il dispositivo su cui gira il modello di embedding.
La CPU lascia le GPU ai modelli di Ollama; indicizzare i frammenti di un referto
e cercare una query per sezione richiede pochi secondi.

`EXTRACTOR_MODE = "agentic_graph"` seleziona la strategia di Agent 1:

- `"full_text"` — passa l'intero referto nel prompt. Nessun rischio di retrieval
  sbagliato, valido finché il referto sta nella finestra di contesto.
- `"rag"` — retrieval a `k` fisso sul referto chunkato.
- `"agentic_graph"` — Agent 1 decide autonomamente quante volte e con quali
  sotto-query interrogare il tool di ricerca, orchestrato come macchina a stati
  esplicita. È la modalità di riferimento. `AGENTIC_MAX_ITERATIONS = 5` limita le
  chiamate al tool per sezione. Come evidenza restituisce i chunk grezzi del
  tool, non il turno finale dell'agente.
- `"raw_record"` — nessun Agent 1: Agent 2 legge il referto intero e non viene
  caricato alcun modello estrattore.

`BRIGHTON_CONTEXT_ENABLED = True` decide se Agent 2 riceve il contesto della
linea guida; a `False` risponde dalla sola evidenza e dalle opzioni.
`SECTION_DESCRIPTIONS_ENABLED = False` decide se sopra le opzioni numerate compare
l'intestazione della sezione presa dalla `description` del campo in `models.py`,
cioè il titolo che la sezione ha nel questionario cartaceo.

### Chunking

`EHR_CHUNK_SIZE = 800`, `EHR_CHUNK_OVERLAP = 150`, `EHR_RETRIEVER_K = 5` per il
referto; `BRIGHTON_CHUNK_SIZE`, `BRIGHTON_CHUNK_OVERLAP`, `BRIGHTON_RETRIEVER_K`
con gli stessi valori per il paper. Le dimensioni sono in caratteri.
`EHR_KB_PERSIST_DIR` e `BRIGHTON_KB_PERSIST_DIR` sono le cartelle di Chroma,
sotto `PROJECT_ROOT`, la cartella che contiene `config.py`. Quella del paper
porta nel nome il modello di embedding (per esempio
`chroma_brighton_kb_multilingual-e5-small`), perché `build_brighton_kb` la
ricarica da disco quando esiste: un indice costruito con un modello non viene
mai interrogato con un altro. Quella dei referti non ne ha bisogno, perché
`build_ehr_kb` la ricostruisce a ogni chiamata.

### `SECTION_ORDER`

`["A1", "A2", "A3_1", "A3_2", "B1_1", "B1_2", "B2", "C", "F", "X"]`. Determina
l'ordine di esecuzione in entrambi i percorsi, il ciclo di `pipeline.py` e la
coda del grafo.

### `SECTION_HINTS` e i suoi interruttori

`SECTION_HINTS` associa a una sezione un testo aggiunto al prompt di Agent 2.
Ogni hint affronta un punto in cui il valutatore legge la domanda diversamente
dal questionario. Le sezioni assenti dal dizionario vengono risposte con le sole
opzioni e il contesto della linea guida.

Due interruttori permettono l'ablazione senza toccare il codice:

- `SECTION_HINTS_ENABLED = True` — interruttore generale.
- `SECTION_HINTS_DISABLED = {"B2"}` — sezioni sospese individualmente. B2 è
  elencata perché il suo hint abbassa l'accuratezza di B2; il testo è conservato
  così l'ablazione è ripetibile.

`section_hint(section_key)` è l'unico accesso e rispetta entrambi gli
interruttori, restituendo stringa vuota quando l'hint non va inviato.
**`hint_fingerprint()`** produce, per ogni sezione che riceve un hint non vuoto,
la coppia lunghezza e prefisso sha256 del testo, più un digest `all` dell'intero
insieme; la usano lo snapshot della pipeline e il verificatore.

Contenuto attuale, in sintesi:

| sezione | cosa dice |
|---|---|
| A1 | contano solo i reperti post-mortem; l'imaging su vivente non è un'autopsia |
| A2 | attenzione estrema alle negazioni prima di termini chirurgici; un esame di imaging eseguito da solo non è una procedura |
| A3_1 | un esame che non ha trovato la TVP ha comunque un risultato (opzione 2); «results unknown» vuol dire referto mancante o non ancora disponibile; un'autopsia non è un esame di imaging |
| A3_2 | si selezionano solo gli esami che hanno confermato la TVP; un esame che non è nessuna delle quattro modalità nominate è "Other" |
| B1_1 | la differenza tra seconda e terza opzione riguarda ciò che sai del paziente, non del documento: l'assenza di un referto non è un referto di assenza |
| B2 | *disattivato* |
| C | usa il limite del laboratorio se il referto lo indica, altrimenti 500 ng/mL; vale solo per un risultato chiamato D-dimero, e basta un'affermazione qualitativa |
| F | la sezione richiede una diagnosi effettivamente riportata; chiede inoltre la riga `DETAILS_PRESENT: yes/no` |
| X | i sintomi non sono diagnosi, e un fattore di rischio non è una spiegazione alternativa |

### `CROSS_SECTION_RULES`

Tre regole applicate dopo che ogni sezione è stata risposta indipendentemente.
Ognuna scatta in uno di due modi:

- `none_option` — scatta quando la sezione sorgente contiene qualunque valore
  diverso da quell'opzione. Un sintomo in B2 implica che B1.1 sia positiva.
- `trigger_value` — scatta su corrispondenza esatta. A3.1 che riporta nessun
  imaging azzera A3.2; A3.1 che riporta imaging non confermativo azzera A3.2
  anch'essa, perché A3.2 registra solo gli studi che **hanno confermato** la DVT.

In entrambi i casi la sezione bersaglio viene sovrascritta con `forced_value`.
Ogni regola porta anche `audit_key` e `override_message`, per lasciare traccia
leggibile nell'audit log.

---

## 2. `models.py`

Definisce gli schemi Pydantic che vincolano l'output di Agent 2, ed è la fonte
unica delle opzioni e del loro ordine: gli altri moduli lo introspezionano invece
di ripetere le stringhe.

`A1_Autopsy`, `A2_SurgicalProcedure`, `A3_1_ImagingOutcome`,
`B1_1_SymptomsReported`, `C_DDimer`, `F_ReportedBySpecialist` e
`X_AlternativeDiagnosis` seguono lo stesso schema: un unico campo
`answer: Literal[...]`. Ogni campo porta anche una `description`, il titolo della
sezione nel questionario, che Agent 2 riceve solo con
`config.SECTION_DESCRIPTIONS_ENABLED` acceso. `Literal` fa rifiutare a Pydantic qualunque
valore che non sia esattamente una delle stringhe elencate, ed è questo vincolo a
rendere sicuro il mapping numero → testo di `agents._match_option`.

`A3_2_ImagingStudies` e `B1_2_DVTType` sono a scelta multipla: `List[Literal[...]]`
con `default_factory=list`, quindi una lista vuota è una risposta valida.
Nessuna delle due ha un'opzione "nessuna delle precedenti", perché il
questionario cartaceo non la prevede.

`B2_NewSymptoms` ha cinque opzioni, l'ultima delle quali è la catch-all
negativa, più un `@model_validator(mode="after")` chiamato `none_is_exclusive`
che rifiuta quella opzione insieme a un sintomo reale. Girando dopo ogni
costruzione, rivalida anche le istanze che `criteria_rules` e `agents`
ricostruiscono, non solo la prima risposta del modello.

`F_ReportedBySpecialist` merita attenzione per l'inversione: `"Yes"` significa
riportata **senza** dettagli, `"No"` significa che i dettagli c'erano **oppure**
che la diagnosi non è stata riportata affatto.

`DVT_CriteriaForm` è il contenitore: `record_id` obbligatorio più dieci campi
opzionali, uno per sezione. Una sezione che la pipeline non è riuscita a
compilare resta `None` invece di bloccare l'intero form.

`SECTION_MODELS` mappa la chiave testuale della sezione alla classe Pydantic.

**`schema_fields.py`** contiene l'unica funzione che legge gli schemi,
**`field_info(section_model)`**, che restituisce `(field_name, options,
is_multi_select, description)`: il nome dell'unico campo, le opzioni nell'ordine
dello schema, se la sezione è a scelta multipla e il titolo della sezione. La
usano `agents.py`, `verifier.py`, `evaluate_predictions.py`,
`export_redcap_csv.py` ed `export_prompts.py`. Importa solo la libreria
standard, così gli strumenti di valutazione non dipendono dallo stack della
pipeline, e riceve la classe come argomento, così funziona anche con lo schema
della miocardite installato come `models`.

---

## 3. `rag_setup.py`

Prepara il lato retrieval: modello di embedding, le due vector store, i loader e
il tool che l'estrattore agentico chiama.

**`get_embeddings()`** restituisce `HuggingFaceEmbeddings` configurato con i
prefissi di ruolo che `multilingual-e5-small` richiede: `"passage: "` per i
chunk indicizzati (`encode_kwargs`, usato da `embed_documents`) e `"query: "` per
le query (`query_encode_kwargs`, usato da `embed_query`). Senza i prefissi la
scheda del modello riporta un retrieval degradato. Il modello gira su
`config.EMBEDDING_DEVICE`, passato come `model_kwargs`; `_run_config.models`
registra nome e dispositivo (`embeddings`, `embeddings_device`).

**`build_brighton_kb(...)`** costruisce o ricarica l'indice del paper. Il
documento è identico per ogni paziente, quindi un indice già su disco viene
ricaricato invece di essere ricalcolato, a meno di `force_rebuild`.

**`build_ehr_kb(...)`** chunka e indicizza il referto di un singolo paziente. La
cartella di persistenza è suffissata con `patient_id`, così due referti non ne
condividono una, e viene **cancellata prima di ricostruire**: senza,
`Chroma.from_texts` appenderebbe a quanto già persistito e rilanciare la pipeline
sullo stesso identificativo accumulerebbe chunk duplicati, diluendo il retrieval
nel tempo.

**`make_ehr_retriever_tool(...)`** avvolge il retriever come tool
`search_patient_record`. La descrizione nomina esplicitamente tutti i domini
clinici toccati dai dieci criteri, perché la stessa descrizione viene riusata
invariata su ogni sezione e una più stretta rischia che il modello non pensi a
cercare un dominio che non vede nominato.

**`load_brighton_pdf_text(...)`** estrae il testo dal PDF e lo tronca
all'intestazione della bibliografia. La lista di riferimenti è circa l'ultimo 40%
del paper ed è puro rumore: i suoi chunk vengono indicizzati e recuperati come
gli altri e arrivano ad Agent 2 come se fossero terminologia di riferimento.
Troncare prima del chunking li rimuove in blocco, comprese le voci spezzate su
più righe che un filtro riga per riga non intercetterebbe. Un paper senza quella
intestazione restituisce il testo intero, quindi si perde la pulizia ma non il
contenuto.

**`clean_brighton_context(...)`** è la seconda linea di difesa: filtra dai chunk
recuperati le righe bibliografiche superstiti (marcatori di citazione seguiti da
un nome, URL, DOI, citazioni di volume). Se il filtro rimuoverebbe tutto,
restituisce l'originale, così un chunk fatto di soli riferimenti produce
comunque qualcosa e non un contesto vuoto.

**`retrieve_brighton_context(brighton_kb, query)`** produce il contesto che
Agente 2 riceve per una sezione: interroga l'indice del paper con la query di
sezione e restituisce i primi `BRIGHTON_RETRIEVER_K` chunk, passati per
`clean_brighton_context`. Con `config.BRIGHTON_CONTEXT_ENABLED` a `False`
restituisce la stringa vuota.

**`load_ehr_text(...)`** legge il referto da un `.txt` in UTF-8.

Tre espressioni regolari del modulo reggono queste funzioni:
`_REFERENCES_HEADING` trova l'intestazione della bibliografia per
`load_brighton_pdf_text`, `_BIBLIOGRAPHY_LINE` riconosce le righe bibliografiche
superstiti per `clean_brighton_context`, `_NUMBERED_HEADING` riconosce le
intestazioni numerate per `_paper_outline`.

---

## 4. `agents.py`

I due agenti e tutto il parsing delle risposte.

### 4.1 Costruzione dei modelli

**`build_llm(model_name=None, temperature=None, num_predict=None)`** è la
factory unica per ogni modello non tool-calling del progetto. Legge i default da
`config.py` e passa `num_gpu`, `num_ctx` e il timeout; il parametro `reasoning` viene inviato **solo**
quando `config.LLM_REASONING` non è `None`, così un modello privo di modalità di
ragionamento non lo riceve affatto.

### 4.2 Agent 1, l'estrattore

**`EXTRACTOR_SYSTEM_PROMPT`** istruisce il modello a essere un copiatore e non un
commentatore: copiare frammenti letterali, riconoscere che il referto può essere
in italiano senza tradurlo, considerare rilevante solo ciò che riguarda lo stesso
test o evento specifico chiesto dal criterio, e rispondere esattamente
`NO RELEVANT EVIDENCE FOUND.` se nulla è pertinente. Il prompt include un
esempio corretto e uno sbagliato.

**`extract_evidence(llm, ehr_vectorstore, criterion_query)`** — modalità `rag`.
Retrieval a `k` fisso, poi il modello copia i frammenti rilevanti dai chunk
recuperati. Se il retrieval non restituisce nulla, salta la chiamata al modello e
ritorna direttamente la stringa di fallback.

**`extract_evidence_full_text(llm, full_ehr_text, criterion_query)`** — modalità
`full_text`. Stessa struttura senza retrieval: il referto intero va nel prompt.

**`AGENTIC_EXTRACTOR_SYSTEM_PROMPT`** estende il prompt condiviso con due blocchi:

1. **TOOL USE** — dice che il referto non è nella conversazione e che è
   obbligatorio chiamare `search_patient_record` almeno una volta prima di
   rispondere. Senza questa istruzione il modello rispondeva la stringa di
   fallback su tutte e dieci le sezioni senza mai cercare.
2. **TRANSCRIPTION RULE** — governa il turno finale dell'agente, che
   `extract_evidence_agentic` non legge: la funzione restituisce l'output grezzo
   del tool. Il paragrafo viene quindi generato e scartato, ed è annotato come
   tale nel codice.

**`extract_evidence_agentic(...)`** costruisce un `AgentExecutor` con
`return_intermediate_steps=True` e `early_stopping_method="force"`. L'evidenza
restituita è l'unione **deduplicata dei chunk grezzi restituiti da ogni chiamata
al tool**, non il turno finale dell'agente: quel turno tende a parafrasare o
tradurre, e una citazione corrotta fa ragionare il valutatore sul testo
sbagliato. La copertura dipende quindi dalle decisioni di ricerca dell'agente:
una sezione dove sceglie una query povera, o non cerca affatto, produce
`NO_EVIDENCE`.

### 4.3 Agent 2, il valutatore

**`EVALUATOR_SYSTEM_PROMPT`** chiede di determinare la risposta dalla sola
evidenza, di consultare i sinonimi Brighton, e soprattutto di prestare estrema
attenzione alle negazioni in entrambe le direzioni: non trattare come presente
ciò che è esplicitamente negato, e non assumere assente ciò che semplicemente non
è menzionato. Un secondo paragrafo tratta le domande su un metodo specifico:
vieta di inferire che quel metodo sia stato eseguito solo perché la DVT è stata
confermata da un metodo diverso.

`evaluate_section` legge campo, opzioni, tipo e titolo della sezione con
`schema_fields.field_info`: è ciò che la rende generica per tutte e dieci le
sezioni senza un ramo per ciascuna.

**`_build_reasoning_prompt(...)`** compone il prompt. Le opzioni sono numerate e
il modello risponde **due volte**: il testo dell'opzione su `FINAL_OPTION` e il
suo numero su `FINAL_ANSWER`. Due risposte invece di una rendono visibile un
disaccordo, dato che il modello a volte nomina un'opzione e scrive il numero di
un'altra. Il prompt chiede inoltre esplicitamente che ogni opzione elencata sia
tracciabile a una frase del ragionamento, e non inclusa per default o per
margine di sicurezza. Con `config.SECTION_DESCRIPTIONS_ENABLED` acceso, il titolo
della sezione viene inserito sopra le opzioni.

Nelle sezioni multi-scelta le istruzioni mostrano con un esempio che i numeri
seguono l'ordine dei testi. L'esempio lo costruisce `_multi_select_example()` da
**`MULTI_SELECT_EXAMPLE`**: con il valore della TVP, `"Leg swelling or pitting
oedema"` (l'opzione 2 di B2), l'esempio nomina quell'opzione; con `None` usa una
frase che non nomina alcuna opzione («if the second option you name is listed as
4. above, the second number must be 4»). Il braccio della miocardite imposta
`None`, così nessuna opzione del suo questionario viene citata nelle istruzioni.

Il blocco della linea guida è annunciato dall'intestazione
`Reference synonyms/terminology (Brighton)`.

Per le sezioni elencate in **`EMPTY_ANSWER_SECTIONS`**, per nome della classe
dello schema, cioè A3.2 e B1.2 della TVP, il prompt aggiunge come dire che nulla
si applica: `FINAL_OPTION: none` e `FINAL_ANSWER: none`. Il questionario stampato
non ha per loro un'opzione "nessuna delle precedenti", e senza questa istruzione,
non avendo modo di esprimere una risposta vuota, il modello selezionava tutte le
opzioni. Solo queste sezioni accettano la risposta `none`; in ogni altra sezione
`none` non corrisponde ad alcuna opzione e la sezione viene ritentata. Il braccio
della miocardite imposta un insieme vuoto: in E ed F ogni referto ha un'opzione,
quella dell'esame non eseguito o quella dell'esame senza anomalie.

**`_has_none_option(options)`** dice se una sezione ha un'opzione esplicita
«none of the above», riconosciuta dal prefisso `_NONE_OPTION_PREFIX`.
**`_accepts_empty(section_model, options)`** è vera per le sezioni di
`EMPTY_ANSWER_SECTIONS` senza un'opzione del genere: è ciò che decide sia
l'istruzione `none` nel prompt sia l'accettazione della risposta vuota.
**`_is_no_selection(raw_value)`**
riconosce una riga di risposta che significa «nessuna opzione»: le parole
accettate sono in `_NO_SELECTION_ANSWERS` (`none`, `nothing`, `nessuna`, `n/a` e
simili).

**`_extract_final_answer_line`** e **`_extract_labeled_line`** leggono le due
righe. Entrambe considerano solo l'etichetta che apre una riga, dopo eventuali
spazi o marcatori markdown (`>`, `*`, `#`, `-`): un'etichetta dentro una frase è il
modello che cita il prompt, non che risponde. Fra quelle prendono l'**ultima**
occorrenza, non la prima, perché il modello a volte ripete l'istruzione prima di
rispondere. La differenza sta nel
fallimento: `FINAL_ANSWER` mancante solleva un'eccezione, `FINAL_OPTION` mancante
restituisce `None` e il chiamante rinuncia al solo controllo incrociato.

**`_match_option(raw_value, valid_options, cutoff=0.75)`** mappa un frammento di
risposta su un'opzione valida, in tre passi. Primo, l'indice numerico, che è ciò
che il prompt chiede. Secondo, il testo dell'opzione, confrontato dopo aver tolto
la punteggiatura finale da entrambi i lati, restituendo comunque la forma esatta
dello schema. Terzo, un fuzzy match con soglia, **stampato esplicitamente** come
avviso: un fuzzy match silenzioso rischia di atterrare sull'opzione opposta per
negazione, dato che "confirmed DVT" e "didn't confirm DVT" sono testualmente
vicine e semanticamente opposte.

**`evaluate_section(...)`** orchestra Agent 2 per una sezione, identica in tutte
le modalità. Ritorna una tripla `(istanza, reasoning_text, conflict)`.

`conflict` è `None` quando le due righe concordano, altrimenti un dizionario con
un campo `kind` che nomina il tipo di disaccordo:

- `text_vs_number` — le due righe indicano opzioni diverse. **Vince il numero**;
  il testo è un controllo incrociato, non una fonte che possa da sola
  invalidare una risposta.
- `none_vs_text` — `FINAL_ANSWER` dice che nulla si applica mentre
  `FINAL_OPTION` elenca opzioni. Vince la risposta vuota.
- `none_with_options` — la stessa riga contiene sia "none" sia opzioni nominate.
  Le opzioni nominate vengono tenute.

Il modello a volte seleziona "None of the above" insieme a reperti reali, che il
validatore di B2 rifiuta: la funzione ripara scartando l'opzione catch-all e
tenendo i reperti, perché lasciar passare l'errore costerebbe l'intera sezione
una volta esauriti i tentativi.

In caso di fallimento del parsing, il prompt viene **esteso** con il testo
dell'errore e la sezione ritentata, fino a `max_retries + 1` tentativi. Il
messaggio d'errore non contiene la risposta fallita: quello di
`_extract_final_answer_line` dice solo che manca la riga `FINAL_ANSWER`, quindi
ogni tentativo aggiunge al prompt circa 320 caratteri e non l'intera risposta
precedente. Se
falliscono tutti, la funzione solleva un `RuntimeError` a cui **allega l'ultima
risposta del modello** come attributo `last_response`: una sezione fallita è
l'unico caso in cui il chiamante non ha altra copia di ciò che il modello ha
scritto.

Il parametro facoltativo `token_counts` è una lista a cui la funzione aggiunge,
per ogni tentativo, i conteggi che Ollama restituisce con la risposta:
`{"prompt": prompt_eval_count, "output": eval_count}`, letti da
`response_metadata`. La lista si riempie anche quando tutti i tentativi
falliscono. `pipeline.py` e `agentic_graph.py` la passano e la scrivono
nell'audit log della sezione sotto `agent2_tokens`, anche quando la sezione
fallisce: la somma di prompt e risposta confrontata con `LLM_NUM_CTX` dice se
una chiamata è rimasta nella finestra di contesto.

---

## 5. `criteria_rules.py`

Le regole cross-section, applicate sopra le risposte di Agent 2 dopo che tutte
le sezioni sono state compilate. Nessuna funzione qui chiama un modello: una
regola sostituisce la risposta di una sezione con un valore derivato dalla
risposta di un'altra. Ogni override lascia una nota `[SYSTEM OVERRIDE]` nel testo
di ragionamento, così una risposta forzata non è mai indistinguibile da una
prodotta dal modello, e il suo effetto si ricostruisce dagli audit log senza
rieseguire nulla.

**`apply_cross_section_rules(form_data, audit_log)`** applica
`CROSS_SECTION_RULES` una volta sola, dopo che tutte le sezioni sono state
compilate, indipendentemente dalla modalità che le ha prodotte. Richiede la
presenza della sola sezione **sorgente**: una risposta forzata derivata da una
sezione mancante sarebbe infondata, mentre il bersaglio può essere `None`, dato
che il valore forzato viene dalla regola e la classe si legge da
`SECTION_MODELS`. Questo permette a una regola di riempire una sezione che una
valutazione fallita aveva lasciato vuota. Sovrascrive solo quando il valore
corrente differisce da quello forzato, per non riempire il log di voci in cui
Agent 2 era già d'accordo. Quando sovrascrive, aggiunge alla voce della sezione
nell'audit log la nota `[SYSTEM OVERRIDE]` e la chiave `overridden_by`, con il
nome della sezione sorgente (`A3_1` o `B2`); `verifier.verify_record` la usa per
assegnare alla sezione il verdetto della sorgente.

---

## 6. `agentic_graph.py`

Implementa `EXTRACTOR_MODE == "agentic_graph"` come macchina a stati LangGraph
esplicita, dove ogni passo è una funzione con il proprio stato in ingresso e in
uscita, invece che un ciclo Python.

Forma del grafo:

```
select_next -> {search_record, finalize} -> answer_criterion -> select_next
```

**`build_agentic_llm()`** costruisce il modello tool-calling, con gli stessi
parametri di generazione di `agents.build_llm`, `num_ctx` compreso. È separata da
`agents.build_llm` perché questo è l'unico ruolo che lega un tool e quindi
l'unico che ha bisogno di un modello che supporti l'API di tool calling di
Ollama.

**`GraphState`** è il `TypedDict` che circola tra i nodi: `record_id`,
`remaining_sections`, `current_section`, `form_data`, `audit_log`, `done`.

**`_select_next(state)`** preleva la sezione successiva dalla coda, restituendo
una nuova lista invece di mutarla in place, perché lo stato del grafo è trattato
come immutabile da un passo all'altro. Quando la coda è vuota imposta
`done=True`.

**`_route_after_select(state)`** è la funzione di routing dell'arco
condizionale: `finalize` se `done`, altrimenti `search_record`.

**`_make_search_node(...)`** e **`_make_answer_node(...)`** sono factory e non
nodi diretti, perché un nodo LangGraph riceve solo lo stato mentre questi due
passi hanno bisogno anche del modello, del tool e delle query. Il nodo di
ricerca cronometra anche i fallimenti, così una sezione lenta perché ha
continuato a ritentare resta visibile nel log. Il nodo di risposta recupera il
contesto della linea guida e chiama Agent 2.

**`build_graph(...)`** collega i quattro nodi e compila la macchina a stati. Il
modello di ricerca e quello di risposta sono due parametri distinti, perché
solo il primo deve legare un tool.

**`_finalize(state)`** è un passthrough deliberato: le regole cross-section
vengono applicate una sola volta da `pipeline.run_pipeline` dopo che il grafo ha
restituito, così ogni modalità passa dallo stesso codice invece che da una copia.

**`run_agentic_graph_pipeline(...)`** è il punto d'ingresso. Il limite di
ricorsione è calcolato da `len(SECTION_ORDER) * 3 + 10`, perché ogni sezione
attraversa tre nodi. Restituisce `(form_data, audit_log)` nella stessa forma del
ciclo semplice, senza aver costruito il form né applicato le regole
cross-section.

---

## 7. `verifier.py`

L'Agent 3. Per ogni sezione chiede al modello del valutatore se la risposta finale
del modulo è corretta data l'evidenza, e registra nell'audit log della sezione il
verdetto (YES o NO) e la probabilità assegnata a YES. Gira dopo le regole
cross-section, quindi giudica la risposta che finisce nel CSV, ed è acceso da
`config.VERIFIER_ENABLED`. Non cambia mai una risposta.

**Le varianti.** Per ogni voce di `config.VERIFIER_INPUTS` la sezione riceve una
richiesta separata; le varianti ammesse sono in **`VARIANTS`**:

- `answer`: evidenza, opzioni numerate e risposta finale;
- `reasoning`: le stesse cose più il ragionamento dell'Agent 2 come registrato
  nell'audit log (campo `reasoning`);
- `answer_hint` e `reasoning_hint`: come le due precedenti, più l'hint della
  sezione, quello che `config.section_hint` dà all'Agent 2.

Nessuna variante riceve il contesto della linea guida. L'hint va in testa al
messaggio sotto l'intestazione `Section guidance:`, seguito da
**`HINT_FORMAT_NOTE`**, che chiede di ignorarne le istruzioni sul formato della
risposta: alcuni hint ne contengono, come la riga `DETAILS_PRESENT` di F nella
TVP, che contraddirebbe «Reply YES or NO». Una sezione senza hint invia la
richiesta senza quel blocco. Le varianti girano nella stessa run, quindi
giudicano le stesse identiche risposte. Una variante non elencata in `VARIANTS`
ferma `verify_record` con un errore.

**Il punteggio.** La richiesta chiede YES o NO senza spiegazioni, a
`config.VERIFIER_TEMPERATURE`. Con `logprobs` attivo Ollama restituisce, per il
token generato, la probabilità delle `TOP_LOGPROBS = 10` alternative più probabili.
Il verdetto è la parola generata; il punteggio è p(YES) / (p(YES) + p(NO)) in
quella posizione, il P(True) di Kadavath et al. (2022). Una sola richiesta dà
quindi un verdetto binario e un punteggio continuo.

**Costanti.** `OLLAMA_URL` è l'indirizzo del server, letto dalla variabile
d'ambiente `OLLAMA_HOST` come fa Ollama stesso, con `http://localhost:11434` come
default. `NUM_PREDICT = 4` è il tetto di token: YES o NO più un margine per uno
spazio o un segno di punteggiatura. `SYSTEM_PROMPT` presenta il compito: valutare
la risposta data da un altro revisore, rispondendo solo YES o NO.

**`_chat(user_prompt)`** invia una richiesta a `/api/chat` di Ollama, non
attraverso `ChatOllama`, perché servono i campi `logprobs`. Il modello è quello
di **`verifier_model()`**: `VERIFIER_MODEL_NAME`, oppure `EVALUATOR_LLM_MODEL_NAME`
quando è `None`, nel qual caso non se ne carica un secondo sulla GPU.
**`verifier_settings()`** restituisce le impostazioni che `_run_config.verifier`
registra: `enabled`, `model`, `inputs`, `hints` (il digest `all` di
`config.hint_fingerprint()` quando c'è una variante `_hint`, altrimenti `None`),
`temperature` e `num_ctx`. Il parametro
`think` (`LLM_REASONING`) è inviato solo quando il giudice è il modello
dell'Agent 2, per cui è impostato. Temperatura e finestra di contesto sono quelle
del verificatore (`VERIFIER_TEMPERATURE`, `VERIFIER_NUM_CTX`).

**`_format_answer(answer, options, multi)`** scrive la risposta come opzioni
numerate: per una sezione multipla le opzioni selezionate, oppure «No option
selected.».

**`build_prompt(evidence, options, multi, answer, reasoning=None, hint=None)`**
compone il messaggio: l'hint con `HINT_FORMAT_NOTE` solo per le varianti `_hint`,
evidenza, opzioni, risposta, il ragionamento solo per le varianti `reasoning` e
`reasoning_hint`, per le sezioni multiple la precisazione che la risposta è corretta
solo se seleziona tutte e sole le opzioni sostenute dall'evidenza, e la domanda
finale.

**`read_verdict(logprobs)`** cerca il primo token YES o NO e restituisce verdetto,
P(YES) e metodo `verdict`; `no_verdict` quando nessuna delle due parole è stata
generata.

**`verify_section(section_key, answer, evidence, reasoning, variant)`** esegue una
richiesta e restituisce `agree`, `p_agree` (quattro decimali), `method`,
`prompt_tokens` (il `prompt_eval_count` di Ollama), `overflow` e `seconds`.
`overflow` è vero quando prompt e tetto di token raggiungono `VERIFIER_NUM_CTX`:
oltre quel limite Ollama scarta l'inizio del prompt, cioè l'evidenza.

**`verify_record(form_data, audit_log)`** è il punto d'ingresso che
`pipeline.run_pipeline` chiama. Aggiunge a ogni voce dell'audit log la chiave
`verifier`, un dizionario variante → risultato, e stampa ogni verdetto con
**`_report`**. Una sezione con la chiave `overridden_by`, la cui risposta è stata
scritta da una regola cross-section, non riceve richieste: il ragionamento
dell'Agent 2 sostiene la risposta che la regola ha sostituito, quindi la sezione
prende verdetto e P(YES) della sezione sorgente, con metodo `rule:<sorgente>`. Lo
fa in un secondo passaggio, perché una sorgente può venire dopo la sezione che la
regola cambia (B2 viene dopo B1_1). Una sezione senza risposta riceve il metodo
`no_answer`, una senza ragionamento `no_reasoning` per la variante `reasoning`,
una richiesta fallita `error` con il messaggio in `error`. Un fallimento
dell'Agent 3 non interrompe la run.

---

## 8. `pipeline.py`

Orchestra un referto e dispatcha sulle quattro modalità: `agentic_graph` passa
dal grafo di `agentic_graph.py`, le altre tre dal ciclo per sezione di questo
modulo.

**`SECTION_QUERIES`** associa a ogni sezione la query che dice ad Agent 1 cosa
cercare, e che serve anche a recuperare il contesto dalla linea guida. Ogni
query nomina il contenuto delle opzioni della sezione con parole che compaiono
nei referti, con i termini della definizione di caso Brighton solo dove nominano
la stessa cosa (per esempio *DVT of lower or upper limbs* per B1.2, *alternate
etiology* per X). Sono in
inglese; il modello di embedding multilingue le confronta con referti in
italiano.

**`_ollama_version()`** legge la versione del binario Ollama. È registrata perché
due versioni di Ollama portano due versioni di llama.cpp, e con esse kernel di
quantizzazione diversi: a temperatura 0 basta a far cambiare un token che il
modello aveva quasi in parità, quindi run prodotte sotto versioni diverse non
sono direttamente confrontabili. Non solleva mai: una versione mancante costa la
provenienza, non la run.

**`_hint_fingerprint()`** restituisce `config.hint_fingerprint()`: per ogni
sezione che riceve un hint non vuoto, la coppia lunghezza e prefisso sha256 del
testo, più un digest `all` dell'intero insieme. Gli interruttori dicono quali hint sono stati inviati, non cosa
dicevano, quindi senza questo due run i cui hint sono stati riscritti in mezzo
porterebbero la stessa firma.

**`_query_fingerprint()`** fa lo stesso per `SECTION_QUERIES`, che è insieme il
brief da cui l'Agente 1 parte e la chiave con cui si recupera il contesto della
linea guida. Nessun altro campo dello snapshot ne registra il testo, quindi
senza questo due run le cui query sono state riscritte in mezzo porterebbero la
stessa firma.

**`_run_config_snapshot(...)`** cattura tutto ciò che determina cosa una run
produce: modalità, contesto della linea guida, intestazioni di sezione,
hint e query con i rispettivi fingerprint, le sezioni che ricevono la clausola
sulla risposta vuota (`empty_answer_sections`, da `agents.EMPTY_ANSWER_SECTIONS`,
che l'impronta degli hint non copre), le impostazioni dell'Agent 3
(`verifier`, da `verifier.verifier_settings()`: `enabled`, `model`, `inputs`,
`hints`, `temperature`, `num_ctx`), modelli per ruolo (con
l'estrattore **effettivo**: `AGENTIC_LLM_MODEL_NAME` in modalità agentica,
nessuno in `raw_record`), ambiente, parametri di generazione compreso
`num_ctx`, parametri di retrieval. Viene scritto nell'audit log sotto la chiave
`RUN_CONFIG_KEY = "_run_config"`, scelta per non poter collidere con un nome di
sezione, così un file di risultati è auto-descrittivo mesi dopo.

**`run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)`** costruisce una
sola volta embedding e valutatore, carica i due testi, costruisce la KB Brighton
sempre e quella del referto solo dove serve. Il modello di Agent 1 viene costruito **dentro** il ramo che
lo usa, così una modalità che non lo interroga, come `raw_record`, non lo carica
in VRAM.

Nel ciclo per-sezione, ogni sezione passa da Agent 1 (in `raw_record` l'evidenza
è il referto intero), dal recupero del contesto Brighton ripulito e da Agent 2.
Il fallimento di una sezione non compromette le altre: il campo resta
`None`, l'errore va in `section_log["error"]` e l'ultima risposta del modello, se
disponibile, in `section_log["reasoning"]`.

Alla fine applica le regole cross-section una volta sola; se
`config.VERIFIER_ENABLED` è acceso chiama `verifier.verify_record` sulle
risposte finali; poi aggiunge lo snapshot di configurazione e costruisce
`DVT_CriteriaForm`.

---

## 9. `aggregation.py` e `main.py`

**`aggregation.form_to_json_summary(form)`** serializza il form con
`exclude_none=True`: le sezioni lasciate `None` da una valutazione fallita
vengono omesse invece che scritte come `null`, così una risposta mancante è
assente e non somiglia a una risposta di "nessuno". Riceve anche l'audit log: se
l'Agent 3 ha girato, copia verdetto e P(YES) di ogni sezione e variante sotto la
chiave `VERIFIER_KEY = "_verifier"`, tramite **`verifier_verdicts(audit_log)`**,
che usa anche `replay_verifier.py`. Così i valori restano nel file dei risultati
anche quando l'audit log non viene salvato.
Il trattino basso la tiene separata dalle chiavi delle sezioni, le sole che
`evaluate_predictions`, `compare_runs` ed `export_redcap_csv` leggono. Con
l'Agent 3 spento il riepilogo è identico a prima.

**`main.py`** esegue la pipeline su un singolo referto, con i percorsi ancorati
alla posizione del file e non alla directory di lancio. Scrive due file che
condividono lo stesso timestamp, il risultato e l'audit log, così una coppia è
sempre associabile e rilanciare non sovrascrive mai una run precedente.

---

## 10. `run_synthetic_records.py`

Esegue la pipeline su ogni referto del corpus, con la stessa convenzione di nomi
di `main.py` così la valutazione li trova con i percorsi di default. Legge i
referti da `RECORDS_DIR` (`data/synthetic_records/`) e il paper da
`BRIGHTON_PDF_PATH`. **`run_one(record_id, record_path, output_dir)`** esegue un
referto e scrive i suoi due file; **`main()`** li scorre in ordine di nome, e il
fallimento di un referto non ferma il batch.

`--output-dir` sceglie la cartella di destinazione, `DEFAULT_OUTPUT_DIR`
(`./output`) se omesso. Ogni braccio sperimentale va in una cartella sua, perché
`evaluate_predictions` ed `export_redcap_csv` tengono il file più recente per
referto e due bracci nella stessa cartella si nasconderebbero a vicenda.

`--only` restringe il batch ai referti il cui identificativo contiene una delle
stringhe date, per ricontrollarne pochi dopo una modifica al prompt senza pagare
l'intero set. **Un run parziale non è un run**: la valutazione tiene il file più
recente per referto, quindi il punteggio mescolerebbe i risultati con quelli
prodotti prima dagli altri referti. Un identificativo scritto male fa uscire lo
script invece di eseguire zero referti, che altrimenti sembrerebbe una run
riuscita con zero record.

Prima di ogni referto un controllo non bloccante segnala un `.txt` senza ground
truth corrispondente, che altrimenti resterebbe silenziosamente non valutato. Il
tempo rimanente è stimato dalla media corrente e non dall'ultimo referto, perché
la durata varia con quante chiamate al tool l'estrattore agentico decide di fare.

---

## 11. `generate_synthetic_records.py`

Definisce il corpus sintetico e ne ricava la ground truth. `SCENARIOS` contiene
i 40 scenari, `OUTPUT_DIR` è `data/synthetic_records/`, `STYLE_VARIANTS` contiene
solo l'identificativo `v2`, il suffisso di ogni identificativo di referto. Lo
script non chiama alcun modello.

Le funzioni:

- **`build_ground_truth(record_id, scenario)`** costruisce le risposte di
  riferimento salvate accanto al referto;
- **`_expected_markers(...)`**, **`check_record(...)`** e
  **`check_existing_records()`** sono il controllo di fedeltà descritto sotto, e
  `_FACT_MARKERS` ne contiene i marcatori;
- **`main()`** riscrive la ground truth di tutti gli scenari, o solo di quelli
  nominati con `--only`, e con `--check` controlla il corpus su disco senza
  scrivere nulla.

**Ground truth per costruzione.** Ogni scenario porta sia i fatti clinici sia le
risposte corrette per tutte e dieci le sezioni, scritte a mano con le stringhe
esatte di `models.py`. Nessun modello indovina mai il riferimento, ed è questo a
renderlo utilizzabile come tale. I JSON di ground truth vengono riscritti a ogni
esecuzione senza `--check`.

**I referti.** Sono redatti fuori dallo script, a partire dai fatti di ogni
scenario, e controllati contro di essi prima di entrare nel corpus. Le
`writer_notes` di uno scenario indicano vincoli che il referto rispetta (cosa
non deve dire) e non sono fatti da ritrovare nel testo.

**Il controllo di fedeltà.** `_expected_markers(scenario)` ricava dai fatti
dello scenario quali marcatori il referto deve contenere; `check_record(text,
scenario)` verifica il testo contro quella lista e restituisce l'elenco dei
problemi; `check_existing_records()` esegue il controllo su tutto il corpus su
disco (`--check`).

**Il limite noto.** Il controllo è deterministico e intercetta la troncatura e i
fatti mancanti, non la violazione semantica: un referto può contenere il valore
giusto e descriverlo male. Intercettarlo richiederebbe un secondo modello come
giudice, deliberatamente non costruito per mantenere il controllo deterministico.

**Un secondo limite, sui dati.** Un referto va da 317 a 1185 caratteri, cioè uno
o due chunk da 800, mentre il retriever ne chiede 5: il retrieval restituisce
ogni volta il referto intero, e cambia solo l'ordine dei chunk. Su questo dataset
le modalità con retrieval non selezionano: l'estrattore riceve lo stesso testo
che in `full_text`, diviso in chunk.

---

## 12. `evaluate_predictions.py`

Valuta una run contro la ground truth. Non esegue la pipeline. Legge predizioni e
riferimenti da directory passate a riga di comando, così lo stesso script serve
il corpus sintetico e qualunque altro insieme annotato. Le predizioni sono
associate al riferimento tramite il campo `record_id` interno al JSON e non dal
nome del file; quando un referto ha più file, vince il più recente.

Metriche, per sezione e complessive:

- **Accuratezza exact-match**: la risposta è giusta solo se coincide con il
  riferimento.
- **Baseline di maggioranza** e guadagno su di essa: è il pavimento che una
  sezione deve superare per portare informazione.
- **Kappa di Cohen**, con la risposta intera trattata come una sola etichetta,
  così una sezione multi-scelta è valutata sull'insieme esatto che ha prodotto.
- **TP/TN/FP/FN, precisione, richiamo e F1 per opzione** e non per sezione:
  senza questo, una sezione multi-scelta risposta a metà conterebbe come
  semplicemente sbagliata. Precisione e richiamo ignorano deliberatamente i veri
  negativi, che sono la maggioranza di ogni conteggio dato che la maggior parte
  delle opzioni non si applica alla maggior parte dei referti.

La riga complessiva è riportata due volte: **micro** mette in comune ogni opzione
di ogni sezione, quindi una sezione con più opzioni pesa di più; **macro** media
le cifre per sezione, quindi ogni sezione conta una volta.

Una sezione lasciata `None`, o un referto senza output, è riportata come mancante
ed esclusa dalle metriche invece di essere contata come errore. È una scelta da
tenere presente leggendo i numeri: un modello che fallisce molte sezioni ottiene
un punteggio ottimisticamente alto.

Dipende da scikit-learn e da `models.py`, ma non da langchain o Ollama: gira
senza lo stack della pipeline installato: le opzioni delle sezioni vengono da
`schema_fields.field_info`, che importa solo la libreria standard.

Le funzioni:

- **`load_ground_truth(directory)`** e **`load_predictions(predictions_dir)`**
  leggono riferimenti e predizioni; le predizioni saltano gli `*_audit_log.json` e
  sono lette in ordine di nome, quindi cronologico, così vince la più recente.
  I default sono `DEFAULT_GROUND_TRUTH_DIR` (`data/synthetic_records/`) e
  `DEFAULT_PREDICTIONS_DIR` (`output/`);
- **`_selected_set(value)`** riduce una risposta a un insieme di opzioni, così
  scelta singola e multipla si confrontano allo stesso modo;
- **`_binary_rows(pairs, options)`** trasforma le coppie in due matrici 0/1, una
  colonna per opzione, la forma che scikit-learn si aspetta;
- **`_majority_baseline(pairs)`** e **`_score_section(pairs, options)`**
  calcolano le metriche di una sezione;
- **`evaluate(ground_truth, predictions)`** produce il report completo;
- **`print_report(report)`** lo stampa, con `_pct`, `_num` e `_gain` per la
  formattazione;
- **`main()`** salva il report in `reports/evaluation_<timestamp>.json`
  (`REPORTS_DIR`, creata se manca).

`print_report` calcola la colonna del guadagno complessivo come differenza fra
accuratezza macro e baseline macro senza controllare che esistano: se nessuna
sezione è stata confrontata, entrambe sono `None` e la stampa si interrompe con
un `TypeError`.

---

## 13. `compare_runs.py`

Confronta **due run tra loro** e riporta quante risposte sono cambiate. È il
complemento di `evaluate_predictions.py`, che confronta una run contro la ground
truth.

La temperatura è 0, quindi la pipeline è nominalmente deterministica, ma Ollama
non garantisce generazioni identiche bit per bit tra chiamate. Questo script
quantifica il pavimento di rumore sotto una metrica prodotta da una sola run: se
due run identiche già divergono sull'N% delle sezioni, qualunque differenza di
accuratezza inferiore a N% tra due configurazioni non è un risultato.

Con una directory confronta i due file più recenti per referto; con due, il più
recente di ciascuna. Riporta anche l'accuratezza di ciascuna run, così una
differenza di stabilità si legge accanto a una differenza di accuratezza. Una
sezione mancante in **entrambe** le run viene esclusa invece che contata come
invariata, per non gonfiare la stabilità con sezioni che non hanno mai prodotto
nulla.

Le funzioni:

- **`load_run_files(directory)`** raccoglie i file di output per referto,
  ordinati dal più vecchio; salta gli audit log e i file il cui nome non porta un
  timestamp leggibile secondo `_OUTPUT_NAME_RE`;
- **`pair_up(dir_a, dir_b)`** forma le coppie da confrontare, con la regola a una
  o due cartelle descritta sopra;
- **`load_ground_truth()`** legge i riferimenti da `GROUND_TRUTH_DIR`, usati solo
  per le due colonne di accuratezza;
- **`_selected_set(value)`** riduce una risposta a un insieme di opzioni;
- **`compare(pairs, ground_truth)`** confronta le due run sezione per sezione;
- **`print_report(report)`** stampa il risultato, chiudendo con la cifra del
  pavimento di rumore;
- **`main()`** accetta una o due cartelle (`./output` se omesse) e salva il report
  in `reports/run_stability_<timestamp>.json` (`REPORTS_DIR`, la stessa cartella
  di `evaluate_predictions.py`).

---

## 14. `export_redcap_csv.py`

Converte l'output della pipeline nel CSV che REDCap importa, così il Level of
Certainty può essere calcolato dal progetto REDCap stesso.

Scrive il formato che il Data Import Tool si aspetta: nomi delle variabili come
intestazioni, codici numerici come valori, e le caselle come colonne
`<campo>___<codice>` che contengono 0 o 1.

`SECTION_FIELDS` mappa ogni sezione su un campo REDCap e su un tipo:

- `radio` — una colonna con la posizione 1-based dell'opzione.
- `checkbox` — una colonna per opzione. **Ogni casella viene scritta
  esplicitamente, comprese quelle non spuntate**: REDCap legge una cella vuota
  come "lascia invariato", quindi una risposta di "nessuna di queste" va inviata
  come una riga di zeri effettivi.
- `yesno` — una colonna codificata 1/0 anziché 1/2. La sezione F è l'unica, ed è
  l'unico punto in cui il codice non segue la posizione nello schema.

I codici delle opzioni sono letti dallo schema Pydantic e non ripetuti qui, così
una modifica a `models.py` non può produrre silenziosamente un CSV i cui codici
puntano alle opzioni sbagliate.

Una sezione che la pipeline non ha risposto produce celle **vuote**, non un
default: scrivere zeri ovunque affermerebbe che ogni opzione è stata valutata e
scartata, che è diverso da "non lo sappiamo". Lo script avvisa esplicitamente
quali referti hanno almeno una sezione non risposta.

Come `evaluate_predictions.py`, tiene il file più recente per referto: una run
sperimentale lasciata in `output/` diventa il CSV che va a REDCap. Va controllato
`_run_config.models.evaluator` sul file più recente prima di esportare.

È volutamente autonomo: importa solo `models.py`, quindi gira senza langchain né
Ollama.

Le funzioni:

- **`_section_options(section_key)`** restituisce le opzioni nell'ordine dello
  schema, lette con `schema_fields.field_info`, e
  **`_option_code(section_key, answer)`** restituisce la posizione 1-based di
  un'opzione come codice;
- **`build_column_order(skip_empty_fields)`** produce le colonne nell'ordine
  dell'export grezzo di REDCap. I campi del modulo che la pipeline non compila,
  testo libero e date, stanno in `UNFILLED_FIELDS_BEFORE` e
  `UNFILLED_FIELDS_AFTER`, posizionati rispetto ai campi dei criteri;
  `--skip-empty-fields` li omette;
- **`row_from_result(result, skip_empty_fields)`** converte un file di risultato in
  una riga, e imposta `criteria_form_complete` a `FORM_COMPLETE_VALUE = "2"`.
  Per le sezioni `yesno`, cioè F, usa `_YESNO_CODES` (`Yes` → 1, `No` → 0) al
  posto di `_option_code`;
- **`load_latest_results(source)`** tiene il file più recente per referto,
  riconoscendolo dal nome con `_OUTPUT_NAME_RE`; accetta una cartella o un file;
- **`write_csv(results, destination, skip_empty_fields)`** scrive il file;
- **`main()`** legge `input_dir` (default `output`) e scrive `destination`
  (default `redcap_import.csv`).

---

## 15. `export_prompts.py`

Scrive su file ogni prompt, query e hint, così la tesi li include con
`\lstinputlisting` invece che con una copia a mano. È in `.gitignore`, quindi non
è versionato. I testi vengono letti dai moduli che li contengono: il testo
incluso nella tesi e quello che ricevono i modelli sono lo stesso oggetto.

- **`collect()`** raccoglie ogni testo da esportare, per nome di file: i prompt
  dei due agenti, lo scheletro del messaggio di Agent 2, i due blocchi di
  istruzioni finali, la descrizione del tool di ricerca e un file per hint;
- **`_message_skeleton()`** e **`_answer_instructions(multi_select)`** passano da
  `agents._build_reasoning_prompt` con segnaposto al posto dei contenuti, così
  etichette, righe vuote e ordine sono quelli che il codice produce;
- **`_example_section(multi_select)`** sceglie la sezione da cui costruire le
  istruzioni finali: la prima a scelta singola di `config.SECTION_ORDER`, oppure
  la prima a scelta multipla che può restare vuota, l'unico tipo che porta la
  clausola sulla risposta vuota (A3_2 nella TVP), o la prima a scelta multipla
  quando nessuna può restarlo (la miocardite, dove il blocco esce senza
  clausola).
  Quando il questionario non ha sezioni di un tipo, il file corrispondente non
  viene scritto: la miocardite non ha `answer_single.txt`;
- **`_retriever_description()`** legge la descrizione dal tool costruito, non dal
  sorgente;
- **`_section_label(section_key)`** scrive la sezione come la nomina il
  questionario (A3_1 diventa A3.1);
- **`_digest(text)`** calcola lunghezza e primi 12 caratteri esadecimali dello
  sha256, la forma usata negli audit log;
- **`main()`** scrive i file in `DEFAULT_OUTPUT_DIR` (`prompts_export/`) o in
  `--output-dir`, più `queries.txt`, `queries.tex` (l'intera tabella),
  `hints.tex` e un `manifest.txt` con due digest per file, quello del testo nel
  codice e quello del file su disco. `--tex-prefix`, `--tex-query-width` e
  `--tex-style` regolano i `.tex` generati.

Tutto viene letto quando `main` gira, non all'importazione (`pipeline.SECTION_QUERIES`,
`pipeline._hint_fingerprint`, `pipeline._query_fingerprint`), così
`myo/export_prompts_myo.py` esporta i testi della miocardite installando lo schema
e chiamando `run_myo.apply_domain` prima di `main`.

---

## 16. Il pacchetto `myo/`

Il braccio di trasferimento: esegue la stessa pipeline sulle sezioni A1, B, C, E
ed F del questionario Brighton della miocardite (modulo
CriteriaForm_SEVALIDMYOCARDITIS), su 17 casi dummy con ground truth scritta da due
autori esterni al progetto. Nessun modulo della pipeline viene modificato:
`myo/` sostituisce in memoria schema, sezioni, prompt e configurazione prima di
chiamarla.

**`models_myo.py`** contiene lo schema. `A1_CardiacSymptoms` (7 opzioni),
`B_PhysicalExam` (6), `C_InflammatoryMarkers` (7), `E_Electrocardiogram` (14) e
`F_Echocardiogram` (10) sono tutte a scelta multipla, `List[Literal[...]]`, con le
stringhe copiate dal modulo, virgole e asterischi compresi; le note del modulo che
rimandano ai range di normalità e i campi «Highest measured value» di C non fanno
parte delle opzioni. A1, B e C hanno un'opzione «none of the above», E ed F no.
`MYO_CriteriaForm` è il contenitore, esportato anche col nome `DVT_CriteriaForm`
perché `pipeline.py` importa quel nome, e `SECTION_MODELS` ha le chiavi `A1`, `B`,
`C`, `E` ed `F`.

**`project_path.py`** aggiunge la cartella del progetto a `sys.path`, così gli
script di `myo/`, lanciati come `python myo/<script>.py`, trovano i moduli
condivisi. **`myo_schema.py`** lo importa e registra `models_myo` come modulo
`models`: `pipeline`, `agentic_graph`, `criteria_rules`, `verifier` ed
`evaluate_predictions` leggono lo schema al momento dell'importazione, quindi la
sostituzione deve avvenire prima. Entrambi si importano per il loro effetto, come
primo import di progetto, marcati `# isort: skip` perché un riordino automatico
degli import non li sposti dopo gli altri.

**`prompts_myo.py`** contiene i prompt con i termini della miocardite:
`EXTRACTOR_SYSTEM_PROMPT_TEMPLATE` e `AGENTIC_EXTRACTOR_SUFFIX_TEMPLATE`, che
portano il segnaposto `{no_evidence}`, `EVALUATOR_SYSTEM_PROMPT` e
`RETRIEVER_TOOL_DESCRIPTION`. Quest'ultima nomina i domini delle cinque sezioni
(sintomi cardiaci, esame obiettivo, marcatori di infiammazione, ECG,
ecocardiogramma) con i reperti scritti nelle parole del modulo.
`MULTI_SELECT_EXAMPLE = None` toglie dalle istruzioni di risposta l'opzione della
TVP usata come esempio, ed `EMPTY_ANSWER_SECTIONS = set()` toglie a E ed F
l'istruzione su come lasciare vuota la sezione (sezione 4.3).

Contiene anche gli hint di sezione: **`HINT_PREAMBLE`**, anteposto a ogni hint, e
i testi di `_SECTION_HINT_TEXTS`, uniti in **`SECTION_HINTS`**. Riprendono le
regole dei prompt usati in passato sulla miocardite (`MYO_prompts_EN.pdf`), dalle
istruzioni zero-shot e dalle «Reasoning Guidelines» della versione chain of
thought, senza il loro formato di risposta, più alcune frasi dalla definizione di
caso (Sexson Tejtel et al. 2022) e dalla Companion Guide SPEAC. La provenienza,
frase per frase:

| sezione | frase dell'hint | fonte |
|---|---|---|
| tutte | valutare ogni opzione indipendentemente; non inferire né aggiungere reperti | prompt precedenti, istruzioni 1-2, riformulate senza «exact terms» perché i referti sono in italiano |
| A1 | termini simili non sono equivalenti; un'aritmia all'ECG non è palpitazione | prompt precedenti, A1, nota 1 |
| A1 | dolore addominale, alla spalla o alla schiena sono sintomi di A2, non dolore toracico | prompt precedenti, A2, note 1 e 5, e il modulo; riformulata perché il paper (§4.4) li chiama varianti del dolore toracico |
| A1 | sintomo non menzionato = unknown; «none of the above» solo con assenza esplicita | prompt precedenti, A1, regole 3-4; la regola su «none» è una scelta di formulazione |
| B | se c'è lo sfregamento, selezionarlo senza dedurne gli altri segni; «confirmed … without details» solo se l'esame conferma la pericardite senza reperti | prompt precedenti, B, note e Reasoning Guidelines |
| B | nessun reperto riportato = unknown | prompt precedenti, B, Reasoning Guidelines |
| B | «none of the above» solo se i segni sono stati cercati e non trovati | formulazione nuova |
| B | i toni attenuati comprendono i toni ovattati (*muffled*) | Companion Guide, Tabella 4.3.B |
| C | soglie di VES, PCR, PCR-hs | prompt precedenti, C (le stesse note del modulo); per il D-dimero il limite del laboratorio, perché il modulo non dà soglia |
| C | valore più alto; conversioni di unità | prompt precedenti, C.1-C.4; Companion Guide, §4.5 |
| C | non dedurre aumento o assenza dai sintomi; «confirmed …» solo se dichiarato | prompt precedenti, C, Reasoning Guidelines |
| C | «None of the above» richiede almeno un marcatore dosato e tutti normali; nessuno citato = «None tested» | testo dell'opzione 5 del modulo e prompt precedenti, C |
| C | troponina, CK-MB, leucociti e procalcitonina non sono marcatori di questa sezione | formulazione nuova; il paper (Tabella 4) elenca i leucociti fra gli esami dell'infiammazione |
| E | solo menzioni dirette; ECG non eseguito = solo la prima opzione; un pattern specifico va alla sua opzione | prompt precedenti, E, Reasoning Guidelines; la regola su «No abnormalities» delle Reasoning Guidelines riguarda il «No» sulle singole righe del formato sì/no/sconosciuto e non ha un corrispettivo nel formato a caselle |
| E | tachicardia sinusale, SVT, FA, TV, FV appartengono a «Paroxysmal …» | esempi nel testo dell'opzione (dal paper, Tabella 5); i PVC no, perché l'opzione 7 li nomina e la regola sul pattern specifico li porta lì |
| E | «Nonspecific …» solo se il referto le chiama aspecifiche o se sono considerate tipicamente aspecifiche | prompt precedenti, E.2 |
| F | definizione dell'opzione 2 (ipocinesia, acinesia, FE ridotta) e della dilatazione | prompt precedenti, F, istruzione 4, quasi alla lettera; *chamber enlargement* dalle Reasoning Guidelines |
| F | definizione delle anomalie segmentali; versamento o infiammazione pericardica = opzione 8 | prompt precedenti, F, Reasoning Guidelines, riformulate |
| F | ECO non eseguito = solo la prima opzione; referto che dice l'ECO normale = «No abnormalities» | prompt precedenti, F, Reasoning Guidelines; della seconda regola solo la metà che non contraddice le istruzioni 2-3, come condizione sufficiente |
| F | «New»: senza un esame precedente, nuova se migliora o peggiora ai controlli | paper, Tabella 1, nota c (definizione CDC 2003), riformulata |
| F | raccolta anomala di liquido pericardico: versamento anteriore e posteriore, o ampio versamento posteriore isolato | paper, Tabella 1 (CDC, pericardite confermata), riformulata |

La riga delle Reasoning Guidelines di F che chiede di selezionare entrambe le
opzioni sulla funzione non è inclusa.

**`convert_cases.py`** trasforma `MYO_dummy_cases.xlsx` in un `.txt` e un
`_ground_truth.json` per caso, in `RECORDS_DIR` (`myo/data/records/`).
`record_id` unisce `Case_id` e `Variation_id`. `ANSWER_COLUMNS` indica le colonne
delle risposte; **`parse_answer(section_key, raw)`** divide una cella sui `+` e
riconosce ogni pezzo come opzione dello schema, direttamente o tramite
`OPTION_ALIASES`, dopo che **`_collapse(text)`** ha uniformato spazi e
maiuscole. Un pezzo che non corrisponde a nessuna opzione ferma la conversione,
invece di essere scartato. **`_valid_options(section_key)`** legge le opzioni
dallo schema. Richiede `openpyxl`, che non è in `requirements.txt`.

**`run_myo.py`** esegue la pipeline sui casi. Importa `myo_schema` prima dei
moduli di progetto. `SECTION_ORDER = ["A1", "B", "C", "E", "F"]` segue l'ordine
del modulo; `MYO_TEMPERATURE = 0.5` è la temperatura di Agent 1 e Agent 2, senza
seme, così ogni run campiona in modo indipendente. `SECTION_QUERIES` ha una query
per sezione composta dal nome della sezione e da tutte le sue opzioni, con le sole
parole del modulo, così nessuna query porta lessico preso dai referti su cui gira.
**`apply_domain(guideline=False, temperature=MYO_TEMPERATURE)`** imposta sezioni e
temperatura, modalità `agentic_graph`, gli hint di `prompts_myo.SECTION_HINTS`,
nessuna regola cross-section; sostituisce i tre prompt degli agenti,
`agents.MULTI_SELECT_EXAMPLE`, `agents.EMPTY_ANSWER_SECTIONS`, le query di
sezione e il tool di ricerca, e indica
per il paper un indice suo, `myo/vectorstores/chroma_brighton_myo`, perché
`build_brighton_kb` ricaricherebbe l'indice del paper della TVP senza leggere il
testo che riceve. Da riga di comando: `--guideline` accende il contesto della
linea guida, `--temperature` sostituisce `MYO_TEMPERATURE`, `--num-ctx`
sostituisce `LLM_NUM_CTX`, `--brighton-pdf` indica il paper (`myo/main.pdf` di
default), più `--records-dir`, `--output-dir` e `--only`. L'Agent 3 gira con le
impostazioni di `config.py`.

**`evaluate_myo.py`** valuta una run con `evaluate_predictions` sulle cinque
sezioni: importa `myo_schema` prima dei moduli di progetto e punta le cartelle di
default su `myo/data/records` e `myo/output_myo`. Ripete la lista delle sezioni di
`run_myo` invece di importarla, così la valutazione non carica la pipeline.

**`export_prompts_myo.py`** esporta prompt, query e hint della miocardite con
`export_prompts.py`, in `myo/prompts_export/` di default (sezione 15).

**`run_dvt_stripped.py`** è il termine di confronto. Importa `project_path`, non
`myo_schema`, perché lavora sullo schema della TVP; esegue il corpus TVP con
**`strip()`**, che spegne in memoria hint, contesto, intestazioni di
sezione e regole cross-section, gli stessi componenti che il
braccio della miocardite non ha. Poi chiama `run_synthetic_records.main()`, di
cui accetta gli argomenti.

---

## 17. `evaluate_verifier.py`

Valuta il verificatore (sezione 7) contro la ground truth. Per ogni sezione con un
risultato del verificatore, la risposta è giusta o sbagliata rispetto al
riferimento, e il verificatore l'ha approvata o no con un certo P(YES). Riporta,
per variante, in totale e per sezione: l'AUROC di P(YES) fra risposte giuste e
sbagliate (le parità contano mezzo punto), gli errori approvati e le risposte
giuste respinte, più il numero di richieste senza verdetto e di quelle oltre la
finestra di contesto. Le sezioni il cui verdetto è stato copiato da una regola
(`rule:<sorgente>`) sono contate, perché è il verdetto che il modulo porta.

- **`load_runs(run_dir)`** legge, per referto, il file più recente e il suo audit
  log;
- **`auroc(right, wrong)`** calcola l'AUROC dalla formula a coppie; `None` se uno
  dei due gruppi è vuoto;
- **`collect(runs, ground_truth)`** produce una riga per referto, sezione e
  variante;
- **`summarise(rows)`** calcola le metriche di un gruppo di righe;
- **`compare_variants(rows, first="answer", second="reasoning")`** elenca le
  sezioni in cui due varianti danno verdetti diversi, divise per correttezza
  della risposta: un errore approvato solo dalla variante `reasoning` è il caso
  in cui il ragionamento dell'Agent 2 ha convinto il verificatore. `evaluate` la
  chiama anche per `answer` contro `answer_hint` e `reasoning` contro
  `reasoning_hint` (`hint_disagreements`), vuote quando le varianti mancano;
- **`evaluate(rows)`**, **`print_report(report)`** e **`main()`**: il report va in
  `reports/verifier_<cartella>_<timestamp>.json`. `--ground-truth` indica i
  riferimenti, `myo/data/records` per la miocardite.

## 18. `agreement_runs.py`

Misura la concordanza fra due o più run della stessa configurazione, prodotte a
temperatura maggiore di 0. Le run fanno da valutatori e i referti da soggetti,
come nell'analisi di concordanza fra abstractor del paper VAC4EU. Riporta per
sezione e in totale:

- la kappa di Fleiss fra le run, con l'insieme di opzioni come categoria (la
  stessa convenzione a corrispondenza esatta di `evaluate_predictions`), non
  pesata;
- l'AC1 di Gwet, più stabile della kappa quando una risposta domina; il numero di
  categorie è quello delle risposte distinte osservate nella sezione, perché una
  sezione a scelta multipla non ha un elenco fisso di categorie;
- la quota di referti su cui tutte le run danno la stessa risposta;
- l'accuratezza di ogni run e quella del voto a maggioranza (una risposta data da
  più di metà delle run; un referto senza maggioranza conta come errore ed è
  contato a parte).

La concordanza come segnale di confidenza: ogni risposta riceve la quota di run
che hanno dato la stessa risposta (1/3, 2/3 o 1 con tre run), e l'AUROC di quella
quota fra risposte giuste e sbagliate dice se una risposta stabile è più spesso
giusta. Se le run portano il verificatore, accanto c'è l'AUROC di P(YES) sulle
stesse risposte, per variante. Una sezione lasciata senza risposta conta come
categoria `<missing>` e come errore.

- **`fleiss_kappa(table)`** e **`gwet_ac1(table)`** ricevono un `Counter` per
  soggetto; la kappa è `None` quando la concordanza attesa vale 1. La kappa è stata
  confrontata con `statsmodels.stats.inter_rater.fleiss_kappa` su tabelle casuali,
  con risultati identici alla sesta cifra;
- **`collect(run_dirs, ground_truth)`** legge i referti presenti in tutte le run e
  nella ground truth;
- **`summarise(rows)`**, **`evaluate(rows, sections)`**, **`print_report`** e
  **`main()`**: il report, con l'elenco dei referti su cui le run divergono, va in
  `reports/agreement_<prima cartella>_<timestamp>.json`.

## 19. `run_myo_t05.sh`

Script bash per mari che esegue le tre run della miocardite a temperatura 0,5 e
le valuta. Prima della prima run controlla che Ollama risponda, che il modello
valutatore sia scaricato e, con lo stesso codice della run (`run_myo.apply_domain`
e le funzioni di impronta di `pipeline.py`), che impronta degli hint, impronta
delle query, temperatura e verificatore siano quelli attesi (`EXPECTED_HINTS`,
`EXPECTED_QUERIES`) e che nessuna sezione riceva la clausola sulla risposta
vuota; altrimenti si ferma. Esegue le run una dopo l'altra in
`myo/output_<RUN_TAG>_r1`, `r2` e `r3` (`RUN_TAG` vale `t05` di default), con un
log per run in `logs/`, e rifiuta una
cartella che contiene già file salvo `FORCE=1`. Dopo ogni run **`check_run`**
rilegge gli audit log e stampa temperatura, tetto di token e finestra, impronte,
sezioni con la clausola sulla risposta vuota, metodi del verificatore,
richieste oltre la finestra, mediana dei secondi per richiesta e sezioni fallite.
Alla fine lancia `myo/evaluate_myo.py` ed `evaluate_verifier.py` su ogni run e
`agreement_runs.py` sulle tre. Variabili d'ambiente: `T0=1` aggiunge una run a
temperatura 0 (`myo/output_<RUN_TAG>_t0`), `STRIPPED=1` la TVP senza hint
(`output_stripped_verifier`), `EVAL_ONLY=1` salta le run, `PYTHON` indica
l'interprete; attiva `.venv` se esiste.

## 20. `replay_verifier.py`

Riapplica il verificatore a una run già su disco, con il modello indicato da
`--model`, senza rieseguire gli Agenti 1 e 2: il verificatore non cambia nessuna
risposta, e l'audit log conserva evidenza e ragionamento, il file dei risultati
le risposte finali. Due giudici si confrontano così sulle stesse risposte.

Per ogni coppia di file dei risultati e audit log della cartella di origine toglie
i verdetti presenti, chiama `verifier.verify_record` sulle risposte salvate (la
stessa funzione della pipeline, quindi stesso prompt, stessa eredità per le
sezioni riscritte da una regola e stessi parametri) con `VERIFIER_MODEL_NAME`
impostato al modello scelto e `VERIFIER_INPUTS` alle varianti di `--inputs` (di
default quelle di `config.py`), e scrive i due file con gli stessi nomi nella
cartella `--output-dir`. Le risposte sono copiate senza modifiche. In
`_run_config.verifier` registra le impostazioni di `verifier.verifier_settings()`,
più `replayed_from` (la cartella di origine), `replayed_at`, `hostname` e
`ollama_version` della rilettura. `evaluate_verifier.py` e `agreement_runs.py`
leggono la cartella prodotta come una run.

`--domain myo` installa lo schema e gli hint della miocardite
(`prompts_myo.SECTION_HINTS`, nessuno disattivato) prima di importare il
verificatore e prende `SECTION_ORDER` dalle sue sezioni. Le varianti `_hint`
ricevono gli hint del codice attuale, non quelli della run: lo script stampa il
loro digest e dice se coincide con `section_hints_fingerprint` della run. Prima del primo referto
una richiesta di prova controlla che il modello risponda YES o NO con le
probabilità dei token; senza, non c'è P(YES) e lo script si ferma. Rifiuta una
cartella di destinazione che contiene già file, salvo `--force`.

