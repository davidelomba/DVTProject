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
| `rag_setup.py` | embedding, vector store, loader, tool di ricerca |
| `agents.py` | i due agenti e il parsing delle risposte |
| `criteria_rules.py` | regole cross-section |
| `agentic_graph.py` | macchina a stati LangGraph della modalità agentica |
| `confidence.py` | Agent 3, la confidenza di ogni risposta finale |
| `pipeline.py` | orchestrazione di un referto |
| `aggregation.py` | serializzazione del form |
| `main.py` | entry point su un singolo referto |
| `run_synthetic_records.py` | esecuzione batch sul corpus |
| `generate_synthetic_records.py` | generazione e audit del corpus sintetico |
| `evaluate_predictions.py` | valutazione contro la ground truth |
| `compare_runs.py` | confronto tra due run, misura del rumore |
| `export_redcap_csv.py` | conversione dei risultati in CSV per REDCap |
| `export_prompts.py` | esportazione di prompt, query e hint in file per la tesi (non versionato) |
| `myo/` | il braccio di trasferimento sul questionario della miocardite |

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

- `LLM_TEMPERATURE = 0.0` — output deterministico.
- `LLM_NUM_PREDICT = 1024` — tetto di token. Le due righe di risposta chiudono
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
- `LLM_NUM_CTX = 4096` — la finestra di contesto che Ollama alloca per ogni
  richiesta, in token. È fissata qui perché il default di Ollama non compare in
  nessun audit log; 4096 è il valore che `ollama ps` riporta per qwen3.6:27b su
  mari, quindi quello già in vigore.
- `LLM_REQUEST_TIMEOUT = 180` secondi.

Due costanti governano l'Agent 3 (sezione 7):

- `CONFIDENCE_ENABLED = False` — acceso, dopo le regole cross-section ogni
  sezione riceve un valore di confidenza fra 0 e 1 nel proprio audit log. Spento,
  la pipeline non fa alcuna richiesta in più.
- `CONFIDENCE_SKIP = {"F"}` — sezioni lasciate senza valore. La risposta di F
  dipende da una domanda invertita e dalla presenza di una diagnosi riportata, e
  l'Agent 3 giudica entrambe senza ragionare. Togliendo `"F"` la sezione viene
  valutata attraverso la sua riga `DETAILS_PRESENT`.

### Embedding e modalità di estrazione

`EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"`: multilingue per
necessità, dato che le query di retrieval sono in inglese e i referti in
italiano, e i referti non vengono tradotti automaticamente per non rischiare che
una traduzione distorca negazioni o terminologia in modo non verificabile.

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
sotto `PROJECT_ROOT`, la cartella che contiene `config.py`.

### `GUIDELINE_ANCHORS` e il suo interruttore

`GUIDELINE_ANCHORS` associa a ogni sezione le etichette del passaggio del paper
che ne definisce il criterio: un'intestazione numerata, che vale fino alla
successiva che non sia una sua sottosezione, o una didascalia di tabella, che
vale fino alla prossima intestazione. `GUIDELINE_ANCHORS_ENABLED`, spento,
lascia il recupero per somiglianza.

Il recupero usa come chiave la query di sezione, cioè una stringa scritta per
nominare un reperto dentro una cartella clinica e non un criterio dentro un
articolo; `RISULTATI_SPERIMENTALI.md` riporta quali passaggi raggiungono di
fatto Agente 2 per questa via.

Le etichette seguono la struttura del questionario. A1 e A2 puntano alla 4.1 e
alla 5.2.2, che definiscono la diagnosi patologica e nominano insieme la
procedura chirurgica e quella per cateterismo. B1.1, B1.2 e B2 puntano alla
Table 3, la case definition da cui il questionario deriva: il suo livello 2
enuncia la presumed diagnosis di una sindrome, TVP degli arti inferiori o
superiori, che è ciò che B1.1 e B1.2 registrano, e i segni aspecifici
dell'estremità che sono le opzioni di B2. A3.2 punta alla Table 1, che elenca le
tecniche per sede ed è la sua lista di opzioni.

Le ancore stanno dentro la finestra di contesto: il prompt più lungo è quello di
B1.2, e `RISULTATI_SPERIMENTALI.md` riporta la misura e il margine.

`F` non compare nella mappa: il suo criterio chiede se una diagnosi è stata
riportata da uno specialista e se è accompagnata da dettagli, e il paper non ha
un passaggio che lo definisca. Quella sezione ripiega sul recupero anche a
interruttore acceso.

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

- `SECTION_HINTS_ENABLED = True` — interruttore generale. L'hint di F chiede
  la riga `DETAILS_PRESENT` che `confidence.score_details` legge: con gli hint
  spenti l'Agent 3 valuta F come ogni altra sezione a scelta singola.
- `SECTION_HINTS_DISABLED = {"B2"}` — sezioni sospese individualmente. B2 è
  elencata perché il suo hint abbassa l'accuratezza di B2; il testo è conservato
  così l'ablazione è ripetibile.

`section_hint(section_key)` è l'unico accesso e rispetta entrambi gli
interruttori, restituendo stringa vuota quando l'hint non va inviato.

Contenuto attuale, in sintesi:

| sezione | cosa dice |
|---|---|
| A1 | contano solo i reperti post-mortem; l'imaging su vivente non è un'autopsia |
| A2 | attenzione estrema alle negazioni prima di termini chirurgici |
| A3_1 | un risultato negativo è comunque un risultato; «result not reported» vuol dire referto mancante; un'autopsia non è un esame di imaging |
| A3_2 | un esame che non è nessuna delle quattro modalità nominate è "Other" |
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
che la diagnosi non è stata riportata affatto. È la convenzione su cui poggia
`confidence.score_details`.

`DVT_CriteriaForm` è il contenitore: `record_id` obbligatorio più dieci campi
opzionali, uno per sezione. Una sezione che la pipeline non è riuscita a
compilare resta `None` invece di bloccare l'intero form.

`SECTION_MODELS` mappa la chiave testuale della sezione alla classe Pydantic.

---

## 3. `rag_setup.py`

Prepara il lato retrieval: modello di embedding, le due vector store, i loader e
il tool che l'estrattore agentico chiama.

**`get_embeddings()`** restituisce `HuggingFaceEmbeddings` configurato con i
prefissi di ruolo che `multilingual-e5-small` richiede: `"passage: "` per i
chunk indicizzati (`encode_kwargs`, usato da `embed_documents`) e `"query: "` per
le query (`query_encode_kwargs`, usato da `embed_query`). Senza i prefissi la
scheda del modello riporta un retrieval degradato.

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

**`retrieve_brighton_context(...)`** produce il contesto che Agente 2 riceve per
una sezione. Con `config.GUIDELINE_ANCHORS_ENABLED` a `False` interroga l'indice
del paper con la query di sezione e restituisce i primi `BRIGHTON_RETRIEVER_K`
chunk; a `True` restituisce il passaggio ancorato, e ripiega sul recupero per le
sezioni che un'ancora non ce l'hanno.

**`section_is_anchored(...)`** dice se una sezione legge un'ancora. La leggono
sia `retrieve_brighton_context`, per scegliere il percorso, sia i suoi
chiamanti, per annunciare il blocco ad Agente 2: l'espressione sta in un posto
solo e le due decisioni non possono divergere.

**`_paper_outline(...)`**, **`_anchor_span(...)`** e
**`resolve_guideline_anchors(...)`** risolvono le etichette di
`config.GUIDELINE_ANCHORS` in testo. `_paper_outline` individua le intestazioni
numerate e tiene solo quelle il cui numero supera l'ultimo tenuto: le righe
delle tabelle delle tecniche sono numerate allo stesso modo, e il confronto
monotono le scarta. `_anchor_span` fa terminare un'intestazione alla prima
successiva che non sia una sua sottosezione, e una didascalia di tabella, che
l'outline non copre, alla prima intestazione successiva di qualunque livello.
`resolve_guideline_anchors` concatena i passaggi di una sezione nell'ordine in
cui sono elencati e li passa per `clean_brighton_context`; una sezione le cui
etichette il paper non contiene resta fuori dalla mappa.

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

**`_get_field_info(section_model)`** introspeziona lo schema e restituisce
`(field_name, options, is_multi_select, description)`, dove `description` è il
titolo della sezione. È ciò che rende `evaluate_section` generica per tutte e
dieci le sezioni senza un ramo per ciascuna; la usa anche `confidence.py`.

**`_build_reasoning_prompt(...)`** compone il prompt. Le opzioni sono numerate e
il modello risponde **due volte**: il testo dell'opzione su `FINAL_OPTION` e il
suo numero su `FINAL_ANSWER`. Due risposte invece di una rendono visibile un
disaccordo, dato che il modello a volte nomina un'opzione e scrive il numero di
un'altra. Il prompt chiede inoltre esplicitamente che ogni opzione elencata sia
tracciabile a una frase del ragionamento, e non inclusa per default o per
margine di sicurezza. Con `config.SECTION_DESCRIPTIONS_ENABLED` acceso, il titolo
della sezione viene inserito sopra le opzioni.

Il blocco della linea guida è annunciato da un'intestazione che dipende da cosa
contiene: i chunk recuperati sono vocabolario da consultare e restano
`Reference synonyms/terminology (Brighton)`, un passaggio ancorato enuncia il
criterio e diventa `Guideline passage defining this criterion (Brighton)`.
L'intestazione dice al modello cosa farne, e annunciare come sinonimi una
definizione la fa trattare da glossario.

Per le sezioni multi-scelta prive di un'opzione "nessuna delle precedenti", cioè
A3.2 e B1.2, il prompt aggiunge come dire che nulla si applica: `FINAL_OPTION:
none` e `FINAL_ANSWER: none`. Senza questa istruzione, non avendo modo di
esprimere una risposta vuota, il modello selezionava tutte le opzioni. La cosa è
esplicitata solo dove serve, così B2 continua a usare la propria opzione 5.

**`_has_none_option(options)`** dice se una sezione ha un'opzione esplicita
«none of the above», riconosciuta dal prefisso `_NONE_OPTION_PREFIX`: è ciò che
decide se il prompt aggiunge l'istruzione `none`. **`_is_no_selection(raw_value)`**
riconosce una riga di risposta che significa «nessuna opzione»: le parole
accettate sono in `_NO_SELECTION_ANSWERS` (`none`, `nothing`, `nessuna`, `n/a` e
simili).

**`_extract_final_answer_line`** e **`_extract_labeled_line`** leggono le due
righe. Entrambe prendono l'**ultima** occorrenza, non la prima, perché il modello
a volte ripete l'istruzione prima di rispondere. La differenza sta nel
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
dell'errore e la sezione ritentata, fino a `max_retries + 1` tentativi. Se
falliscono tutti, la funzione solleva un `RuntimeError` a cui **allega l'ultima
risposta del modello** come attributo `last_response`: una sezione fallita è
l'unico caso in cui il chiamante non ha altra copia di ciò che il modello ha
scritto.

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
nome della sezione sorgente (`A3_1` o `B2`); `confidence.score_record` la usa per
assegnare la confidenza.

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
contesto della linea guida, ancore comprese se accese, e chiama Agent 2.

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

## 7. `confidence.py`

L'Agent 3. Per ogni sezione calcola un numero fra 0 e 1, la **confidenza** della
risposta finale del modulo, e lo scrive nell'audit log della sezione. Gira dopo
le regole cross-section, quindi valuta la risposta che finisce nel CSV, ed è
acceso da `config.CONFIDENCE_ENABLED`. Non cambia mai una risposta.

**Come si ottiene il numero.** Il modello di Agent 2 riceve di nuovo l'evidenza, il
contesto della linea guida e l'hint della sezione, con le opzioni numerate, ma
**non** la risposta di Agent 2, e deve rispondere subito, senza ragionare. Con
`logprobs` attivo Ollama restituisce, per ogni token generato, la sua probabilità
e quella delle `TOP_LOGPROBS = 10` alternative più probabili nella stessa
posizione. La confidenza è la probabilità che il modello assegna alla risposta
del modulo. La richiesta esclude il ragionamento perché, dopo un ragionamento, il
numero finale è determinato dal testo che lo precede, e la sua probabilità dice
quanto il modello è coerente con quel testo più che quanto l'evidenza sostiene la
risposta.

**Costanti.** `OLLAMA_URL` è l'indirizzo del server, letto dalla variabile
d'ambiente `OLLAMA_HOST` come fa Ollama stesso, con `http://localhost:11434` come
default. `SYSTEM_PROMPT` è il messaggio di sistema della richiesta: chiede di
rispondere dalla sola evidenza e senza spiegare.

**`_chat(user_prompt, num_predict)`** invia una richiesta a `/api/chat` di
Ollama, non attraverso `ChatOllama`, perché servono i campi `logprobs` della
risposta. Modello (`EVALUATOR_LLM_MODEL_NAME`), temperatura, `num_gpu`, `num_ctx`
e ragionamento sono quelli di Agent 2; cambia solo il tetto di token. Usare lo
stesso modello evita di caricarne un secondo sulla GPU.

**`_build_prompt(evidence, context, hint, options, multi)`** compone il messaggio:
contesto della linea guida, hint con l'avvertenza di ignorarne le istruzioni di
formato, evidenza, opzioni numerate, e l'istruzione finale. Per una sezione a
scelta singola chiede il solo numero; per una a scelta multipla una riga
`N: YES` o `N: NO` per ogni opzione.

**`_mass(token, accept)`** somma la probabilità delle alternative di una posizione
che soddisfano un criterio, così «2» e « 2», che sono token diversi, contano
insieme. **`_number_distribution(token, n)`** usa `_mass` per ottenere la
probabilità di ciascun numero di opzione valido, normalizzata sui soli numeri
validi.

**`score_single(logprobs, n, chosen)`**: sezioni a scelta singola. Legge il primo
token che è un numero di opzione valido e restituisce la quota di probabilità
che cade sull'opzione scelta. Se il modello ha scritto altro prima del numero, il
metodo diventa `single_after_text`, perché la probabilità è condizionata da quel
testo.

**`score_multi(logprobs, n, chosen)`**: sezioni a scelta multipla (A3_2, B1_2,
B2). Per ogni opzione legge p(YES) sul token che segue `N:`, normalizzato su
YES e NO. La confidenza è la probabilità dell'**insieme esatto** scelto: il
prodotto di p(YES) per le opzioni selezionate e di p(NO) per le altre. È coerente
con la valutazione a insieme esatto, ma tratta le opzioni come indipendenti, e
un prodotto di più fattori è sistematicamente più basso del valore di una sezione
a scelta singola.

**`score_details(logprobs, options, chosen)`**: F, quando il suo hint fa scrivere
prima la riga `DETAILS_PRESENT`. La risposta di F non si ricava dal solo yes/no
di quella riga: dettagli presenti implicano No, ma dettagli assenti implicano Yes
solo se una diagnosi è riportata, altrimenti No. La funzione legge p(yes) e p(no)
sul token dopo `DETAILS_PRESENT:` e la distribuzione q del numero che segue, che
tiene conto della diagnosi, ma solo per il ramo effettivamente scritto:

| il modello scrive | risultato | metodo |
|---|---|---|
| `no` | P(Yes) = p(no)·q(Yes\|no); P(No) = p(yes) + p(no)·q(No\|no) | `details_exact`: il ramo `yes` è fissato dall'hint |
| `yes` | P(risposta) ≥ p(yes)·q(risposta\|yes) | `details_lower_bound`: il ramo `no` non è stato generato |

**`score_section(section_key, answer, evidence, context)`** esegue una sezione:
costruisce il prompt con `config.section_hint`, chiama `_chat` con un tetto di 32
token per le sezioni a scelta singola (la riga `DETAILS_PRESENT` di F ne occupa
diversi) e di `8 × opzioni + 8` per quelle multiple, e sceglie il metodo di
calcolo. Su F prova `score_details` se la risposta contiene `DETAILS_PRESENT`, e
ripiega su `score_single` altrimenti. Restituisce `confidence` (arrotondata a
quattro decimali, o `None`), `confidence_method` e `agent3_seconds`.

**`score_record(form_data, audit_log)`** è il punto d'ingresso che
`pipeline.run_pipeline` chiama. Scorre `config.SECTION_ORDER`, prende la risposta
finale di ogni sezione da `form_data` e aggiunge i tre campi alla sua voce
dell'audit log, stampando il valore a schermo con **`_report`**. Una sezione con
la chiave `overridden_by`, cioè la cui risposta è stata scritta da una regola
cross-section, non riceve una richiesta: prende la confidenza della sezione
sorgente, con `confidence_method` `rule:<sorgente>`, per esempio `rule:A3_1`.
Lo fa in un secondo passaggio, perché una sorgente può venire dopo la sezione
che la regola cambia (B2 viene dopo B1_1). La ragione è che quella risposta non è
di Agent 2 ma della regola, e dipende interamente dalla sorgente: interrogato
sulla sola A3_2, il modello vede l'esame eseguito e non sa che A3.1 ha stabilito
che non ha confermato la TVP. Assegna `confidence: None` con il
motivo in `confidence_method` in tre casi: la sezione è in
`config.CONFIDENCE_SKIP` (`skipped`), la sezione non ha risposta (`no_answer`),
la richiesta fallisce (`error`, con il messaggio in `confidence_error`). Un
fallimento dell'Agent 3 non interrompe la run.

I valori possibili di `confidence_method` sono quindi `single`,
`single_after_text`, `multi`, `details_exact`, `details_lower_bound`,
`rule:<sorgente>`, e, quando
il numero manca, `no_number`, `incomplete_lines`, `skipped`, `no_answer`,
`error`. Su F, se `score_details` non trova la riga o il numero, il metodo
registrato è quello del ripiego su `score_single`.

---

## 8. `pipeline.py`

Orchestra un referto e dispatcha sulle quattro modalità: `agentic_graph` passa
dal grafo di `agentic_graph.py`, le altre tre dal ciclo per sezione di questo
modulo.

**`SECTION_QUERIES`** associa a ogni sezione la query che dice ad Agent 1 cosa
cercare, e che serve anche a recuperare il contesto dalla linea guida. La query
di `X` è formulata seguendo il linguaggio della Tabella 2 del paper: la
formulazione precedente non recuperava mai quella tabella, che finiva a B2 perché
la riga della TVP è scritta in parole di sintomo che corrispondono quasi
esattamente alla query di B2.

**`_ollama_version()`** legge la versione del binario Ollama. È registrata perché
due versioni di Ollama portano due versioni di llama.cpp, e con esse kernel di
quantizzazione diversi: a temperatura 0 basta a far cambiare un token che il
modello aveva quasi in parità, quindi run prodotte sotto versioni diverse non
sono direttamente confrontabili. Non solleva mai: una versione mancante costa la
provenienza, non la run.

**`_hint_fingerprint()`** produce, per ogni sezione che riceve un hint non vuoto,
la coppia lunghezza e prefisso sha256 del testo, più un digest `all` dell'intero
insieme. Gli interruttori dicono quali hint sono stati inviati, non cosa
dicevano, quindi senza questo due run i cui hint sono stati riscritti in mezzo
porterebbero la stessa firma.

**`_query_fingerprint()`** fa lo stesso per `SECTION_QUERIES`, che è insieme il
brief da cui l'Agente 1 parte e la chiave con cui si recupera il contesto della
linea guida. Nessun altro campo dello snapshot ne registra il testo, quindi
senza questo due run le cui query sono state riscritte in mezzo porterebbero la
stessa firma.

**`_anchor_fingerprint(...)`** fa lo stesso per le ancore, digerendo però il
testo risolto e non le etichette: le etichette nominano intestazioni, e il testo
a cui arrivano dipende da come il PDF si è estratto, quindi da sole non dicono
cosa Agente 2 abbia letto. Le sezioni senza ancora non compaiono.

**`_run_config_snapshot(...)`** cattura tutto ciò che determina cosa una run
produce: modalità, contesto della linea guida, intestazioni di sezione,
hint, query e ancore con i rispettivi fingerprint, le impostazioni dell'Agent 3
(`confidence.enabled` e `confidence.skipped_sections`), modelli per ruolo (con
l'estrattore **effettivo**: `AGENTIC_LLM_MODEL_NAME` in modalità agentica,
nessuno in `raw_record`), ambiente, parametri di generazione compreso
`num_ctx`, parametri di retrieval. Viene scritto nell'audit log sotto la chiave
`RUN_CONFIG_KEY = "_run_config"`, scelta per non poter collidere con un nome di
sezione, così un file di risultati è auto-descrittivo mesi dopo.

**`run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)`** costruisce una
sola volta embedding e valutatore, carica i due testi, costruisce la KB Brighton
sempre e quella del referto solo dove serve. Risolve le ancore della linea guida
dal testo del paper. Il modello di Agent 1 viene costruito **dentro** il ramo che
lo usa, così una modalità che non lo interroga, come `raw_record`, non lo carica
in VRAM.

Nel ciclo per-sezione, ogni sezione passa da Agent 1 (in `raw_record` l'evidenza
è il referto intero), dal recupero del contesto Brighton ripulito e da Agent 2.
Il fallimento di una sezione non compromette le altre: il campo resta
`None`, l'errore va in `section_log["error"]` e l'ultima risposta del modello, se
disponibile, in `section_log["reasoning"]`.

Alla fine applica le regole cross-section una volta sola; se
`config.CONFIDENCE_ENABLED` è acceso chiama `confidence.score_record` sulle
risposte finali; poi aggiunge lo snapshot di configurazione e costruisce
`DVT_CriteriaForm`.

---

## 9. `aggregation.py` e `main.py`

**`aggregation.form_to_json_summary(form)`** serializza il form con
`exclude_none=True`: le sezioni lasciate `None` da una valutazione fallita
vengono omesse invece che scritte come `null`, così una risposta mancante è
assente e non somiglia a una risposta di "nessuno". Riceve anche l'audit log: se
l'Agent 3 ha girato, copia la confidenza di ogni sezione sotto la chiave
`CONFIDENCE_KEY = "_confidence"`, con `null` per le sezioni non valutate. Così il
valore resta nel file dei risultati anche quando l'audit log non viene salvato.
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

Genera i referti sintetici in italiano con la ground truth corrispondente.
`SCENARIOS` contiene i 40 scenari, `OUTPUT_DIR` è `data/synthetic_records/`.
Il modello scrittore è `WRITER_MODEL_NAME = "qwen2.5:7b-instruct"`, a
`WRITER_TEMPERATURE = 0.8` e con `WRITER_NUM_PREDICT = 3072`, ben sopra la
lunghezza richiesta perché un tetto raggiunto a metà tronca il referto;
`WRITER_SYSTEM_PROMPT` gli chiede di scrivere come un medico di pronto soccorso.

Le funzioni, nell'ordine in cui lavorano:

- **`directive_for(scenario, style)`** sceglie la direttiva di stile: quella
  ricca di dettagli, o quella senza dettagli per gli scenari con `F = "Yes"`;
- **`facts_to_prompt(scenario, style)`** compone il messaggio per lo scrittore:
  direttiva, fatti clinici e le eventuali `writer_notes` dello scenario;
- **`build_ground_truth(record_id, scenario)`** costruisce le risposte di
  riferimento salvate accanto al referto;
- **`generate_record(llm, scenario, style)`** chiede un referto allo scrittore,
  senza controllarlo;
- **`generate_checked_record(...)`**, **`_expected_markers(...)`**,
  **`check_record(...)`** e **`check_existing_records()`** sono il controllo di
  fedeltà descritto sotto, e `_FACT_MARKERS` ne contiene i marcatori;
- **`main()`** riscrive sempre la ground truth; scrive i referti mancanti solo con
  `--generate` (sovrascrive gli esistenti solo con `--force`), controlla il
  corpus su disco con `--check`, e `--only` restringe il lavoro agli scenari
  nominati.

**Ground truth per costruzione.** Ogni scenario porta sia i fatti clinici sia le
risposte corrette per tutte e dieci le sezioni, scritte a mano con le stringhe
esatte di `models.py`. Nessun modello indovina mai il riferimento, ed è questo a
renderlo utilizzabile come tale. I JSON di ground truth vengono sempre riscritti,
dato che produrli non coinvolge alcun modello.

**I referti.** Quelli attualmente su disco non sono stati prodotti dal modello
scrittore: sono stati redatti da un modello generalista esterno a ogni ruolo
della pipeline, a partire dai fatti di ogni scenario e rivisti contro di essi,
dopo che quelli generati erano stati ripetutamente trovati in contraddizione con
la propria ground truth. Lo scrittore resta disponibile per nuovi scenari, a
temperatura non nulla per variazione lessicale. I referti vengono scritti solo se
mancanti, salvo `--force`, quindi un run normale non può sovrascriverli.

**Gli stili.** `STYLE_VARIANTS` chiede le caratteristiche strutturali che i
referti ospedalieri italiani condividono, mai le etichette o le formulazioni
esatte: uno scrittore copia gli esempi che riceve, quindi prescriverle
produrrebbe varianti quasi identiche di un unico documento, sovradattate a un
solo clinico. Esiste una direttiva separata per gli scenari la cui ground truth
è `F = "Yes"`, cioè diagnosi riportata **senza** dettagli: un solo parametro
vitale o reperto renderebbe il referto dettagliato e ribalterebbe la risposta
corretta di F.

**Il controllo di fedeltà.** `_expected_markers(scenario)` ricava dai fatti
dello scenario quali marcatori il referto deve contenere; `check_record(text,
scenario)` verifica il testo contro quella lista e restituisce l'elenco dei
problemi; `generate_checked_record(...)` rigenera finché il controllo rifiuta,
fino a `WRITER_MAX_ATTEMPTS`, e ogni tentativo è un vero ricampionamento grazie
alla temperatura non nulla. `check_existing_records()` esegue gli stessi
controlli su un corpus già su disco senza chiamare alcun modello
(`--check`).

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

- **Accuratezza exact-match** con intervallo di Wilson al 95%. Wilson e non
  l'intervallo normale, che su una sezione quasi perfetta esce oltre 1.0 e
  collassa a larghezza zero esattamente a 1.0.
- **Baseline di maggioranza** e guadagno su di essa: è il pavimento che una
  sezione deve superare per portare informazione.
- **Kappa di Cohen**, con la risposta intera trattata come una sola etichetta,
  così una sezione multi-scelta è valutata sull'insieme esatto che ha prodotto.
- **TP/TN/FP/FN, precisione, richiamo e F1 per opzione** e non per sezione:
  senza questo, una sezione multi-scelta risposta a metà conterebbe come
  semplicemente sbagliata. Precisione e richiamo ignorano deliberatamente i veri
  negativi, che sono la maggioranza di ogni conteggio dato che la maggior parte
  delle opzioni non si applica alla maggior parte dei referti.
- **Matrice di confusione**, solo per le sezioni a scelta singola, dove una
  predizione è una classe. Mostra **quali** opzioni vengono scambiate tra loro.

La riga complessiva è riportata due volte: **micro** mette in comune ogni opzione
di ogni sezione, quindi una sezione con più opzioni pesa di più; **macro** media
le cifre per sezione, quindi ogni sezione conta una volta.

Una sezione lasciata `None`, o un referto senza output, è riportata come mancante
ed esclusa dalle metriche invece di essere contata come errore. È una scelta da
tenere presente leggendo i numeri: un modello che fallisce molte sezioni ottiene
un punteggio ottimisticamente alto.

Dipende da scikit-learn e da `models.py`, ma non da langchain o Ollama: gira
senza lo stack della pipeline installato. Per questo **`_field_info(section)`**
reintroduce qui l'introspezione dello schema invece di importarla da `agents.py`.

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
- **`_wilson_interval(...)`**, **`_majority_baseline(pairs)`** e
  **`_score_section(...)`** calcolano le metriche di una sezione;
- **`evaluate(ground_truth, predictions)`** produce il report completo;
- **`print_report(report, show_matrices)`** e **`print_confusion_matrices(report)`**
  lo stampano, con `_pct`, `_num`, `_ci` e `_gain` per la formattazione;
  `--no-matrices` omette le matrici;
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

- **`_section_options(section_key)`** legge le opzioni nell'ordine dello schema, e
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

---

## 16. Il pacchetto `myo/`

Il braccio di trasferimento: esegue la stessa pipeline sulle sezioni E ed F del
questionario Brighton della miocardite, su 17 casi dummy con ground truth scritta
da due autori esterni al progetto. Nessun modulo della pipeline viene modificato:
`myo/` sostituisce in memoria schema, sezioni, prompt e configurazione prima di
chiamarla.

**`models_myo.py`** contiene lo schema. `E_Electrocardiogram` (14 opzioni) e
`F_Echocardiogram` (10 opzioni) sono entrambe a scelta multipla,
`findings: List[Literal[...]]`, con le stringhe copiate dal questionario, virgole
e asterischi compresi. `MYO_CriteriaForm` è il contenitore, esportato anche col
nome `DVT_CriteriaForm` perché `pipeline.py` importa quel nome, e
`SECTION_MODELS` ha le chiavi `E` e `F`.

**`prompts_myo.py`** contiene i prompt con i termini della miocardite:
`EXTRACTOR_SYSTEM_PROMPT_TEMPLATE` e `AGENTIC_EXTRACTOR_SUFFIX_TEMPLATE`, che
portano il segnaposto `{no_evidence}`, `EVALUATOR_SYSTEM_PROMPT` e
`RETRIEVER_TOOL_DESCRIPTION`.

**`convert_cases.py`** trasforma `MYO_dummy_cases.xlsx` in un `.txt` e un
`_ground_truth.json` per caso, in `RECORDS_DIR` (`myo/data/records/`).
`record_id` unisce `Case_id` e `Variation_id`. `ANSWER_COLUMNS` indica le colonne
delle risposte; **`parse_answer(section_key, raw)`** divide una cella sui `+` e
riconosce ogni pezzo come opzione dello schema, direttamente o tramite
`OPTION_ALIASES`, dopo che **`_collapse(text)`** ha uniformato spazi e
maiuscole. Un pezzo che non corrisponde a nessuna opzione ferma la conversione,
invece di essere scartato. **`_valid_options(section_key)`** legge le opzioni
dallo schema.

**`run_myo.py`** esegue la pipeline sui casi. Installa `models_myo` come modulo
`models` prima di importare `pipeline`, che lega lo schema all'importazione.
**`apply_domain(guideline=False)`** imposta sezioni `E` ed `F`, modalità
`agentic_graph`, svuota hint e regole cross-section, sostituisce i tre
prompt degli agenti, le query di sezione (`SECTION_QUERIES`, scritte a partire
dalla lista delle opzioni) e il tool di ricerca, e indica per il paper un indice
suo, `myo/vectorstores/chroma_brighton_myo`, perché `build_brighton_kb`
ricaricherebbe l'indice del paper della TVP senza leggere il testo che riceve.
Da riga di comando: `--guideline` accende il contesto della linea guida,
`--num-ctx` sostituisce `LLM_NUM_CTX`, `--brighton-pdf` indica il paper
(`myo/main.pdf` di default), più `--records-dir`, `--output-dir` e `--only`.

**`evaluate_myo.py`** valuta una run con `evaluate_predictions`, dopo aver
installato `models_myo` come `models` e puntato le cartelle di default su
`myo/data/records` e `myo/output_myo`.

**`run_dvt_stripped.py`** è il termine di confronto: esegue il corpus TVP con
**`strip()`**, che spegne in memoria hint, contesto, ancore, intestazioni di
sezione e regole cross-section, gli stessi componenti che il
braccio della miocardite non ha. Poi chiama `run_synthetic_records.main()`, di
cui accetta gli argomenti.
