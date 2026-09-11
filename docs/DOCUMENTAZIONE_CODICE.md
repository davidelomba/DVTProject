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
| `config.py` | costanti, prompt hint, gate, regole cross-section |
| `models.py` | schema Pydantic delle 10 sezioni, unica fonte di verità sulle opzioni |
| `rag_setup.py` | embedding, vector store, loader, tool di ricerca |
| `agents.py` | i due agenti e il parsing delle risposte |
| `criteria_rules.py` | post-processing deterministico |
| `agentic_graph.py` | macchina a stati LangGraph della modalità agentica |
| `pipeline.py` | orchestrazione di un referto |
| `aggregation.py` | serializzazione del form |
| `main.py` | entry point su un singolo referto |
| `run_synthetic_records.py` | esecuzione batch sul corpus |
| `generate_synthetic_records.py` | generazione e audit del corpus sintetico |
| `evaluate_predictions.py` | valutazione contro la ground truth |
| `compare_runs.py` | confronto tra due run, misura del rumore |
| `export_redcap_csv.py` | conversione dei risultati in CSV per REDCap |

---

## 1. `config.py`

Modulo di sole costanti, letto da tutti gli altri. È il pannello di controllo
della pipeline: ogni esperimento fatto finora è stato una modifica a questo file.

### Modelli e generazione

Tre ruoli, tre costanti separate, così ognuno si cambia indipendentemente:

- `LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"` — Agent 1 nelle modalità
  `full_text` e `rag`.
- `EVALUATOR_LLM_MODEL_NAME = "qwen3.6:27b"` — Agent 2, in ogni modalità.
- `AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"` — il solo passo di
  ricerca in modalità `agentic_graph`. Serve un modello separato perché Llama 3
  base non supporta il tool calling nativo di Ollama, che risponde
  `model does not support tools` se gli si lega un tool; Llama 3.1 sì.

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
- `LLM_REQUEST_TIMEOUT = 180` secondi.

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
  esplicita. È la modalità di riferimento; le altre due sono le baseline.
  `AGENTIC_MAX_ITERATIONS = 5` limita le chiamate al tool per sezione.

### Chunking

`EHR_CHUNK_SIZE = 800`, `EHR_CHUNK_OVERLAP = 150`, `EHR_RETRIEVER_K = 5` per il
referto; `BRIGHTON_CHUNK_SIZE`, `BRIGHTON_CHUNK_OVERLAP`, `BRIGHTON_RETRIEVER_K`
con gli stessi valori per il paper. `EHR_KB_PERSIST_DIR` e
`BRIGHTON_KB_PERSIST_DIR` sono le cartelle di Chroma.

### `SECTION_ORDER`

`["A1", "A2", "A3_1", "A3_2", "B1_1", "B1_2", "B2", "C", "F", "X"]`. Determina
l'ordine di esecuzione in entrambi i percorsi, il ciclo di `pipeline.py` e la
coda del grafo.

### `SECTION_GATES_ENABLED`

Tre interruttori per il post-processing deterministico, così un'ablazione non
richiede modifiche al codice:

```python
{"keyword": True, "details": False, "absent_pulses": True}
```

`details` è spento: la sua mappatura tratta l'assenza di dettagli come una
conclusione nuda, il che è falso quando nessuna diagnosi è stata riportata.

Le regole cross-section sono deliberatamente **fuori** da questi interruttori e
si applicano sempre: codificano la struttura del modulo, non una debolezza del
modello.

### `SECTION_KEYWORD_GATES`

Tre voci, `A1`, `A2` e `X`. Ognuna ha una lista di `keywords` e un
`default_option_text`, cioè la risposta negativa della sezione. Se Agent 2 dà una
risposta che le parole chiave sono autorizzate a controllare e nessuna di quelle
parole compare nell'evidenza, la risposta viene riportata al default.

Il campo opzionale `gated_options` nomina le risposte su cui le parole chiave
hanno voce. `A2` lo usa per elencare la sola opzione "thrombectomy": senza,
l'opzione "Other procedure done that confirmed presence of DVT" veniva respinta
su ogni referto perché non conteneva parole di trombectomia.

Il gate è **unidirezionale per costruzione**: può solo rimuovere un positivo non
supportato, mai aggiungerne uno mancante. La presenza di una parola chiave non
implica una risposta positiva, dato che l'evidenza potrebbe negare la procedura.

`X` elenca le condizioni concorrenti della Tabella 2 del paper. La maggior parte
condivide una radice greco-latina tra italiano e inglese (`cellulit-`,
`vasculit-`, `cirrosi`/`cirrhosis`); dove non accade sono elencati entrambi i
termini.

### `SECTION_HINTS` e i suoi interruttori

`SECTION_HINTS` associa a una sezione un testo aggiunto al prompt di Agent 2.
Ogni hint affronta un punto in cui il valutatore legge la domanda diversamente
dal questionario. Le sezioni assenti dal dizionario vengono risposte con le sole
opzioni e il contesto della linea guida.

Due interruttori permettono l'ablazione senza toccare il codice:

- `SECTION_HINTS_ENABLED = True` — interruttore generale. Nota che l'hint di F
  chiede la riga `DETAILS_PRESENT` che il details gate legge, quindi spegnere
  gli hint disattiva di fatto anche quel gate.
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
| A3_1 | come leggere l'esito dell'imaging |
| A3_2 | un esame che non è nessuna delle quattro modalità nominate è "Other" |
| B1_1 | la differenza tra seconda e terza opzione riguarda ciò che sai del paziente, non del documento: l'assenza di un referto non è un referto di assenza |
| B2 | *disattivato* |
| C | usa l'intervallo del laboratorio se il referto lo indica, altrimenti 500 ng/mL |
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

`A1_Autopsy`, `A2_SurgicalProcedure`, `A3_1_ImagingOutcome`, `C_DDimer`,
`F_ReportedBySpecialist`, `X_AlternativeDiagnosis` seguono lo stesso schema: un
unico campo `answer: Literal[...]`. `Literal` fa rifiutare a Pydantic qualunque
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
`criteria_rules.apply_details_gate`.

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

**`load_ehr_text(...)`** legge il referto da un `.txt` in UTF-8.

---

## 4. `agents.py`

I due agenti e tutto il parsing delle risposte.

### 4.1 Costruzione dei modelli

**`build_llm(model_name=None, temperature=None, num_predict=None)`** è la
factory unica per ogni modello non tool-calling del progetto. Legge i default da
`config.py` e passa `num_gpu`; il parametro `reasoning` viene inviato **solo**
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
`(field_name, options, is_multi_select)`. È ciò che rende `evaluate_section`
generica per tutte e dieci le sezioni senza un ramo per ciascuna.

**`_build_reasoning_prompt(...)`** compone il prompt. Le opzioni sono numerate e
il modello risponde **due volte**: il testo dell'opzione su `FINAL_OPTION` e il
suo numero su `FINAL_ANSWER`. Due risposte invece di una rendono visibile un
disaccordo, dato che il modello a volte nomina un'opzione e scrive il numero di
un'altra. Il prompt chiede inoltre esplicitamente che ogni opzione elencata sia
tracciabile a una frase del ragionamento, e non inclusa per default o per
margine di sicurezza.

Per le sezioni multi-scelta prive di un'opzione "nessuna delle precedenti", cioè
A3.2 e B1.2, il prompt aggiunge come dire che nulla si applica: `FINAL_OPTION:
none` e `FINAL_ANSWER: none`. Senza questa istruzione, non avendo modo di
esprimere una risposta vuota, il modello selezionava tutte le opzioni. La cosa è
esplicitata solo dove serve, così B2 continua a usare la propria opzione 5.

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

Reti di sicurezza deterministiche applicate sopra l'output dei due agenti.
Nessuna funzione qui chiama un modello: ognuna o mantiene la risposta di Agent 2
o la sostituisce con un valore derivato meccanicamente dall'evidenza o da
un'altra sezione. Ogni override lascia una nota `[SYSTEM OVERRIDE]` nel testo di
ragionamento, così una risposta forzata non è mai indistinguibile da una
prodotta dal modello — ed è questa proprietà che rende possibile ricostruire
l'effetto di un gate dagli audit log senza rieseguire nulla.

**`apply_keyword_gate(...)`** implementa `SECTION_KEYWORD_GATES`. Legge il campo
unico dello schema, così funziona su scelta singola e multipla senza un ramo per
ciascuna, determina se la risposta è fra quelle che le parole chiave possono
controllare, e in caso di assenza di ogni parola chiave ricostruisce l'istanza sul
default negativo. La ricostruzione passa dal costruttore Pydantic e non da
`setattr`, così il valore forzato viene rivalidato contro lo schema.

**`apply_details_gate(...)`** riguarda la sola sezione F. Legge la riga
`DETAILS_PRESENT` che l'hint di F chiede al modello — un giudizio fattuale,
non una mappatura sullo schema — e ne deriva meccanicamente l'etichetta:
dettagli presenti implica `"No"`, assenti implica `"Yes"`. Se la riga manca, la
funzione non fa nulla e si fida del modello anziché far fallire la sezione.
Attualmente il gate è spento in `config.SECTION_GATES_ENABLED`.

**`apply_absent_pulses_gate(...)`** rimuove da B2 la sola opzione
`"Absent pulses in legs or arms"` quando nell'evidenza non compare alcun esame
dei polsi. Il modello la sceglieva sulla base del solo linguaggio dell'imaging,
ragionando che un flusso assente al Doppler implichi polsi assenti: sono reperti
diversi, uno di imaging vascolare e uno di esame obiettivo. È ristretto a questa
singola coppia sezione-opzione e non generalizzato, perché l'evidenza che la
distingue è quasi non ambigua — le parole `polso`, `polsi`, `pulse` compaiono o
no — mentre le altre opzioni di B2 e le modalità di A3.2 variano troppo nella
formulazione perché una lista corta di parole chiave sia sicura.

**`apply_section_gates(...)`** applica in ordine i gate abilitati. Con tutti
disattivati è la funzione identità, cioè la risposta grezza del modello.

**`apply_cross_section_rules(form_data, audit_log)`** applica
`CROSS_SECTION_RULES` una volta sola, dopo che tutte le sezioni sono state
compilate, indipendentemente dalla modalità che le ha prodotte. Richiede la
presenza della sola sezione **sorgente**: una risposta forzata derivata da una
sezione mancante sarebbe infondata, mentre il bersaglio può essere `None`, dato
che il valore forzato viene dalla regola e la classe si legge da
`SECTION_MODELS`. Questo permette a una regola di riempire una sezione che una
valutazione fallita aveva lasciato vuota. Sovrascrive solo quando il valore
corrente differisce da quello forzato, per non riempire il log di voci in cui
Agent 2 era già d'accordo.

---

## 6. `agentic_graph.py`

Implementa `EXTRACTOR_MODE == "agentic_graph"` come macchina a stati LangGraph
esplicita, dove ogni passo è una funzione con il proprio stato in ingresso e in
uscita, invece che un ciclo Python.

Forma del grafo:

```
select_next -> {search_record, finalize} -> answer_criterion -> select_next
```

**`build_agentic_llm()`** costruisce il modello tool-calling. È separata da
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
continuato a ritentare resta visibile nel log. Il nodo di risposta applica gli
stessi gate per-sezione di ogni altra modalità.

**`_finalize(state)`** è un passthrough deliberato: le regole cross-section
vengono applicate una sola volta da `pipeline.run_pipeline` dopo che il grafo ha
restituito, così ogni modalità passa dallo stesso codice invece che da una copia.

**`run_agentic_graph_pipeline(...)`** è il punto d'ingresso. Il limite di
ricorsione è calcolato da `len(SECTION_ORDER) * 3 + 10`, perché ogni sezione
attraversa tre nodi. Restituisce `(form_data, audit_log)` nella stessa forma del
ciclo semplice, senza aver costruito il form né applicato le regole
cross-section.

---

## 7. `pipeline.py`

Orchestra un referto e dispatcha sulle tre modalità.

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

**`_run_config_snapshot()`** cattura tutto ciò che determina cosa una run
produce: modalità, gate, hint e loro fingerprint, modelli per ruolo (con
l'estrattore **effettivo**, che in modalità agentica è un modello diverso),
ambiente, parametri di generazione, parametri di retrieval. Viene scritto
nell'audit log sotto `_run_config`, chiave scelta per non poter collidere con un
nome di sezione, così un file di risultati è auto-descrittivo mesi dopo.

**`run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)`** costruisce una
sola volta embedding e valutatore, carica i due testi, costruisce la KB Brighton
sempre e quella del referto solo dove serve. Il modello di Agent 1 viene
costruito **dentro** il ramo che lo usa, così una modalità che non lo interroga
non lo carica in VRAM.

Nel ciclo per-sezione, ogni sezione passa da Agent 1, dal recupero del contesto
Brighton ripulito, da Agent 2 e dai gate. Il fallimento di una sezione non
compromette le altre: il campo resta `None`, l'errore va in `section_log["error"]`
e l'ultima risposta del modello, se disponibile, in `section_log["reasoning"]`.

Alla fine applica le regole cross-section una volta sola, aggiunge lo snapshot di
configurazione e costruisce `DVT_CriteriaForm`.

---

## 8. `aggregation.py` e `main.py`

**`aggregation.form_to_json_summary(form)`** serializza il form con
`exclude_none=True`: le sezioni lasciate `None` da una valutazione fallita
vengono omesse invece che scritte come `null`, così una risposta mancante è
assente e non somiglia a una risposta di "nessuno".

**`main.py`** esegue la pipeline su un singolo referto, con i percorsi ancorati
alla posizione del file e non alla directory di lancio. Scrive due file che
condividono lo stesso timestamp, il risultato e l'audit log, così una coppia è
sempre associabile e rilanciare non sovrascrive mai una run precedente.

---

## 9. `run_synthetic_records.py`

Esegue la pipeline su ogni referto del corpus, con la stessa convenzione di nomi
di `main.py` così la valutazione li trova con i percorsi di default.

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

## 10. `generate_synthetic_records.py`

Genera i referti sintetici in italiano con la ground truth corrispondente.

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

**Un secondo limite, sui dati.** Un referto è di circa 1000 caratteri, cioè circa
un chunk, mentre il retriever ne chiede 5: il retrieval restituisce ogni volta il
referto intero. Le tre modalità di estrazione danno quindi ad Agent 2 lo stesso
input e non sono confrontabili su questo dataset.

---

## 11. `evaluate_predictions.py`

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
senza lo stack della pipeline installato.

---

## 12. `compare_runs.py`

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

---

## 13. `export_redcap_csv.py`

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
