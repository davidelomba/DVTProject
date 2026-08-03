# Documentazione del codice — DVTProject

Descrizione approfondita, file per file e funzione per funzione, di tutto il codice del progetto: `config.py`, `models.py`, `rag_setup.py`, `agents.py`, `pipeline.py`, `criteria_rules.py`, `agentic_graph.py`, `aggregation.py`, `main.py`.

Nota: `agentic_graph.py` era in origine un file sperimentale standalone (`experimental_agentic_graph_pipeline.py`), non collegato al resto della pipeline. È stato successivamente integrato come quarta modalità ufficiale (`config.EXTRACTOR_MODE == "agentic_graph"`), e il SOFT GATE / le regole cross-section — prima duplicate identiche sia in `pipeline.py` sia nel file experimental — sono state estratte in `criteria_rules.py`, unica fonte condivisa da entrambi i percorsi di esecuzione.

---

## 1. `config.py`

Nessuna funzione: è un modulo di sole costanti, letto da tutti gli altri file. Va inteso come il "pannello di controllo" centrale della pipeline.

**Blocco LLM locale.** `LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"` fissa il modello Ollama usato da Agent 2 (evaluator, in ogni modalità) e da Agent 1 nelle modalità `full_text`/`rag`. Il commento sopra spiega la motivazione: modelli medici specializzati (es. OpenBioLLM) sono stati scartati perché, con prompt di sistema complessi, tendevano a non rispettare i vincoli di formato (JSON, frasi trigger fisse), mentre un modello generico instruction-tuned si è dimostrato più affidabile. `LLM_TEMPERATURE = 0.0` rende l'output deterministico (stesso input → stessa risposta, niente campionamento casuale). `LLM_NUM_PREDICT = 512` è il tetto massimo di token generabili per risposta, per evitare che il modello continui a generare testo indefinitamente. `LLM_REQUEST_TIMEOUT = 180` (secondi) dà margine su hardware locale più lento prima di considerare la richiesta fallita.

`AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"` (aggiunta con l'integrazione della modalità a grafo): modello separato, usato **solo** dal nodo di ricerca di `agentic_graph.py`. Il commento spiega perché serve: `LLM_MODEL_NAME` (Llama 3 base) non supporta il tool-calling nativo di Ollama (Ollama risponde "model does not support tools", HTTP 400, se le si prova a legare un tool) — solo Llama 3.1+ lo supporta. Agent 2 non lega mai un tool, quindi continua a usare `LLM_MODEL_NAME` indipendentemente da `EXTRACTOR_MODE`. Prima dell'integrazione questa costante viveva come variabile locale nel file experimental; ora è parte della configurazione centrale.

**Blocco embeddings.** `EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"`: modello di embedding leggero e multilingue, scelto perché i referti sono in italiano e — spiega il commento — non vengono tradotti automaticamente in inglese prima dell'elaborazione, per evitare che una traduzione automatica distorca negazioni o terminologia clinica in modo non verificabile.

**Blocco modalità estrattore.** `EXTRACTOR_MODE = "full_text"` seleziona quale strategia di estrazione usa `pipeline.run_pipeline` (vedi sezione 5), tra tre possibili:
- `"full_text"`: passa l'intero referto al modello (default, il più testato).
- `"rag"`: retrieval a chunk fissi via `agents.extract_evidence`.
- `"agentic_graph"`: Agent 1 esplora il referto autonomamente con un tool di ricerca (`agents.extract_evidence_agentic`), orchestrato come macchina a stati esplicita con **LangGraph** (`agentic_graph.py`), e con un modello separato (`AGENTIC_LLM_MODEL_NAME`) per il solo passo di ricerca, dato che `LLM_MODEL_NAME` non supporta il tool-calling.

`"agentic_graph"` non è ancora validata come affidabile quanto `full_text`: la qualità/copertura del retrieval può variare da un'esecuzione all'altra perché è il modello stesso a decidere autonomamente come/quanto cercare. `AGENTIC_MAX_ITERATIONS = 5` limita quante volte l'agente può richiamare il tool prima di essere forzato a rispondere. (Una precedente modalità `"agentic"` — stessa ricerca autonoma ma dentro il semplice ciclo `for` di `pipeline.py`, senza LangGraph — è stata rimossa: `"agentic_graph"` la sostituisce interamente.)

**Blocco chunking EHR.** `EHR_CHUNK_SIZE = 800` e `EHR_CHUNK_OVERLAP = 150` sono i parametri di `RecursiveCharacterTextSplitter` usati per spezzare il referto in frammenti quando serve una vector store (modalità `rag`/`agentic_graph`); `EHR_RETRIEVER_K = 5` indica quanti chunk recuperare per query; `EHR_KB_PERSIST_DIR = "./chroma_ehr_kb"` è la cartella base dove Chroma salva l'indice (poi suffissata per paziente, vedi `rag_setup.build_ehr_kb`).

**Blocco chunking Brighton.** Stessa logica ma per il paper Brighton (`BRIGHTON_CHUNK_SIZE = 800`, `BRIGHTON_CHUNK_OVERLAP = 150`, `BRIGHTON_RETRIEVER_K = 3`, `BRIGHTON_KB_PERSIST_DIR = "./chroma_brighton_kb"`), knowledge base statica che non cambia tra pazienti.

**`SECTION_ORDER`.** Lista ordinata delle 10 sezioni del questionario da compilare: `["A1", "A2", "A3_1", "A3_2", "B1_1", "B1_2", "B2", "C", "F", "X"]`. Determina sia l'ordine di esecuzione nel ciclo di `pipeline.run_pipeline`, sia (in modalità `agentic_graph`) l'ordine di consumo della coda `remaining_sections`.

**`SECTION_KEYWORD_GATES`.** Dizionario con due voci, `A1` e `A2`. Ognuna definisce una lista di `keywords` (es. per A1: `["autops", "autoptic", "postmortem", "post-mortem", "necrosc"]`) e un `default_option_text` (la risposta negativa di quella sezione). È il "guardrail" deterministico applicato tramite `criteria_rules.apply_keyword_gate` (vedi sezione 6): se Agent 2 sceglie una risposta diversa dal default ma nel testo dell'evidenza non compare nessuna delle parole chiave, la risposta viene forzata al default, perché si presume un'allucinazione del modello (es. dedurre un'autopsia mai menzionata).

**`SECTION_HINTS`.** Dizionario di istruzioni extra in linguaggio naturale, iniettate nel prompt di Agent 2 solo per la sezione corrispondente, per correggere errori sistematici osservati empiricamente:
- `A1`: ribadisce che la domanda riguarda *solo* l'autopsia post-mortem, non l'imaging su paziente vivo.
- `A2`: richiama l'attenzione sulle negazioni prima di termini chirurgici.
- `A3_2`: impone di selezionare solo le modalità di imaging esplicitamente citate per *questo* paziente, senza farsi condizionare dall'elenco generico di modalità nel contesto Brighton.
- `B1_1`: distingue quando selezionare "nessun sintomo riportato" (negazione esplicita) da "sconosciuto" (assenza totale di informazione).
- `B2`: impone di selezionare solo i sintomi esplicitamente documentati per *questo* paziente (non quelli elencati genericamente nel contesto Brighton), e distingue esplicitamente "flusso assente" (reperto di imaging sul flusso venoso) da "polsi assenti" (reperto d'esame obiettivo arterioso, non equivalente).
- `C`: fissa la soglia di default del D-dimero a 500 ng/mL quando il referto non specifica il limite del laboratorio.
- `F`: chiarisce che la sezione richiede "Sì" solo se la diagnosi è riportata *senza* alcun dettaglio clinico di supporto.
- `X`: avverte di non confondere sintomi (dolore, edema) con una diagnosi alternativa vera e propria.

**`CROSS_SECTION_RULES`.** Lista di regole applicate *dopo* che tutte le sezioni sono state compilate indipendentemente, tramite `criteria_rules.apply_cross_section_rules` (vedi sezione 6). Attualmente una sola regola: se in `b2` (chiave minuscola, cioè il campo del form) è presente qualunque valore diverso da `"None of the above were present or it is unknown if any of 1-4 were present"`, allora `b1_1` viene forzato al valore `"≥1 symptom or sign of DVT was reported"`. Ogni regola ha anche `audit_key` (per scrivere una nota nell'audit log) e `override_message` (testo human-readable spiegato nel log).

---

## 2. `models.py`

Definisce gli schemi Pydantic che vincolano l'output di Agent 2. Non ci sono funzioni "attive": ogni classe è uno schema dichiarativo.

**`A1_Autopsy`, `A2_SurgicalProcedure`, `A3_1_ImagingOutcome`, `C_DDimer`, `F_ReportedBySpecialist`, `X_AlternativeDiagnosis`** seguono tutte lo stesso pattern: un unico campo `answer: Literal[opzione1, opzione2, ...]`, con `Field(description=...)` che documenta a quale domanda del questionario corrisponde. `Literal` obbliga Pydantic a rifiutare qualunque valore che non sia *esattamente* una delle stringhe elencate — è questo vincolo che rende sicuro il mapping numero→testo fatto in `agents._match_option`.

**`A3_2_ImagingStudies` e `B1_2_DVTType`** sono a scelta multipla: il campo (`studies` o `types`) è `List[Literal[...]]` con `default_factory=list` (lista vuota di default se nessuna opzione è selezionata, invece di `None`).

**`B2_NewSymptoms`** è il caso più complesso: campo `symptoms: List[Literal[...]]` con 5 opzioni, di cui l'ultima è `"None of the above were present or it is unknown if any of 1-4 were present"`. Il metodo `none_is_exclusive`, decorato con `@model_validator(mode="after")`, viene eseguito automaticamente da Pydantic dopo la costruzione dell'oggetto: controlla se l'opzione "nessuno dei precedenti" è presente insieme ad altre nella lista (`len(self.symptoms) > 1`) e, se sì, solleva `ValueError` — impedendo uno stato logicamente incoerente (es. "calf pain" + "nessuno dei precedenti" insieme).

**`DVT_CriteriaForm`** è il contenitore finale: un campo obbligatorio `record_id: str` più un campo opzionale (`... | None = None`) per ciascuna delle 10 sezioni, tipizzato con la classe Pydantic corrispondente. È l'oggetto restituito da `pipeline.run_pipeline`, indipendentemente da quale `EXTRACTOR_MODE` sia stato usato.

**`SECTION_MODELS`** è il dizionario `{"A1": A1_Autopsy, "A2": A2_SurgicalProcedure, ...}` che permette di risalire dalla chiave testuale della sezione (es. `"A3_1"`) alla classe Pydantic da istanziare, senza bisogno di un lungo `if/elif`.

---

## 3. `rag_setup.py`

Costruisce le due knowledge base vettoriali (Chroma) e i loader dei testi sorgente.

**`get_embeddings()`**: restituisce un'istanza di `HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, encode_kwargs={"prompt": "passage: "}, query_encode_kwargs={"prompt": "query: "})`. Il modello `intfloat/multilingual-e5-small` richiede, per un uso corretto, che i testi indicizzati siano prefissati con `"passage: "` e le query con `"query: "` — senza questi prefissi la qualità del ranking per similarità peggiora. Funzione isolata così che il modello di embedding venga istanziato una sola volta e passato esplicitamente alle altre funzioni, invece di essere ricreato ogni volta.

**`build_brighton_kb(brighton_pdf_text, embeddings=None, force_rebuild=False)`**: se `embeddings` non è passato, lo crea con `get_embeddings()`. Controlla se la cartella `config.BRIGHTON_KB_PERSIST_DIR` esiste già sul disco e `force_rebuild` è `False`: in tal caso ricarica l'indice esistente con `Chroma(persist_directory=..., embedding_function=embeddings)` invece di ricalcolare gli embedding (risparmio di tempo, dato che il paper Brighton non cambia mai). Altrimenti crea uno `splitter = RecursiveCharacterTextSplitter(chunk_size=config.BRIGHTON_CHUNK_SIZE, chunk_overlap=config.BRIGHTON_CHUNK_OVERLAP)`, lo usa per spezzare il testo in `chunks`, e costruisce l'indice da zero con `Chroma.from_texts(...)`, allegando a ogni chunk il metadato `{"source": "brighton_dvt_synonyms"}`.

**`build_ehr_kb(patient_record_text, patient_id, embeddings=None)`**: stessa logica di chunking ma per il referto del singolo paziente, con `config.EHR_CHUNK_SIZE`/`EHR_CHUNK_OVERLAP`. La cartella di persistenza è `f"{config.EHR_KB_PERSIST_DIR}_{patient_id}"` — suffissata per paziente, così run su pazienti diversi non si sovrascrivono a vicenda. Prima di ricreare l'indice, controlla `if os.path.isdir(persist_dir): shutil.rmtree(persist_dir)` — cancella esplicitamente la cartella se esiste già: il commento spiega che senza questa cancellazione, rilanciare la pipeline sullo stesso `patient_id` (tipico durante il debug) accumulerebbe chunk duplicati nello stesso indice ad ogni run, diluendo silenziosamente la qualità del retrieval nel tempo.

**`make_ehr_retriever_tool(ehr_vectorstore)`**: crea `ehr_retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})` e lo avvolge con `create_retriever_tool(ehr_retriever, "search_patient_record", ...)`, con una descrizione che copre esplicitamente tutti e 10 i domini di criteri del questionario (non solo "sintomi, referti chirurgici, date e risultati di laboratorio" come nella prima versione) — restituendo un oggetto `Tool` di LangChain, utilizzabile da un agente tool-calling. Usato quando `EXTRACTOR_MODE == "agentic_graph"` (chiamato in `pipeline.py`, il tool risultante è poi passato al nodo di ricerca di `agentic_graph.py`). Il docstring nota che richiede il pacchetto base `langchain` (non solo i sotto-pacchetti `langchain-core`/`community`/`ollama`).

**`load_brighton_pdf_text(pdf_path)`**: apre il PDF con `PdfReader(pdf_path)` e concatena `page.extract_text() or ""` per ogni pagina, unendo tutto con `"\n".join(...)`. L'`or ""` gestisce il caso in cui `extract_text()` restituisca `None` per una pagina non testuale.

**`load_ehr_text(txt_path)`**: apertura file di testo semplice in lettura UTF-8, restituisce il contenuto intero come stringa. Il docstring lascia una nota per sé stessi: se in futuro i referti arriveranno in PDF o DOCX, andrà aggiunto un loader equivalente qui (riusando l'approccio di `load_brighton_pdf_text` per PDF, o `python-docx` per Word).

---

## 4. `agents.py`

Il cuore della logica dei due agenti. Contiene sia le funzioni di estrazione (Agent 1) sia quella di valutazione (Agent 2).

### 4.1 Setup comune

**`build_llm(temperature=None)`**: costruisce e restituisce un `ChatOllama` con `model=config.LLM_MODEL_NAME`, temperatura di default `config.LLM_TEMPERATURE` (sovrascrivibile passando un valore esplicito), `num_predict=config.LLM_NUM_PREDICT` e `request_timeout=config.LLM_REQUEST_TIMEOUT`. È la factory usata per Agent 2 in ogni modalità, e per Agent 1 nelle modalità `full_text`/`rag`. In modalità `agentic_graph`, Agent 1 usa invece `agentic_graph.build_agentic_llm()` (modello diverso, tool-capable).

**`test_structured_output_support(llm, sample_model)`**: funzione diagnostica non usata dalla pipeline principale. Prova a chiamare `llm.with_structured_output(sample_model)` (l'API nativa di LangChain per output strutturato) e verifica con `isinstance(result, sample_model)` se il modello ha effettivamente rispettato lo schema. Se solleva un'eccezione, la cattura, stampa un avviso e restituisce `False`. Serve a spiegare (e verificare a runtime) perché il progetto non usa questa via diretta, preferendo invece il parsing manuale di `FINAL_ANSWER: <numero>` fatto in `evaluate_section`.

### 4.2 Agent 1 — Extractor

**`EXTRACTOR_SYSTEM_PROMPT`**: prompt di sistema condiviso da tutte e tre le funzioni di estrazione. Istruisce il modello a estrarre *solo* frammenti letterali dal referto (niente parafrasi, niente introduzioni/conclusioni), a riconoscere che il referto può essere in italiano, a preservare la lingua originale del frammento, e a rispondere esattamente con la stringa `"NO RELEVANT EVIDENCE FOUND."` se nulla è pertinente.

**`extract_evidence(llm, ehr_vectorstore, criterion_query)`** (modalità `rag`): crea un retriever con `k=config.EHR_RETRIEVER_K`, esegue `retriever.invoke(criterion_query)` per ottenere i documenti più simili, concatena i loro `page_content` con separatore `"\n---\n"` in `context`. Se `context` è vuoto dopo strip, ritorna direttamente una stringa di fallback senza nemmeno interpellare il modello. Altrimenti costruisce i messaggi (`system` = `EXTRACTOR_SYSTEM_PROMPT`, `human` = criterio + frammenti recuperati) e ritorna `response.content` dell'invocazione LLM. Non usata dalla pipeline di default (`EXTRACTOR_MODE = "full_text"`), pensata per referti troppo lunghi da passare per intero.

**`extract_evidence_full_text(llm, full_ehr_text, criterion_query)`** (modalità default): stessa struttura ma senza alcun retrieval — il messaggio "human" contiene l'intero `full_ehr_text` concatenato al criterio. Più semplice e priva del rischio di "retrieval miss" (chunk sbagliato recuperato), finché il referto sta nella finestra di contesto del modello.

**`AGENTIC_EXTRACTOR_SYSTEM_PROMPT`**: è `EXTRACTOR_SYSTEM_PROMPT` con in coda due blocchi di istruzioni aggiuntivi, introdotti in momenti diversi per correggere bug reali osservati:
1. **TOOL USE**: dice esplicitamente al modello che il referto *non* è nel messaggio, che esiste un tool chiamato `search_patient_record`, che è *obbligatorio* chiamarlo almeno una volta, e che non può rispondere `"NO RELEVANT EVIDENCE FOUND."` senza averlo prima interrogato. Motivo: senza questa istruzione, il modello rispondeva quella stringa di fallback su *tutte* le 10 sezioni senza mai invocare il tool.
2. **TRANSCRIPTION RULE**: vieta esplicitamente di parafrasare, tradurre o riassumere ciò che il tool restituisce, imponendo di ricopiare i frammenti letteralmente. Motivo: in test reali, senza questa regola, la risposta finale dell'agente dopo l'uso del tool tendeva a "raccontare" ciò che aveva trovato invece di citarlo, corrompendo le risposte a valle (es. una diagnosi "No" trasformata in "Yes", o un'ecocolordoppler etichettata erroneamente come "compression ultrasonography").

Usata dal nodo di ricerca di `agentic_graph.py` (modalità `agentic_graph`) tramite `extract_evidence_agentic`, sotto.

**`extract_evidence_agentic(llm, ehr_tool, ehr_vectorstore, criterion_query, max_iterations=3)`** (usata in modalità `agentic_graph`, chiamata dal nodo di ricerca di `agentic_graph.py`): costruisce un `ChatPromptTemplate` a tre messaggi — `system` (`AGENTIC_EXTRACTOR_SYSTEM_PROMPT`), `human` (placeholder `{input}`), `placeholder` (`{agent_scratchpad}`, dove LangChain inserisce la cronologia delle chiamate al tool). Crea `agent = create_tool_calling_agent(llm, [ehr_tool], prompt)` e lo avvolge in un `AgentExecutor` con `max_iterations=max_iterations`, `early_stopping_method="force"` (se il limite di iterazioni viene raggiunto, l'agente è forzato a produrre comunque una risposta invece di continuare a chiamare il tool) e `return_intermediate_steps=True`. Invoca l'executor passando come `input` sia il criterio da investigare sia un promemoria esplicito di usare il tool prima di rispondere.

A differenza della prima versione, **non** ritorna più `result["output"]` (il testo finale, in linguaggio naturale, prodotto dall'agente) — questo per un bug osservato ripetutamente su più run reali: l'agente a volte si fermava dopo che una ricerca aveva recuperato un chunk irrilevante per quella sezione (tipicamente accadeva sulla sezione B2, dove il chunk con i sintomi soggettivi del paziente veniva scartato a favore del chunk con D-dimero/ecocolordoppler, pur essendo entrambi recuperabili dallo stesso vector store). L'evidenza restituita ora è invece l'unione di due fonti grezze:
1. `agent_chunks`: il testo grezzo effettivamente restituito da ogni chiamata al tool durante la ricerca autonoma, letto da `result["intermediate_steps"]` (una lista di coppie `(azione, osservazione)`) — non il riassunto finale dell'agente, che poteva parafrasare o scartare informazioni.
2. `floor_chunks`: una retrieval deterministica a `top_k` fisso (`ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K}).invoke(criterion_query)`) sulla query di sezione — la stessa identica ricerca che farebbe la modalità `rag`, eseguita sempre e comunque, indipendentemente da cosa l'agente abbia deciso di cercare.

Le due liste vengono unite (`floor_chunks + agent_chunks`) e deduplicate per corrispondenza esatta di stringa, poi ricongiunte con `"\n---\n"`. In questo modo il retrieval a query fissa fa da "pavimento" di sicurezza: anche se in un run l'agente si ferma dopo una ricerca povera, l'evidenza include comunque il chunk rilevante che la ricerca deterministica avrebbe comunque trovato — la ricerca autonoma dell'agente può solo aggiungere copertura extra rispetto a quel minimo, mai toglierla.

### 4.3 Agent 2 — Evaluator

**`EVALUATOR_SYSTEM_PROMPT`**: istruisce il modello a determinare la risposta corretta basandosi solo sull'evidenza data, a consultare i sinonimi Brighton quando rilevanti, e soprattutto a prestare "estrema attenzione" alle negazioni (non trattare come presente qualcosa esplicitamente negato; non assumere assente qualcosa semplicemente non menzionato). Un secondo paragrafo tratta il caso specifico delle domande su UN metodo preciso (autopsia, un tipo di intervento chirurgico, una modalità di imaging): vieta di inferire che quel metodo specifico sia stato eseguito solo perché la DVT è stata confermata con un metodo *diverso* menzionato altrove nell'evidenza.

**`_get_field_info(section_model)`**: dato uno schema Pydantic di sezione, recupera il nome del suo unico campo (`field_name = next(iter(section_model.model_fields.keys()))`) e la relativa annotazione di tipo. Se l'annotazione è una `list` (controllato con `get_origin(annotation) is list`), estrae il tipo `Literal` interno con `get_args(annotation)[0]` e ne ricava le opzioni con `get_args(inner)`, segnalando `is_multi_select=True`. Altrimenti tratta l'annotazione come `Literal` diretto e ritorna `is_multi_select=False`. Questa funzione è ciò che permette a `evaluate_section` di essere generica per tutte le 10 sezioni senza bisogno di un `if` per ciascuna.

**`_build_reasoning_prompt(evidence_text, brighton_context, options, multi_select, extra_instructions="")`**: costruisce il prompt testuale per Agent 2. Le opzioni vengono numerate (`f"{i}. {opt}"` per ognuna, a partire da 1) — scelta motivata dal commento: far rispondere il modello con un *numero* invece che con il testo dell'opzione evita fuzzy-match sbagliati tra opzioni quasi identiche salvo una negazione. Il prompt include, in ordine: l'evidenza, il contesto Brighton (se presente), le istruzioni extra per-sezione (se presenti), il blocco delle opzioni numerate, e infine l'istruzione di formato finale — diversa a seconda che sia una domanda a singola scelta (`FINAL_ANSWER: <numero>`) o multi-select (`FINAL_ANSWER: <numero>; <numero>; ...`).

**`_extract_final_answer_line(text)`**: usa una regex (`r"FINAL_ANSWER:\s*(.+)"`) con `re.findall` per trovare tutte le occorrenze di quella riga nel testo di risposta, e prende specificamente *l'ultima* (`matches[-1]`) — non la prima — perché il commento nota che il modello a volte ripete/cita l'istruzione stessa prima di rispondere davvero. Se non trova alcuna occorrenza, solleva `ValueError`.

**`_match_option(raw_value, valid_options, cutoff=0.75)`**: converte il valore grezzo estratto in una delle stringhe valide dello schema. Prima pulisce la stringa (`strip`, rimozione di `-` iniziali e `.`/`;` finali). Se il risultato è numerico (`isdigit()`), lo interpreta come indice 1-based: se nel range valido ritorna `valid_options[idx - 1]`, altrimenti solleva `ValueError` con messaggio esplicito sull'indice fuori range. Se non è numerico, prova un match esatto con una delle opzioni; se fallisce anche quello, tenta un fuzzy match con `difflib.get_close_matches` (soglia di similarità `cutoff=0.75`), stampando un avviso esplicito perché un fuzzy match silenzioso rischierebbe di far atterrare la risposta sull'opzione opposta per negazione. Se nessuna via funziona, solleva `ValueError` finale.

**`evaluate_section(llm, section_model, evidence_text, brighton_context="", extra_instructions="", max_retries=2)`**: orchestratore di Agent 2 per una singola sezione, usato identicamente da tutte e tre le modalità (chiamato direttamente dal ciclo di `pipeline.py` per `full_text`/`rag`, e dal nodo `answer_criterion` di `agentic_graph.py` per `agentic_graph`). Chiama `_get_field_info` per ottenere nome campo/opzioni/multi-select, poi `_build_reasoning_prompt` per costruire il prompt iniziale. Entra in un ciclo `for attempt in range(max_retries + 1)` (quindi fino a 3 tentativi totali): invoca l'LLM con `system=EVALUATOR_SYSTEM_PROMPT` e `human=prompt`, prova a estrarre e mappare la risposta (`_extract_final_answer_line` + `_match_option`, gestendo separatamente il caso multi-select — dove ogni elemento separato da `;` viene mappato singolarmente, poi deduplicato preservando l'ordine con l'idioma `seen = set(); [m for m in matched if not (m in seen or seen.add(m))]`). Se tutto va a buon fine, ritorna subito `(section_model(**{field_name: matched}), content)` — istanzia lo schema Pydantic (che valida di nuovo il valore) insieme al testo di ragionamento completo del modello. Se il parsing/matching fallisce, cattura l'eccezione, la salva in `last_error`, e **aggiunge** al prompt (non lo sostituisce) un paragrafo che spiega l'errore e ripete l'istruzione di formato, prima di ritentare. Se tutti i tentativi falliscono, solleva `RuntimeError` con l'ultimo errore registrato.

---

## 5. `pipeline.py`

Orchestratore principale, usato da `main.py`, che dispatcha su tutte e tre le modalità di `config.EXTRACTOR_MODE`.

**`SECTION_QUERIES`**: dizionario `{sezione: query_testuale}` — per ognuna delle 10 sezioni definisce cosa Agent 1 deve cercare nel referto (es. per `C`: `"D-dimer value, test date, laboratory upper limit of normal"`). Usata sia come query di estrazione sia come query di retrieval sul Brighton KB, sia dal ciclo di `pipeline.py` sia (passata esplicitamente come parametro) da `agentic_graph.run_agentic_graph_pipeline`.

**`run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)`**: funzione principale, restituisce `(form, audit_log)`.

*Setup iniziale*: crea `embeddings` e `llm`, carica il testo del PDF Brighton e del referto, costruisce `brighton_kb` (sempre). Se `config.EXTRACTOR_MODE` è `"rag"` o `"agentic_graph"`, costruisce anche `ehr_kb`; se è `"agentic_graph"`, costruisce in più il tool di ricerca `ehr_tool` via `make_ehr_retriever_tool` — altrimenti queste variabili restano `None` e non vengono usate.

*Dispatch su `agentic_graph`*: se `config.EXTRACTOR_MODE == "agentic_graph"`, la funzione costruisce il modello di ricerca separato (`search_llm = build_agentic_llm()`, importata in cima al modulo da `agentic_graph.py`) e delega l'intera esecuzione delle 10 sezioni a `run_agentic_graph_pipeline(...)`, passando anche `ehr_vectorstore=ehr_kb` (lo stesso oggetto Chroma già costruito sopra, usato dal nodo di ricerca per il retrieval deterministico di sicurezza -- vedi sezione 4.2) oltre a `ehr_tool`. Ritorna `(form_data, audit_log)` con la stessa forma prodotta dal ciclo `for` (vedi sezione 7). L'import da `agentic_graph.py` è un normale import di modulo, in cima a `pipeline.py`: non c'è alcun ciclo da evitare, dato che `agentic_graph.py` non importa nulla da `pipeline.py` (`SECTION_QUERIES` gli viene passato come parametro esplicito, non importato).

*Ciclo principale* (per le altre due modalità, `full_text` e `rag`): per ogni `section_key` in `config.SECTION_ORDER`, recupera lo schema (`SECTION_MODELS[section_key]`) e la query (`SECTION_QUERIES[section_key]`), poi entra in un blocco `try`:
1. **Agent 1**: a seconda di `config.EXTRACTOR_MODE`, chiama `extract_evidence` o `extract_evidence_full_text`, cronometrando il tempo con `time.time()`. Il risultato (`evidence`) viene salvato in `section_log["evidence"]`.
2. **Contesto Brighton**: interroga `brighton_kb.as_retriever(search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}).invoke(query)`, concatena i `page_content` in `brighton_context`.
3. **Agent 2**: chiama `evaluate_section` passando anche `config.SECTION_HINTS.get(section_key, "")` come istruzioni extra. Cronometra e stampa il tempo impiegato.
4. **Gate a parole chiave**: chiama `criteria_rules.apply_keyword_gate(section_key, section_result, evidence, reasoning_text)` (vedi sezione 6), che ritorna eventualmente un `section_result`/`reasoning_text` corretti se il gate scatta.
5. Salva `section_log["reasoning"]`, `section_log["result"]` (via `.model_dump()`) e popola `form_data[section_key.lower()]` con l'oggetto Pydantic.

Se una qualunque eccezione viene sollevata durante questi passi, il blocco `except` stampa il traceback, imposta `form_data[section_key.lower()] = None` e registra l'errore in `section_log["error"]` — così il fallimento di una sezione non compromette le altre. In ogni caso, `audit_log[section_key] = section_log` viene eseguito fuori dal try/except.

*Regole cross-section*: dopo il branch (indipendentemente da quale modalità l'abbia prodotto), chiama `form_data = criteria_rules.apply_cross_section_rules(form_data, audit_log)` (vedi sezione 6) — un'unica chiamata condivisa da tutte e tre le modalità, invece di una copia duplicata dentro ciascun percorso di esecuzione.

*Chiusura*: costruisce `form = DVT_CriteriaForm(**form_data)` e ritorna `(form, audit_log)`.

---

## 6. `criteria_rules.py`

Modulo nuovo, introdotto per eliminare la duplicazione di codice tra `pipeline.py` (ciclo `for` per `full_text`/`rag`) e `agentic_graph.py` (nodi del grafo per `agentic_graph`): prima queste due funzioni di sicurezza erano copiate identiche in entrambi i file; ora hanno un'unica fonte, importata da entrambi.

**`apply_keyword_gate(section_key, section_result, evidence, reasoning_text)`**: implementa `config.SECTION_KEYWORD_GATES`. Recupera il nome dell'unico campo di `section_result` (`list(type(section_result).model_fields.keys())[0]`) e il valore scelto da Agent 2. Determina `is_positive` confrontando quel valore con il `default_option_text` della sezione (gestendo sia il caso singolo sia lista, con `any(...)` per le liste). Se `is_positive`, controlla se una qualunque delle `keywords` compare (case-insensitive) in `evidence`; se **nessuna** compare, ricostruisce `section_result` forzandolo al default negativo — usando il costruttore Pydantic (`type(section_result)(**{...})`), non `setattr`, così il valore forzato ripassa comunque dalla validazione dello schema — e appende una nota `"[SYSTEM OVERRIDE]"` a `reasoning_text`. Ritorna sempre la coppia `(section_result, reasoning_text)`, invariata se il gate non scatta o la sezione non ne ha uno.

**`apply_cross_section_rules(form_data, audit_log)`**: implementa `config.CROSS_SECTION_RULES`. Per ogni regola, recupera `if_result`/`then_result` da `form_data` (saltando se uno dei due è `None`, ad esempio perché quella sezione è fallita), normalizza `if_answers` a lista se non lo è già, e controlla `has_non_default = any(ans != rule["none_option"] for ans in if_answers)`. Se vero e il valore corrente di `then_result` è diverso da `rule["forced_value"]`, ricostruisce quel campo (stesso principio: costruttore Pydantic, non `setattr`) con il valore forzato e aggiunge una nota `"[SYSTEM OVERRIDE]"` al reasoning già presente in `audit_log[rule["audit_key"]]`. Muta e ritorna `form_data`.

Entrambe le funzioni sono invocate una sola volta per punto di applicazione: `apply_keyword_gate` dentro il ciclo per-sezione (sia quello di `pipeline.py` sia il nodo `answer_criterion` di `agentic_graph.py`); `apply_cross_section_rules` una sola volta in `pipeline.run_pipeline`, dopo che `form_data` è stato prodotto — da qualunque modalità, incluso `agentic_graph` — e prima di costruire `DVT_CriteriaForm`.

---

## 7. `agentic_graph.py`

Implementazione della modalità `EXTRACTOR_MODE == "agentic_graph"`: Agent 1 esplora il referto autonomamente con un tool di ricerca, orchestrato come macchina a stati esplicita con **LangGraph** invece che con un ciclo Python semplice. In origine era un file sperimentale standalone (`experimental_agentic_graph_pipeline.py`) che non modificava né si integrava con `pipeline.py`; ora è importato normalmente (in cima al modulo) da `pipeline.py` ed è a tutti gli effetti l'unica modalità agentic di produzione, selezionabile semplicemente impostando `config.EXTRACTOR_MODE = "agentic_graph"` e lanciando `python main.py` come al solito. (Una precedente modalità `"agentic"`, che faceva la stessa ricerca autonoma ma dentro il semplice ciclo `for` di `pipeline.py` senza LangGraph, è stata rimossa: questa modalità la sostituisce interamente.)

**`build_agentic_llm()`**: costruisce e restituisce un `ChatOllama` con `model=config.AGENTIC_LLM_MODEL_NAME` (non più una costante locale come nella versione sperimentale, ma letta da `config.py`), riusando gli stessi `temperature`/`num_predict`/`request_timeout` di `config.py`. Usato *solo* per il nodo di ricerca (`search_record`); il nodo di valutazione (`answer_criterion`) riceve invece il modello standard `agents.build_llm()`, passato dall'esterno da `pipeline.py`.

**`GraphState`**: `TypedDict` che definisce la forma dello stato che circola tra i nodi del grafo: `record_id`, `remaining_sections` (coda delle sezioni ancora da fare), `current_section`, `form_data`, `audit_log`, `done` (flag di terminazione).

**`_select_next(state)`**: nodo del grafo. Se `remaining_sections` è vuota, ritorna lo stato con `current_section=None, done=True`. Altrimenti "pop" (senza mutare in place: `remaining[1:]`) del primo elemento, lo assegna a `current_section`, stampa l'intestazione `=== Section {sezione} ===` e ritorna lo stato aggiornato con `done=False`.

**`_route_after_select(state)`**: funzione di routing condizionale, non un nodo vero e proprio — ritorna la stringa `"finalize"` se `state["done"]`, altrimenti `"search_record"`. Usata da `graph.add_conditional_edges`.

**`_make_search_node(llm, ehr_tool, ehr_vectorstore, section_queries)`**: *factory* che chiude su `llm`, `ehr_tool`, `ehr_vectorstore` (l'oggetto Chroma passato da `pipeline.py`, usato per il retrieval deterministico di sicurezza -- vedi sezione 4.2) e sul dizionario `section_queries` (passato esplicitamente da `pipeline.py`, non più importato direttamente da `pipeline.SECTION_QUERIES` per evitare l'accoppiamento diretto al momento dell'import) e ritorna la funzione-nodo `search_record`. Quest'ultima recupera la query per `current_section`, chiama `extract_evidence_agentic` (con `max_iterations=config.AGENTIC_MAX_ITERATIONS`) dentro un `try/except` (in caso di eccezione stampa l'errore e imposta `evidence=None` invece di interrompere il grafo), e salva `{"query": ..., "evidence": ...}` in `audit_log[section_key]`.

**`_make_answer_node(llm, brighton_kb, section_queries)`**: factory analoga per il nodo `answer_criterion`. Recupera l'evidenza salvata dal nodo precedente; se è vuota/assente (`if not evidence`), salta direttamente la valutazione, imposta il campo a `None` e registra un errore esplicito ("Agent 1 (agentic) produced no evidence"). Altrimenti: recupera il contesto Brighton (stessa logica di `pipeline.py`), chiama `evaluate_section`, applica il gate a parole chiave chiamando **`criteria_rules.apply_keyword_gate`** (non più una copia locale duplicata, come nella versione sperimentale), e infine popola `form_data`/`audit_log`. Un `try/except` esterno cattura eventuali fallimenti di Agent 2 senza interrompere il grafo.

**`_finalize(state)`**: nodo finale del grafo. Nella versione sperimentale applicava qui le regole cross-section (con una copia locale duplicata della logica di `pipeline.py`); ora è un semplice passthrough che ritorna lo stato invariato, perché `pipeline.run_pipeline` applica `criteria_rules.apply_cross_section_rules` una sola volta, dopo aver ricevuto `form_data` da questo modulo — stessa logica, unica fonte, indipendentemente dalla modalità.

**`build_graph(search_llm, answer_llm, ehr_tool, ehr_vectorstore, brighton_kb, section_queries)`**: assembla il grafo. Crea `StateGraph(GraphState)`, aggiunge i 4 nodi (`select_next`, `search_record`, `answer_criterion`, `finalize`), imposta `select_next` come punto di ingresso (`set_entry_point`), collega `select_next` con un arco condizionale (`add_conditional_edges`) che instrada verso `search_record` o `finalize` in base a `_route_after_select`, poi gli archi fissi `search_record → answer_criterion → select_next` (chiudendo il ciclo) e `finalize → END`. Ritorna il grafo compilato (`graph.compile()`), pronto per essere invocato.

**`run_agentic_graph_pipeline(record_id, evaluator_llm, search_llm, ehr_tool, ehr_vectorstore, brighton_kb, section_queries)`**: punto d'ingresso chiamato da `pipeline.run_pipeline` (sostituisce la vecchia `run_experimental_pipeline`, che invece costruiva da sé embeddings/KB e restituiva direttamente un `DVT_CriteriaForm` già completo di regole cross-section). Costruisce il grafo con `build_graph(...)`, prepara `initial_state` con `remaining_sections = list(config.SECTION_ORDER)`, e invoca il grafo con `recursion_limit = len(config.SECTION_ORDER) * 3 + 10` (margine esplicito perché il limite di default di LangGraph, 25, sarebbe insufficiente per 10 sezioni × 3 passi ciascuna). Ritorna direttamente `(final_state["form_data"], final_state["audit_log"])` — **non** un `DVT_CriteriaForm` e **senza** aver applicato le regole cross-section: entrambi questi passi restano centralizzati in `pipeline.run_pipeline`, uguali per tutte le modalità.

---

## 8. `aggregation.py`

Un solo modulo minimale.

**`form_to_json_summary(form)`**: chiama `form.model_dump(exclude_none=True)` — serializza il form Pydantic in un dizionario, escludendo tutti i campi rimasti `None` (cioè le sezioni fallite durante `run_pipeline`), così l'output JSON pulito contiene solo le sezioni effettivamente compilate.

---

## 9. `main.py`

Entry point della pipeline, unico per tutte e tre le modalità (basta cambiare `config.EXTRACTOR_MODE`, non serve toccare `main.py`).

**`main()`**: fissa `record_id = "PATIENT_001"` e i percorsi hardcoded del referto (`./patient_001.txt`) e del PDF Brighton (`./1-s2.0-S0264410X22010854-main.pdf`). Chiama `run_pipeline(...)`, ottenendo `(form, audit_log)`. Serializza il form con `form_to_json_summary`, lo stampa a schermo in JSON indentato. Crea la cartella `./output` se non esiste (`os.makedirs(..., exist_ok=True)`). Salva due file distinti: `output/PATIENT_001.json` (il riassunto pulito) e `output/PATIENT_001_audit_log.json` (l'intero audit log, incluse le sezioni fallite) — quest'ultimo tenuto separato apposta, come spiega il commento, così non deve essere condiviso a valle ma resta disponibile per verificare manualmente una risposta specifica senza dover rilanciare tutta la pipeline. Il blocco `if __name__ == "__main__": main()` rende lo script eseguibile direttamente.
