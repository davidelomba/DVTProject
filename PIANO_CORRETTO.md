# Piano d'azione corretto — Compilazione automatica modulo DVT (SEVALID_DVT)

## 0. Differenze rispetto al piano originale

Il piano originale (Step 1-5) è strutturalmente corretto ma copre solo i Criteri B e C
con Pydantic, assume tool-calling agentico affidabile su un modello 8B quantizzato senza
verifica preventiva, e si ferma alla produzione dei JSON per sezione senza definire come
questi si combinino nel Level of Certainty (LOC) finale, che è presumibilmente l'output
clinico reale del form. Le correzioni sotto integrano questi tre punti.

## 1. Setup ambiente (invariato, con un test aggiuntivo)

Stesso stack: Ollama + Llama3-OpenBioLLM-8B o BioMistral (GGUF Q4, ~5.5 GB RAM),
`HuggingFaceEmbeddings` con `BAAI/bge-small-en-v1.5` (~130 MB), Chroma come vector store.

**Aggiunta — Step 0 obbligatorio prima di costruire la pipeline:**
verificare che `.with_structured_output()` di LangChain funzioni nativamente con il
modello scelto via Ollama. Molti modelli clinici fine-tuned (OpenBioLLM, BioMistral)
non sono addestrati per function-calling strutturato: se il test fallisce, il fallback
è forzare l'output JSON via prompt + parsing/validazione Pydantic con retry, invece di
affidarsi al tool-calling nativo. Vedi `agents.py` per l'implementazione del test e del
fallback.

## 2. Mappatura completa delle "crocette" in Pydantic

Il modulo REDCap reale contiene più sezioni di quelle coperte nel piano originale.
Inventario completo delle domande a scelta (crocette), escludendo tutti i campi di
testo libero (date, descrizioni, "specify other"):

| Sezione | Domanda | Tipo | Opzioni |
|---|---|---|---|
| A1 | Autopsia | scelta singola | 3 |
| A2 | Procedura chirurgica | scelta singola | 3 |
| A3.1 | Imaging — esito | scelta singola | 3 |
| A3.2 | Imaging — quali studi | multi-select | 5 (incl. "Other", testo escluso) |
| B1.1 | Sintomi riportati | scelta singola | 3 |
| B1.2 | Tipo di DVT | multi-select | 2 |
| B2 | Sintomi clinici nuovi | multi-select | 5 (incl. "None...") |
| C | D-Dimero | scelta singola | 3 |
| F | Riportato da specialista | scelta singola (Yes/No) | 2 |
| X | Diagnosi alternativa | scelta singola | 2 |

Tutte queste sono ora modellate in `models.py` (nel piano originale erano presenti
solo B e C). Aggiunto inoltre un **validator** su B2: se è selezionata l'opzione
"None of the above...", nessun'altra opzione può essere co-selezionata (mutua
esclusività logica non garantita dallo schema Pydantic di base).

## 3. Sistema RAG (invariato nell'impostazione, due flussi)

- **KB statica**: paper Brighton (sinonimi DVT) → Chroma, non cambia mai.
- **KB dinamica**: cartella clinica del paziente → chunking (`RecursiveCharacterTextSplitter`,
  chunk ~800, overlap ~150) → Chroma temporaneo per singolo paziente.

## 4. Architettura dei due agenti — con fallback di robustezza

**Agente 1 (Estrattore):** di default usa **retrieval diretto** (similarity search +
prompt con contesto iniettato), non un agente ReAct con tool-loop. Il tool-calling
agentico resta disponibile come opzione (`agents.py` la implementa) ma va attivato solo
se il test dello Step 0 conferma che il modello lo gestisce in modo affidabile — su un
8B quantizzato il rischio di parsing incoerente del tool-call è concreto.

**Agente 2 (Valutatore):** invariato nell'impostazione — `.with_structured_output()`
sul modello Pydantic della sezione corrente, temperatura 0. Aggiunti **few-shot
examples** nel prompt per la gestione delle negazioni ("nessun edema" → None), che era
già segnalata come criticità nel piano originale ma senza contromisura concreta.

Entrambi gli agenti operano a `temperature=0` (nel piano originale era specificato solo
per l'Agente 2).

## 5. Loop di esecuzione per sezione

Esteso a tutte le 10 domande a crocette (non solo A1, B, C come nel piano originale):
A1 → A2 → A3.1 → A3.2 → B1.1 → B1.2 → B2 → C → F → X.

## 6. Output finale: solo compilazione, nessuna classificazione LOC (aggiornato)

Precisazione dell'obiettivo: il progetto richiede **solo la compilazione delle
crocette** a partire dalla cartella clinica, non la classificazione del Level of
Certainty complessivo. Il punto sollevato in una versione precedente di questo
piano (mancanza dell'algoritmo di combinazione Brighton) non è quindi più
rilevante per lo scope del lavoro: non va implementato.

`aggregation.py` è stato semplificato di conseguenza — si limita a serializzare
il `DVT_CriteriaForm` popolato dalla pipeline in un JSON con tutte le crocette
compilate. Questo è l'output finale del progetto.

## 7. Validazione (NUOVO — mancante nel piano originale)

Prima di presentare i risultati, servirebbe un confronto su un piccolo set di cartelle
annotate manualmente (anche 10-15), con almeno l'accuratezza per singolo criterio e,
se disponibile un secondo annotatore, il Cohen's kappa. Questo è lo step che dà un dato
quantitativo difendibile davanti al professore, oltre alla descrizione architetturale.
Non implementato nel codice fornito (richiede il dataset), ma predisposto come step
successivo naturale una volta che la pipeline produce output stabili.

## 8. Nota sui dati sensibili

Le cartelle cliniche sono dati sanitari sensibili: assicurati che l'uso locale
(Ollama, niente chiamate a API esterne) sia coerente con le policy del tuo ateneo/
comitato etico per il trattamento dati, se applicabile al tuo progetto.
