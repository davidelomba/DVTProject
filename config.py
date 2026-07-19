"""
Configurazione centrale della pipeline DVT.
Modifica qui i parametri invece di sparpagliarli nei vari moduli.
"""

# --- Modello LLM locale (via Ollama) ---
# NOTA: non e' un modello della libreria ufficiale Ollama, ma caricato dalla community.
# Nome esatto verificato: koesn/llama3-openbiollm-8b (tag q4_K_M ~4.9GB, coerente
# col budget RAM). Assicurati di fare `ollama pull koesn/llama3-openbiollm-8b`
# (o il tag specifico, es. `:q4_K_M`) prima di eseguire la pipeline.
LLM_MODEL_NAME = "koesn/llama3-openbiollm-8b"  # oppure "biomistral" se preferisci quello
LLM_TEMPERATURE = 0.0  # deterministico per entrambi gli agenti (estrattore + valutatore)

# --- Embeddings locali (leggeri, non pesano sulla RAM del LLM) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Chunking per la cartella clinica (EHR) ---
EHR_CHUNK_SIZE = 800
EHR_CHUNK_OVERLAP = 150
EHR_RETRIEVER_K = 5

# --- Percorsi Chroma persistenti ---
BRIGHTON_KB_PERSIST_DIR = "./chroma_brighton_kb"   # KB statica, non cambia mai
EHR_KB_PERSIST_DIR = "./chroma_ehr_kb"             # KB dinamica, una per paziente/run

# --- Estrattore: usare l'agente con tool-calling reale o retrieval diretto? ---
# Di default False: su modelli 8B quantizzati il tool-calling agentico è spesso
# inaffidabile (vedi PIANO_CORRETTO.md, punto 4). Metti True solo dopo aver
# verificato con test_structured_output_support() in agents.py che il modello
# regge il tool-calling in modo consistente.
USE_AGENTIC_EXTRACTOR = False

# --- Sezioni del questionario, nell'ordine in cui vanno processate ---
# (usato dal loop in pipeline.py)
SECTION_ORDER = [
    "A1", "A2", "A3_1", "A3_2",
    "B1_1", "B1_2", "B2",
    "C", "F", "X",
]