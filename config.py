"""
Central configuration for the DVT pipeline.
Change parameters here instead of scattering them across modules.
"""

# --- Local LLM (via Ollama) ---
# FINAL DECISION (after extensive empirical testing, see ACTION_PLAN.md):
# - koesn/llama3-openbiollm-8b (community GGUF conversion): failed -- echoed
#   the prompt back or hallucinated unrelated clinical cases on multi-
#   constraint instructions.
# - hf.co/aaditya/OpenBioLLM-Llama3-8B-GGUF:Q4_K_M (official GGUF from the
#   original OpenBioLLM author): reasoned correctly in isolation, but
#   consistently failed to follow ANY additional formatting instruction
#   (JSON output, a fixed "FINAL_ANSWER:" line) once the system prompt grew
#   past ~100 characters -- it would default to explaining in prose instead,
#   regardless of instruction content. It also occasionally reasoned
#   incorrectly on the harder multi-select schema (B2) depending on prompt
#   phrasing.
# - llama3:8b-instruct-q4_0 (official, generic instruction-tuned model):
#   reliably reasons correctly AND follows the "reason, then write a
#   FINAL_ANSWER: ... line" instruction, on both a simple single-choice
#   schema and a harder multi-select schema. Chosen as the final model.
# Trade-off: this is NOT a clinically fine-tuned model. Domain knowledge is
# instead supplied via the RAG context (Brighton synonyms + extracted EHR
# evidence) and the system prompt's negation-handling guidance, rather than
# from the model's own training. This is a deliberate, documented choice,
# not an oversight.
LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"
LLM_TEMPERATURE = 0.0  # deterministic for both agents (extractor + evaluator)
LLM_NUM_PREDICT = 512   # token cap: prevents runaway generation from costing minutes
LLM_REQUEST_TIMEOUT = 180  # seconds; this model has been observed to take >140s on some calls

# --- Local embeddings (lightweight, do not affect the LLM's RAM budget) ---
# Multilingual model, not English-only: clinical records are in Italian.
# Deliberately NOT translating the records to English instead -- machine
# translation is an extra, non-deterministic, non-auditable step that risks
# distorting exactly the kind of language this pipeline is most sensitive to
# (negations, clinical terminology), on top of adding RAM/latency cost.
# Keeping records in their original language + a multilingual embedding model
# avoids that failure mode entirely.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# --- Chunking for the clinical record (EHR) ---
EHR_CHUNK_SIZE = 800
EHR_CHUNK_OVERLAP = 150
EHR_RETRIEVER_K = 5

# --- Chunking for the Brighton paper KB (static, English source PDF) ---
BRIGHTON_CHUNK_SIZE = 800
BRIGHTON_CHUNK_OVERLAP = 150
BRIGHTON_RETRIEVER_K = 3

# --- Persistent Chroma paths ---
BRIGHTON_KB_PERSIST_DIR = "./chroma_brighton_kb"   # static KB, never changes
EHR_KB_PERSIST_DIR = "./chroma_ehr_kb"             # dynamic KB, one per patient/run

# --- Extractor: use real tool-calling agent or direct retrieval? ---
# Default False: agentic tool-calling was never validated as reliable during
# testing (all testing focused on the evaluator's structured-output problem).
# Direct retrieval remains the safer default.
USE_AGENTIC_EXTRACTOR = False

# --- Questionnaire sections, in the order they should be processed ---
SECTION_ORDER = [
    "A1", "A2", "A3_1", "A3_2",
    "B1_1", "B1_2", "B2",
    "C", "F", "X",
]
