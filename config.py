"""
Central configuration for the DVT pipeline.
Change parameters here instead of scattering them across modules.
"""

# --- Local LLM (via Ollama) ---
# NOTE: not an official Ollama library model, uploaded by the community.
# Verified exact name: koesn/llama3-openbiollm-8b (tag q4_K_M ~4.9GB, consistent
# with the RAM budget). Make sure to run `ollama pull koesn/llama3-openbiollm-8b`
# (or the specific tag, e.g. `:q4_K_M`) before running the pipeline.
LLM_MODEL_NAME = "koesn/llama3-openbiollm-8b"  # or "biomistral" if you prefer that one
LLM_TEMPERATURE = 0.0  # deterministic for both agents (extractor + evaluator)

# --- Local embeddings (lightweight, do not affect the LLM's RAM budget) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Chunking for the clinical record (EHR) ---
EHR_CHUNK_SIZE = 800
EHR_CHUNK_OVERLAP = 150
EHR_RETRIEVER_K = 5

# --- Persistent Chroma paths ---
BRIGHTON_KB_PERSIST_DIR = "./chroma_brighton_kb"   # static KB, never changes
EHR_KB_PERSIST_DIR = "./chroma_ehr_kb"             # dynamic KB, one per patient/run

# --- Extractor: use real tool-calling agent or direct retrieval? ---
# Default False: on quantized 8B models, agentic tool-calling is often
# unreliable (see ACTION_PLAN.md, point 4). Set to True only after verifying
# with test_structured_output_support() in agents.py that the model handles
# tool-calling consistently.
USE_AGENTIC_EXTRACTOR = False

# --- Questionnaire sections, in the order they should be processed ---
# (used by the loop in pipeline.py)
SECTION_ORDER = [
    "A1", "A2", "A3_1", "A3_2",
    "B1_1", "B1_2", "B2",
    "C", "F", "X",
]
