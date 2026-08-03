"""
Central configuration for the clinical extraction pipeline.
Drives both retrieval (RAG) parameters and LLM reasoning constraints.
"""

# --- Local LLM (via Ollama) ---
# FINAL DECISION (after extensive empirical testing):
# - Specialized medical/bio LLMs (e.g., OpenBioLLM variants) often struggled
#   with strict formatting constraints (JSON output, fixed trigger phrases)
#   once the system prompt grew too complex, defaulting to prose instead.
# - Generic instruction-tuned models (llama3:8b-instruct-q4_0) reliably
#   reason correctly AND follow formatting instructions, including on
#   multi-select schemas. Chosen as the final model.
LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"
LLM_TEMPERATURE = 0.0  # deterministic output for both agents
LLM_NUM_PREDICT = 512   # token cap: prevents runaway generation
LLM_REQUEST_TIMEOUT = 180  # seconds; allows time for reasoning on slower hardware

# Separate tool-calling-capable model, used ONLY by Agent 1's autonomous
# search step in "agentic_graph" mode (see agentic_graph.py). LLM_MODEL_NAME
# above does NOT support Ollama's native tool-calling API (confirmed:
# Ollama returns "model does not support tools", HTTP 400, when a tool is
# bound to it) -- base Llama 3 never got tool-calling support in Ollama,
# only Llama 3.1+ does. Agent 2 (evaluator) never binds tools, so it keeps
# using LLM_MODEL_NAME regardless of EXTRACTOR_MODE.
# Requires: ollama pull llama3.1:8b-instruct-q4_0
AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"

# --- Local embeddings ---
# Multilingual model, since clinical records here are in Italian. Records are
# NOT machine-translated to English first: translation is a non-deterministic,
# non-auditable extra step that risks distorting negations and terminology.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# --- Extractor mode (Agent 1) ---
# "full_text": passes the entire clinical record to the LLM (default, see
#   agents.extract_evidence_full_text). Simple and reliable as long as the
#   record fits the model's context window.
# "rag": chunked/embedded retrieval (agents.extract_evidence). Needed for
#   clinical records too long to pass whole.
# "agentic_graph": tool-calling extraction (agents.extract_evidence_agentic),
#   Agent 1 explores the record autonomously -- orchestrated as an explicit
#   LangGraph state machine (agentic_graph.py), using a separate tool-
#   calling-capable model for the search step (AGENTIC_LLM_MODEL_NAME below)
#   since LLM_MODEL_NAME doesn't support Ollama tool-calling. NOT yet
#   validated as reliable -- retrieval quality/coverage can vary between
#   runs since Agent 1 decides autonomously how to search. Requires the
#   base `langchain` package and `langgraph`.
EXTRACTOR_MODE = "full_text"
AGENTIC_MAX_ITERATIONS = 5  # cap on tool calls per section, used by "agentic_graph" mode

# --- Chunking for the clinical record (EHR) ---
# Only used when EXTRACTOR_MODE is "rag" or "agentic_graph" (both need the
# record chunked/embedded into a vector store; full_text mode does not).
EHR_CHUNK_SIZE = 800
EHR_CHUNK_OVERLAP = 150
EHR_RETRIEVER_K = 5
EHR_KB_PERSIST_DIR = "./chroma_ehr_kb"

# --- Chunking for the static reference KB (Brighton guidelines PDF) ---
BRIGHTON_CHUNK_SIZE = 800
BRIGHTON_CHUNK_OVERLAP = 150
BRIGHTON_RETRIEVER_K = 5
BRIGHTON_KB_PERSIST_DIR = "./chroma_brighton_kb"  # static KB, rarely changes

# --- Questionnaire sections, in execution order ---
SECTION_ORDER = [
    "A1", "A2", "A3_1", "A3_2",
    "B1_1", "B1_2", "B2",
    "C", "F", "X",
]

# --- Deterministic keyword gates (safety net for specific procedures) ---
# For criteria asking about ONE specific method (autopsy, surgery), the LLM
# can hallucinate a positive answer even when the method is never mentioned.
# If none of the keywords appear in Agent 1's evidence, the section is
# forced to its negative default without an LLM call; if a keyword IS
# present, the LLM still evaluates normally (presence alone doesn't imply
# a positive answer -- the evidence could equally negate it).
SECTION_KEYWORD_GATES = {
    "A1": {
        "keywords": ["autops", "autoptic", "postmortem", "post-mortem", "necrosc"],
        "default_option_text": "No autopsy done, unknown if done, or done but results unavailable"
    },
    "A2": {
        "keywords": ["thrombectom", "trombectom", "embolectom"],
        "default_option_text": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"
    }
}

# --- Section-specific prompt hints ---
# The generic evaluator instructions don't cover every criterion the model
# tends to conflate or invert (e.g. confusing a risk factor with an
# alternative diagnosis). Appended to that section's prompt only.
SECTION_HINTS = {
    "A1": (
        "CRITICAL: This question refers EXCLUSIVELY to post-mortem autopsy findings. "
        "Imaging studies (ultrasound, Doppler, CT, MRI) performed on a living patient "
        "are NOT autopsies. If the patient is alive or no post-mortem autopsy is mentioned, "
        "you MUST select: 'No autopsy done, unknown if done, or done but results unavailable'."
    ),
    "A2": (
        "CRITICAL FOR A2: Pay extreme attention to ANY negations preceding surgical terms. "
        "If the clinical record explicitly states that a surgical procedure was denied, "
        "not performed, or is absent, you MUST NOT select options 1 or 2. "
    ),
    "A3_2": (
        "CRITICAL FOR A3.2: Select ONLY the specific imaging modality(ies) EXPLICITLY "
        "stated to have been performed on THIS patient in the evidence text. The "
        "Brighton reference context lists ALL modalities that CAN in general confirm "
        "a DVT -- it is background terminology, NOT a description of what was actually "
        "done for this patient. Do NOT select an option just because it appears in the "
        "Brighton reference list; select it only if the evidence itself names that "
        "specific test. "
    ),
    "B1_1": (
        "CRITICAL DISTINCTION FOR NEGATIONS: "
        "Select the second option ('There was no report of a recognized DVT syndrome') "
        "ONLY IF the clinical record EXPLICITLY states that there are no signs or symptoms "
        "(e.g., 'no leg swelling', 'no calf pain'). "
        "Select the third option ('It is unknown...') IF there is NO information at all "
        "regarding the presence or absence of DVT signs/symptoms in the extracted text."
    ),
    "B2": (
        "CRITICAL FOR B2: Select ONLY the specific symptoms/signs EXPLICITLY documented "
        "in the evidence text for THIS patient -- do not select an option just because "
        "the Brighton reference context lists it as a symptom generally associated with "
        "DVT. Pay special attention to NOT confuse 'absent blood flow' seen on an imaging "
        "study (e.g. Italian 'flusso assente' on a Doppler/ultrasound, a finding about "
        "venous flow) with 'absent pulses' (a distinct physical examination finding about "
        "arterial pulses) -- these are NOT the same thing; only select 'Absent pulses in "
        "legs or arms' if pulse examination is explicitly mentioned as absent."
    ),
    "C": (
        "D-DIMER REFERENCE RANGE RULE: "
        "To determine if the D-Dimer value is normal or exceeded, use the laboratory-specific "
        "upper limit if it is explicitly mentioned in the text. "
        "If the lab's reference range is NOT available in the text, you MUST use 500 ng/mL "
        "as the default upper limit of normal (i.e., a value < 500 is normal, >= 500 exceeds the limit)."
    ),
    "F": (
        "CRITICAL: This criterion asks if the diagnosis was reported 'WITHOUT details'. "
        "If the evidence provides specific clinical details about the diagnostic tests used, "
        "such as imaging findings, affected anatomy, or laboratory results, you MUST select 'No'. "
        "Select 'Yes' ONLY if the diagnosis is stated with absolutely no supporting clinical details."
    ),
    "X": (
        "CRITICAL ERROR PREVENTION: Symptoms (like pain, swelling, edema) are NOT alternative diagnoses. "
        "Do NOT guess or assume an alternative diagnosis just because symptoms are present. "
        "You must select 'An alternative diagnosis was found' ONLY IF the clinical record explicitly names "
        "a DIFFERENT medical condition (e.g., Cellulitis, Baker's cyst, muscle tear) as the TRUE cause "
        "of the illness INSTEAD of DVT. "
        "If the clinical record diagnoses DVT, or if no other specific competing disease is confirmed, "
        "you MUST select the option: 'No alternative diagnosis was found to explain the acute illness'."
    ),
}


# --- Cross-section dependency rules ---
# Evaluated after all sections are independently filled in. Each entry: if
# `if_section`'s field has any value other than `none_option`, force
# `then_section`'s field to `forced_value`.
CROSS_SECTION_RULES = [
    {
        "if_section": "b2",           # form_data key (lowercase)
        "none_option": "None of the above were present or it is unknown if any of 1-4 were present",
        "then_section": "b1_1",       # form_data key (lowercase)
        "forced_value": "\u22651 symptom or sign of DVT was reported",
        "audit_key": "B1_1",          # audit_log key (matches SECTION_ORDER casing)
        "override_message": (
            "B1.1 was automatically updated to '\u22651 symptom or sign of DVT was reported' "
            "because specific symptoms were detected in Section B2, enforcing the "
            "questionnaire's structural dependency rule."
        ),
    }
]
