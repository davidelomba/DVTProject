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

# --- Deterministic keyword gates ---
# For sections asking whether ONE SPECIFIC method/procedure was performed
# (autopsy, surgery), the LLM (llama3:8b-instruct-q4_0, 8B quantized) was
# observed to hallucinate a full positive narrative even when the evidence
# never mentions the method at all -- e.g. for A1 it wrote "the evidence
# explicitly states that an autopsy was performed" when the extracted
# evidence contained no such statement (see PATIENT_001 audit log). The
# system prompt instruction against this alone did not reliably prevent it.
#
# As a deterministic safety net: if NONE of the configured keywords appear
# (case-insensitive) in Agent 1's extracted evidence for that section, skip
# the LLM call entirely and default straight to the configured "not done /
# unknown" option. If a keyword IS present, the LLM still evaluates
# normally (its presence doesn't by itself confirm a positive answer -- the
# evidence could equally say the method was NOT done).
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

# --- Section-specific clarifications appended to the evaluator's prompt ---
# Generic instructions in EVALUATOR_SYSTEM_PROMPT don't cover criteria whose
# meaning is easy for the model to invert or conflate. X is the clearest
# case found so far: the model reasoned that DVT symptoms (and a DVT risk
# factor, recent travel) were themselves "an alternative diagnosis
# explaining the acute clinical picture" -- conflating support FOR the DVT
# diagnosis with a diagnosis INSTEAD OF it (see PATIENT_001 audit log).
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

    "B1_1": (
        "CRITICAL DISTINCTION FOR NEGATIONS: "
        "Select the second option ('There was no report of a recognized DVT syndrome') "
        "ONLY IF the clinical record EXPLICITLY states that there are no signs or symptoms "
        "(e.g., 'no leg swelling', 'no calf pain'). "
        "Select the third option ('It is unknown...') IF there is NO information at all "
        "regarding the presence or absence of DVT signs/symptoms in the extracted text."
    ),
    
    "C": (
        "D-DIMER REFERENCE RANGE RULE: "
        "To determine if the D-Dimer value is normal or exceeded, use the laboratory-specific "
        "upper limit if it is explicitly mentioned in the text. "
        "If the lab's reference range is NOT available in the text, you MUST use 500 ng/mL "
        "as the default upper limit of normal (i.e., a value < 500 is normal, >= 500 exceeds the limit)."
    ),

    "F": (
        "CRITICAL: This criterion asks if the DVT was reported 'WITHOUT details'. "
        "If the evidence provides specific details about the diagnostic tests used, "
        "such as imaging findings, specific veins affected (e.g., 'vena poplitea'), or laboratory results, "
        "you MUST select 'No'. Select 'Yes' ONLY if the diagnosis is stated with absolutely no supporting clinical details."
    ),

    "X": (
        "Look for an explicitly stated ALTERNATIVE medical condition or disease (e.g., Baker's cyst, "
        "muscle tear, cellulitis, fracture) that the physician identified as the TRUE cause of the symptoms INSTEAD of DVT. "
        "If the final diagnosis is indeed DVT, or if no other competing disease is mentioned in the evidence, "
        "you MUST select the option indicating 'No alternative diagnosis was found to explain the acute illness'."
    ),
}


# --- Cross-section dependency rules ---
# Applied after all sections have been evaluated independently.
# Each entry: if `if_section`'s field has any value other than `none_option`,
# force `then_section`'s field to `forced_value`. Add new entries here when
# additional inter-section dependencies are identified in the questionnaire.
CROSS_SECTION_RULES = [
    {
        # Brighton dependency: B2 has >=1 reported symptom -> B1.1 must say ">=1 symptom reported".
        "if_section": "b2",           # form_data key (lowercase)
        "none_option": "None of the above were present or it is unknown if any of 1-4 were present",
        "then_section": "b1_1",       # form_data key (lowercase)
        "forced_value": "\u22651 symptom or sign of DVT was reported",
        "audit_key": "B1_1",          # audit_log key (matches SECTION_ORDER casing)
        "override_message": (
            "B1.1 was automatically updated to '\u22651 symptom or sign of DVT was reported' "
            "because symptoms were detected in Section B2, enforcing the "
            "questionnaire's dependency rule."
        ),
    }
]
