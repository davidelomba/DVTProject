"""
Central configuration for the clinical extraction pipeline.
Drives both retrieval (RAG) parameters and LLM reasoning constraints.
"""

# --- Local LLM (via Ollama) ---
# One model name per pipeline role. All three are built through the same
# agents.build_llm(model_name=...) factory, so trying a different model for
# any single role is a one-line change here -- no code changes needed
# elsewhere. LLM_TEMPERATURE/LLM_NUM_PREDICT/LLM_REQUEST_TIMEOUT are shared
# generation settings applied to whichever model is built for a given role.
LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"  # Agent 1 (extractor), all modes
EVALUATOR_LLM_MODEL_NAME = "cniongolo/biomistral"  # Agent 2 (evaluator), all modes
AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"  # Agent 1's search step, "agentic_graph" mode only
LLM_TEMPERATURE = 0.0  # deterministic output for all agents
LLM_NUM_PREDICT = 512   # token cap: prevents runaway generation
LLM_REQUEST_TIMEOUT = 180  # seconds; allows time for reasoning on slower hardware

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
EXTRACTOR_MODE = "agentic_graph"  # "full_text", "rag", or "agentic_graph"
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
    },
    "X": {
        # Names of the actual differential-diagnosis conditions from the Brighton
        # Table 2 list (physical trauma, cardiovascular, and "other conditions"
        # categories) -- a risk factor (e.g. a long flight) or a denied/absent
        # symptom is NOT one of these, so it can never trigger this gate; only
        # the record naming one of these specific competing conditions can.
        # Most terms share a Latin/Greek root across Italian and English
        # (cellulit-, vasculit-, cirrosi/cirrhosis, nefrosic-/nephrotic); a few
        # do not, so both the English and Italian term are listed explicitly,
        # same exception already made for "thrombectom"/"trombectom" in A2.
        "keywords": [
            "cellulit", "baker", "fractur", "frattur", "compartment", "compartimental",
            "vasculit", "cirrhosis", "cirrosi", "nephrotic", "nefrosic",
            "lymphatic", "linfatic", "heart failure", "scompenso cardiaco",
            "muscle tear", "strappo muscolare", "hematoma", "ematoma",
            "septic arthritis", "artrite settica", "fistula", "fistola",
            "dependent edema", "edema declive",
        ],
        "default_option_text": "No alternative diagnosis was found to explain the acute illness"
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
        "DISTINGUISHING ULTRASOUND MODALITIES: 'Compression ultrasonography' is a "
        "grayscale (B-mode) test that checks whether the vein collapses under probe "
        "pressure -- it does not by itself assess blood flow. 'Doppler/Duplex "
        "Ultrasound' is any ultrasound exam that also assesses blood flow using the "
        "Doppler effect (color Doppler, duplex, echo-color-Doppler, flow/velocity "
        "studies, absent/present flow signal). If the evidence describes a flow or "
        "Doppler/color exam, select 'Doppler/Duplex Ultrasound', NOT 'Compression "
        "ultrasonography', even if compression is not explicitly mentioned. If the "
        "evidence clearly documents both vein compressibility AND blood flow "
        "assessment, select both options."
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
        "CRITICAL FOR B2: Select every option EXPLICITLY documented in the evidence for "
        "THIS patient -- the options are independent, not mutually exclusive, so check "
        "all of them. Do not select an option just because the Brighton reference context "
        "lists it as a symptom generally associated with DVT. "
        "'Absent blood flow' on an imaging/Doppler study is NOT the same as 'absent "
        "pulses' on physical examination -- only select 'Absent pulses in legs or arms' "
        "if a pulse examination itself is reported as absent, never merely because an "
        "imaging study reports no flow in a vein. "
        "A mention of increased local skin temperature or warmth alone is enough to "
        "select 'Redness, warmth, or pain in one or more extremities', even without "
        "redness, and even if calf pain is already selected separately. "
        "Before finalizing your answer, re-read the evidence once more against every "
        "option -- it is easy to stop as soon as you find one or two matches and miss "
        "a symptom mentioned elsewhere in the same text."
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
        "Select 'Yes' ONLY if the diagnosis is stated with absolutely no supporting clinical details. "
        "NOTE: judge this by whether a specific finding is present -- an anatomical location, a "
        "measurement/value, or a described result -- NOT by whether the extracted evidence fragment "
        "also happens to repeat the NAME of the test/procedure that produced it. The evidence you "
        "receive is an extracted fragment and may omit the test's name even when a specific finding "
        "is present; the absence of the test's name alone is NOT evidence that the diagnosis lacked "
        "details. "
        "Also, immediately before FINAL_OPTION, write one more line: 'DETAILS_PRESENT: yes' or "
        "'DETAILS_PRESENT: no' -- yes if the evidence contains any specific finding as described "
        "above, no only if the diagnosis is stated as a bare conclusion with no specific finding of "
        "any kind. Answer this independently of what you are about to write on FINAL_OPTION."
    ),
    "X": (
        "CRITICAL ERROR PREVENTION: Symptoms (like pain, swelling, edema) are NOT alternative diagnoses. "
        "Do NOT guess or assume an alternative diagnosis just because symptoms are present. "
        "You must select 'An alternative diagnosis was found' ONLY IF the clinical record explicitly names "
        "a DIFFERENT medical condition (e.g., Cellulitis, Baker's cyst, muscle tear) as the TRUE cause "
        "of the illness INSTEAD of DVT. "
        "RISK FACTORS ARE NOT ALTERNATIVE DIAGNOSES EITHER: things that make DVT MORE likely -- e.g. "
        "recent immobilization (a long flight, hospitalization, bed rest), recent surgery, hormonal "
        "therapy/pregnancy, active cancer, or a family history of clotting -- are risk factors FOR DVT, "
        "not competing explanations that replace it. Mentioning one of these does NOT justify selecting "
        "'An alternative diagnosis was found'; if anything, it supports DVT being the correct diagnosis. "
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
