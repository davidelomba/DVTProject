"""
Central configuration for the clinical extraction pipeline: model names and
parameters, retrieval settings, section order, the deterministic gates and
the per-section prompt hints.
"""

from pathlib import Path

# Project root directory, used to locate persistent vector stores and other resources
PROJECT_ROOT = Path(__file__).resolve().parent

# LLM model names and parameters
LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"  # Agent 1 (extractor), all modes
EVALUATOR_LLM_MODEL_NAME = "qwen3.6:27b"  # Agent 2 (evaluator), all modes
AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"  # Agent 1's search step, "agentic_graph" mode only
LLM_TEMPERATURE = 0.0  # deterministic output for all agents
LLM_NUM_PREDICT = 1024  # token cap: prevents runaway generation. The two
                        # answer lines close the response, so a cap the model
                        # reaches first costs the whole section.
# How many layers to place on the GPU. 999 means all of them: Ollama's own
# split left a model 7% on CPU with 5 GB of VRAM still free, and layers on
# CPU dominate the time per token.
LLM_NUM_GPU = 999
# Thinking mode. False turns it off on models that have one: with it on the
# model can spend the whole token cap reasoning and return empty content,
# since the reasoning does not travel in the response body. None sends
# nothing to Ollama, leaving the model's own default.
LLM_REASONING = False
LLM_REQUEST_TIMEOUT = 180  # seconds; allows time for reasoning on slower hardware

# Multilingual embedding model. Used in every mode: the Brighton store is
# always built, the EHR one only for rag and agentic_graph.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Extractor mode (Agent 1):
# "full_text": sends the entire clinical record to the LLM
# "rag": chunked/embedded retrieval. Needed for clinical records too long to pass whole
# "agentic_graph": tool-calling extraction
#   Agent 1 explores the record autonomously, orchestrated as an explicit
#   LangGraph state machine (agentic_graph.py), using a separate tool-
#   calling-capable model for the search step (AGENTIC_LLM_MODEL_NAME).
EXTRACTOR_MODE = "agentic_graph"  # "full_text", "rag", or "agentic_graph"
AGENTIC_MAX_ITERATIONS = 5  # cap on tool calls per section, used by "agentic_graph" mode

# Chunking for the clinical record (EHR)
# Only used when EXTRACTOR_MODE is "rag" or "agentic_graph"
EHR_CHUNK_SIZE = 800
EHR_CHUNK_OVERLAP = 150
EHR_RETRIEVER_K = 5
EHR_KB_PERSIST_DIR = str(PROJECT_ROOT / "vectorstores" / "chroma_ehr_kb")

# Chunking for the static reference KB (Brighton PDF)
BRIGHTON_CHUNK_SIZE = 800
BRIGHTON_CHUNK_OVERLAP = 150
BRIGHTON_RETRIEVER_K = 5
BRIGHTON_KB_PERSIST_DIR = str(PROJECT_ROOT / "vectorstores" / "chroma_brighton_kb")  # static KB, rarely changes

# Questionnaire sections, in execution order
SECTION_ORDER = [
    "A1", "A2", "A3_1", "A3_2",
    "B1_1", "B1_2", "B2",
    "C", "F", "X",
]

# Per-section deterministic gates: on/off switches
# Switches the post-processing applied to each section's answer (see
# criteria_rules.apply_section_gates); setting one to False skips that
# gate, leaving the model's answer unchanged.
# CROSS_SECTION_RULES below are excluded on purpose and always apply.
# TODO: consider setting all gates to False when a big LLM is used.
SECTION_GATES_ENABLED = {
    # Reverts a positive answer when the evidence never names the procedure
    # (A1, A2, X: see SECTION_KEYWORD_GATES below).
    "keyword": True,
    # Derives section F's Yes/No from the model's own DETAILS_PRESENT line.
    "details": True,
    # Drops B2's "Absent pulses" when no pulse examination is in the evidence.
    "absent_pulses": True,
}

# Deterministic keyword gates
# Sections asking whether one specific thing is present (A1 autopsy, A2
# surgery, X an alternative diagnosis): a small model can answer positively
# even when the evidence never names it. If none of the keywords appear in
# Agent 1's evidence the section is forced to its negative default, without
# an LLM call. A keyword being present changes nothing on its own: the
# evidence could be negating it, so the model still evaluates normally.
# An optional "gated_options" names the answers the keywords can speak for.
# Without it every answer other than the default is checked against them.
# TODO: consider deleting this gates when a big LLM is used.
SECTION_KEYWORD_GATES = {
    "A1": {
        "keywords": ["autops", "autoptic", "postmortem", "post-mortem", "necrosc"],
        "default_option_text": "No autopsy done, unknown if done, or done but results unavailable"
    },
    "A2": {
        "keywords": ["thrombectom", "trombectom", "embolectom"],
        "default_option_text": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done",
        # These words name thrombectomy only. Checking the "Other procedure"
        # option against them rejects it on every record.
        "gated_options": ["Thrombectomy related to DVT performed"],
    },
    "X": {
        # Names of competing conditions from Brighton Table 2. Most share a
        # Latin/Greek root across Italian and English (cellulit-, vasculit-,
        # cirrosi/cirrhosis); where they do not, both terms are listed.
        "keywords": [
            "cellulit", "baker", "fractur", "frattur", "compartment", "compartimental",
            "vasculit", "cirrhosis", "cirrosi", "nephrotic", "nefrosic",
            "lymphatic", "linfatic", "heart failure", "scompenso cardiaco",
            "muscle tear", "strappo muscolare", "hematoma", "ematoma",
            "septic arthritis", "artrite settica", "fistula", "fistola",
            "dependent edema", "edema declive",
            "achilles", "achille", "external compression", "compressione esterna",
            "congenital vascular", "malformazione vascolare",
        ],
        "default_option_text": "No alternative diagnosis was found to explain the acute illness"
    }
}

# Section-specific prompt hints
# These sentences are injected into the prompt for each section, in order to help the model
# avoid common pitfalls and focus on the most important reasoning points.
#TODO: try to make these hints more concise when a big LLM will be used.
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
    "A3_1": (
        "CRITICAL FOR A3.1: the options differ on ONE question -- was an imaging study "
        "actually PERFORMED on this patient?\n"
        "- Option 2 requires BOTH that an imaging study was actually carried out AND "
        "that its result did not show DVT. Select it only if the evidence names an "
        "imaging test that was really done.\n"
        "- Option 3 covers every other case: no imaging test was done, it is not known "
        "whether one was done, or one was done but its result is not reported. If the "
        "evidence states that no imaging was performed, or says nothing at all about "
        "imaging, the correct answer is option 3, NOT option 2.\n"
        "WATCH OUT: a NEGATIVE result is still a result. An imaging study reporting "
        "compressible veins, no thrombosis, absence of DVT or DVT ruled out WAS "
        "performed and did not confirm DVT -> option 2. Option 3's 'result not "
        "reported' means the report is missing or unavailable, NOT that the report "
        "says there is no DVT.\n"
        "An autopsy is not an imaging study. A DVT diagnosis reported without any "
        "imaging report does not by itself mean that imaging was done."
    ),
    "A3_2": (
        "CRITICAL FOR A3.2: select every imaging modality the evidence itself states was "
        "performed on THIS patient, and only those. The Brighton context lists modalities "
        "that CAN confirm a DVT in general: it is terminology, not a record of what was "
        "done here.\n"
        "WATCH OUT: the Brighton table names a single bundled modality, 'Compression "
        "ultrasonography with and without Doppler'. This questionnaire SPLITS it into two "
        "separate options, so that phrase is never an answer on its own. Decide by what "
        "the exam actually did.\n"
        "Match the modality the evidence names, not the one most commonly used for DVT:\n"
        "- ultrasound assessing blood flow (color Doppler, duplex, echo-color-Doppler, "
        "ecocolordoppler, flow or velocity study, absent/present flow signal) -> "
        "'Doppler/Duplex Ultrasound', NOT 'Compression ultrasonography', even when the "
        "evidence does not mention compression\n"
        "- ultrasound that only tests whether the vein collapses under the probe, with no "
        "mention of flow -> 'Compression ultrasonography'\n"
        "- contrast venography / phlebography / 'flebografia' -> 'Contrast venography'\n"
        "- CT or MR venography / 'TC venografia' / 'RM venografia' -> 'CT or MR venography'\n"
        "- any other named imaging test (e.g. impedance plethysmography, a whole-body CT "
        "done for another purpose) -> 'Other'"
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
        "THIS patient. The options are independent, not mutually exclusive; evaluate each "
        "one separately. Do not select an option just because the Brighton reference "
        "context lists it as a symptom generally associated with DVT.\n"
        "- 'Calf pain or tenderness': select ONLY if pain or tenderness specifically IN THE "
        "CALF is described.\n"
        "- 'Redness, warmth, or pain in one or more extremities': select ONLY if redness, "
        "warmth, or pain in an extremity is described that is NOT itself calf-specific "
        "tenderness.\n"
        "- These two are DIFFERENT findings, never select both just because one applies. "
        "Each needs its OWN evidence.\n"
        "- EXAMPLE: evidence says 'pain and redness in the right arm' (no mention of calf or "
        "leg anywhere) -> select ONLY 'Redness, warmth, or pain in one or more extremities'. "
        "Do NOT also select 'Calf pain or tenderness', there is no calf involved.\n"
        "- 'Absent pulses in legs or arms': select ONLY if a pulse examination itself is "
        "reported as absent. 'Absent blood flow' on an imaging/Doppler study is NOT the same "
        "thing; never select this merely because an imaging study reports no flow in a vein."
    ),
    "C": (
        "D-DIMER REFERENCE RANGE RULE: "
        "To determine if the D-Dimer value is normal or exceeded, use the laboratory-specific "
        "upper limit if it is explicitly mentioned in the text. "
        "If the lab's reference range is NOT available in the text, you MUST use 500 ng/mL "
        "as the default upper limit of normal (i.e., a value < 500 is normal, >= 500 exceeds the limit). "
        "IMPORTANT: only apply this rule to a result that explicitly refers to the D-Dimer test by "
        "name (e.g. 'D-dimero', 'D-Dimer'). Other lab findings, such as leukocytosis, elevated CRP/PCR, "
        "white blood cell count, etc. are NOT D-Dimer and must NOT be used to infer a D-Dimer "
        "result. If the D-Dimer test itself is never mentioned in the evidence, select 'D-dimer not "
        "tested, or tested but results unknown or not available', even if other lab abnormalities are "
        "present. A qualitative statement that the D-Dimer exceeded (or was within) the lab's limit "
        "(e.g. 'D-dimero superiore al limite di laboratorio') is sufficient evidence on its own. "
    ),
    "F": (
        "FIRST LINE OF YOUR RESPONSE, before any reasoning, write exactly: "
        "'DETAILS_PRESENT: yes' or 'DETAILS_PRESENT: no'. Write 'yes' if the evidence "
        "contains ANY specific clinical finding, for example an imaging result (even a NEGATIVE "
        "one, e.g. 'no evidence of thrombosis'), a lab value (e.g. leukocytosis, CRP, "
        "D-dimer), or a named anatomical site; write 'no' only if the diagnosis is "
        "stated as a bare conclusion with no supporting finding of any kind. Then "
        "continue with your reasoning as instructed below.\n"
        "This criterion asks if the diagnosis was reported 'WITHOUT details'. If such "
        "details are present you MUST select 'No', even when the finding is not itself "
        "about DVT or the final diagnosis is a different condition (e.g. cellulitis). "
        "Select 'Yes' ONLY for a bare conclusion with no supporting detail. "
        "NOTE: judge this by whether a specific finding is present, NOT by whether the "
        "evidence fragment also repeats the NAME of the test that produced it; the "
        "fragment may omit the test's name even when a finding is present."
    ),
    "X": (
        "CRITICAL FOR X: ask yourself ONE question -- does the record name a disease other "
        "than DVT as the cause of this illness?\n"
        "- If yes, THAT disease is the alternative diagnosis (e.g. cellulitis, Baker's "
        "cyst, muscle tear): select 'An alternative diagnosis was found that explained the "
        "acute illness'. This holds even when the record presents it as the final confirmed "
        "diagnosis rather than as one hypothesis among several; 'alternative' means "
        "alternative to DVT, not alternative to the record's own conclusion.\n"
        "- If no, select 'No alternative diagnosis was found to explain the acute illness'.\n"
        "Symptoms (pain, swelling, edema) are not diagnoses. Risk factors that make DVT more "
        "likely (immobilization, recent surgery, pregnancy, active cancer) are not competing "
        "explanations either."
    ),
}


# Cross-section dependency rules
# Applied after every section has been answered independently. A rule with
# "none_option" fires when if_section holds anything else (B2 has a symptom,
# so B1.1 must be positive); one with "trigger_value" fires on an exact match
# (A3.1 says no imaging, so A3.2 is cleared). Regardless of the trigger,
# then_section is overwritten with forced_value.
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
    },
    {
        "if_section": "a3_1",         # form_data key (lowercase)
        "trigger_value": "No imaging studies done, unknown if done, or done but results unknown",
        "then_section": "a3_2",       # form_data key (lowercase)
        "forced_value": [],
        "audit_key": "A3_2",          # audit_log key (matches SECTION_ORDER casing)
        "override_message": (
            "A3.2 was automatically cleared because Section A3.1 reported that no "
            "imaging study was done, enforcing the questionnaire's structural "
            "dependency rule (A3.2 only applies when A3.1 confirms an imaging study "
            "was performed)."
        ),
    },
    {
        # A3.2 asks which studies CONFIRMED DVT, so a study that was performed
        # and did not confirm it belongs to no answer here.
        "if_section": "a3_1",
        "trigger_value": "≥1 imaging study was done but didn't confirm DVT",
        "then_section": "a3_2",
        "forced_value": [],
        "audit_key": "A3_2",
        "override_message": (
            "A3.2 was automatically cleared because Section A3.1 reported that the "
            "imaging study did not confirm DVT, and A3.2 records only the studies "
            "that confirmed it."
        ),
    },
]
