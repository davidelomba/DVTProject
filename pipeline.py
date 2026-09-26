"""
Main pipeline.
For each section of the form:
  1. Agent 1 extracts the relevant evidence from the clinical record (full text by default)
  2. Agent 2 reasons over the evidence and fills in the checkbox
Independent per-section results are then merged, cross-section dependency
rules are applied and, when config.CONFIDENCE_ENABLED is set, Agent 3 scores
the confidence of each final answer. The final DVT_CriteriaForm is returned
together with a full audit log.
"""

import hashlib
import socket
import subprocess
import time
import traceback

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import (
    get_embeddings,
    build_brighton_kb,
    build_ehr_kb,
    make_ehr_retriever_tool,
    load_brighton_pdf_text,
    load_ehr_text,
    resolve_guideline_anchors,
    retrieve_brighton_context,
    section_is_anchored,
)
from agents import build_llm, evaluate_section, extract_evidence, extract_evidence_full_text
from criteria_rules import apply_cross_section_rules
from agentic_graph import build_agentic_llm, run_agentic_graph_pipeline
from confidence import score_record


# Retrieval/extraction query for each section: tells Agent 1 what to look
# for in the clinical record. Refine based on the language and terminology
# of the EHRs actually used (e.g. Italian abbreviations).
SECTION_QUERIES = {
    "A1": "autopsy report, necropsy, post-mortem examination, autoptic findings",
    "A2": "thrombectomy, surgical procedure related to DVT",
    "A3_1": "ultrasound, CT, MRI, venography: imaging outcome for deep vein thrombosis",
    "A3_2": "type of imaging study performed: compression ultrasonography, doppler, venography",
    "B1_1": "reported symptoms or signs of deep vein thrombosis",
    "B1_2": "deep vein thrombosis lower extremity or upper extremity",
    "B2": "calf pain or tenderness, leg swelling or pitting oedema, redness, "
          "warmth or pain in any extremity, absent pulses in legs or arms",
    "C": "D-dimer value, test date, laboratory upper limit of normal",
    "F": "diagnosis of deep vein thrombosis reported by specialist",
    # Worded after Table 2 of the guideline, which lists these conditions under
    # the symptoms they mimic. The earlier wording sent that table to B2, whose
    # query names the same symptoms.
    "X": "possible alternative etiologies; clinical syndromes to be differentiated "
         "from thrombosis; conditions explaining calf pain, redness, warmth or "
         "ankle edema other than DVT",
}

# Key under which run_pipeline records the settings a run was produced with.
# Chosen so it cannot collide with a section name (config.SECTION_ORDER holds
# A1, A2, A3_1, ...), since the audit log is otherwise keyed by section.
RUN_CONFIG_KEY = "_run_config"


def _ollama_version() -> str:
    """The version string of the Ollama binary on PATH.

    Recorded because two Ollama versions ship two versions of llama.cpp, and
    with them different quantisation kernels: at temperature 0 that is enough
    to flip a token the model had nearly tied, so runs produced under different
    versions are not directly comparable.

    Returns:
        The version, or a short marker when the binary is absent or does not
        answer. Never raises: a missing version costs provenance, not a run.
    """

    try:
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or out.stderr.strip() or "unknown"
    except Exception:
        return "unavailable"


def _hint_fingerprint() -> dict:
    """Identifies the hint text each section was answered with.

    `section_hints_enabled` and `section_hints_disabled` say which hints were
    sent, not what they said, so two runs whose hints were rewritten between
    them carry the same signature. Hashing the text tells them apart without
    copying the hints into every audit log.

    Returns:
        Section key -> "<chars> <first 12 hex of sha256>", for the sections
        that received a non-empty hint. `all` digests the whole set, so two
        runs can be compared on one string.
    """

    digest = hashlib.sha256()
    per_section = {}
    for section_key in config.SECTION_ORDER:
        hint = config.section_hint(section_key)
        digest.update(f"{section_key}:{hint}\n".encode("utf-8"))
        if hint:
            own = hashlib.sha256(hint.encode("utf-8")).hexdigest()[:12]
            per_section[section_key] = f"{len(hint)} {own}"
    per_section["all"] = digest.hexdigest()[:12]
    return per_section


def _query_fingerprint() -> dict:
    """Identifies the section queries a run was produced with.

    SECTION_QUERIES is the brief Agent 1 works from and the key that retrieves
    the guideline context. No other field records its text, so without this two
    runs whose queries were rewritten between them carry the same signature.

    Returns:
        Section key -> "<chars> <first 12 hex of sha256>". `all` digests the
        whole set, so two runs can be compared on one string.
    """

    digest = hashlib.sha256()
    per_section = {}
    for section_key in config.SECTION_ORDER:
        query = SECTION_QUERIES[section_key]
        digest.update(f"{section_key}:{query}\n".encode("utf-8"))
        own = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        per_section[section_key] = f"{len(query)} {own}"
    per_section["all"] = digest.hexdigest()[:12]
    return per_section


def _anchor_fingerprint(guideline_anchors: dict) -> dict:
    """Identifies the guideline passages a run was produced with.

    Digests the resolved passages rather than the labels of
    config.GUIDELINE_ANCHORS: a heading reaches different text depending on how
    the PDF extracted, so the labels alone do not identify what Agent 2 read.

    Args:
        guideline_anchors: the mapping rag_setup.resolve_guideline_anchors
            returned.

    Returns:
        Section key -> "<chars> <first 12 hex of sha256>", for the sections that
        resolved. `all` digests the whole set.
    """

    digest = hashlib.sha256()
    per_section = {}
    for section_key in config.SECTION_ORDER:
        passage = guideline_anchors.get(section_key)
        if passage is None:
            continue
        digest.update(f"{section_key}:{passage}\n".encode("utf-8"))
        own = hashlib.sha256(passage.encode("utf-8")).hexdigest()[:12]
        per_section[section_key] = f"{len(passage)} {own}"
    per_section["all"] = digest.hexdigest()[:12]
    return per_section


def _run_config_snapshot(guideline_anchors: dict = None) -> dict:
    """Captures the settings that determine what a run produces.

    Recorded in the audit log so an output file is self-describing: without it
    there is no way to tell, months later, which hints and context a given
    result was produced with, which model answered, or which extraction mode
    was used -- all of which are varied between experiments.

    Args:
        guideline_anchors: the resolved anchors, digested into the snapshot.

    Returns:
        A JSON-serialisable dict of the relevant config values.
    """

    # Which model reads the record, by mode. "raw_record" loads none, since the
    # record reaches Agent 2 unfiltered.
    if config.EXTRACTOR_MODE == "agentic_graph":
        extractor_model = config.AGENTIC_LLM_MODEL_NAME
    elif config.EXTRACTOR_MODE == "raw_record":
        extractor_model = None
    else:
        extractor_model = config.LLM_MODEL_NAME

    return {
        "extractor_mode": config.EXTRACTOR_MODE,
        "brighton_context_enabled": config.BRIGHTON_CONTEXT_ENABLED,
        "section_descriptions_enabled": config.SECTION_DESCRIPTIONS_ENABLED,
        "section_hints_enabled": config.SECTION_HINTS_ENABLED,
        "section_hints_disabled": sorted(config.SECTION_HINTS_DISABLED),
        "section_hints_fingerprint": _hint_fingerprint(),
        "section_queries_fingerprint": _query_fingerprint(),
        "guideline_anchors_enabled": config.GUIDELINE_ANCHORS_ENABLED,
        "guideline_anchors_fingerprint": _anchor_fingerprint(guideline_anchors or {}),
        # Always applied, never switchable: recorded so a reader does not have
        # to know that to interpret the run.
        "cross_section_rules_applied": True,
        "confidence": {
            "enabled": config.CONFIDENCE_ENABLED,
            "skipped_sections": sorted(config.CONFIDENCE_SKIP),
        },
        "models": {
            "extractor": extractor_model,
            "evaluator": config.EVALUATOR_LLM_MODEL_NAME,
            "embeddings": config.EMBEDDING_MODEL_NAME,
        },
        # Two runs of the same records on different machines have been observed
        # to differ, so a result is only comparable to another produced here.
        "environment": {
            "hostname": socket.gethostname(),
            "ollama_version": _ollama_version(),
        },
        "generation": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_NUM_PREDICT,
            "num_gpu": config.LLM_NUM_GPU,
            "num_ctx": config.LLM_NUM_CTX,
            "reasoning": config.LLM_REASONING,
        },
        "retrieval": {
            "ehr_chunk_size": config.EHR_CHUNK_SIZE,
            "ehr_chunk_overlap": config.EHR_CHUNK_OVERLAP,
            "ehr_retriever_k": config.EHR_RETRIEVER_K,
            "brighton_retriever_k": config.BRIGHTON_RETRIEVER_K,
        },
    }


def run_pipeline(record_id: str, patient_ehr_path: str, brighton_pdf_path: str):
    """Fills in the whole questionnaire for one clinical record.

    Args:
        record_id: identifier carried into the results.
        patient_ehr_path: the .txt clinical record.
        brighton_pdf_path: the reference guideline PDF.

    Returns:
        (form, audit_log). audit_log is keyed by section (A1, A2, ...) and
        holds, for each, the evidence Agent 1 extracted, the guideline context
        given to Agent 2 and Agent 2's full reasoning, so a wrong answer stays
        diagnosable without re-running. It also carries a RUN_CONFIG_KEY entry
        describing the settings the run was produced with.
    """
    # Built once and reused across every section, not per loop iteration.
    # Each role reads its own constant, so any one can be swapped from
    # config.py alone. Agent 1's model is built in the branch that uses it, so
    # a mode that never queries it does not load it into VRAM.
    embeddings = get_embeddings()
    evaluator_llm = build_llm(config.EVALUATOR_LLM_MODEL_NAME)

    # Load both source texts once: the static Brighton reference paper and
    # this run's patient record.
    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    patient_ehr_text = load_ehr_text(patient_ehr_path)

    # Brighton KB is always needed (every mode consults it for synonyms/context).
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)

    # Resolved from the same text the KB is built from, so the anchored and the
    # retrieved context are two readings of one document.
    guideline_anchors = resolve_guideline_anchors(brighton_text)

    # "rag" and "agentic_graph" both need the EHR chunked/embedded into a
    # vector store; "full_text" passes the raw record directly per section.
    # Only "agentic_graph" additionally needs the retriever wrapped as a
    # tool the LLM can call autonomously.
    ehr_kb = None
    ehr_tool = None
    if config.EXTRACTOR_MODE in ("rag", "agentic_graph"):
        ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=record_id, embeddings=embeddings)
        if config.EXTRACTOR_MODE == "agentic_graph":
            ehr_tool = make_ehr_retriever_tool(ehr_kb)

    if config.EXTRACTOR_MODE == "agentic_graph":
        # Separate tool-calling-capable model, used only for Agent 1's
        # autonomous search step (see config.AGENTIC_LLM_MODEL_NAME).
        search_llm = build_agentic_llm()

        # Delegates the whole per-section loop to the LangGraph state
        # machine; returns the same (form_data, audit_log) shape as the
        # plain loop below, just not yet passed through the cross-section
        # rules (applied once, uniformly, further down).
        form_data, audit_log = run_agentic_graph_pipeline(
            record_id, evaluator_llm=evaluator_llm, search_llm=search_llm,
            ehr_tool=ehr_tool, ehr_vectorstore=ehr_kb, brighton_kb=brighton_kb,
            section_queries=SECTION_QUERIES, guideline_anchors=guideline_anchors,
        )
    else:
        # Built only by the modes that call Agent 1, so "raw_record" keeps the
        # extractor out of VRAM.
        llm = build_llm() if config.EXTRACTOR_MODE in ("rag", "full_text") else None

        form_data = {"record_id": record_id}
        audit_log = {}
        # Section keys map to DVT_CriteriaForm fields via lower-casing
        # (e.g. "A3_1" -> "a3_1").

        # Sequentially fill in every section of the questionnaire, in the
        # fixed order defined by config.SECTION_ORDER.
        for section_key in config.SECTION_ORDER:
            section_model = SECTION_MODELS[section_key]
            query = SECTION_QUERIES[section_key]

            print(f"\n=== Section {section_key} ===", flush=True)
            section_log = {"query": query}
            # Agent 2's token counts, one entry per attempt.
            token_counts = []

            try:
                # Agent 1: extraction mode per config.EXTRACTOR_MODE
                print(f"[{section_key}] Agent 1 (extractor, mode={config.EXTRACTOR_MODE}) searching the clinical record...", flush=True)
                t0 = time.time()

                # "agentic_graph" is dispatched separately, above, so only the
                # other three reach this loop. "raw_record" passes the record
                # unchanged, so every section receives the same text.
                if config.EXTRACTOR_MODE == "rag":
                    evidence = extract_evidence(llm, ehr_kb, query)
                elif config.EXTRACTOR_MODE == "full_text":
                    evidence = extract_evidence_full_text(llm, patient_ehr_text, query)
                else:
                    evidence = patient_ehr_text
                elapsed = time.time() - t0
                print(f"[{section_key}] Agent 1 done in {elapsed:.1f}s", flush=True)
                section_log["evidence"] = evidence
                section_log["agent1_seconds"] = round(elapsed, 1)

                brighton_context = retrieve_brighton_context(
                    brighton_kb, query, section_key, guideline_anchors
                )
                section_log["brighton_context"] = brighton_context

                # Agent 2: evaluation constrained to the section's Pydantic schema
                print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
                t0 = time.time()
                extra_instructions = config.section_hint(section_key)
                section_result, reasoning_text, answer_conflict = evaluate_section(
                    evaluator_llm, section_model, evidence, brighton_context, extra_instructions,
                    context_is_anchored=section_is_anchored(section_key, guideline_anchors),
                    token_counts=token_counts,
                )
                elapsed = time.time() - t0
                print(f"[{section_key}] Agent 2 done in {elapsed:.1f}s", flush=True)
                section_log["agent2_seconds"] = round(elapsed, 1)

                # None unless the model's two answer lines disagreed; kept as a
                # review flag for this section (see agents.evaluate_section).
                section_log["answer_conflict"] = answer_conflict

                section_log["reasoning"] = reasoning_text
                section_log["result"] = section_result.model_dump()

                form_data[section_key.lower()] = section_result
                print(f"[{section_key}] filled in: {section_result}", flush=True)

            except Exception as exc:

                # One section failing shouldn't lose work already done on others.
                print(f"[{section_key}] FAILED -> leaving this field as None. Traceback:", flush=True)
                traceback.print_exc()
                form_data[section_key.lower()] = None
                section_log["error"] = str(exc)
                # Kept so the answer that could not be parsed stays readable.
                section_log["reasoning"] = getattr(exc, "last_response", None)

            # Written on failure too, since a truncated prompt is one way a
            # section ends up with no parseable answer.
            section_log["agent2_tokens"] = token_counts
            audit_log[section_key] = section_log

    # Cross-section dependency rules (see config.CROSS_SECTION_RULES), applied
    # once here regardless of which EXTRACTOR_MODE produced form_data;
    # shared with agentic_graph.py via criteria_rules.py.
    form_data = apply_cross_section_rules(form_data, audit_log)

    # Agent 3, on the final answers, so after the rules above.
    if config.CONFIDENCE_ENABLED:
        score_record(form_data, audit_log)

    # Added last, so it cannot be mistaken for a section by the loops above.
    audit_log[RUN_CONFIG_KEY] = _run_config_snapshot(guideline_anchors)

    form = DVT_CriteriaForm(**form_data)
    return form, audit_log
