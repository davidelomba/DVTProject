"""
Main pipeline.
For each section of the form:
  1. Agent 1 extracts the relevant evidence from the clinical record (full text by default)
  2. Agent 2 reasons over the evidence and fills in the checkbox
Independent per-section results are then merged, cross-section dependency
rules are applied and the final DVT_CriteriaForm is returned together with
a full audit log.
"""

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
    clean_brighton_context,
)
from agents import build_llm, evaluate_section, extract_evidence, extract_evidence_full_text
from criteria_rules import apply_section_gates, apply_cross_section_rules
from agentic_graph import build_agentic_llm, run_agentic_graph_pipeline


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
    "B2": "calf pain, swelling, oedema, redness, warmth, absent pulses",
    "C": "D-dimer value, test date, laboratory upper limit of normal",
    "F": "diagnosis of deep vein thrombosis reported by specialist",
    "X": "alternative diagnosis explaining the acute clinical picture",
}

# Key under which run_pipeline records the settings a run was produced with.
# Chosen so it cannot collide with a section name (config.SECTION_ORDER holds
# A1, A2, A3_1, ...), since the audit log is otherwise keyed by section.
RUN_CONFIG_KEY = "_run_config"


def _run_config_snapshot() -> dict:
    """Captures the settings that determine what a run produces.

    Recorded in the audit log so an output file is self-describing: without it
    there is no way to tell, months later, whether a given result came from a
    run with the deterministic gates enabled, which model answered, or which
    extraction mode was used -- all of which are varied between experiments.

    Returns:
        A JSON-serialisable dict of the relevant config values.
    """

    return {
        "extractor_mode": config.EXTRACTOR_MODE,
        "section_gates_enabled": dict(config.SECTION_GATES_ENABLED),
        # Always applied, never switchable: recorded so a reader does not have
        # to know that to interpret the run.
        "cross_section_rules_applied": True,
        # The model that actually extracted, not the constant: in
        # "agentic_graph" Agent 1 runs on AGENTIC_LLM_MODEL_NAME and
        # LLM_MODEL_NAME is never queried.
        "models": {
            "extractor": (
                config.AGENTIC_LLM_MODEL_NAME
                if config.EXTRACTOR_MODE == "agentic_graph"
                else config.LLM_MODEL_NAME
            ),
            "evaluator": config.EVALUATOR_LLM_MODEL_NAME,
            "embeddings": config.EMBEDDING_MODEL_NAME,
        },
        "generation": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_NUM_PREDICT,
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
    # Each role reads its own constant, so any one of them can be swapped from
    # config.py alone. The extractor is built further down, only in the branch
    # that uses it: "agentic_graph" reaches the record through
    # config.AGENTIC_LLM_MODEL_NAME instead, and loading a model Ollama then
    # never queries costs several GB of VRAM.
    embeddings = get_embeddings()
    evaluator_llm = build_llm(config.EVALUATOR_LLM_MODEL_NAME)

    # Load both source texts once: the static Brighton reference paper and
    # this run's patient record.
    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    patient_ehr_text = load_ehr_text(patient_ehr_path)

    # Brighton KB is always needed (every mode consults it for synonyms/context).
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)

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
            section_queries=SECTION_QUERIES,
        )
    else:
        # Agent 1's model for the two non-agentic modes.
        llm = build_llm()

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

            try:
                # Agent 1: extraction mode per config.EXTRACTOR_MODE
                print(f"[{section_key}] Agent 1 (extractor, mode={config.EXTRACTOR_MODE}) searching the clinical record...", flush=True)
                t0 = time.time()

                # Only "rag" and "full_text" reach this loop; "agentic_graph"
                # is dispatched separately, above, before this loop runs.
                if config.EXTRACTOR_MODE == "rag":
                    evidence = extract_evidence(llm, ehr_kb, query)
                else:
                    evidence = extract_evidence_full_text(llm, patient_ehr_text, query)
                elapsed = time.time() - t0
                print(f"[{section_key}] Agent 1 done in {elapsed:.1f}s", flush=True)
                section_log["evidence"] = evidence
                section_log["agent1_seconds"] = round(elapsed, 1)

                # Brighton context relevant to this section, with
                # bibliography stripped (see rag_setup.clean_brighton_context).
                brighton_docs = brighton_kb.as_retriever(search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}).invoke(query)
                brighton_context = clean_brighton_context(
                    "\n".join(d.page_content for d in brighton_docs)
                )
                section_log["brighton_context"] = brighton_context

                # Agent 2: evaluation constrained to the section's Pydantic schema
                print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
                t0 = time.time()
                extra_instructions = config.SECTION_HINTS.get(section_key, "")
                section_result, reasoning_text, answer_conflict = evaluate_section(
                    evaluator_llm, section_model, evidence, brighton_context, extra_instructions
                )
                elapsed = time.time() - t0
                print(f"[{section_key}] Agent 2 done in {elapsed:.1f}s", flush=True)
                section_log["agent2_seconds"] = round(elapsed, 1)

                # None unless the model's two answer lines disagreed; kept as a
                # review flag for this section (see agents.evaluate_section).
                section_log["answer_conflict"] = answer_conflict

                # Per-section deterministic gates, individually switchable in
                # config.SECTION_GATES_ENABLED and shared with agentic_graph.py.
                section_result, reasoning_text = apply_section_gates(
                    section_key, section_result, evidence, reasoning_text
                )

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

            audit_log[section_key] = section_log

    # Cross-section dependency rules (see config.CROSS_SECTION_RULES), applied
    # once here regardless of which EXTRACTOR_MODE produced form_data;
    # shared with agentic_graph.py via criteria_rules.py.
    form_data = apply_cross_section_rules(form_data, audit_log)

    # Added last, so it cannot be mistaken for a section by the loops above.
    audit_log[RUN_CONFIG_KEY] = _run_config_snapshot()

    form = DVT_CriteriaForm(**form_data)
    return form, audit_log
