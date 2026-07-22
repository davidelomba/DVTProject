"""
Main pipeline: for each section of the form,
  1. Agent 1 extracts the relevant evidence from the clinical record (full text by default)
  2. Agent 2 reasons over the evidence and fills in the checkbox
Independent per-section results are then merged, cross-section dependency
rules are applied, and the final DVT_CriteriaForm is returned together with
a full audit log.
"""

import time
import traceback

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import get_embeddings, build_brighton_kb, build_ehr_kb, make_ehr_retriever_tool, load_brighton_pdf_text, load_ehr_text
from agents import build_llm, evaluate_section, extract_evidence, extract_evidence_full_text, extract_evidence_agentic


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


def run_pipeline(record_id: str, patient_ehr_path: str, brighton_pdf_path: str):
    """
    Returns (form, audit_log).

    audit_log is a dict keyed by section (A1, A2, ...) with the evidence
    Agent 1 extracted, the Brighton context given to Agent 2, and Agent 2's
    full reasoning text for whichever attempt succeeded. This makes a wrong
    answer diagnosable after the fact, without re-running the pipeline.
    """
    embeddings = get_embeddings()
    llm = build_llm()

    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    patient_ehr_text = load_ehr_text(patient_ehr_path)
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)

    # Only "rag" and "agentic" modes need the EHR chunked/embedded into a
    # vector store; "full_text" passes the raw record directly per section.
    ehr_kb = None
    ehr_tool = None
    if config.EXTRACTOR_MODE in ("rag", "agentic"):
        ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=record_id, embeddings=embeddings)
        if config.EXTRACTOR_MODE == "agentic":
            ehr_tool = make_ehr_retriever_tool(ehr_kb)

    form_data = {"record_id": record_id}
    audit_log = {}
    # Section keys map to DVT_CriteriaForm fields via lower-casing
    # (e.g. "A3_1" -> "a3_1"); no explicit mapping dict needed.

    for section_key in config.SECTION_ORDER:
        section_model = SECTION_MODELS[section_key]
        query = SECTION_QUERIES[section_key]

        print(f"\n=== Section {section_key} ===", flush=True)
        section_log = {"query": query}

        try:
            # Agent 1: extraction mode per config.EXTRACTOR_MODE (see config.py)
            print(f"[{section_key}] Agent 1 (extractor, mode={config.EXTRACTOR_MODE}) searching the clinical record...", flush=True)
            t0 = time.time()
            if config.EXTRACTOR_MODE == "agentic":
                evidence = extract_evidence_agentic(llm, ehr_tool, query, max_iterations=config.AGENTIC_MAX_ITERATIONS)
            elif config.EXTRACTOR_MODE == "rag":
                evidence = extract_evidence(llm, ehr_kb, query)
            else:
                evidence = extract_evidence_full_text(llm, patient_ehr_text, query)
            print(f"[{section_key}] Agent 1 done in {time.time() - t0:.1f}s", flush=True)
            section_log["evidence"] = evidence

            # Brighton context (synonyms) relevant to this section
            brighton_docs = brighton_kb.as_retriever(search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}).invoke(query)
            brighton_context = "\n".join(d.page_content for d in brighton_docs)
            section_log["brighton_context"] = brighton_context

            # Agent 2: evaluation constrained to the section's Pydantic schema
            print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
            t0 = time.time()
            extra_instructions = config.SECTION_HINTS.get(section_key, "")
            section_result, reasoning_text = evaluate_section(
                llm, section_model, evidence, brighton_context, extra_instructions
            )
            print(f"[{section_key}] Agent 2 done in {time.time() - t0:.1f}s", flush=True)

            # --- Deterministic keyword gate (see config.SECTION_KEYWORD_GATES) ---
            gate_info = config.SECTION_KEYWORD_GATES.get(section_key)
            if gate_info:
                field_name = list(type(section_result).model_fields.keys())[0]
                llm_chosen_answer = getattr(section_result, field_name)

                if isinstance(llm_chosen_answer, list):
                    is_positive = any(ans != gate_info["default_option_text"] for ans in llm_chosen_answer)
                else:
                    is_positive = (llm_chosen_answer != gate_info["default_option_text"])

                if is_positive:
                    evidence_lower = evidence.lower()
                    has_keyword = any(kw.lower() in evidence_lower for kw in gate_info["keywords"])

                    if not has_keyword:
                        print(f"[{section_key}] SOFT GATE TRIGGERED: LLM hallucinated a positive answer. Reverting to default.", flush=True)
                        # Rebuild via the model (not setattr) to preserve validation.
                        section_result = type(section_result)(**{field_name: gate_info["default_option_text"]})
                        reasoning_text += (
                            f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
                            f"but no triggering keywords {gate_info['keywords']} were found in the evidence. "
                            f"Answer was automatically reverted to the negative default."
                        )
            # -----------------------------------------------------------------

            section_log["reasoning"] = reasoning_text
            section_log["result"] = section_result.model_dump()

            form_data[section_key.lower()] = section_result
            print(f"[{section_key}] filled in: {section_result}", flush=True)

        except Exception as exc:
            # One section failing shouldn't lose work already done on others.
            print(f"[{section_key}] FAILED -- leaving this field as None. Traceback:", flush=True)
            traceback.print_exc()
            form_data[section_key.lower()] = None
            section_log["error"] = str(exc)

        audit_log[section_key] = section_log

    # --- Cross-section dependency rules (see config.CROSS_SECTION_RULES) ---
    for rule in config.CROSS_SECTION_RULES:
        if_result = form_data.get(rule["if_section"])
        then_result = form_data.get(rule["then_section"])

        if if_result is None or then_result is None:
            continue

        if_field = list(type(if_result).model_fields.keys())[0]
        then_field = list(type(then_result).model_fields.keys())[0]

        if_answers = getattr(if_result, if_field)
        if not isinstance(if_answers, list):
            if_answers = [if_answers]

        has_non_default = any(ans != rule["none_option"] for ans in if_answers)

        if has_non_default:
            current_value = getattr(then_result, then_field)
            if current_value != rule["forced_value"]:
                print(
                    f"[CROSS-SECTION RULE] '{rule['if_section']}' triggered. "
                    f"Forcing '{rule['then_section']}' to '{rule['forced_value']}'.",
                    flush=True,
                )
                form_data[rule["then_section"]] = type(then_result)(
                    **{then_field: rule["forced_value"]}
                )
                audit_key = rule["audit_key"]
                if audit_key in audit_log:
                    audit_log[audit_key]["reasoning"] = (
                        audit_log[audit_key].get("reasoning", "")
                        + f"\n\n[SYSTEM OVERRIDE]: {rule['override_message']}"
                    )
    # -------------------------------------------------------------------------

    form = DVT_CriteriaForm(**form_data)
    return form, audit_log
