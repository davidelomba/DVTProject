"""
Main pipeline: for each section of the form,
  1. Agent 1 extracts the relevant evidence from the clinical record (RAG on EHR)
  2. Agent 2 consults the Brighton KB and fills in the checkbox (structured output)
At the end, merges all sections into a DVT_CriteriaForm and produces the JSON.
"""

import time
import traceback

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import get_embeddings, build_brighton_kb, build_ehr_kb, load_brighton_pdf_text, load_ehr_text
from agents import build_llm, extract_evidence, evaluate_section
from aggregation import form_to_json_summary


# Retrieval query for each section: guides Agent 1 on what to look for in the
# clinical record. Refine based on the actual language of the EHRs you'll use
# (e.g. local abbreviations, Italian vs English).
SECTION_QUERIES = {
    "A1": "autopsy report, pathologic evidence of deep vein thrombosis",
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
    Agent 1 retrieved, the Brighton context given to Agent 2, and Agent 2's
    full free-form reasoning text (including its FINAL_ANSWER line) for
    whichever attempt succeeded. This is what makes a case like "the model
    reasoned correctly but reported the wrong option/index" (or the
    reverse) diagnosable after the fact, instead of requiring the whole
    pipeline to be re-run in debug mode to find out.
    """
    # Run test_step0.py separately before processing real patients, to confirm
    # the model in config.py is currently behaving reliably -- not repeated
    # here on every run since it takes several minutes on its own.
    embeddings = get_embeddings()
    llm = build_llm()

    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    patient_ehr_text = load_ehr_text(patient_ehr_path)
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)
    ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=record_id, embeddings=embeddings)

    form_data = {"record_id": record_id}
    audit_log = {}
    field_name_map = {
        "A1": "a1", "A2": "a2", "A3_1": "a3_1", "A3_2": "a3_2",
        "B1_1": "b1_1", "B1_2": "b1_2", "B2": "b2",
        "C": "c", "F": "f", "X": "x",
    }

    for section_key in config.SECTION_ORDER:
        section_model = SECTION_MODELS[section_key]
        query = SECTION_QUERIES[section_key]

        print(f"\n=== Section {section_key} ===", flush=True)
        section_log = {"query": query}

        try:
            # Agent 1: evidence extraction (direct retrieval by default, see config)
            print(f"[{section_key}] Agent 1 (extractor) searching the clinical record...", flush=True)
            t0 = time.time()
            if getattr(config, "USE_AGENTIC_EXTRACTOR", False):
                from rag_setup import make_ehr_retriever_tool
                from agents import extract_evidence_agentic
                ehr_tool = make_ehr_retriever_tool(ehr_kb)
                evidence = extract_evidence_agentic(llm, ehr_tool, query)
            else:
                evidence = extract_evidence(llm, ehr_kb, query)
            print(f"[{section_key}] Agent 1 done in {time.time() - t0:.1f}s", flush=True)
            section_log["evidence"] = evidence

            # Brighton context (synonyms) relevant to this section
            brighton_docs = brighton_kb.as_retriever(search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}).invoke(query)
            brighton_context = "\n".join(d.page_content for d in brighton_docs)
            section_log["brighton_context"] = brighton_context

            # Agent 2: evaluation constrained to the section's Pydantic schema.
            # (Soft Gate: Always call LLM, check for hallucinations afterwards)
            print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
            t0 = time.time()
            extra_instructions = config.SECTION_HINTS.get(section_key, "")
            section_result, reasoning_text = evaluate_section(
                llm, section_model, evidence, brighton_context, extra_instructions
            )
            print(f"[{section_key}] Agent 2 done in {time.time() - t0:.1f}s", flush=True)

            # --- SOFT GATE CROSS-CHECK ---
            gate_info = getattr(config, "SECTION_KEYWORD_GATES", {}).get(section_key)
            if gate_info:
                # Estraiamo dinamicamente il nome del campo (es. 'answer')
                field_name = list(type(section_result).model_fields.keys())[0]
                llm_chosen_answer = getattr(section_result, field_name)
                
                # Verifichiamo se l'LLM ha dato una risposta positiva (diversa dal default)
                if isinstance(llm_chosen_answer, list):
                    is_positive = any(ans != gate_info["default_option_text"] for ans in llm_chosen_answer)
                else:
                    is_positive = (llm_chosen_answer != gate_info["default_option_text"])
                
                if is_positive:
                    evidence_lower = evidence.lower()
                    has_keyword = any(kw.lower() in evidence_lower for kw in gate_info["keywords"])
                    
                    if not has_keyword:
                        print(f"[{section_key}] SOFT GATE TRIGGERED: LLM hallucinated a positive answer. Reverting to default.", flush=True)
                        
                        # Sovrascriviamo la risposta col default negativo
                        setattr(section_result, field_name, gate_info["default_option_text"])
                        
                        # Tracciamo l'intervento nell'audit log per trasparenza
                        reasoning_text += (
                            f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
                            f"but no triggering keywords {gate_info['keywords']} were found in the evidence. "
                            f"Answer was automatically reverted to the negative default."
                        )
            # -------------------------------

            section_log["reasoning"] = reasoning_text
            section_log["result"] = section_result.model_dump()

            form_data[field_name_map[section_key]] = section_result
            print(f"[{section_key}] filled in: {section_result}", flush=True)

        except Exception as exc:
            # Partial-failure resilience: one section failing (e.g. retries
            # exhausted in evaluate_section, or an unexpected error) should
            # not lose the work already done on other sections. The field
            # is left as None (DVT_CriteriaForm allows this on every field).
            print(f"[{section_key}] FAILED -- leaving this field as None. Traceback:", flush=True)
            traceback.print_exc()
            form_data[field_name_map[section_key]] = None
            section_log["error"] = str(exc)

        audit_log[section_key] = section_log

    # ---------------------------------------------------------------------
    # HARD-CODED LOGIC RULE: B2 -> B1.1 Dependency
    # "If at least one of the first four answers in B2 is selected, 
    # it implies that in B1.1 the first answer should be selected as well."
    # ---------------------------------------------------------------------
    b2_result = form_data.get("b2")
    b1_1_result = form_data.get("b1_1")

    if b2_result is not None and b1_1_result is not None:
        b2_field = list(type(b2_result).model_fields.keys())[0]
        b1_1_field = list(type(b1_1_result).model_fields.keys())[0]
        
        b2_answers = getattr(b2_result, b2_field)
        
        has_actual_symptoms = any(
            "None of the above" not in ans for ans in b2_answers
        )
        
        if has_actual_symptoms:
            forced_b1_1_answer = "≥1 symptom or sign of DVT was reported"
            
            if getattr(b1_1_result, b1_1_field) != forced_b1_1_answer:
                print(f"[CROSS-SECTION RULE] B2 has symptoms. Forcing B1.1 to '{forced_b1_1_answer}'", flush=True)
                setattr(b1_1_result, b1_1_field, forced_b1_1_answer)
                
                # Tracciamo la correzione nell'audit log
                if "B1_1" in audit_log:
                    audit_log["B1_1"]["reasoning"] += (
                        "\n\n[SYSTEM OVERRIDE]: B1.1 was automatically updated to "
                        "'≥1 symptom or sign of DVT was reported' because symptoms "
                        "were detected in Section B2, enforcing the questionnaire's dependency rule."
                    )

    form = DVT_CriteriaForm(**form_data)
    return form, audit_log


if __name__ == "__main__":
    import json
    import os

    # Replace these two paths with the actual locations of your files.
    record_id_example = "PATIENT_001"
    patient_ehr_path_example = "./patient_001.txt"          # plain .txt clinical record
    brighton_pdf_path_example = "./1-s2.0-S0264410X22010854-main.pdf"  # Brighton paper PDF

    form, audit_log = run_pipeline(record_id_example, patient_ehr_path_example, brighton_pdf_path_example)

    summary = form_to_json_summary(form)

    print("\n--- Filled-in JSON (checkboxes) ---")
    print(json.dumps(summary, indent=2))

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{record_id_example}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Audit log: evidence, Brighton context, and full model reasoning for
    # every section, including failed ones. Kept as a separate file (not
    # merged into the clean output JSON) so it doesn't need to be shared
    # downstream, but is available whenever a specific answer needs to be
    # checked without re-running the pipeline.
    audit_path = os.path.join(output_dir, f"{record_id_example}_audit_log.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {os.path.abspath(output_path)}")
    print(f"Audit log saved to: {os.path.abspath(audit_path)}")