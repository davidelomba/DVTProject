"""
Main pipeline: for each section of the form,
  1. Agent 1 extracts the relevant evidence from the clinical record (RAG on EHR)
  2. Agent 2 consults the Brighton KB and fills in the checkbox (structured output)
At the end, merges all sections into a DVT_CriteriaForm and produces the JSON.
"""

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import get_embeddings, build_brighton_kb, build_ehr_kb, load_brighton_pdf_text
from agents import build_llm, test_structured_output_support, extract_evidence, evaluate_section
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


def run_pipeline(record_id: str, patient_ehr_text: str, brighton_pdf_path: str) -> DVT_CriteriaForm:
    embeddings = get_embeddings()
    llm = build_llm()

    # Step 0: verify structured-output compatibility BEFORE processing real patients.
    # sample_model can be any of the available ones, e.g. C_DDimer (simple schema).
    from models import C_DDimer
    if not test_structured_output_support(llm, C_DDimer):
        print(
            "[Step 0] WARNING: with_structured_output is not reliable with "
            f"{config.LLM_MODEL_NAME}. Consider the JSON-mode fallback "
            "described in agents.evaluate_section() before proceeding on real data."
        )

    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)
    ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=record_id, embeddings=embeddings)

    form_data = {"record_id": record_id}
    field_name_map = {
        "A1": "a1", "A2": "a2", "A3_1": "a3_1", "A3_2": "a3_2",
        "B1_1": "b1_1", "B1_2": "b1_2", "B2": "b2",
        "C": "c", "F": "f", "X": "x",
    }

    for section_key in config.SECTION_ORDER:
        section_model = SECTION_MODELS[section_key]
        query = SECTION_QUERIES[section_key]

        # Agent 1: evidence extraction (direct retrieval by default, see config)
        if config.USE_AGENTIC_EXTRACTOR:
            from rag_setup import make_ehr_retriever_tool
            from agents import extract_evidence_agentic
            ehr_tool = make_ehr_retriever_tool(ehr_kb)
            evidence = extract_evidence_agentic(llm, ehr_tool, query)
        else:
            evidence = extract_evidence(llm, ehr_kb, query)

        # Brighton context (synonyms) relevant to this section
        brighton_docs = brighton_kb.as_retriever(search_kwargs={"k": 3}).invoke(query)
        brighton_context = "\n".join(d.page_content for d in brighton_docs)

        # Agent 2: evaluation constrained to the section's Pydantic schema
        section_result = evaluate_section(llm, section_model, evidence, brighton_context)
        form_data[field_name_map[section_key]] = section_result

        print(f"[{section_key}] filled in: {section_result}")

    form = DVT_CriteriaForm(**form_data)
    return form


if __name__ == "__main__":
    # Example run -- replace the placeholders with real paths.
    record_id_example = "PATIENT_001"
    patient_ehr_text_example = (
        "The patient shows no pitting edema. Reports calf pain in the left leg "
        "that started two days ago. No autopsy report available."
    )
    brighton_pdf_path_example = "/path/to/brighton_dvt_synonyms.pdf"

    form = run_pipeline(record_id_example, patient_ehr_text_example, brighton_pdf_path_example)

    print("\n--- Filled-in JSON (checkboxes) ---")
    print(form_to_json_summary(form))
