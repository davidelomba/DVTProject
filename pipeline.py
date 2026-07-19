"""
Pipeline principale: per ogni sezione del form,
  1. Agente 1 estrae le evidenze pertinenti dalla cartella clinica (RAG su EHR)
  2. Agente 2 consulta il Brighton KB e compila la crocetta (structured output)
Alla fine unisce tutte le sezioni in un DVT_CriteriaForm e produce il JSON.
La classificazione finale in LOC e' delegata ad aggregation.py (vedi TODO li').
"""

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import get_embeddings, build_brighton_kb, build_ehr_kb, load_brighton_pdf_text
from agents import build_llm, test_structured_output_support, extract_evidence, evaluate_section
from aggregation import form_to_json_summary


# Query di retrieval per ciascuna sezione: guidano l'Agente 1 su cosa cercare
# nella cartella clinica. Da affinare in base al linguaggio reale delle EHR
# che userai (es. abbreviazioni locali, lingua italiana vs inglese).
SECTION_QUERIES = {
    "A1": "referto autoptico, autopsia, evidenza patologica di trombosi venosa profonda",
    "A2": "trombectomia, intervento chirurgico correlato a DVT",
    "A3_1": "ecografia, TC, RM, venografia: esito imaging per trombosi venosa profonda",
    "A3_2": "tipo di esame di imaging eseguito: ecografia compressiva, doppler, venografia",
    "B1_1": "sintomi o segni riportati di trombosi venosa profonda",
    "B1_2": "trombosi venosa profonda arto inferiore o arto superiore",
    "B2": "dolore al polpaccio, gonfiore, edema, arrossamento, calore, polsi assenti",
    "C": "valore D-dimero, data del test, limite superiore di normalita' del laboratorio",
    "F": "diagnosi di trombosi venosa profonda riportata da specialista",
    "X": "diagnosi alternativa che spiega il quadro clinico acuto",
}


def run_pipeline(record_id: str, patient_ehr_text: str, brighton_pdf_path: str) -> DVT_CriteriaForm:
    embeddings = get_embeddings()
    llm = build_llm()

    # Step 0: verifica compatibilita' structured output PRIMA di processare pazienti reali.
    # sample_model qualsiasi tra quelli disponibili, es. C_DDimer (schema semplice).
    from models import C_DDimer
    if not test_structured_output_support(llm, C_DDimer):
        print(
            "[Step 0] ATTENZIONE: with_structured_output non e' affidabile con "
            f"{config.LLM_MODEL_NAME}. Valutare il fallback JSON-mode descritto "
            "in agents.evaluate_section() prima di procedere su dati reali."
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

        # Agente 1: estrazione evidenze (retrieval diretto di default, vedi config)
        if config.USE_AGENTIC_EXTRACTOR:
            from rag_setup import make_ehr_retriever_tool
            from agents import extract_evidence_agentic
            ehr_tool = make_ehr_retriever_tool(ehr_kb)
            evidence = extract_evidence_agentic(llm, ehr_tool, query)
        else:
            evidence = extract_evidence(llm, ehr_kb, query)

        # Contesto Brighton (sinonimi) pertinente per questa sezione
        brighton_docs = brighton_kb.as_retriever(search_kwargs={"k": 3}).invoke(query)
        brighton_context = "\n".join(d.page_content for d in brighton_docs)

        # Agente 2: valutazione vincolata allo schema Pydantic della sezione
        section_result = evaluate_section(llm, section_model, evidence, brighton_context)
        form_data[field_name_map[section_key]] = section_result

        print(f"[{section_key}] compilato: {section_result}")

    form = DVT_CriteriaForm(**form_data)
    return form


if __name__ == "__main__":
    # Esempio di esecuzione -- sostituire i placeholder con i percorsi reali.
    record_id_example = "PATIENT_001"
    patient_ehr_text_example = (
        "Il paziente non presenta edema foveolare. Riferisce dolore al polpaccio "
        "sinistro insorto due giorni fa. Nessun referto autoptico disponibile."
    )
    brighton_pdf_path_example = "/path/to/brighton_dvt_synonyms.pdf"

    form = run_pipeline(record_id_example, patient_ehr_text_example, brighton_pdf_path_example)

    print("\n--- JSON compilato (crocette) ---")
    print(form_to_json_summary(form))
