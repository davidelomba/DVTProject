"""
Diagnostic script for the agentic (tool-calling) extractor.

Compares extract_evidence_agentic against extract_evidence_full_text on the
same patient record and the same section queries, so you can see whether
the agentic version finds equivalent (or better/worse) evidence, and
whether it behaves reliably (no runaway tool-calling loops, no malformed
tool invocations) before trusting it inside the full pipeline.

Requires: pip install langchain (base package, not just langchain-core/
langchain-community/langchain-ollama).

Usage:
    python debug_agentic_extractor.py
"""

import time

import config
from agents import build_llm, extract_evidence_full_text, extract_evidence_agentic
from rag_setup import get_embeddings, build_ehr_kb, make_ehr_retriever_tool, load_ehr_text

# Edit this to point to your actual patient file.
PATIENT_EHR_PATH = "./patient_001.txt"
RECORD_ID = "PATIENT_001_AGENTIC_TEST"

# A subset of section queries -- enough to compare behavior without waiting
# for all 10 sections. Add more from pipeline.SECTION_QUERIES if needed.
TEST_QUERIES = {
    "A1": "autopsy report, necropsy, post-mortem examination, autoptic findings",
    "C": "D-dimer value, test date, laboratory upper limit of normal",
    "B2": "calf pain, swelling, oedema, redness, warmth, absent pulses",
}


def run_comparison():
    llm = build_llm()
    patient_ehr_text = load_ehr_text(PATIENT_EHR_PATH)

    # The agentic extractor needs the record chunked/embedded and wrapped
    # as a searchable tool -- full_text mode below reuses the same llm but
    # needs neither of these.
    print("Building EHR knowledge base for the agentic extractor...")
    embeddings = get_embeddings()
    ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=RECORD_ID, embeddings=embeddings)
    ehr_tool = make_ehr_retriever_tool(ehr_kb)
    print("Done.\n")

    for section_key, query in TEST_QUERIES.items():
        print(f"{'=' * 70}\nSection {section_key} -- query: {query}\n{'=' * 70}")

        # Baseline: whole record passed directly, no retrieval involved.
        print("\n--- full_text mode (current default) ---")
        t0 = time.time()
        result_full = extract_evidence_full_text(llm, patient_ehr_text, query)
        print(f"Done in {time.time() - t0:.1f}s")
        print(result_full)

        # Comparison: model autonomously decides how to search the tool.
        # Wrapped in try/except since this path is explicitly unvalidated
        # and can fail in ways full_text mode can't (e.g. runaway tool use).
        print("\n--- agentic mode (tool-calling) ---")
        t0 = time.time()
        try:
            result_agentic = extract_evidence_agentic(
                llm, ehr_tool, query, max_iterations=config.AGENTIC_MAX_ITERATIONS
            )
            print(f"Done in {time.time() - t0:.1f}s")
            print(result_agentic)
        except Exception as exc:
            print(f"FAILED after {time.time() - t0:.1f}s: {type(exc).__name__} -- {exc}")

        print()


if __name__ == "__main__":
    run_comparison()
