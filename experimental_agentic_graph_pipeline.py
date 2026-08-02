"""
EXPERIMENTAL pipeline variant: Agent 1 (extractor) explores the clinical
record autonomously for EVERY section, orchestrated as a LangGraph state
graph instead of the plain Python loop in pipeline.py.

This file does NOT modify pipeline.py, agents.py, models.py, config.py or
rag_setup.py -- it only imports and reuses their existing functions/objects.
It is meant to be run standalone (python experimental_agentic_graph_pipeline.py)
to compare its output/behavior against the default pipeline (main.py) before
considering whether to fold any of this into the main pipeline.

Design (see conversation / README context for the two options considered):
- Section ORDER is kept fixed (config.SECTION_ORDER), same as pipeline.py.
  This isolates the actual experimental variable: how Agent 1 looks at the
  record for each section (tool-calling / agentic search, see
  agents.extract_evidence_agentic) instead of full_text.
- Graph nodes:
    select_next     -> pops the next section off the queue (deterministic).
    search_record   -> Agent 1, agentic: autonomously decides how many times
                        (and with which sub-queries) to call the EHR search
                        tool before returning evidence for the current section.
    answer_criterion -> Agent 2 (agents.evaluate_section, unchanged), plus
                        the same deterministic keyword gate used in
                        pipeline.py (config.SECTION_KEYWORD_GATES), since
                        that safety net isn't exposed as a standalone
                        function to import.
    finalize        -> applies config.CROSS_SECTION_RULES (same logic as
                        pipeline.py, reimplemented here since it also isn't
                        exposed as a standalone function) and returns.
  select_next -> {search_record, finalize} -> answer_criterion -> select_next (loop)

Requires the `langgraph` package (already listed in requirements.txt) and
the base `langchain` package (needed by agents.extract_evidence_agentic).

Model note: config.LLM_MODEL_NAME ("llama3:8b-instruct-q4_0") does not
support Ollama's native tool-calling API (confirmed: Ollama returns "model
does not support tools", HTTP 400, when a tool is bound to it). Llama 3
(base) never got tool-calling support in Ollama; only Llama 3.1+ does. So
Agent 1's agentic search here uses a SEPARATE model (AGENTIC_LLM_MODEL_NAME
below, "llama3.1:8b-instruct-q4_0") while Agent 2 (evaluator, no tools
involved) keeps using config.LLM_MODEL_NAME via agents.build_llm(),
unchanged. Requires: ollama pull llama3.1:8b-instruct-q4_0

Caveat: NOT validated. Agent 1's agentic search is explicitly marked
unreliable in agents.py/config.py; running it across all 10 sections (instead
of the 3-section smoke test in debug_agentic_extractor.py) is itself part of
the experiment.
"""

import json
import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import config
from models import DVT_CriteriaForm, SECTION_MODELS
from rag_setup import (
    get_embeddings,
    build_brighton_kb,
    build_ehr_kb,
    make_ehr_retriever_tool,
    load_brighton_pdf_text,
    load_ehr_text,
)
from agents import build_llm, extract_evidence_agentic, evaluate_section
from pipeline import SECTION_QUERIES  # reused as-is, not duplicated
from aggregation import form_to_json_summary


# ---------------------------------------------------------------------------
# Separate tool-calling-capable model for Agent 1's agentic search
# ---------------------------------------------------------------------------
# config.LLM_MODEL_NAME does not support Ollama's native tool-calling API
# (see module docstring). This constant picks a model that does, used ONLY
# by the search_record node. Swap it for another tool-capable model (e.g.
# "mistral:7b") if preferred -- see https://ollama.com/search?c=tools
#
# Requires: ollama pull llama3.1:8b-instruct-q4_0
AGENTIC_LLM_MODEL_NAME = "llama3.1:8b-instruct-q4_0"


def build_agentic_llm() -> ChatOllama:
    """Tool-calling-capable model, used only by the search_record node.
    Agent 2 (answer_criterion) keeps using agents.build_llm() unchanged,
    since it never binds any tool."""
    return ChatOllama(
        model=AGENTIC_LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    record_id: str
    remaining_sections: list
    current_section: Optional[str]
    form_data: dict
    audit_log: dict
    done: bool


# ---------------------------------------------------------------------------
# Cross-section rules (reimplemented from pipeline.py; not imported because
# pipeline.py applies them inline inside run_pipeline rather than exposing
# them as a standalone function)
# ---------------------------------------------------------------------------

def _apply_cross_section_rules(form_data: dict, audit_log: dict) -> dict:
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
    return form_data


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _select_next(state: GraphState) -> GraphState:
    remaining = state["remaining_sections"]
    if not remaining:
        return {**state, "current_section": None, "done": True}

    next_section = remaining[0]
    print(f"\n=== Section {next_section} ===", flush=True)
    return {
        **state,
        "current_section": next_section,
        "remaining_sections": remaining[1:],
        "done": False,
    }


def _route_after_select(state: GraphState) -> str:
    return "finalize" if state["done"] else "search_record"


def _make_search_node(llm, ehr_tool):
    def search_record(state: GraphState) -> GraphState:
        section_key = state["current_section"]
        query = SECTION_QUERIES[section_key]

        print(f"[{section_key}] Agent 1 (agentic search) exploring the record...", flush=True)
        try:
            evidence = extract_evidence_agentic(
                llm, ehr_tool, query, max_iterations=config.AGENTIC_MAX_ITERATIONS
            )
        except Exception as exc:
            print(f"[{section_key}] Agent 1 FAILED: {exc}", flush=True)
            evidence = None

        audit_log = dict(state["audit_log"])
        audit_log[section_key] = {"query": query, "evidence": evidence}
        return {**state, "audit_log": audit_log}

    return search_record


def _make_answer_node(llm, brighton_kb):
    def answer_criterion(state: GraphState) -> GraphState:
        section_key = state["current_section"]
        section_model = SECTION_MODELS[section_key]
        query = SECTION_QUERIES[section_key]
        section_log = dict(state["audit_log"].get(section_key, {"query": query}))
        evidence = section_log.get("evidence")

        form_data = dict(state["form_data"])
        audit_log = dict(state["audit_log"])

        if not evidence:
            print(f"[{section_key}] No evidence from Agent 1 -- leaving field as None.", flush=True)
            form_data[section_key.lower()] = None
            section_log["error"] = "Agent 1 (agentic) produced no evidence."
            audit_log[section_key] = section_log
            return {**state, "form_data": form_data, "audit_log": audit_log}

        try:
            brighton_docs = brighton_kb.as_retriever(
                search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}
            ).invoke(query)
            brighton_context = "\n".join(d.page_content for d in brighton_docs)
            section_log["brighton_context"] = brighton_context

            print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
            extra_instructions = config.SECTION_HINTS.get(section_key, "")
            section_result, reasoning_text = evaluate_section(
                llm, section_model, evidence, brighton_context, extra_instructions
            )

            # --- Same deterministic keyword gate as pipeline.py ---
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
                        print(f"[{section_key}] SOFT GATE TRIGGERED: reverting to default.", flush=True)
                        section_result = type(section_result)(**{field_name: gate_info["default_option_text"]})
                        reasoning_text += (
                            f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
                            f"but no triggering keywords {gate_info['keywords']} were found in the evidence. "
                            f"Answer was automatically reverted to the negative default."
                        )
            # -------------------------------------------------------

            section_log["reasoning"] = reasoning_text
            section_log["result"] = section_result.model_dump()
            form_data[section_key.lower()] = section_result
            print(f"[{section_key}] filled in: {section_result}", flush=True)

        except Exception as exc:
            print(f"[{section_key}] Agent 2 FAILED -- leaving field as None: {exc}", flush=True)
            form_data[section_key.lower()] = None
            section_log["error"] = str(exc)

        audit_log[section_key] = section_log
        return {**state, "form_data": form_data, "audit_log": audit_log}

    return answer_criterion


def _finalize(state: GraphState) -> GraphState:
    form_data = _apply_cross_section_rules(dict(state["form_data"]), dict(state["audit_log"]))
    return {**state, "form_data": form_data}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(search_llm, answer_llm, ehr_tool, brighton_kb):
    """search_llm: tool-calling-capable model, used by Agent 1 (search_record).
    answer_llm: config.LLM_MODEL_NAME model, used by Agent 2 (answer_criterion),
    which needs no tool support."""
    graph = StateGraph(GraphState)

    graph.add_node("select_next", _select_next)
    graph.add_node("search_record", _make_search_node(search_llm, ehr_tool))
    graph.add_node("answer_criterion", _make_answer_node(answer_llm, brighton_kb))
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("select_next")
    graph.add_conditional_edges(
        "select_next",
        _route_after_select,
        {"search_record": "search_record", "finalize": "finalize"},
    )
    graph.add_edge("search_record", "answer_criterion")
    graph.add_edge("answer_criterion", "select_next")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_experimental_pipeline(record_id: str, patient_ehr_path: str, brighton_pdf_path: str):
    """Same return shape as pipeline.run_pipeline: (form, audit_log)."""
    embeddings = get_embeddings()
    llm = build_llm()                  # Agent 2 (evaluator) -- unchanged model, no tools needed
    agentic_llm = build_agentic_llm()  # Agent 1 (agentic search) -- tool-calling-capable model

    brighton_text = load_brighton_pdf_text(brighton_pdf_path)
    patient_ehr_text = load_ehr_text(patient_ehr_path)
    brighton_kb = build_brighton_kb(brighton_text, embeddings=embeddings)

    # Agentic search always needs the EHR chunked/embedded (see rag_setup.py).
    ehr_kb = build_ehr_kb(patient_ehr_text, patient_id=record_id, embeddings=embeddings)
    ehr_tool = make_ehr_retriever_tool(ehr_kb)

    app = build_graph(agentic_llm, llm, ehr_tool, brighton_kb)

    initial_state: GraphState = {
        "record_id": record_id,
        "remaining_sections": list(config.SECTION_ORDER),
        "current_section": None,
        "form_data": {"record_id": record_id},
        "audit_log": {},
        "done": False,
    }

    # 3 graph steps per section (select_next, search_record, answer_criterion)
    # plus entry/finalize overhead; default langgraph limit (25) is too low
    # for 10 sections.
    recursion_limit = len(config.SECTION_ORDER) * 3 + 10
    final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

    form = DVT_CriteriaForm(**final_state["form_data"])
    return form, final_state["audit_log"]


def main():
    record_id = "PATIENT_001_AGENTIC_GRAPH"
    patient_ehr_path = "./patient_001.txt"
    brighton_pdf_path = "./1-s2.0-S0264410X22010854-main.pdf"

    form, audit_log = run_experimental_pipeline(record_id, patient_ehr_path, brighton_pdf_path)
    summary = form_to_json_summary(form)

    print("\n--- Filled-in JSON (checkboxes) ---")
    print(json.dumps(summary, indent=2))

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Distinct filenames from main.py's output, so runs don't overwrite each other.
    output_path = os.path.join(output_dir, f"{record_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    audit_path = os.path.join(output_dir, f"{record_id}_audit_log.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {os.path.abspath(output_path)}")
    print(f"Audit log saved to: {os.path.abspath(audit_path)}")


if __name__ == "__main__":
    main()
