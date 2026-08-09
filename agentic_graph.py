"""
Agentic extraction pipeline orchestrated as an explicit LangGraph state
machine, used when config.EXTRACTOR_MODE == "agentic_graph" (see
pipeline.run_pipeline for the dispatch).

Agent 1 autonomously decides how many times, and with which sub-queries,
to call the EHR search tool for each section (see
agents.extract_evidence_agentic). Control flow itself is expressed as an
explicit graph of nodes/edges instead of a plain Python loop, which makes
each step (select next section / search / answer / finalize) independently
inspectable and testable.

Section ORDER is kept fixed (config.SECTION_ORDER), same as every other
mode: the only thing that varies across modes is how Agent 1 gathers
evidence, not which sections get evaluated or in what order.

Graph shape:
    select_next -> {search_record, finalize} -> answer_criterion -> select_next (loop)

The deterministic keyword gate (criteria_rules.apply_keyword_gate) is
applied inside answer_criterion, exactly like every other mode. Cross-section
dependency rules (criteria_rules.apply_cross_section_rules) are NOT applied
inside this module -- they run once, uniformly for every EXTRACTOR_MODE,
in pipeline.run_pipeline after this graph returns. That keeps a single
source of truth for both safety nets across all execution modes.

Model note: config.LLM_MODEL_NAME ("llama3:8b-instruct-q4_0") does not
support Ollama's native tool-calling API (confirmed: Ollama returns "model
does not support tools", HTTP 400, when a tool is bound to it). Only
Llama 3.1+ has tool-calling support in Ollama, not base Llama 3. So Agent
1's search step here uses a separate model, config.AGENTIC_LLM_MODEL_NAME,
while Agent 2 (evaluator, never binds a tool) uses config.EVALUATOR_LLM_MODEL_NAME
via agents.build_llm(model_name=...) -- both models are built once in
pipeline.py and passed in, see run_agentic_graph_pipeline below.
Requires: ollama pull llama3.1:8b-instruct-q4_0 -- and the `langgraph` package.

"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import config
from models import SECTION_MODELS
from agents import extract_evidence_agentic, evaluate_section
from criteria_rules import apply_keyword_gate, apply_details_gate, apply_absent_pulses_gate


def build_agentic_llm() -> ChatOllama:
    """Tool-calling-capable model, used only by the search_record node.
    Agent 2 (answer_criterion) is built separately in pipeline.py via
    agents.build_llm(config.EVALUATOR_LLM_MODEL_NAME), since it never binds
    any tool. See config.AGENTIC_LLM_MODEL_NAME."""
    return ChatOllama(
        model=config.AGENTIC_LLM_MODEL_NAME,
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
# Graph nodes
# ---------------------------------------------------------------------------

def _select_next(state: GraphState) -> GraphState:
    remaining = state["remaining_sections"]
    # No sections left: signal termination so routing sends us to "finalize".
    if not remaining:
        return {**state, "current_section": None, "done": True}

    # Pop the next section off the queue without mutating the list in
    # place (LangGraph state should be treated as immutable per step).
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


def _make_search_node(llm, ehr_tool, ehr_vectorstore, section_queries: dict):
    def search_record(state: GraphState) -> GraphState:
        section_key = state["current_section"]
        query = section_queries[section_key]

        print(f"[{section_key}] Agent 1 (agentic search) exploring the record...", flush=True)
        try:
            # Agent 1 decides autonomously how/whether to call the search
            # tool, UNION a deterministic fixed-query retrieval floor
            # against ehr_vectorstore -- see agents.extract_evidence_agentic
            # for why the floor was added (B2 repeatedly missing its
            # symptom sentence despite the agent's own search).
            evidence = extract_evidence_agentic(
                llm, ehr_tool, ehr_vectorstore, query, max_iterations=config.AGENTIC_MAX_ITERATIONS
            )
        except Exception as exc:
            # A failed search shouldn't crash the whole graph -- the next
            # node (answer_criterion) treats evidence=None as "no evidence".
            print(f"[{section_key}] Agent 1 FAILED: {exc}", flush=True)
            evidence = None

        # Copy-then-mutate: LangGraph state updates should return a new
        # dict rather than mutating the incoming one in place.
        audit_log = dict(state["audit_log"])
        audit_log[section_key] = {"query": query, "evidence": evidence}
        return {**state, "audit_log": audit_log}

    return search_record


def _make_answer_node(llm, brighton_kb, section_queries: dict):
    def answer_criterion(state: GraphState) -> GraphState:
        section_key = state["current_section"]
        section_model = SECTION_MODELS[section_key]
        query = section_queries[section_key]
        section_log = dict(state["audit_log"].get(section_key, {"query": query}))
        evidence = section_log.get("evidence")

        form_data = dict(state["form_data"])
        audit_log = dict(state["audit_log"])

        # No evidence (search failed or found nothing): skip Agent 2
        # entirely rather than asking it to evaluate an empty answer.
        if not evidence:
            print(f"[{section_key}] No evidence from Agent 1 -- leaving field as None.", flush=True)
            form_data[section_key.lower()] = None
            section_log["error"] = "Agent 1 (agentic) produced no evidence."
            audit_log[section_key] = section_log
            return {**state, "form_data": form_data, "audit_log": audit_log}

        try:
            # Retrieve the Brighton reference context (synonyms/terminology)
            # relevant to this section's query -- same retrieval used by
            # every other EXTRACTOR_MODE.
            brighton_docs = brighton_kb.as_retriever(
                search_kwargs={"k": config.BRIGHTON_RETRIEVER_K}
            ).invoke(query)
            brighton_context = "\n".join(d.page_content for d in brighton_docs)
            section_log["brighton_context"] = brighton_context

            print(f"[{section_key}] Agent 2 (evaluator) filling in the schema...", flush=True)
            # Per-section prompt hints (config.SECTION_HINTS), if any exist
            # for this section.
            extra_instructions = config.SECTION_HINTS.get(section_key, "")
            section_result, reasoning_text = evaluate_section(
                llm, section_model, evidence, brighton_context, extra_instructions
            )

            # Same deterministic keyword gate used by every other mode
            # (see pipeline.py) -- single source of truth in criteria_rules.py.
            section_result, reasoning_text = apply_keyword_gate(
                section_key, section_result, evidence, reasoning_text
            )
            # Same section-F details gate used by every other mode (see
            # criteria_rules.apply_details_gate) -- single source of truth.
            section_result, reasoning_text = apply_details_gate(
                section_key, section_result, reasoning_text
            )
            # Same B2 "Absent pulses" gate used by every other mode (see
            # criteria_rules.apply_absent_pulses_gate) -- single source of truth.
            section_result, reasoning_text = apply_absent_pulses_gate(
                section_key, section_result, evidence, reasoning_text
            )

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
    # Cross-section dependency rules are applied once, uniformly for every
    # EXTRACTOR_MODE, in pipeline.run_pipeline after this graph returns --
    # not here, so that logic has a single source of truth (criteria_rules.py).
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(search_llm, answer_llm, ehr_tool, ehr_vectorstore, brighton_kb, section_queries: dict):
    """search_llm: tool-calling-capable model, used by Agent 1 (search_record).
    answer_llm: config.EVALUATOR_LLM_MODEL_NAME model, used by Agent 2
    (answer_criterion), which needs no tool support. ehr_vectorstore: the same Chroma object
    ehr_tool wraps, passed separately so search_record can also run the
    deterministic retrieval floor (see agents.extract_evidence_agentic)."""
    graph = StateGraph(GraphState)

    # Register the 4 nodes described in the module docstring.
    graph.add_node("select_next", _select_next)
    graph.add_node("search_record", _make_search_node(search_llm, ehr_tool, ehr_vectorstore, section_queries))
    graph.add_node("answer_criterion", _make_answer_node(answer_llm, brighton_kb, section_queries))
    graph.add_node("finalize", _finalize)

    # select_next -> {search_record, finalize} -> answer_criterion -> select_next (loop)
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
# Entry point used by pipeline.run_pipeline
# ---------------------------------------------------------------------------

def run_agentic_graph_pipeline(
    record_id: str,
    evaluator_llm,
    search_llm,
    ehr_tool,
    ehr_vectorstore,
    brighton_kb,
    section_queries: dict,
):
    """
    Runs every section of config.SECTION_ORDER through the graph above.

    Returns (form_data, audit_log) with the SAME shape produced by the
    plain per-section loop in pipeline.py (form_data keyed by lowercased
    section, audit_log keyed by the original casing) -- NOT yet wrapped
    into a DVT_CriteriaForm and NOT yet passed through
    criteria_rules.apply_cross_section_rules; pipeline.run_pipeline does
    both of those once, uniformly, regardless of which mode produced
    form_data.
    """
    app = build_graph(search_llm, evaluator_llm, ehr_tool, ehr_vectorstore, brighton_kb, section_queries)

    # Every section starts unfilled; the queue drives select_next's loop.
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

    return final_state["form_data"], final_state["audit_log"]
