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

The per-section gates (criteria_rules.apply_section_gates) are applied inside
answer_criterion, exactly as in every other mode. Cross-section dependency
rules are NOT applied here: they run once in pipeline.run_pipeline after this
graph returns, so both safety nets have a single source of truth across all
execution modes.

Model note: base Llama 3 does not support Ollama's tool-calling API (binding a
tool to it returns HTTP 400, "model does not support tools"); only Llama 3.1+
does. Agent 1's search step therefore uses config.AGENTIC_LLM_MODEL_NAME,
while Agent 2 never binds a tool and uses config.EVALUATOR_LLM_MODEL_NAME.
Both models are built once in pipeline.py and passed into
run_agentic_graph_pipeline.

Requires the `langgraph` package and a tool-calling model pulled in Ollama.
"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import config
from models import SECTION_MODELS
from agents import extract_evidence_agentic, evaluate_section
from criteria_rules import apply_section_gates


def build_agentic_llm() -> ChatOllama:
    """Builds the tool-calling model used by the search_record node.

    Separate from agents.build_llm because this is the only role that binds a
    tool, and therefore the only one that needs a model supporting Ollama's
    tool-calling API (config.AGENTIC_LLM_MODEL_NAME).

    Returns:
        A configured ChatOllama instance.
    """
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
    """State threaded through every node of the graph.

    remaining_sections is consumed one at a time by _select_next; form_data and
    audit_log accumulate the results, keyed as pipeline.run_pipeline expects
    them; done tells _route_after_select when to stop looping.
    """

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
    """Takes the next section off the queue, or signals that none are left.

    Args:
        state: the current graph state.

    Returns:
        A new state with current_section set, or with done=True when the queue
        is empty so _route_after_select ends the loop.
    """
    remaining = state["remaining_sections"]
    if not remaining:
        return {**state, "current_section": None, "done": True}

    # Returned as a new list rather than popped in place: graph state is
    # treated as immutable from one step to the next.
    next_section = remaining[0]
    print(f"\n=== Section {next_section} ===", flush=True)
    return {
        **state,
        "current_section": next_section,
        "remaining_sections": remaining[1:],
        "done": False,
    }


def _route_after_select(state: GraphState) -> str:
    """Routes to the next node after _select_next: another section or the end."""
    return "finalize" if state["done"] else "search_record"


def _make_search_node(llm, ehr_tool, ehr_vectorstore, section_queries: dict):
    """Builds the search_record node, closing over its dependencies.

    A factory rather than a plain node because LangGraph nodes take only the
    state, while this step also needs the model, the tool and the queries.

    Args:
        llm: the tool-calling model from build_agentic_llm.
        ehr_tool: the retriever wrapped as a tool.
        ehr_vectorstore: passed through to agents.extract_evidence_agentic.
        section_queries: section key -> retrieval query.

    Returns:
        The node function.
    """

    def search_record(state: GraphState) -> GraphState:
        """Runs Agent 1 for the current section and records what it found."""
        section_key = state["current_section"]
        query = section_queries[section_key]

        print(f"[{section_key}] Agent 1 (agentic search) exploring the record...", flush=True)
        try:
            evidence = extract_evidence_agentic(
                llm, ehr_tool, ehr_vectorstore, query, max_iterations=config.AGENTIC_MAX_ITERATIONS
            )
        except Exception as exc:
            # A failed search must not crash the graph: the next node treats
            # a missing evidence value as "no evidence found".
            print(f"[{section_key}] Agent 1 FAILED: {exc}", flush=True)
            evidence = None

        audit_log = dict(state["audit_log"])
        audit_log[section_key] = {"query": query, "evidence": evidence}
        return {**state, "audit_log": audit_log}

    return search_record


def _make_answer_node(llm, brighton_kb, section_queries: dict):
    """Builds the answer_criterion node, closing over its dependencies.

    Args:
        llm: the evaluator model (Agent 2).
        brighton_kb: vector store of the reference guideline.
        section_queries: section key -> retrieval query, reused here to fetch
            the guideline context for the section.

    Returns:
        The node function, which runs Agent 2 and then the same deterministic
        gates every other execution mode applies.
    """

    def answer_criterion(state: GraphState) -> GraphState:
        """Runs Agent 2 and the deterministic gates for the current section.

        A section that fails is left as None and its error recorded, so one bad
        section does not lose the work already done on the others.
        """
        section_key = state["current_section"]
        section_model = SECTION_MODELS[section_key]
        query = section_queries[section_key]
        section_log = dict(state["audit_log"].get(section_key, {"query": query}))
        evidence = section_log.get("evidence")

        form_data = dict(state["form_data"])
        audit_log = dict(state["audit_log"])

        # Search failed or found nothing: skip Agent 2 rather than ask it to
        # reason over an empty evidence string.
        if not evidence:
            print(f"[{section_key}] No evidence from Agent 1 -- leaving field as None.", flush=True)
            form_data[section_key.lower()] = None
            section_log["error"] = "Agent 1 (agentic) produced no evidence."
            audit_log[section_key] = section_log
            return {**state, "form_data": form_data, "audit_log": audit_log}

        try:
            # Guideline terminology for this section, retrieved exactly as in
            # every other execution mode.
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

            # The same per-section gates pipeline.py applies: criteria_rules.py
            # is the single source of truth for every mode.
            section_result, reasoning_text = apply_section_gates(
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
    """Terminal node: returns the state unchanged.

    Deliberately a passthrough. Cross-section rules are applied once by
    pipeline.run_pipeline after this graph returns, so every execution mode
    goes through the same code rather than a copy of it.
    """
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(search_llm, answer_llm, ehr_tool, ehr_vectorstore, brighton_kb, section_queries: dict):
    """Wires the four nodes into the state machine and compiles it.

    Args:
        search_llm: tool-calling model for Agent 1 (search_record).
        answer_llm: model for Agent 2 (answer_criterion), needs no tools.
        ehr_tool: the record retriever wrapped as a tool.
        ehr_vectorstore: the store ehr_tool wraps, passed separately because
            agents.extract_evidence_agentic takes it as its own argument.
        brighton_kb: vector store of the reference guideline.
        section_queries: section key -> retrieval query.

    Returns:
        The compiled graph, ready to invoke with an initial GraphState.
    """
    graph = StateGraph(GraphState)

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
    """Runs every section of config.SECTION_ORDER through the graph.

    Args:
        record_id: identifier carried into the results.
        evaluator_llm: model for Agent 2.
        search_llm: tool-calling model for Agent 1.
        ehr_tool: the record retriever wrapped as a tool.
        ehr_vectorstore: the store ehr_tool wraps.
        brighton_kb: vector store of the reference guideline.
        section_queries: section key -> retrieval query.

    Returns:
        (form_data, audit_log), shaped exactly like the plain per-section loop
        in pipeline.py: form_data keyed by lowercased section, audit_log by the
        original casing. Neither the cross-section rules nor the final
        DVT_CriteriaForm are applied here; pipeline.run_pipeline does both once
        for whichever mode produced the data.
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
