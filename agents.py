"""
Agent 1 (Extractor) and Agent 2 (Evaluator).

Agent 2's design went through three iterations during empirical testing
(see ACTION_PLAN.md and the debug_*.py scripts for the full record):
  1. LangChain's .with_structured_output() (tool-calling): unreliable --
     returned schema-valid but factually wrong answers, or failed outright
     on multi-select schemas.
  2. JSON-mode prompting (ask for a JSON object, parse manually): the model
     would stop producing JSON entirely once the system prompt included any
     real clinical guidance (Brighton synonyms, negation handling), falling
     back to plain prose instead.
  3. FINAL (current): let the model reason freely in plain text with full
     clinical guidance, then require just ONE fixed-format line at the end
     ("FINAL_ANSWER: ..."), extracted with a regex and matched against the
     schema's valid options via exact-then-fuzzy matching. This is the only
     approach that reliably preserved both correct reasoning AND parseable
     output, with llama3:8b-instruct-q4_0 as the model (see config.py for
     why this model was chosen over the clinically fine-tuned alternatives).
"""

import difflib
import re
from typing import get_args, get_origin

from langchain_ollama import ChatOllama

import config


def build_llm(temperature: float = None) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
    )


def test_structured_output_support(llm: ChatOllama, sample_model) -> bool:
    """
    Kept for reference/comparison purposes (this is what the Step 0 test
    originally checked). No longer used by the main pipeline, since
    evaluate_section() below does not rely on .with_structured_output()
    at all -- see the module docstring for why.
    """
    try:
        structured_llm = llm.with_structured_output(sample_model)
        result = structured_llm.invoke("Fill in the schema with plausible example data.")
        return isinstance(result, sample_model)
    except Exception as exc:
        print(f"[with_structured_output check] not reliably supported: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agent 1: Extractor -- direct retrieval (default, non-agentic)
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """You are a clinical extractor. You receive fragments
of a clinical record related to the assigned criterion. Extract exact phrases,
values, and dates that are relevant. Do not draw conclusions, do not
summarize, do not infer. If you find no relevant information, state so
explicitly instead of making anything up. Return only the raw evidence as
text, not JSON."""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """Direct retrieval: similarity search over the EHR KB + extraction prompt."""
    retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    docs = retriever.invoke(criterion_query)
    context = "\n---\n".join(d.page_content for d in docs)

    if not context.strip():
        return "No relevant fragment found in the clinical record for this criterion."

    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterion to investigate: {criterion_query}\n\nClinical record fragments:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


def extract_evidence_agentic(llm: ChatOllama, ehr_tool, criterion_query: str) -> str:
    """
    Agentic version (real tool-calling) -- available but not validated during
    testing (all testing focused on the evaluator, not the extractor). Enable
    via config.USE_AGENTIC_EXTRACTOR only after separately verifying it on
    your chosen model.
    """
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [ehr_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[ehr_tool], verbose=False)
    result = executor.invoke({"input": f"Criterion to investigate: {criterion_query}"})
    return result["output"]


# ---------------------------------------------------------------------------
# Agent 2: Evaluator -- reason freely, then extract deterministically
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """You are a clinical validator. Determine the
correct answer based on the evidence given. Consult the known synonyms from
the Brighton paper when relevant (e.g. VTE can include DVT). Pay extreme
attention to negations: if a symptom or finding is explicitly described as
absent, denied, or ruled out, do not treat it as present. If the evidence
does not mention a symptom or finding at all, do not assume it is absent."""


def _get_field_info(section_model):
    """
    Returns (field_name, valid_options, is_multi_select) for a section's
    Pydantic model, by inspecting its single field's type annotation:
    - Literal[...] -> single choice
    - List[Literal[...]] -> multi-select
    """
    field_name = next(iter(section_model.model_fields.keys()))
    annotation = section_model.model_fields[field_name].annotation

    if get_origin(annotation) is list:
        inner = get_args(annotation)[0]
        options = list(get_args(inner))
        return field_name, options, True

    options = list(get_args(annotation))
    return field_name, options, False


def _build_reasoning_prompt(evidence_text: str, brighton_context: str, options: list[str], multi_select: bool) -> str:
    options_block = "\n".join(f"- {opt}" for opt in options)
    prompt = f"Evidence: {evidence_text}"
    if brighton_context:
        prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    prompt += f"\n\nOptions:\n{options_block}\n\n"

    if multi_select:
        prompt += (
            "First explain your reasoning in a few sentences. Then, on the "
            "very last line, write exactly:\nFINAL_ANSWER: <option 1>; <option 2>; ...\n"
            "listing every option that applies, copied exactly as shown above, "
            "separated by semicolons. If only one applies, list just that one."
        )
    else:
        prompt += (
            "First explain your reasoning in a few sentences. Then, on the "
            "very last line, write exactly:\nFINAL_ANSWER: <the one option that applies>\n"
            "copied exactly as shown above."
        )
    return prompt


def _extract_final_answer_line(text: str) -> str:
    match = re.search(r"FINAL_ANSWER:\s*(.+)", text)
    if not match:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return match.group(1).strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.6) -> str:
    """
    Exact match first, then fuzzy match against the valid enum options.
    Strips common stray formatting the model sometimes adds (leading "- ",
    trailing punctuation) before comparing.
    """
    cleaned = raw_value.strip().lstrip("-").strip().rstrip(".;").strip()

    if cleaned in valid_options:
        return cleaned

    close = difflib.get_close_matches(cleaned, valid_options, n=1, cutoff=cutoff)
    if close:
        return close[0]

    raise ValueError(f"Could not match '{raw_value}' to any of {valid_options}")


def evaluate_section(
    llm: ChatOllama,
    section_model,
    evidence_text: str,
    brighton_context: str = "",
    max_retries: int = 2,
):
    """
    Fills in the current section's schema: the model reasons freely (no
    format constraints beyond the final line), then the FINAL_ANSWER line
    is extracted and matched against the schema's valid options. See the
    module docstring for why this replaced two earlier, less reliable
    approaches.
    """
    field_name, options, multi_select = _get_field_info(section_model)
    prompt = _build_reasoning_prompt(evidence_text, brighton_context, options, multi_select)

    last_error = None
    for attempt in range(max_retries + 1):
        response = llm.invoke([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        content = response.content

        try:
            raw_final = _extract_final_answer_line(content)

            if multi_select:
                raw_items = [item.strip() for item in raw_final.split(";") if item.strip()]
                matched = [_match_option(item, options) for item in raw_items]
                seen = set()
                matched = [m for m in matched if not (m in seen or seen.add(m))]
                return section_model(**{field_name: matched})

            matched = _match_option(raw_final, options)
            return section_model(**{field_name: matched})

        except Exception as exc:
            last_error = exc
            prompt += (
                f"\n\n(Your previous attempt failed: {exc}. Make sure to "
                f"include the FINAL_ANSWER line exactly as instructed, "
                f"copying the option text exactly as shown in the list above.)"
            )

    raise RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")
