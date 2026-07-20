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
does not mention a symptom or finding at all, do not assume it is absent.

Some questions ask specifically about ONE method or procedure (e.g. autopsy,
a specific surgical procedure, a specific imaging modality). For these:
only select an option stating that the method was performed and/or
confirmed DVT if the evidence EXPLICITLY states that THIS SPECIFIC method
was used. Do not infer that a method was performed, or that it confirmed
DVT, just because DVT was confirmed through a DIFFERENT method mentioned
elsewhere in the evidence (e.g. do not treat DVT confirmed by ultrasound as
evidence that an autopsy was performed or confirmed anything). If the
evidence does not mention that specific method at all, the correct answer
is the "not done / unknown" option for that method, even if DVT was
confirmed by other means."""


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


def _build_reasoning_prompt(
    evidence_text: str,
    brighton_context: str,
    options: list[str],
    multi_select: bool,
    extra_instructions: str = "",
) -> str:
    # Options are numbered and the model is asked to answer with the NUMBER(S),
    # not by copying the option text. This sidesteps a real failure mode found
    # during testing: several questions have "mirror" options that differ only
    # by a negation (e.g. A1: "...showed presence of DVT" vs "...showed no
    # evidence of DVT"; A3_1: "...and confirmed DVT" vs "...but didn't confirm
    # DVT"). If the model paraphrases even slightly instead of copying
    # verbatim, a text-similarity fuzzy match (difflib) can latch onto the
    # semantically OPPOSITE option, because the two strings are textually very
    # close despite meaning the opposite thing. An index has no "near miss"
    # failure mode of that kind.
    options_block = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))
    prompt = f"Evidence: {evidence_text}"
    if brighton_context:
        prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    if extra_instructions:
        prompt += f"\n\n{extra_instructions}"
    prompt += f"\n\nOptions:\n{options_block}\n\n"

    if multi_select:
        prompt += (
            "First explain your reasoning in a few sentences. Then, on the "
            "very last line of your entire response, write exactly:\n"
            "FINAL_ANSWER: <number>; <number>; ...\n"
            "listing the NUMBER of every option that applies, separated by "
            "semicolons (e.g. \"FINAL_ANSWER: 1; 3\"). If only one applies, "
            "list just that one number. Do not write the option text on this "
            "line, only the number(s)."
        )
    else:
        prompt += (
            "First explain your reasoning in a few sentences. Then, on the "
            "very last line of your entire response, write exactly:\n"
            "FINAL_ANSWER: <number>\n"
            "with the NUMBER of the one option that applies (e.g. "
            "\"FINAL_ANSWER: 2\"). Do not write the option text on this line, "
            "only the number."
        )
    return prompt


def _extract_final_answer_line(text: str) -> str:
    """
    Returns the content after the LAST "FINAL_ANSWER:" occurrence in the
    response. Using the last (not first) match matters because the model's
    free-form reasoning sometimes references the instruction itself (e.g.
    "I will follow the FINAL_ANSWER: format as requested") before actually
    answering; re.search would previously grab that mention instead of the
    real answer on the closing line.
    """
    matches = re.findall(r"FINAL_ANSWER:\s*(.+)", text)
    if not matches:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return matches[-1].strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.75) -> str:
    """
    Primary path: the raw value is the option's 1-based index (what the
    prompt now asks for), which is unambiguous and immune to the
    negation-pair confusion described in _build_reasoning_prompt.

    Fallback path (only if the model didn't follow the numeric-answer
    instruction and wrote text instead): exact text match first, then a
    fuzzy match at a stricter cutoff than before (0.75, was 0.6). Every
    fuzzy match is logged with its score so it can be audited -- silent
    fuzzy matching on clinically opposite options was the original risk.
    """
    cleaned = raw_value.strip().lstrip("-").strip().rstrip(".;").strip()

    # Numeric index path (expected/primary case).
    index_candidate = cleaned.rstrip(".").strip()
    if index_candidate.isdigit():
        idx = int(index_candidate)
        if 1 <= idx <= len(valid_options):
            return valid_options[idx - 1]
        raise ValueError(
            f"Index {idx} out of range for {len(valid_options)} options: {valid_options}"
        )

    # Text fallback (model didn't answer with a number).
    if cleaned in valid_options:
        return cleaned

    close = difflib.get_close_matches(cleaned, valid_options, n=1, cutoff=cutoff)
    if close:
        print(
            f"[WARNING] Fuzzy text match used (no valid index given): "
            f"'{raw_value}' -> '{close[0]}'. Verify this is correct, especially "
            f"for negation-paired options.",
            flush=True,
        )
        return close[0]

    raise ValueError(f"Could not match '{raw_value}' to any of {valid_options}")


def evaluate_section(
    llm: ChatOllama,
    section_model,
    evidence_text: str,
    brighton_context: str = "",
    extra_instructions: str = "",
    max_retries: int = 2,
):
    """
    Fills in the current section's schema: the model reasons freely (no
    format constraints beyond the final line), then the FINAL_ANSWER line
    is extracted and matched against the schema's valid options. See the
    module docstring for why this replaced two earlier, less reliable
    approaches.

    extra_instructions (optional): section-specific clarification appended
    to the prompt, for criteria whose meaning the model tends to invert or
    conflate with generic instructions alone (see config.SECTION_HINTS).

    Returns a (section_model_instance, reasoning_text) tuple. reasoning_text
    is the model's full free-form response (including the FINAL_ANSWER
    line) for the attempt that succeeded -- callers should persist this
    alongside the JSON output. Without it, a wrong final answer is
    unauditable after the fact: there is no way to tell whether the model's
    clinical reasoning was correct and only the number/index was
    misreported, or whether the reasoning itself was wrong -- and no way to
    tell without re-running the whole pipeline in debug mode.
    """
    field_name, options, multi_select = _get_field_info(section_model)
    prompt = _build_reasoning_prompt(evidence_text, brighton_context, options, multi_select, extra_instructions)

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
                return section_model(**{field_name: matched}), content

            matched = _match_option(raw_final, options)
            return section_model(**{field_name: matched}), content

        except Exception as exc:
            last_error = exc
            prompt += (
                f"\n\n(Your previous attempt failed: {exc}. Remember: on "
                f"the very last line of your response, write exactly "
                f"'FINAL_ANSWER: <number>' (or several numbers separated by "
                f"';' for a multi-select question), using ONLY the option's "
                f"number from the list above. Do not write the option text "
                f"on that line.)"
            )

    raise RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")
