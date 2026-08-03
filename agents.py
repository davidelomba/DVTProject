"""
Agent 1 (Extractor) and Agent 2 (Evaluator).

Agent 1 pulls the relevant fragment(s) from the clinical record for a given
criterion. Agent 2 reasons over that evidence in plain text and ends its
response with a fixed-format line ("FINAL_ANSWER: <number>"), which is
parsed and mapped to the schema's valid options -- more reliable than
asking the model for structured/JSON output directly (see config.py).
"""

import difflib
import re
from typing import get_args, get_origin
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import config


def build_llm(temperature: float = None) -> ChatOllama:
    """Builds the configured ChatOllama instance."""
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
    )


def test_structured_output_support(llm: ChatOllama, sample_model) -> bool:
    """Diagnostic helper: checks if the model supports .with_structured_output().
    Not used by the main pipeline (see agents.py module docstring for why)."""
    try:
        structured_llm = llm.with_structured_output(sample_model)
        result = structured_llm.invoke("Fill in the schema with plausible example data.")
        return isinstance(result, sample_model)
    except Exception as exc:
        print(f"[with_structured_output check] not reliably supported: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agent 1: Extractor
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """You are a clinical extractor.
Extract ONLY exact raw sentences or fragments from the clinical record that are explicitly relevant to the requested criterion.

CRITICAL RULES:
1. The clinical record may be written in ITALIAN. Match Italian medical terminology and procedure names.
2. Do NOT write any introductory or concluding sentences.
3. Extract the complete raw fragment in its original language (Italian or English) to preserve full contextual meaning.
4. If no fragment in the text is relevant to the requested criterion, output exactly: "NO RELEVANT EVIDENCE FOUND."
"""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """
    RAG-based extraction: similarity search over a chunked/embedded clinical
    record, then the LLM extracts the relevant fragments from the retrieved
    chunks. NOT used by the default pipeline (see extract_evidence_full_text)
    -- kept for clinical records too long to fit whole in the model's context
    window, where chunked retrieval becomes necessary again.
    """
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


def extract_evidence_full_text(llm: ChatOllama, full_ehr_text: str, criterion_query: str) -> str:
    """
    Default extraction path: passes the entire clinical record to the LLM
    instead of retrieving chunks. Simpler and avoids retrieval misses, and
    is a reasonable default as long as the record comfortably fits in the
    model's context window (see extract_evidence for the RAG alternative).
    """
    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterion to investigate: {criterion_query}\n\nFull Clinical Record:\n{full_ehr_text}"),
    ]
    response = llm.invoke(messages)
    return response.content


# Used only by extract_evidence_agentic below. EXTRACTOR_SYSTEM_PROMPT (shared
# with extract_evidence/extract_evidence_full_text) assumes the record text is
# already included in the human message -- true for those two, but NOT for the
# agentic path, where the record is only reachable through search_patient_record.
# Without an explicit instruction to call the tool first, a real-patient test
# (10-section run against an actual EHR with unambiguous findings for most
# criteria) showed the model just answered "NO RELEVANT EVIDENCE FOUND." on
# EVERY section, without ever invoking the tool -- a total retrieval failure,
# not a precision problem.
AGENTIC_EXTRACTOR_SYSTEM_PROMPT = EXTRACTOR_SYSTEM_PROMPT + """

TOOL USE: You have access to a tool called `search_patient_record` that searches
the patient's clinical record. The record is NOT included in this conversation
-- you can only see it by calling this tool. You MUST call `search_patient_record`
at least once, using the criterion as your search query (you may call it again
with a reformulated or narrower query if the first result doesn't seem
relevant). Only after calling the tool and reviewing its results may you decide
whether relevant evidence exists. Do NOT answer "NO RELEVANT EVIDENCE FOUND."
without having called the tool at least once.
"""


def extract_evidence_agentic(llm: ChatOllama, ehr_tool, criterion_query: str, max_iterations: int = 3) -> str:
    """
    Agentic extraction: the model gets a retrieval tool and decides itself
    whether/how many times to call it, instead of a single fixed-query call.
    Requires the base `langchain` package (pip install langchain).

    max_iterations caps how many tool calls the agent can make before being
    forced to answer -- a safety net given this model's tendency (seen
    during evaluator testing) to run away with generation when not tightly
    constrained. NOT validated as reliable yet; test with a standalone
    script (e.g. comparing its output against extract_evidence_full_text
    on the same evidence) before trusting it in the full pipeline.

    Uses AGENTIC_EXTRACTOR_SYSTEM_PROMPT (not the shared EXTRACTOR_SYSTEM_PROMPT)
    -- see that constant's comment: without an explicit instruction to call the
    tool, the model was found to skip it entirely and default straight to the
    negative fallback string on every section.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENTIC_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [ehr_tool], prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[ehr_tool],
        verbose=False,
        max_iterations=max_iterations,
        early_stopping_method="force",
    )
    result = executor.invoke({
        "input": (
            f"Criterion to investigate: {criterion_query}\n\n"
            "Use the search_patient_record tool to look in the patient's "
            "clinical record before answering."
        )
    })
    return result["output"]


# ---------------------------------------------------------------------------
# Agent 2: Evaluator
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """You are a clinical validator. Determine the
correct answer based on the evidence given. Consult the known synonyms from
the Brighton paper when relevant (e.g. VTE can include DVT). Pay extreme
attention to negations: if a symptom, finding, or procedure is explicitly 
described as absent, denied, negated, or ruled out, do not treat it as present. 
If the evidence does not mention a symptom or finding at all, do not assume 
it is absent.

Some questions ask specifically about ONE method or procedure (e.g. autopsy,
a specific surgical procedure, a specific imaging modality). For these:
only select an option stating that the method was performed and/or
confirmed DVT if the evidence EXPLICITLY states that THIS SPECIFIC method
was used. Do not infer that a method was performed, or that it confirmed
DVT, just because DVT was confirmed through a DIFFERENT method mentioned
elsewhere in the evidence (e.g. do not treat DVT confirmed by ultrasound as
evidence that an autopsy was performed or confirmed anything). Furthermore, 
if the evidence EXPLICITLY NEGATES the procedure (e.g., states no surgery 
was performed), or does not mention that specific method at all, the correct 
answer is the "not done / unknown" option for that method, even if DVT was
confirmed by other means."""


def _get_field_info(section_model):
    """Returns (field_name, valid_options, is_multi_select) for a section's
    Pydantic model, based on whether its field is Literal[...] or List[Literal[...]]."""
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
    """
    Builds the evaluator prompt. Options are numbered and the model answers
    with the number(s), not the option text -- this avoids fuzzy-matching
    a paraphrased answer onto the wrong option when two options are near-
    identical except for a negation (e.g. "confirmed DVT" vs "didn't confirm DVT").
    """
    options_block = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))
    prompt = f"Evidence: {evidence_text}"
    if brighton_context:
        prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    if extra_instructions:
        prompt += f"\n\n{extra_instructions}"
    prompt += f"\n\nOptions:\n{options_block}\n\n"

    base_instruction = "First explain your reasoning in a few sentences. Then, on the very last line of your entire response, write exactly:\n"

    if multi_select:
        prompt += (
            base_instruction +
            "FINAL_ANSWER: <number>; <number>; ...\n"
            "listing the NUMBER of every option that applies, separated by "
            "semicolons (e.g. \"FINAL_ANSWER: 1; 3\"). If only one applies, "
            "list just that one number. Do not write the option text on this "
            "line, only the number(s)."
        )
    else:
        prompt += (
            base_instruction +
            "FINAL_ANSWER: <number>\n"
            "with the NUMBER of the one option that applies (e.g. "
            "\"FINAL_ANSWER: 2\"). Do not write the option text on this line, "
            "only the number."
        )
    return prompt


def _extract_final_answer_line(text: str) -> str:
    """Returns the content after the LAST 'FINAL_ANSWER:' occurrence (not the
    first, since the model sometimes references the instruction itself before
    actually answering)."""
    matches = re.findall(r"FINAL_ANSWER:\s*(.+)", text)
    if not matches:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return matches[-1].strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.75) -> str:
    """
    Maps the extracted value to a valid option. Primary path: numeric index
    (what the prompt asks for). Fallback: exact text match, then fuzzy match
    (logged, since a silent fuzzy match risks landing on a negation-opposite
    option) -- only used if the model didn't answer with a bare number.
    """
    cleaned = raw_value.strip().lstrip("-").strip().rstrip(".;").strip()

    index_candidate = cleaned.rstrip(".").strip()
    if index_candidate.isdigit():
        idx = int(index_candidate)
        if 1 <= idx <= len(valid_options):
            return valid_options[idx - 1]
        raise ValueError(
            f"Index {idx} out of range for {len(valid_options)} options: {valid_options}"
        )

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
    Fills in one section's schema from the evidence. Returns
    (section_model_instance, reasoning_text) -- reasoning_text is the
    model's full response, kept so a wrong answer can later be audited
    without re-running the pipeline (see pipeline.py's audit_log).
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
                matched = [m for m in matched if not (m in seen or seen.add(m))]  # dedupe, keep order
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
