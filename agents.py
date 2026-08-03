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
        # Allows a one-off override; defaults to the deterministic
        # config.LLM_TEMPERATURE (0.0) used everywhere else.
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
    # Fixed top-k similarity search against the EHR vector store (no
    # autonomous decision-making by the model, unlike extract_evidence_agentic).
    retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    docs = retriever.invoke(criterion_query)
    context = "\n---\n".join(d.page_content for d in docs)

    # Short-circuit: nothing retrieved, so skip the LLM call entirely.
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
    # Whole record embedded directly in the prompt -- no retrieval step,
    # so there is no "wrong chunk retrieved" failure mode in this mode.
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
#
# Second, separate issue found once retrieval started working: the agent's
# final answer (the free-form chat turn after calling the tool) tends to
# paraphrase/translate/summarize the tool's results instead of quoting them
# verbatim, even though EXTRACTOR_SYSTEM_PROMPT already asks for raw fragments
# only. Two concrete wrong downstream answers were traced back to this:
# (1) section F flipped from correct ("No") to wrong ("Yes") because the
# agent's "evidence" was a paraphrase ("La diagnosi... e' stata riferita dallo
# specialista") that dropped clinical details present in the source text but
# not carried into that paraphrase; (2) section A3_2 mislabeled the imaging
# modality as "compression ultrasonography" because the agent's own English
# rendering of "ecocolordoppler venoso" introduced that (incorrect) term --
# Agent 2 then reasoned correctly over an already-corrupted quote. The
# "TRANSCRIPTION" paragraph below re-states the no-paraphrasing rule more
# forcefully, specific to this failure mode.
AGENTIC_EXTRACTOR_SYSTEM_PROMPT = EXTRACTOR_SYSTEM_PROMPT + """

TOOL USE: You have access to a tool called `search_patient_record` that searches
the patient's clinical record. The record is NOT included in this conversation
-- you can only see it by calling this tool. You MUST call `search_patient_record`
at least once, using the criterion as your search query (you may call it again
with a reformulated or narrower query if the first result doesn't seem
relevant). Only after calling the tool and reviewing its results may you decide
whether relevant evidence exists. Do NOT answer "NO RELEVANT EVIDENCE FOUND."
without having called the tool at least once.

TRANSCRIPTION RULE FOR YOUR FINAL ANSWER: once you have enough information to
answer, your final answer must consist ONLY of the exact original sentence(s)
copied verbatim (word-for-word) from the tool's results, in their original
language. Do NOT paraphrase, translate, summarize, reword, or add your own
interpretation of what a finding means -- copying the wrong words (e.g.
translating "ecocolordoppler" as "compression ultrasonography" instead of
quoting it as-is) or dropping details present in the source text will directly
cause the next step to reach a wrong conclusion. Do NOT add framing sentences
like "Based on the search results..." or "Here is the relevant evidence...".
If multiple fragments are relevant, list each verbatim, one per line, with no
other commentary.
"""


def extract_evidence_agentic(
    llm: ChatOllama,
    ehr_tool,
    ehr_vectorstore,
    criterion_query: str,
    max_iterations: int = 3,
) -> str:
    """
    Agentic extraction: the model gets a retrieval tool and decides itself
    whether/how many times to call it, instead of a single fixed-query call.
    Requires the base `langchain` package (pip install langchain).

    max_iterations caps how many tool calls the agent can make before being
    forced to answer -- a safety net given this model's tendency (seen
    during evaluator testing) to run away with generation when not tightly
    constrained. NOT validated as reliable yet.

    IMPORTANT: the returned evidence is built from the RAW chunks returned
    by every tool call the agent made (result["intermediate_steps"]), UNION
    a deterministic fixed-query retrieval against ehr_vectorstore (the same
    top-k search "rag" mode would do) -- NOT from the agent's own final
    chat-turn text (previously returned as result["output"]). Two real
    failure modes motivated this:
    (1) the agent's final answer sometimes paraphrased/translated/summarized
    what the tool found instead of quoting it verbatim, corrupting
    downstream answers (traced to sections F and A3_2);
    (2) even with verbatim quoting enforced, the agent sometimes stopped
    searching after a tool call that happened to rank an irrelevant chunk
    above the actually relevant one for that section's query (traced to
    section B2 repeatedly missing its symptom sentence while B1_1, same
    run, found it) -- a retrieval *coverage* problem, not a wording one.
    The deterministic floor guarantees the fixed-query top-k chunks are
    always included regardless of what the agent's own search decided to
    do; the agent's own tool calls (using its own reformulated queries) can
    only ever add more coverage on top of that floor, never less.

    Uses AGENTIC_EXTRACTOR_SYSTEM_PROMPT (not the shared EXTRACTOR_SYSTEM_PROMPT)
    -- see that constant's comment: without an explicit instruction to call the
    tool, the model was found to skip it entirely and default straight to the
    negative fallback string on every section.
    """

    # {agent_scratchpad} is where LangChain injects the running history of
    # tool calls/results as the agent iterates.
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENTIC_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    # Binds the search tool to the model and wires up the ReAct-style
    # tool-calling loop (LangChain's create_tool_calling_agent).
    agent = create_tool_calling_agent(llm, [ehr_tool], prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[ehr_tool],
        verbose=False,
        max_iterations=max_iterations,
        # If max_iterations is hit, force a final answer instead of
        # raising/looping forever.
        early_stopping_method="force",
        # Needed to recover the RAW tool outputs below (result["intermediate_steps"]),
        # instead of only the agent's own final chat-turn text.
        return_intermediate_steps=True,
    )
    # Explicit reminder to use the tool, on top of the system prompt's TOOL
    # USE paragraph -- belt-and-suspenders against the "never searches" bug.
    result = executor.invoke({
        "input": (
            f"Criterion to investigate: {criterion_query}\n\n"
            "Use the search_patient_record tool to look in the patient's "
            "clinical record before answering."
        )
    })

    # Raw text actually returned by each tool call (e.g. the retriever
    # tool's own formatted chunk dump) -- not the agent's paraphrase of it.
    agent_chunks = [
        str(observation) for _, observation in result.get("intermediate_steps", [])
    ]

    # Deterministic floor: the same fixed-query top-k retrieval "rag" mode
    # uses, run unconditionally regardless of what the agent itself searched for.
    floor_retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    floor_chunks = [d.page_content for d in floor_retriever.invoke(criterion_query)]

    # Union, deduplicated (exact-string match), floor first since it's
    # guaranteed relevant to the criterion query.
    seen = set()
    combined = [
        c for c in floor_chunks + agent_chunks if c.strip() and not (c in seen or seen.add(c))
    ]

    if not combined:
        return "NO RELEVANT EVIDENCE FOUND."

    return "\n---\n".join(combined)


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
    # Every section schema has exactly one field; grab it generically
    # instead of hardcoding "answer" vs "studies" vs "symptoms" etc.
    field_name = next(iter(section_model.model_fields.keys()))
    annotation = section_model.model_fields[field_name].annotation

    # Multi-select fields are typed List[Literal[...]] -- unwrap the list
    # to get at the inner Literal's options.
    if get_origin(annotation) is list:
        inner = get_args(annotation)[0]
        options = list(get_args(inner))
        return field_name, options, True

    # Single-choice fields are typed Literal[...] directly.
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
    # Options numbered 1..N; the model answers with the NUMBER, not the
    # option text (see docstring: avoids fuzzy-matching onto the wrong
    # near-identical, negation-flipped option).
    options_block = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))
    prompt = f"Evidence: {evidence_text}"
    if brighton_context:
        prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    # Per-section hint (config.SECTION_HINTS), if this section has one.
    if extra_instructions:
        prompt += f"\n\n{extra_instructions}"
    prompt += f"\n\nOptions:\n{options_block}\n\n"

    base_instruction = "First explain your reasoning in a few sentences. Then, at the end of your response, write exactly two more lines, in this order:\n"

    # Output format differs for multi-select (semicolon-separated) vs
    # single-choice (one item) -- parsed back out by _extract_final_answer_line/
    # _extract_labeled_line/_match_option below.
    #
    # FINAL_OPTION is asked for IN ADDITION to FINAL_ANSWER (not instead of):
    # a real failure mode was traced where the model's own prose reasoning
    # correctly identified the right option (e.g. "venous ultrasound"), but
    # the NUMBER it wrote on the old single FINAL_ANSWER line pointed at a
    # different, wrong option -- a mapping slip between reasoning and index,
    # not a misunderstanding of the evidence (traced on section A3_2, where
    # "Compression ultrasonography" and "Doppler/Duplex Ultrasound" both
    # contain "ultrasound"/"ultrasonography" and are easy to conflate by
    # index alone). Asking the model to also copy the option's exact text
    # gives evaluate_section a second, independently-derived answer to
    # cross-check the number against -- see evaluate_section below.
    if multi_select:
        prompt += (
            base_instruction +
            "FINAL_OPTION: <option text>; <option text>; ...\n"
            "copying VERBATIM, for every option that applies, the exact text of that "
            "option as written in the numbered list above (not a paraphrase), separated "
            "by semicolons.\n"
            "FINAL_ANSWER: <number>; <number>; ...\n"
            "with the NUMBER of every option that applies, separated by semicolons, in "
            "the SAME order as the FINAL_OPTION line -- each number must correspond to "
            "the same option you just named there (e.g. \"FINAL_OPTION: Leg swelling or "
            "pitting oedema\" must be followed by \"FINAL_ANSWER: 2\" if that option is "
            "listed as 2. above). If only one applies, write just that one option/number "
            "on each line."
        )
    else:
        prompt += (
            base_instruction +
            "FINAL_OPTION: <option text>\n"
            "copying VERBATIM the exact text of the one option that applies, as written "
            "in the numbered list above (not a paraphrase).\n"
            "FINAL_ANSWER: <number>\n"
            "with the NUMBER of that SAME option (e.g. if FINAL_OPTION copies option 2's "
            "text, FINAL_ANSWER must be 2)."
        )
    return prompt


def _extract_final_answer_line(text: str) -> str:
    """Returns the content after the LAST 'FINAL_ANSWER:' occurrence (not the
    first, since the model sometimes references the instruction itself before
    actually answering)."""
    # findall (not search): the model sometimes echoes the instruction text
    # itself before the real answer, so multiple matches can occur --
    # matches[-1] below always takes the LAST one.
    matches = re.findall(r"FINAL_ANSWER:\s*(.+)", text)
    if not matches:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return matches[-1].strip()


def _extract_labeled_line(text: str, label: str) -> str | None:
    """Same lookup as _extract_final_answer_line, generalized to any labeled
    line (used for 'FINAL_OPTION:'). Returns None instead of raising when the
    label is absent -- unlike FINAL_ANSWER, FINAL_OPTION is a best-effort
    cross-check (see _build_reasoning_prompt), not a hard requirement, so its
    absence should degrade gracefully rather than fail the whole evaluation."""
    matches = re.findall(rf"{re.escape(label)}:\s*(.+)", text)
    if not matches:
        return None
    return matches[-1].strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.75) -> str:
    """
    Maps the extracted value to a valid option. Primary path: numeric index
    (what the prompt asks for). Fallback: exact text match, then fuzzy match
    (logged, since a silent fuzzy match risks landing on a negation-opposite
    option) -- only used if the model didn't answer with a bare number.
    """
    # Strip stray formatting the model sometimes adds around the number
    # (leading "-", trailing "." or ";").
    cleaned = raw_value.strip().lstrip("-").strip().rstrip(".;").strip()

    # Primary path: the prompt asks for a bare 1-based index.
    index_candidate = cleaned.rstrip(".").strip()
    if index_candidate.isdigit():
        idx = int(index_candidate)
        if 1 <= idx <= len(valid_options):
            return valid_options[idx - 1]
        raise ValueError(
            f"Index {idx} out of range for {len(valid_options)} options: {valid_options}"
        )

    # Fallback 1: model answered with the option text verbatim.
    if cleaned in valid_options:
        return cleaned

    # Fallback 2: fuzzy match, logged explicitly -- a silent fuzzy match
    # risks landing on a negation-opposite option (e.g. "confirmed DVT" vs
    # "didn't confirm DVT" are textually close but semantically opposite).
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
    # Schema introspection + prompt built once, outside the retry loop.
    field_name, options, multi_select = _get_field_info(section_model)
    prompt = _build_reasoning_prompt(evidence_text, brighton_context, options, multi_select, extra_instructions)

    last_error = None
    # Up to max_retries + 1 attempts total (1 initial + retries on parse failure).
    for attempt in range(max_retries + 1):
        response = llm.invoke([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        content = response.content

        try:
            raw_final = _extract_final_answer_line(content)
            # Best-effort second signal (see _build_reasoning_prompt): the
            # option text the model copied verbatim, independent of the
            # number it wrote on FINAL_ANSWER. None if the model skipped it.
            raw_option_text = _extract_labeled_line(content, "FINAL_OPTION")

            if multi_select:
                # Each ';'-separated item mapped to a valid option independently.
                raw_items = [item.strip() for item in raw_final.split(";") if item.strip()]
                matched = [_match_option(item, options) for item in raw_items]

                if raw_option_text:
                    text_items = [item.strip() for item in raw_option_text.split(";") if item.strip()]
                    try:
                        matched_by_text = [_match_option(item, options) for item in text_items]
                    except Exception:
                        matched_by_text = None
                    # Disagreement means the model's own number-mapping slipped
                    # relative to the option text it just copied verbatim --
                    # trust the copied text (see _build_reasoning_prompt for
                    # why this was added: A3_2 correctly named the right
                    # modality in prose but wrote the wrong index).
                    if matched_by_text and matched_by_text != matched:
                        print(
                            f"[WARNING] FINAL_OPTION text {matched_by_text} disagreed with "
                            f"FINAL_ANSWER number {matched} -- trusting the verbatim option "
                            f"text. Raw model output: FINAL_OPTION={raw_option_text!r} "
                            f"FINAL_ANSWER={raw_final!r}",
                            flush=True,
                        )
                        matched = matched_by_text

                seen = set()
                matched = [m for m in matched if not (m in seen or seen.add(m))]  # dedupe, keep order
                # Constructing via section_model(...) re-validates the value
                # against the schema (e.g. B2's none-is-exclusive rule).
                return section_model(**{field_name: matched}), content

            matched = _match_option(raw_final, options)

            if raw_option_text:
                try:
                    matched_by_text = _match_option(raw_option_text, options)
                except Exception:
                    matched_by_text = None
                if matched_by_text and matched_by_text != matched:
                    print(
                        f"[WARNING] FINAL_OPTION text '{matched_by_text}' disagreed with "
                        f"FINAL_ANSWER number '{matched}' -- trusting the verbatim option "
                        f"text. Raw model output: FINAL_OPTION={raw_option_text!r} "
                        f"FINAL_ANSWER={raw_final!r}",
                        flush=True,
                    )
                    matched = matched_by_text

            return section_model(**{field_name: matched}), content

        except Exception as exc:
            # Parsing/matching failed: remember the error and retry with an
            # extended prompt, instead of giving up on the first bad response.
            last_error = exc
            prompt += (
                f"\n\n(Your previous attempt failed: {exc}. Remember: end your "
                f"response with a 'FINAL_OPTION: <option text>' line copying the "
                f"exact option text verbatim, followed by a 'FINAL_ANSWER: <number>' "
                f"line with that same option's number (several, separated by ';', "
                f"for a multi-select question).)"
            )

    # All attempts exhausted without a parseable/valid answer.
    raise RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")
