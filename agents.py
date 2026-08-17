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


def build_llm(model_name: str = None, temperature: float = None,
              num_predict: int = None) -> ChatOllama:
    """Builds a ChatOllama instance for a given pipeline role.

    Single factory for every non-tool-calling model in the project, so adding
    a role means adding a config constant and a call here, not a new function.
    build_agentic_llm (agentic_graph.py) is the one exception, since it needs
    Ollama's tool-calling API.

    Args:
        model_name: Ollama tag; defaults to config.LLM_MODEL_NAME (Agent 1).
            Pass the relevant constant to select another role's model, e.g.
            build_llm(config.EVALUATOR_LLM_MODEL_NAME) for Agent 2.
        temperature: defaults to config.LLM_TEMPERATURE (0.0).
        num_predict: token cap; defaults to config.LLM_NUM_PREDICT. That
            default is sized for the pipeline's short structured answers and
            is too low for callers that emit a whole document in one go --
            generate_synthetic_records.py's writer overrides it, otherwise
            records are silently truncated mid-sentence.

    Returns:
        A configured ChatOllama instance.
    """
    return ChatOllama(
        model=model_name if model_name is not None else config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=num_predict if num_predict is not None else config.LLM_NUM_PREDICT,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Agent 1: Extractor
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """You are a clinical extractor. Copy exact sentences/fragments
from the clinical record that are relevant to the requested criterion. You are a copier,
not a commentator: never explain, label, translate, or justify what you copy.

RULES:
1. The record may be in ITALIAN; match Italian medical terms, but copy fragments exactly
   as written -- do not translate them.
2. A fragment is relevant only if it concerns the SAME specific test/procedure/event the
   criterion asks about, not just the same underlying condition in general (e.g. an
   imaging finding is not evidence for an autopsy or surgery criterion).
3. Output ONLY the copied fragment(s), verbatim -- no preamble, no parenthetical notes,
   no sentence explaining why it's relevant, no repeating the criterion text, and no
   added labels or interpretation (e.g. never call an ultrasound "post-mortem" or
   "autopsy" unless the record itself says so).
4. If nothing is relevant, output exactly: "NO RELEVANT EVIDENCE FOUND."

CORRECT: Eseguito ecocolordoppler venoso degli arti inferiori: trombosi venosa a carico della vena poplitea sinistra.
INCORRECT: Here is the extracted relevant fragment: "Eseguito ecocolordoppler venoso..." (Note: this is relevant because it describes an imaging finding related to the criterion.)
"""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """Extracts evidence via retrieval (config.EXTRACTOR_MODE == "rag").

    Runs a fixed top-k similarity search over the chunked clinical record and
    asks the LLM to copy the relevant fragments out of the retrieved chunks.
    Intended for records too long to pass whole; extract_evidence_full_text is
    the simpler default.

    Args:
        llm: the extractor model.
        ehr_vectorstore: Chroma store holding the chunked record.
        criterion_query: what to look for, from pipeline.SECTION_QUERIES.

    Returns:
        The extracted fragments, or a "no relevant fragment" message.
    """
    # Fixed k, unlike extract_evidence_agentic: the model makes no retrieval
    # decisions in this mode.
    retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    docs = retriever.invoke(criterion_query)
    context = "\n---\n".join(d.page_content for d in docs)

    # Nothing retrieved: skip the LLM call entirely.
    if not context.strip():
        return "No relevant fragment found in the clinical record for this criterion."

    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterion to investigate: {criterion_query}\n\nClinical record fragments:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


def extract_evidence_full_text(llm: ChatOllama, full_ehr_text: str, criterion_query: str) -> str:
    """Extracts evidence from the whole record (config.EXTRACTOR_MODE ==
    "full_text").

    Passes the entire record in the prompt instead of retrieving chunks, which
    removes the "wrong chunk retrieved" failure mode altogether. Valid as long
    as the record fits comfortably in the model's context window.

    Args:
        llm: the extractor model.
        full_ehr_text: the complete clinical record.
        criterion_query: what to look for, from pipeline.SECTION_QUERIES.

    Returns:
        The extracted fragments as returned by the model.
    """
    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterion to investigate: {criterion_query}\n\nFull Clinical Record:\n{full_ehr_text}"),
    ]
    response = llm.invoke(messages)
    return response.content


# Extends the shared extractor prompt for the agentic path, which differs in
# two ways that the base prompt does not cover.
#
# TOOL USE: the base prompt assumes the record is already in the human message.
# Here it is only reachable through search_patient_record, and without an
# explicit instruction to call it the model answers "NO RELEVANT EVIDENCE
# FOUND." on every section without ever searching.
#
# TRANSCRIPTION: the agent's final chat turn tends to paraphrase, translate or
# summarise what the tool returned. A corrupted quote makes Agent 2 reason
# correctly over the wrong text, so the rule is restated forcefully here.
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
interpretation of what a finding means -- copying the wrong words or dropping 
details present in the source text will directly
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
    """Extracts evidence agentically (config.EXTRACTOR_MODE ==
    "agentic_graph").

    The model is given a retrieval tool and decides for itself whether and how
    often to call it, and with which sub-queries, instead of running one fixed
    query. Requires the base `langchain` package.

    The evidence returned is assembled from the RAW chunks of every tool call
    the agent made, not from its own final chat turn: that turn tends to
    paraphrase or translate what the tool found, and a corrupted quote makes
    the evaluator reason over the wrong text.

    Args:
        llm: a tool-calling-capable model (see agentic_graph.build_agentic_llm).
        ehr_tool: the retriever wrapped as a tool, from
            rag_setup.make_ehr_retriever_tool.
        ehr_vectorstore: unused; kept for call-site compatibility. Coverage
            depends entirely on the agent's own search decisions, so a query
            that ranks an irrelevant chunk first can miss evidence a fixed
            top-k search would have found.
        criterion_query: what to look for, from pipeline.SECTION_QUERIES.
        max_iterations: cap on tool calls before the agent is forced to
            answer, bounding runaway generation.

    Returns:
        The concatenated tool results, deduplicated, or "NO RELEVANT EVIDENCE
        FOUND." if the agent never retrieved anything.
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
    # Reminder repeated here on top of the system prompt's TOOL USE paragraph:
    # the model otherwise skips retrieval and answers from nothing.
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

    # Deduplicated (exact-string match) across the agent's own tool calls --
    # the agent can call the tool more than once and get overlapping results.
    seen = set()
    combined = [
        c for c in agent_chunks if c.strip() and not (c in seen or seen.add(c))
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
    """Introspects a section's schema to find its field name and valid options.

    Args:
        section_model: a Pydantic class from models.SECTION_MODELS.

    Returns:
        (field_name, valid_options, is_multi_select). Multi-select sections are
        typed List[Literal[...]], single-choice ones Literal[...] directly.
    """
    # Every section schema has exactly one field, read generically rather than
    # hardcoding "answer" vs "studies" vs "symptoms".
    field_name = next(iter(section_model.model_fields.keys()))
    annotation = section_model.model_fields[field_name].annotation

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
    """Builds the prompt Agent 2 answers for one section.

    Options are numbered and the model replies with the number rather than the
    text, which avoids fuzzy-matching a paraphrased answer onto the wrong
    option when two options differ only by a negation ("confirmed DVT" vs
    "didn't confirm DVT").

    It is also asked to copy the option's exact text on a separate
    FINAL_OPTION line. That gives evaluate_section a second, independently
    derived answer to cross-check the number against: the model has been seen
    naming the right option in its prose while writing a different option's
    number, which no single-line format can detect.

    Args:
        evidence_text: what Agent 1 extracted for this section.
        brighton_context: reference terminology retrieved from the guideline.
        options: the section's valid options, in schema order.
        multi_select: whether several options may apply.
        extra_instructions: the section's hint from config.SECTION_HINTS.

    Returns:
        The prompt string.
    """
    options_block = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))
    prompt = f"Evidence: {evidence_text}"
    if brighton_context:
        prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    # Per-section hint (config.SECTION_HINTS), if this section has one.
    if extra_instructions:
        prompt += f"\n\n{extra_instructions}"
    prompt += f"\n\nOptions:\n{options_block}\n\n"

    base_instruction = "First explain your reasoning in a few sentences. Then, at the end of your response, write exactly two more lines, in this order:\n"

    # Semicolon-separated for multi-select, a single item otherwise; parsed
    # back out by _extract_final_answer_line/_extract_labeled_line/_match_option.
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
            "on each line.\n"
            "Only include an option if your reasoning above explicitly discussed evidence "
            "supporting it. Do NOT include an option 'by default', as a safety margin, or "
            "just because it is listed first -- every option you list on FINAL_OPTION must "
            "be traceable to a specific sentence in your reasoning."
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
    """Reads the FINAL_ANSWER line out of Agent 2's response.

    Args:
        text: the model's full response.

    Returns:
        The content after the LAST "FINAL_ANSWER:" occurrence. The last, not
        the first, because the model sometimes echoes the instruction before
        actually answering.

    Raises:
        ValueError: if no such line exists.
    """
    matches = re.findall(r"FINAL_ANSWER:\s*(.+)", text)
    if not matches:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return matches[-1].strip()


def _extract_labeled_line(text: str, label: str) -> str | None:
    """Reads any labelled line out of Agent 2's response, e.g. FINAL_OPTION.

    Args:
        text: the model's full response.
        label: the label to look for, without the colon.

    Returns:
        The content after the last occurrence, or None if the label is absent.
        Unlike FINAL_ANSWER, FINAL_OPTION is a best-effort cross-check, so a
        missing line degrades gracefully instead of failing the section.
    """
    matches = re.findall(rf"{re.escape(label)}:\s*(.+)", text)
    if not matches:
        return None
    return matches[-1].strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.75) -> str:
    """Maps one raw answer fragment onto a valid option of the section.

    Args:
        raw_value: a single item from FINAL_ANSWER or FINAL_OPTION.
        valid_options: the section's options, in schema order.
        cutoff: similarity threshold for the fuzzy fallback.

    Returns:
        The matching option, exactly as written in the schema.

    Raises:
        ValueError: if the value is an out-of-range index, or matches nothing.
    """
    # Strip formatting the model adds around the value: a leading "-", a
    # trailing "." or ";", and a "1." / "1)" list prefix it tends to keep when
    # copying an option verbatim.
    cleaned = raw_value.strip().lstrip("-").strip().rstrip(".;").strip()
    cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()

    # Primary path: the prompt asks for a bare 1-based index.
    index_candidate = cleaned.rstrip(".").strip()
    if index_candidate.isdigit():
        idx = int(index_candidate)
        if 1 <= idx <= len(valid_options):
            return valid_options[idx - 1]
        raise ValueError(
            f"Index {idx} out of range for {len(valid_options)} options: {valid_options}"
        )

    # Fallback 1: the option text, verbatim. Trailing punctuation is stripped
    # from both sides before comparing, because some options end with a period
    # in models.py that the model naturally omits. The ORIGINAL option is
    # returned, so the result still satisfies the schema's exact Literal value.
    normalized = cleaned.rstrip(".").strip()
    for option in valid_options:
        if normalized == option.rstrip(".").strip():
            return option

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
    """Fills in one section's schema from the evidence (Agent 2).

    Args:
        llm: the evaluator model.
        section_model: the section's Pydantic class.
        evidence_text: what Agent 1 extracted for this section.
        brighton_context: reference terminology retrieved from the guideline.
        extra_instructions: the section's hint from config.SECTION_HINTS.
        max_retries: extra attempts allowed after a parse or validation
            failure; each retry appends the error to the prompt.

    Returns:
        (section instance, reasoning_text). The full response is kept so a
        wrong answer can be audited later without re-running the pipeline.

    Raises:
        RuntimeError: if no attempt produced a parseable, valid answer.
    """
    # Introspection and prompt built once, outside the retry loop.
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
                    # Disagreement normally means the model's index slipped
                    # relative to the text it just copied, so the text wins --
                    # but only when both lines list the SAME number of items.
                    # The model tends to pad FINAL_OPTION with every category
                    # while FINAL_ANSWER names just the one that applies;
                    # trusting the longer list there turns a correct single
                    # answer into a wrong double one. Differing counts point to
                    # padding rather than a slip, so the shorter, more
                    # conservative number-based answer is kept.
                    if matched_by_text and matched_by_text != matched:
                        if len(matched_by_text) != len(matched):
                            print(
                                f"[WARNING] FINAL_OPTION text {matched_by_text} and "
                                f"FINAL_ANSWER number {matched} disagree AND have a "
                                f"different number of items -- keeping the number-based "
                                f"answer (not trusting text, likely padded/truncated). "
                                f"Raw model output: FINAL_OPTION={raw_option_text!r} "
                                f"FINAL_ANSWER={raw_final!r}",
                                flush=True,
                            )
                        else:
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

                # "None of the above" is mutually exclusive with every real
                # finding (enforced by B2's none_is_exclusive validator), and
                # the model does sometimes select both. Left alone, that raises
                # below and the retry -- which only feeds back the raw Pydantic
                # traceback -- keeps failing until the section is lost entirely.
                # Repaired here instead by keeping the specific findings, the
                # more informative of the two contradictory signals.
                if len(matched) > 1:
                    none_options = [m for m in matched if m.lower().startswith("none of the above")]
                    if none_options:
                        print(
                            f"[WARNING] Model selected {none_options} together with real "
                            f"findings {[m for m in matched if m not in none_options]} -- "
                            f"mutually exclusive. Dropping the 'none of the above' option "
                            f"and keeping the specific findings.",
                            flush=True,
                        )
                        matched = [m for m in matched if m not in none_options]

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
