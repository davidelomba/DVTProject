"""
Agent 1 (Extractor) and Agent 2 (Evaluator).

Agent 1 pulls the relevant fragment(s) from the clinical record for a given
criterion. Agent 2 reasons over that evidence in plain text and ends with two
fixed-format lines, FINAL_OPTION naming an option and FINAL_ANSWER giving its
number. The number is the answer; a disagreement between the two is recorded
as a conflict.
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

# Constant string the extractor model is asked to output when it finds no relevant evidence
NO_EVIDENCE = "NO RELEVANT EVIDENCE FOUND."


def build_llm(model_name: str = None, temperature: float = None,
              num_predict: int = None) -> ChatOllama:
    """Builds a ChatOllama instance for a given pipeline role.

    Single factory for every non-tool-calling model in the project.

    Args:
        model_name: Ollama tag; defaults to config.LLM_MODEL_NAME (Agent 1).
            Pass the relevant constant to select another role's model, e.g.
            build_llm(config.EVALUATOR_LLM_MODEL_NAME) for Agent 2.
        temperature: defaults to config.LLM_TEMPERATURE (0.0).
        num_predict: token cap; defaults to config.LLM_NUM_PREDICT.

    Returns:
        A configured ChatOllama instance, with config.LLM_NUM_GPU layers on
        the GPU.
    """
    # Sent only when set, so a model without a thinking mode never receives
    # the parameter.
    reasoning = {} if config.LLM_REASONING is None else {"reasoning": config.LLM_REASONING}
    return ChatOllama(
        model=model_name if model_name is not None else config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=num_predict if num_predict is not None else config.LLM_NUM_PREDICT,
        num_gpu=config.LLM_NUM_GPU,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        **reasoning,
    )


# Agent 1: Extractor

EXTRACTOR_SYSTEM_PROMPT = f"""You are a clinical extractor. Copy exact sentences/fragments
from the clinical record that are relevant to the requested criterion. You are a copier,
not a commentator: never explain, label, translate, or justify what you copy.

RULES:
1. The record may be in ITALIAN; match Italian medical terms, but copy fragments exactly
   as written -- do not translate them.
2. A fragment is relevant only if it concerns the SAME specific test/procedure/event the
   criterion asks about, not just the same underlying condition in general (e.g. an
   imaging finding is not evidence for an autopsy or surgery criterion).
3. Output ONLY the copied fragment(s), verbatim: no preamble, no parenthetical notes,
   no sentence explaining why it's relevant, no repeating the criterion text, and no
   added labels or interpretation (e.g. never call an ultrasound "post-mortem" or
   "autopsy" unless the record itself says so).
4. If nothing is relevant, output exactly: "{NO_EVIDENCE}"

CORRECT: Eseguito ecocolordoppler venoso degli arti inferiori: trombosi venosa a carico della vena poplitea sinistra.
INCORRECT: Here is the extracted relevant fragment: "Eseguito ecocolordoppler venoso..." (Note: this is relevant because it describes an imaging finding related to the criterion.)
"""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """Extracts evidence via retrieval (config.EXTRACTOR_MODE == "rag").

    Runs a fixed top-k similarity search over the chunked clinical record and
    asks the LLM to copy the relevant fragments out of the retrieved chunks.
    Intended for records too long to pass whole.

    Args:
        llm: the extractor model.
        ehr_vectorstore: Chroma store holding the chunked record.
        criterion_query: what to look for, from pipeline.SECTION_QUERIES.

    Returns:
        The extracted fragments, or NO_EVIDENCE if retrieval came back empty.
    """
    # Fixed k, unlike extract_evidence_agentic: the model makes no retrieval
    # decisions in this mode.
    retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    docs = retriever.invoke(criterion_query)
    context = "\n---\n".join(d.page_content for d in docs)

    # Nothing retrieved: skip the LLM call entirely and report the same
    # "no evidence" string the model itself would have been asked to produce.
    if not context.strip():
        return NO_EVIDENCE

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
# explicit instruction to call it the model reports no evidence on every
# section without ever searching.
#
# TRANSCRIPTION: governs the agent's final chat turn, which
# extract_evidence_agentic does not read. It returns the raw tool output
# instead, so the paragraph is generated and discarded.
AGENTIC_EXTRACTOR_SYSTEM_PROMPT = EXTRACTOR_SYSTEM_PROMPT + f"""

TOOL USE: You have access to a tool called `search_patient_record` that searches
the patient's clinical record. The record is NOT included in this conversation,
you can only see it by calling this tool. You MUST call `search_patient_record`
at least once, using the criterion as your search query (you may call it again
with a reformulated or narrower query if the first result doesn't seem
relevant). Only after calling the tool and reviewing its results may you decide
whether relevant evidence exists. Do NOT answer "{NO_EVIDENCE}"
without having called the tool at least once.

TRANSCRIPTION RULE FOR YOUR FINAL ANSWER: once you have enough information to
answer, your final answer must consist ONLY of the exact original sentence(s)
copied verbatim (word-for-word) from the tool's results, in their original
language. Do NOT paraphrase, translate, summarize, reword or add your own
interpretation of what a finding means. Copying the wrong words or dropping 
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
    often to call it and with which sub-queries, instead of running one fixed
    query. Requires the base `langchain` package.

    The evidence returned is assembled from the RAW chunks of every tool call
    the agent made, not from its own final chat turn: that turn tends to
    paraphrase or translate what the tool found and a corrupted quote makes
    the evaluator reason over the wrong text.

    Coverage depends on the agent's own search decisions: a section where it
    chooses a poor query, or does not search at all, yields NO_EVIDENCE.

    Args:
        llm: a tool-calling-capable model (see agentic_graph.build_agentic_llm).
        ehr_tool: the retriever wrapped as a tool, from
            rag_setup.make_ehr_retriever_tool.
        ehr_vectorstore: unused; kept for call-site compatibility.
        criterion_query: what to look for, from pipeline.SECTION_QUERIES.
        max_iterations: cap on tool calls before the agent is forced to
            answer, bounding runaway generation.

    Returns:
        The concatenated tool results, deduplicated or NO_EVIDENCE if the
        agent never retrieved anything.
    """

    # {agent_scratchpad} is where LangChain injects the running history of
    # tool calls/results as the agent iterates.
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENTIC_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    # Binds the search tool to the model and wires up the loop that lets it
    # call the tool and read the results back.
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
    # tool's own formatted chunk dump).
    agent_chunks = [
        str(observation) for _, observation in result.get("intermediate_steps", [])
    ]

    # Deduplicated (exact-string match) across the agent's own tool calls:
    # the agent can call the tool more than once and get overlapping results.
    seen = set()
    combined = [
        c for c in agent_chunks if c.strip() and not (c in seen or seen.add(c))
    ]

    if not combined:
        return NO_EVIDENCE

    return "\n---\n".join(combined)


# Agent 2: Evaluator

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

    Options are numbered, and the model answers twice: the option's NUMBER on a
    FINAL_ANSWER line and its exact TEXT on a FINAL_OPTION line. Two answers
    instead of one make a mismatch visible, since the model sometimes names one
    option and writes another one's number.

    The number is the answer; evaluate_section keeps it and logs any mismatch.

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
    # Per-section hint (config.SECTION_HINTS), if this section has one
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
            "the SAME order as the FINAL_OPTION line: each number must correspond to "
            "the same option you just named there (e.g. \"FINAL_OPTION: Leg swelling or "
            "pitting oedema\" must be followed by \"FINAL_ANSWER: 2\" if that option is "
            "listed as 2. above). If only one applies, write just that one option/number "
            "on each line.\n"
            "Only include an option if your reasoning above explicitly discussed evidence "
            "supporting it. Do NOT include an option 'by default', as a safety margin, or "
            "just because it is listed first -- every option you list on FINAL_OPTION must "
            "be traceable to a specific sentence in your reasoning."
        )
        # Sections with no "none of the above" option among their choices (A3.2,
        # B1.2) still accept an empty answer in the schema, but the numbered list
        # gives the model no way to express one and when nothing applied it was
        # observed selecting every option instead. Spelled out only where the
        # section actually needs it, so B2 keeps using its own option 5.
        if not _has_none_option(options):
            prompt += (
                "\nIf NONE of the numbered options applies, write exactly "
                "'FINAL_OPTION: none' and 'FINAL_ANSWER: none'. Leaving this section "
                "empty is a valid answer; listing every option is NOT how to say that "
                "nothing applies."
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


# Opening words of the explicit "nothing applies" option, where a section has
# one (only B2 does). Used to tell apart the sections that can express an empty
# answer through an option of their own from those that cannot.
_NONE_OPTION_PREFIX = "none of the above"

# What the model may write on FINAL_ANSWER/FINAL_OPTION to say that no option
# applies, in the sections that have no "none of the above" option to select.
# Both languages are accepted because the evaluator answers in English but the
# clinical records may be in any language and it occasionally follows theirs.
_NO_SELECTION_ANSWERS = {
    "none", "no option", "no options", "nothing", "empty",
    "nessuna", "nessuno", "nessuna opzione",
    "n/a", "na", "-", "--",
}


def _has_none_option(options: list[str]) -> bool:
    """Whether a section offers an explicit 'none of the above' option.

    Args:
        options: the section's options, in schema order.

    Returns:
        True if one of them is the catch-all negative option.
    """
    return any(opt.lower().startswith(_NONE_OPTION_PREFIX) for opt in options)


def _is_no_selection(raw_value: str) -> bool:
    """Whether an answer line means 'no option applies' rather than naming one.

    A3.2 and B1.2 accept an empty answer but list no option that expresses one.
    These tokens are what the prompt asks the model to write instead.

    Args:
        raw_value: the whole content of a FINAL_ANSWER or FINAL_OPTION line.

    Returns:
        True if the line is one of the recognised "nothing applies" tokens.
    """
    cleaned = raw_value.strip().lower().strip("\"'.;:,[]() ")
    return cleaned in _NO_SELECTION_ANSWERS


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
        Missing FINAL_OPTION only costs the cross-check, so the caller skips it
        instead of failing the section, unlike a missing FINAL_ANSWER.
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
        ValueError: if the value is an out-of-range index or matches nothing.
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

    # Fallback 1: the option text, verbatim. Trailing punctuation is discarded
    # from both sides before comparing, because some options end with a period
    # in models.py that the model naturally omits. The ORIGINAL option is
    # returned, so the result still satisfies the schema's exact Literal value.
    normalized = cleaned.rstrip(".").strip()
    for option in valid_options:
        if normalized == option.rstrip(".").strip():
            return option

    # Fallback 2: fuzzy match, logged explicitly. A silent fuzzy match
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
        (section instance, reasoning_text, conflict). The full response is kept
        so a wrong answer can be audited later without re-running the pipeline.
        conflict is None when the answer lines agree, otherwise a dict holding
        what each of them said, under a "kind" naming the disagreement:
        "text_vs_number", "none_vs_text" or "none_with_options". Only one is
        returned, the last one found.

    Raises:
        RuntimeError: if no attempt produced a parseable, valid answer.
    """
    # Introspection and prompt built once, outside the retry loop.
    field_name, options, multi_select = _get_field_info(section_model)
    prompt = _build_reasoning_prompt(evidence_text, brighton_context, options, multi_select, extra_instructions)

    last_error = None
    last_content = None
    # Up to max_retries + 1 attempts total (1 initial + retries on parse failure).
    for attempt in range(max_retries + 1):
        response = llm.invoke([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        content = response.content

        try:
            raw_final = _extract_final_answer_line(content)

            # None if the model omitted the line.
            raw_option_text = _extract_labeled_line(content, "FINAL_OPTION")

            # Sections where several options may apply; the single-choice ones
            # are handled after this block.
            if multi_select:
                # True when the section offers no "none of the above" option
                # (A3.2, B1.2) and the model answered with the "none" token.
                if not _has_none_option(options) and _is_no_selection(raw_final):
                    conflict = None
                    # Logged only: FINAL_OPTION still naming options contradicts
                    # the "none" just given.
                    if raw_option_text and not _is_no_selection(raw_option_text):
                        print(
                            f"[WARNING] FINAL_ANSWER says no option applies while "
                            f"FINAL_OPTION lists {raw_option_text!r} -- taking the "
                            f"empty answer.",
                            flush=True,
                        )
                        conflict = {
                            "kind": "none_vs_text",
                            "from_number": [],
                            "from_text": raw_option_text,
                        }
                    return section_model(**{field_name: []}), content, conflict

                # Each ';'-separated item mapped to a valid option independently.
                raw_items = [item.strip() for item in raw_final.split(";") if item.strip()]

                # "none" also arrives as one item among several, which the
                # whole-line check above does not see.
                none_with_options = None
                if not _has_none_option(options):
                    dropped = [item for item in raw_items if _is_no_selection(item)]
                    raw_items = [item for item in raw_items if not _is_no_selection(item)]
                    if not raw_items:
                        return section_model(**{field_name: []}), content, None
                    # The model named an option and said "none" in the same
                    # line. The named options are kept; the contradiction is
                    # recorded so it can be counted across a run.
                    if dropped:
                        print(
                            f"[WARNING] FINAL_ANSWER lists {raw_items} together with "
                            f"{dropped} -- keeping the named options.",
                            flush=True,
                        )
                        none_with_options = {
                            "kind": "none_with_options",
                            "from_number": list(raw_items),
                            "from_text": list(dropped),
                        }

                matched = [_match_option(item, options) for item in raw_items]

                conflict = none_with_options
                if raw_option_text:
                    text_items = [item.strip() for item in raw_option_text.split(";") if item.strip()]
                    try:
                        matched_by_text = [_match_option(item, options) for item in text_items]
                    except Exception:
                        matched_by_text = None
                    # On disagreement the number is kept and the conflict is
                    # recorded.
                    if matched_by_text and matched_by_text != matched:
                        print(
                            f"[WARNING] FINAL_OPTION text {matched_by_text} disagrees "
                            f"with FINAL_ANSWER number {matched} -- keeping the "
                            f"number-based answer. Raw model output: "
                            f"FINAL_OPTION={raw_option_text!r} FINAL_ANSWER={raw_final!r}",
                            flush=True,
                        )
                        conflict = {
                            "kind": "text_vs_number",
                            "from_number": list(matched),
                            "from_text": list(matched_by_text),
                        }

                seen = set()
                matched = [m for m in matched if not (m in seen or seen.add(m))]  # dedupe, keep order

                # The model sometimes selects "None of the above" alongside real
                # findings, which B2's none_is_exclusive validator rejects.
                # Keeping the findings repairs it; leaving it would cost the
                # whole section once the retries run out.
                if len(matched) > 1:
                    none_options = [m for m in matched if m.lower().startswith(_NONE_OPTION_PREFIX)]
                    if none_options:
                        print(
                            f"[WARNING] Model selected {none_options} together with real "
                            f"findings {[m for m in matched if m not in none_options]} -- "
                            f"mutually exclusive. Dropping the 'none of the above' option "
                            f"and keeping the specific findings.",
                            flush=True,
                        )
                        matched = [m for m in matched if m not in none_options]

                # Building the instance is also a check: Pydantic validates the
                # list here, so the edits made above cannot slip through.
                return section_model(**{field_name: matched}), content, conflict

            matched = _match_option(raw_final, options)

            conflict = None
            if raw_option_text:
                try:
                    matched_by_text = _match_option(raw_option_text, options)
                except Exception:
                    matched_by_text = None
                # Same rule as the multi-select branch above: the number is the
                # answer, the text only a cross-check that makes a disagreement
                # visible.
                if matched_by_text and matched_by_text != matched:
                    print(
                        f"[WARNING] FINAL_OPTION text '{matched_by_text}' disagrees with "
                        f"FINAL_ANSWER number '{matched}' -- keeping the number-based "
                        f"answer. Raw model output: FINAL_OPTION={raw_option_text!r} "
                        f"FINAL_ANSWER={raw_final!r}",
                        flush=True,
                    )
                    conflict = {
                        "kind": "text_vs_number",
                        "from_number": matched,
                        "from_text": matched_by_text,
                    }

            return section_model(**{field_name: matched}), content, conflict

        except Exception as exc:
            # Parsing/matching failed: remember the error and retry with an
            # extended prompt, instead of giving up on the first bad response.
            last_error = exc
            last_content = content
            prompt += (
                f"\n\n(Your previous attempt failed: {exc}. Remember: end your "
                f"response with a 'FINAL_OPTION: <option text>' line copying the "
                f"exact option text verbatim, followed by a 'FINAL_ANSWER: <number>' "
                f"line with that same option's number (several, separated by ';', "
                f"for a multi-select question).)"
            )

    # All attempts exhausted without a parseable/valid answer. The last response
    # travels on the exception: a failed section is the one case where the
    # caller has no other copy of what the model wrote.
    error = RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")
    error.last_response = last_content
    raise error
