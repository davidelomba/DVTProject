"""
Core module defining Agent 1 (Extractor) and Agent 2 (Evaluator).

This architecture uses a two-step approach:
  1. Extractor: Retrieves exact evidence from the clinical record based on a specific query.
  2. Evaluator: Receives the evidence and evaluates it against predefined options.
     To maximize reliability, the Evaluator reasons freely in plain text and 
     outputs a fixed-format line ("FINAL_ANSWER: <number>") at the end. This line 
     is then parsed using regex and mapped to the valid schema options.
"""

import difflib
import re
from typing import get_args, get_origin

from langchain_ollama import ChatOllama

import config


def build_llm(temperature: float = None) -> ChatOllama:
    """Initializes and returns the ChatOllama LLM instance based on configuration."""
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
    )


def test_structured_output_support(llm: ChatOllama, sample_model) -> bool:
    """
    Utility function to verify if the configured LLM reliably supports 
    LangChain's native .with_structured_output() method.
    Not used in the primary pipeline, but kept for diagnostic purposes.
    """
    try:
        structured_llm = llm.with_structured_output(sample_model)
        result = structured_llm.invoke("Fill in the schema with plausible example data.")
        return isinstance(result, sample_model)
    except Exception as exc:
        print(f"[with_structured_output check] not reliably supported: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agent 1: Extractor -- Evidence retrieval and exact extraction
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
    Primary extraction method: performs a similarity search over the vector store 
    and uses the LLM to extract exact, relevant fragments.
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
    Instead of searching for snippets using the RAG, pass the ENTIRE report to Agent 1.
    """
    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterion to investigate: {criterion_query}\n\nFull Clinical Record:\n{full_ehr_text}"),
    ]
    response = llm.invoke(messages)
    return response.content

"""
def extract_evidence_agentic(llm: ChatOllama, ehr_tool, criterion_query: str) -> str:
    
    Alternative extraction method using a LangChain tool-calling agent.
    To be used if config.USE_AGENTIC_EXTRACTOR is enabled.
    
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

"""
# ---------------------------------------------------------------------------
# Agent 2: Evaluator -- Free-text reasoning and deterministic extraction
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
    """
    Inspects a Pydantic model to return its single field's name, valid options, 
    and a boolean indicating if it is a multi-select field.
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
    """
    Constructs the prompt instructing the LLM to reason over the evidence and 
    respond with the corresponding index number of the chosen option(s).
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
    """
    Extracts the content immediately following the last occurrence of 'FINAL_ANSWER:' 
    in the LLM's response using regex.
    """
    matches = re.findall(r"FINAL_ANSWER:\s*(.+)", text)
    if not matches:
        raise ValueError(f"No FINAL_ANSWER line found in response: {text!r}")
    return matches[-1].strip()


def _match_option(raw_value: str, valid_options: list[str], cutoff: float = 0.75) -> str:
    """
    Attempts to map the extracted raw value to one of the valid string options.
    Primary path: Assumes the value is a 1-based index (e.g., '1' maps to options[0]).
    Fallback path: Attempts an exact text match, followed by a fuzzy string match.
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

    # Text fallback (triggered if the model didn't answer with a number).
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
    Evaluates the evidence for a specific schema section. Prompts the LLM to 
    reason and extract the answer, automatically retrying if parsing fails.

    Returns:
        tuple: (section_model_instance, reasoning_text) containing the populated 
        Pydantic model and the full response text from the LLM.
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
                # Deduplicate matches while preserving order
                matched = [m for m in matched if not (m in seen or seen.add(m))]
                return section_model(**{field_name: matched}), content

            matched = _match_option(raw_final, options)
            return section_model(**{field_name: matched}), content

        except Exception as exc:
            last_error = exc
            # Append a correction instruction to the prompt before the next attempt
            prompt += (
                f"\n\n(Your previous attempt failed: {exc}. Remember: on "
                f"the very last line of your response, write exactly "
                f"'FINAL_ANSWER: <number>' (or several numbers separated by "
                f"';' for a multi-select question), using ONLY the option's "
                f"number from the list above. Do not write the option text "
                f"on that line.)"
            )

    raise RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")