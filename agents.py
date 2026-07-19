"""
Agent 1 (Extractor) and Agent 2 (Evaluator).

Agent 1: by default does direct retrieval (similarity search + prompt), not a
real agent with a tool-calling loop -- see ACTION_PLAN.md point 4 for why. The
agentic version is still available and can be enabled via
config.USE_AGENTIC_EXTRACTOR after verifying with
test_structured_output_support() that the model handles it reliably.

Agent 2: uses .with_structured_output() to force the response into the
Pydantic model for the current section. The prompt includes few-shot examples
for handling clinical negations, which was flagged as a concern (but not
resolved) in the original plan.
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel

import config


def build_llm(temperature: float = None) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
    )


def test_structured_output_support(llm: ChatOllama, sample_model: type[BaseModel]) -> bool:
    """
    Step 0 of the plan: check whether the chosen model reliably supports
    .with_structured_output() BEFORE building the whole pipeline around that
    assumption. If it fails repeatedly, use the JSON-mode + manual parsing
    fallback (see evaluate_section with retry).
    """
    try:
        structured_llm = llm.with_structured_output(sample_model)
        result = structured_llm.invoke(
            "Fill in the schema with plausible example data."
        )
        return isinstance(result, sample_model)
    except Exception as exc:
        print(f"[Step 0] with_structured_output not reliably supported: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agent 1: Extractor -- default version (direct retrieval, non-agentic)
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """You are a clinical extractor. You receive fragments
of a clinical record related to the assigned criterion. Extract exact phrases,
values, and dates that are relevant. Do not draw conclusions, do not
summarize, do not infer. If you find no relevant information, state so
explicitly instead of making anything up. Return only the raw evidence as
text, not JSON."""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """
    Direct retrieval: similarity search over the EHR KB + extraction prompt.
    This is the default path (config.USE_AGENTIC_EXTRACTOR = False).
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


def extract_evidence_agentic(llm: ChatOllama, ehr_tool, criterion_query: str) -> str:
    """
    Agentic version (real tool-calling) -- enable only after verifying with
    test_structured_output_support() (or an equivalent test for tool-calling)
    that the model handles it consistently. On quantized 8B models the risk
    of malformed tool calls is real: if you see intermittent parsing errors,
    go back to the default path extract_evidence().
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
# Agent 2: Evaluator -- structured output constrained to the Pydantic model
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """You are a clinical validator. Read the evidence
extracted from the clinical record and consult the known synonyms from the
Brighton paper (e.g. VTE can include DVT). Fill in the form choosing
EXCLUSIVELY among the options provided by the schema. Do not add any text
outside the schema.

Pay extreme attention to negations. Examples:
- "no edema" -> the corresponding option must NOT be selected.
- "the patient denies calf pain" -> "Calf pain or tenderness" must NOT be
  selected.
- "no signs of DVT" -> corresponds to "There was no report of a recognized
  DVT syndrome" or equivalently "None of the above...", NOT to a positive
  symptom.
If the evidence does not mention a symptom at all (neither positively nor
negatively), do NOT assume it is absent: use the default/unknown option
provided by the schema.
"""


def evaluate_section(
    llm: ChatOllama,
    section_model,
    evidence_text: str,
    brighton_context: str = "",
    max_retries: int = 2,
):
    """
    Applies .with_structured_output() to constrain the output to the current
    criterion's Pydantic model. If Step 0 found that the model does not
    reliably support structured output, this is the place to add the
    fallback: an explicit prompt "answer only with valid JSON following this
    schema: {schema}" + section_model.model_validate_json() with retry on
    malformed JSON.
    """
    structured_llm = llm.with_structured_output(section_model)

    human_prompt = f"Evidence extracted from the clinical record:\n{evidence_text}"
    if brighton_context:
        human_prompt += f"\n\nReference synonyms/terminology (Brighton):\n{brighton_context}"
    human_prompt += "\n\nFill in the schema for this criterion."

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke([
                ("system", EVALUATOR_SYSTEM_PROMPT),
                ("human", human_prompt),
            ])
        except Exception as exc:  # e.g. ValidationError from B2's model_validator
            last_error = exc
            human_prompt += (
                f"\n\nWARNING: the previous attempt failed validation "
                f"({exc}). Recheck the internal consistency of the answer."
            )
    raise RuntimeError(f"Evaluation failed after {max_retries + 1} attempts: {last_error}")
