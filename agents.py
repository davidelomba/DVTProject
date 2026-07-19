"""
Agente 1 (Estrattore) e Agente 2 (Valutatore).

Agente 1: di default fa retrieval diretto (similarity search + prompt), non un
vero agente con tool-loop -- vedi PIANO_CORRETTO.md punto 4 sul perche'. La versione
agentica e' comunque disponibile e attivabile da config.USE_AGENTIC_EXTRACTOR dopo
aver verificato con test_structured_output_support() che il modello la regge.

Agente 2: usa .with_structured_output() per forzare la risposta nel modello Pydantic
della sezione corrente. Prompt include few-shot per la gestione delle negazioni
cliniche, che era la criticita' esplicitamente segnalata (ma non risolta) nel piano
originale.
"""

from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel

import config


def build_llm(temperature: float = None) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
    )


def test_structured_output_support(llm: ChatOllama, sample_model: type[BaseModel]) -> bool:
    """
    Step 0 del piano corretto: verifica se il modello scelto supporta
    .with_structured_output() in modo affidabile PRIMA di costruire tutta la
    pipeline attorno a quell'assunzione. Se fallisce ripetutamente, usare il
    fallback JSON-mode + parsing manuale (vedi evaluate_section con retry).
    """
    try:
        structured_llm = llm.with_structured_output(sample_model)
        result = structured_llm.invoke(
            "Rispondi compilando lo schema con dati di esempio plausibili."
        )
        return isinstance(result, sample_model)
    except Exception as exc:
        print(f"[Step 0] with_structured_output non supportato in modo affidabile: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agente 1: Estrattore -- versione di default (retrieval diretto, non agentico)
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """Sei un estrattore clinico. Ricevi frammenti di una
cartella clinica relativi al criterio assegnato. Estrai frasi esatte, valori e
date pertinenti. Non trarre conclusioni, non riassumere, non inferire. Se non
trovi informazioni rilevanti, dichiaralo esplicitamente invece di inventare.
Restituisci solo le evidenze grezze come testo, non JSON."""


def extract_evidence(llm: ChatOllama, ehr_vectorstore, criterion_query: str) -> str:
    """
    Retrieval diretto: similarity search sulla KB EHR + prompt di estrazione.
    Questo e' il percorso di default (config.USE_AGENTIC_EXTRACTOR = False).
    """
    retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    docs = retriever.invoke(criterion_query)
    context = "\n---\n".join(d.page_content for d in docs)

    if not context.strip():
        return "Nessun frammento pertinente trovato nella cartella clinica per questo criterio."

    messages = [
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        ("human", f"Criterio da investigare: {criterion_query}\n\nFrammenti cartella clinica:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


def extract_evidence_agentic(llm: ChatOllama, ehr_tool, criterion_query: str) -> str:
    """
    Versione agentica (tool-calling reale) -- attivare solo dopo aver verificato
    con test_structured_output_support() (o test analogo per il tool-calling)
    che il modello la gestisce in modo consistente. Su modelli 8B quantizzati
    il rischio di tool-call malformate e' concreto: se osservi errori di parsing
    intermittenti, torna al percorso di default extract_evidence().
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
    result = executor.invoke({"input": f"Criterio da investigare: {criterion_query}"})
    return result["output"]


# ---------------------------------------------------------------------------
# Agente 2: Valutatore -- structured output vincolato al modello Pydantic
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """Sei un validatore clinico. Leggi le evidenze estratte
dalla cartella clinica e consulta i sinonimi noti dal paper Brighton (es. VTE puo'
includere DVT). Compila il modulo scegliendo ESCLUSIVAMENTE tra le opzioni fornite
dallo schema. Non aggiungere testo fuori schema.

Fai estrema attenzione alle negazioni. Esempi:
- "nessun edema" o "no swelling" -> l'opzione corrispondente NON va selezionata.
- "il paziente nega dolore al polpaccio" -> "Calf pain or tenderness" NON va selezionata.
- "assenza di segni di TVP" -> corrisponde a "There was no report of a recognized
  DVT syndrome" o equivalente "None of the above...", NON a un sintomo positivo.
Se le evidenze non menzionano affatto un sintomo (ne' in positivo ne' in negativo),
NON assumere che sia assente: usa l'opzione di default/unknown prevista dallo schema.
"""


def evaluate_section(
    llm: ChatOllama,
    section_model,
    evidence_text: str,
    brighton_context: str = "",
    max_retries: int = 2,
):
    """
    Applica .with_structured_output() per vincolare l'output al Pydantic
    del criterio corrente. Se lo Step 0 ha rilevato che il modello non supporta
    structured output in modo affidabile, questa funzione e' il punto in cui
    aggiungere il fallback: prompt esplicito "rispondi solo con JSON valido
    secondo questo schema: {schema}" + section_model.model_validate_json()
    con retry in caso di JSON malformato.
    """
    structured_llm = llm.with_structured_output(section_model)

    human_prompt = f"Evidenze estratte dalla cartella clinica:\n{evidence_text}"
    if brighton_context:
        human_prompt += f"\n\nSinonimi/terminologia di riferimento (Brighton):\n{brighton_context}"
    human_prompt += "\n\nCompila lo schema per questo criterio."

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke([
                ("system", EVALUATOR_SYSTEM_PROMPT),
                ("human", human_prompt),
            ])
        except Exception as exc:  # es. ValidationError dal model_validator di B2
            last_error = exc
            human_prompt += (
                f"\n\nATTENZIONE: il tentativo precedente ha fallito la validazione "
                f"({exc}). Ricontrolla la coerenza interna della risposta."
            )
    raise RuntimeError(f"Valutazione fallita dopo {max_retries + 1} tentativi: {last_error}")
