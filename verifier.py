"""
Agent 3 as a verifier: does the model agree with the answer the form holds?

After the cross-section rules, verify_record sends, for
each section and for each input variant in config.VERIFIER_INPUTS, one request
to the model verifier_model() names: config.VERIFIER_MODEL_NAME, or Agent 2's
model when that is None. The request carries the evidence Agent 2 received, the
section's numbered options and the final answer, and asks for YES or NO alone:

  - "answer":         evidence, options, answer;
  - "reasoning":      the same, plus Agent 2's reasoning as logged;
  - "answer_hint":    "answer" plus the section's hint;
  - "reasoning_hint": "reasoning" plus the section's hint.

The hint is the one config.section_hint gives Agent 2, under a heading and
followed by a line telling the model to ignore its formatting instructions. A
section without a hint sends the "_hint" request without that block. No variant
receives the guideline context.

The verdict is the YES or NO the model generated; the score is the probability
the model puts on YES at that position, normalised over YES and NO, the P(True)
of Kadavath et al. (2022). One request therefore gives both a binary verdict and
a continuous score.

A section whose answer a cross-section rule wrote gets no request: Agent 2's
reasoning argues for the answer the rule replaced, so the section takes the
source section's result, with method "rule:<source>".

The request goes straight to Ollama's /api/chat, whose logprobs fields carry the
token probabilities, with its own temperature and context window
(config.VERIFIER_TEMPERATURE, config.VERIFIER_NUM_CTX).
"""

import json
import math
import os
import time
import urllib.request

import config
from models import SECTION_MODELS
from schema_fields import field_info


OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Alternatives Ollama returns per generated token.
TOP_LOGPROBS = 10

# YES or NO, plus a token of margin for a leading space or punctuation.
NUM_PREDICT = 4

VARIANTS = ("answer", "reasoning", "answer_hint", "reasoning_hint")

# Some hints carry instructions on how Agent 2 formats its reply, such as DVT
# F's DETAILS_PRESENT line, which would contradict "Reply YES or NO".
HINT_FORMAT_NOTE = "(Ignore any instruction in this guidance about how to format the answer.)"

SYSTEM_PROMPT = (
    "You review answers to a clinical questionnaire. You receive evidence "
    "extracted from a patient record, the numbered options of one question and "
    "the answer another reviewer gave. Decide whether that answer is correct "
    "given the evidence. Reply with YES or NO only."
)


def verifier_model() -> str:
    """The model the verifier asks: config.VERIFIER_MODEL_NAME, or Agent 2's."""

    return config.VERIFIER_MODEL_NAME or config.EVALUATOR_LLM_MODEL_NAME


def verifier_settings() -> dict:
    """The verifier's settings, as _run_config.verifier records them.

    "hints" is the `all` digest of config.hint_fingerprint() when a "_hint"
    variant is in config.VERIFIER_INPUTS, None otherwise.
    """

    uses_hint = any(v.endswith("_hint") for v in config.VERIFIER_INPUTS)
    return {
        "enabled": config.VERIFIER_ENABLED,
        "model": verifier_model(),
        "inputs": list(config.VERIFIER_INPUTS),
        "hints": config.hint_fingerprint()["all"] if uses_hint else None,
        "temperature": config.VERIFIER_TEMPERATURE,
        "num_ctx": config.VERIFIER_NUM_CTX,
    }


def _chat(user_prompt: str) -> dict:
    """One non-streamed /api/chat request with token probabilities.

    The thinking switch config.LLM_REASONING is sent only when the verifier
    model is Agent 2's, the model it was set for.
    """

    model = verifier_model()
    options = {
        "temperature": config.VERIFIER_TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "num_gpu": config.LLM_NUM_GPU,
        "num_ctx": config.VERIFIER_NUM_CTX,
    }
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {k: v for k, v in options.items() if v is not None},
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    if config.LLM_REASONING is not None and model == config.EVALUATOR_LLM_MODEL_NAME:
        body["think"] = config.LLM_REASONING
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=config.LLM_REQUEST_TIMEOUT) as response:
        return json.loads(response.read())


def _format_answer(answer, options: list, multi: bool) -> str:
    """The answer as numbered options, the way the question lists them."""

    if multi:
        chosen = [o for o in options if o in (answer or [])]
        if not chosen:
            return "No option selected."
        return "Selected options:\n" + "\n".join(f"{options.index(o) + 1}. {o}" for o in chosen)
    return f"{options.index(answer) + 1}. {answer}"


def build_prompt(evidence: str, options: list, multi: bool, answer, reasoning: str = None,
                 hint: str = None) -> str:
    """The user message for one section and one input variant.

    Args:
        evidence: what Agent 1 extracted for the section.
        options: the section's options, in schema order.
        multi: whether the section is multiple-choice.
        answer: the final answer, a string or a list of strings.
        reasoning: Agent 2's reasoning, or None for the variants without it.
        hint: the section's hint, or None for the variants without it.
    """

    parts = []
    if hint:
        parts.append("Section guidance:\n" + hint + "\n" + HINT_FORMAT_NOTE)
    parts += [
        "Evidence from the patient record:\n" + (evidence or ""),
        "Options:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1)),
        "Answer given:\n" + _format_answer(answer, options, multi),
    ]
    if reasoning:
        parts.append("Reasoning given for the answer:\n" + reasoning)
    if multi:
        parts.append("The question allows several options. The answer is correct only "
                     "if it selects every option the evidence supports and no other.")
    parts.append("Is the answer correct given the evidence? Reply YES or NO.")
    return "\n\n".join(parts)


def _mass(token: dict, prefix: str) -> float:
    alternatives = token.get("top_logprobs") or [token]
    return sum(math.exp(a["logprob"]) for a in alternatives
               if a["token"].strip().upper().startswith(prefix))


def read_verdict(logprobs: list):
    """The verdict and P(YES) at the first generated YES or NO.

    Returns:
        (agree, p_agree, method): agree True or False, p_agree from 0 to 1,
        method "verdict"; or (None, None, "no_verdict") when neither word was
        generated.
    """

    for token in logprobs:
        word = token["token"].strip().upper()
        if word.startswith("YES") or word.startswith("NO"):
            yes, no = _mass(token, "YES"), _mass(token, "NO")
            if yes + no == 0:
                break
            return word.startswith("YES"), yes / (yes + no), "verdict"
    return None, None, "no_verdict"


def verify_section(section_key: str, answer, evidence: str, reasoning: str, variant: str) -> dict:
    """One verifier request for one section and one input variant.

    Returns:
        A dict with "agree", "p_agree", "method", "prompt_tokens", "overflow"
        and "seconds".
    """

    _, options, multi, _ = field_info(SECTION_MODELS[section_key])
    prompt = build_prompt(evidence, options, multi, answer,
                          reasoning if variant.startswith("reasoning") else None,
                          config.section_hint(section_key) if variant.endswith("_hint") else None)

    started = time.time()
    reply = _chat(prompt)
    seconds = round(time.time() - started, 1)

    agree, p_agree, method = read_verdict(reply.get("logprobs") or [])

    # Past num_ctx Ollama drops the start of the prompt, which is the evidence
    prompt_tokens = reply.get("prompt_eval_count")
    overflow = prompt_tokens is not None and prompt_tokens + NUM_PREDICT >= config.VERIFIER_NUM_CTX

    return {
        "agree": agree,
        "p_agree": None if p_agree is None else round(p_agree, 4),
        "method": method,
        "prompt_tokens": prompt_tokens,
        "overflow": overflow,
        "seconds": seconds,
    }


def verify_record(form_data: dict, audit_log: dict) -> None:
    """Adds the verifier's results to every section's audit log entry.

    Each entry gains "verifier": {variant: result}. Sections without an answer,
    without a reasoning (for the two "reasoning" variants) or whose request fails get
    agree None with the reason in "method". A failure never affects the answers.

    Args:
        form_data: section key in lower case -> the section's final answer.
        audit_log: section key -> that section's log entry, updated in place.
    """

    unknown = [v for v in config.VERIFIER_INPUTS if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown verifier variants {unknown}; valid: {list(VARIANTS)}")

    inherited = []
    for section_key in config.SECTION_ORDER:
        entry = audit_log.get(section_key)
        if entry is None:
            continue
        result = form_data.get(section_key.lower())
        if entry.get("overridden_by"):
            inherited.append(section_key)
            continue

        entry["verifier"] = {}
        for variant in config.VERIFIER_INPUTS:
            if result is None:
                outcome = {"agree": None, "p_agree": None, "method": "no_answer"}
            elif variant.startswith("reasoning") and not entry.get("reasoning"):
                outcome = {"agree": None, "p_agree": None, "method": "no_reasoning"}
            else:
                values = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                answer = next(iter(values.values()))
                try:
                    outcome = verify_section(section_key, answer, entry.get("evidence"),
                                             entry.get("reasoning"), variant)
                except Exception as exc:
                    outcome = {"agree": None, "p_agree": None, "method": "error",
                               "error": str(exc)}
            entry["verifier"][variant] = outcome
            _report(section_key, variant, outcome)

    for section_key in inherited:
        source = audit_log[section_key]["overridden_by"]
        source_results = (audit_log.get(source) or {}).get("verifier") or {}
        audit_log[section_key]["verifier"] = {
            variant: {"agree": (source_results.get(variant) or {}).get("agree"),
                      "p_agree": (source_results.get(variant) or {}).get("p_agree"),
                      "method": f"rule:{source}"}
            for variant in config.VERIFIER_INPUTS
        }
        for variant in config.VERIFIER_INPUTS:
            _report(section_key, variant, audit_log[section_key]["verifier"][variant])


def _report(section_key: str, variant: str, outcome: dict) -> None:
    verdict = {True: "AGREE", False: "DISAGREE", None: "n/a"}[outcome.get("agree")]
    p = outcome.get("p_agree")
    shown = "n/a" if p is None else f"{p:.4f}"
    print(f"[{section_key}] Agent 3 verifier ({variant}) {verdict}, P(YES) {shown} "
          f"({outcome.get('method')})", flush=True)
