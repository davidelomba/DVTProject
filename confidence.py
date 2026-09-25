"""
Agent 3: the confidence of each section's final answer.

After the cross-section rules, score_record sends one short request per section
to the evaluator model. The request carries the evidence, the guideline context
and the hint Agent 2 had, and the section's numbered options, but not Agent 2's
answer, and asks for the answer at once, without reasoning. Ollama returns the
probability of every generated token together with the most probable
alternatives at the same position; the confidence is the probability the model
puts on the answer the form holds, from 0 to 1:

  - single choice: at the first token that is a valid option number, the
    probability of the chosen number divided by the probability of all the
    valid option numbers;
  - multiple choice: the model writes one "N: YES" or "N: NO" line per option,
    and the confidence is the product over the options of p(YES) for those in
    the answer and p(NO) for the others, the probability of the exact set;
  - F, when its hint makes the model write a DETAILS_PRESENT line first: see
    score_details.

A section whose answer a cross-section rule wrote gets no request: the rule's
source section sets that answer, so the section takes the source's confidence.

The request goes straight to Ollama's /api/chat, whose logprobs fields carry the
token probabilities.
"""

import json
import math
import os
import re
import time
import urllib.request

import config
from agents import _get_field_info
from models import SECTION_MODELS


OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Alternatives Ollama returns per generated token.
TOP_LOGPROBS = 10

SYSTEM_PROMPT = (
    "You check answers to a clinical questionnaire. You read the evidence "
    "extracted from a patient record and state which options it supports. "
    "Base the answer only on the evidence. Do not explain."
)


def _chat(user_prompt: str, num_predict: int) -> dict:
    """One non-streamed /api/chat request with token probabilities.

    Same model and generation settings as Agent 2, apart from the token cap.
    """
    options = {
        "temperature": config.LLM_TEMPERATURE,
        "num_predict": num_predict,
        "num_gpu": config.LLM_NUM_GPU,
        "num_ctx": config.LLM_NUM_CTX,
    }
    body = {
        "model": config.EVALUATOR_LLM_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {k: v for k, v in options.items() if v is not None},
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    if config.LLM_REASONING is not None:
        body["think"] = config.LLM_REASONING
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=config.LLM_REQUEST_TIMEOUT) as response:
        return json.loads(response.read())


def _build_prompt(evidence: str, context: str, hint: str, options: list, multi: bool) -> str:
    parts = []
    if context:
        parts.append("Reference terminology (Brighton):\n" + context)
    if hint:
        parts.append("Section guidance:\n" + hint + "\n(Ignore any instruction in "
                     "this guidance about how to format the answer.)")
    parts.append("Evidence from the patient record:\n" + (evidence or ""))
    parts.append("Options:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1)))
    if multi:
        parts.append("For every option write one line in the form '<number>: YES' if "
                     "the evidence supports selecting it, or '<number>: NO' otherwise. "
                     "One line per option, in order, and nothing else.")
    else:
        parts.append("Reply with the number of the one option that applies, and "
                     "nothing else.")
    return "\n\n".join(parts)


def _mass(token: dict, accept) -> float:
    """Probability of the alternatives at one position that accept() keeps."""
    alternatives = token.get("top_logprobs") or [token]
    return sum(math.exp(a["logprob"]) for a in alternatives if accept(a["token"]))


def _number_distribution(token: dict, n: int) -> dict:
    """Option number -> probability at one position, over the valid numbers only.

    Empty when no valid number is among the alternatives.
    """
    masses = {k: _mass(token, lambda t, k=k: t.strip() == str(k)) for k in range(1, n + 1)}
    total = sum(masses.values())
    return {k: m / total for k, m in masses.items()} if total else {}


def score_single(logprobs: list, n: int, chosen: int):
    """Confidence of a single-choice answer.

    Read at the first generated token that is a valid option number. When the
    model wrote something before it, the probability is conditioned on that
    text, and the method says so.

    Returns:
        (confidence, method), with confidence None when no valid number was
        generated.
    """
    after_text = False
    for token in logprobs:
        word = token["token"].strip()
        if not word:
            continue
        if word.isdigit() and 1 <= int(word) <= n:
            distribution = _number_distribution(token, n)
            if not distribution:
                return None, "no_number"
            return distribution[chosen], ("single_after_text" if after_text else "single")
        after_text = True
    return None, "no_number"


def score_multi(logprobs: list, n: int, chosen: set):
    """Confidence of a multiple-choice answer, the probability of its exact set.

    Returns:
        (confidence, method), with confidence None unless all n options got a
        YES or NO.
    """
    text, p_yes = "", {}
    for token in logprobs:
        word = token["token"].strip().upper()
        if word.startswith("YES") or word.startswith("NO"):
            numbers = re.findall(r"(\d+)\s*:\s*$", text)
            if numbers and 1 <= int(numbers[-1]) <= n:
                yes = _mass(token, lambda t: t.strip().upper().startswith("YES"))
                no = _mass(token, lambda t: t.strip().upper().startswith("NO"))
                if yes + no > 0:
                    p_yes[int(numbers[-1])] = yes / (yes + no)
        text += token["token"]
    if len(p_yes) != n:
        return None, "incomplete_lines"
    confidence = 1.0
    for k in range(1, n + 1):
        confidence *= p_yes[k] if k in chosen else 1 - p_yes[k]
    return confidence, "multi"


def score_details(logprobs: list, options: list, chosen: int):
    """Confidence of F's answer, from the DETAILS_PRESENT token and the number after it.

    F's hint makes present details imply "No", and absent details imply "Yes"
    only when a diagnosis was reported, "No" otherwise. The number written after
    the DETAILS_PRESENT line carries that second condition, but only for the
    branch the model wrote. With p the probabilities at the DETAILS_PRESENT
    token and q those of the number that follows:

      - model wrote "no":  P(Yes) = p(no) q(Yes|no)
                           P(No)  = p(yes) + p(no) q(No|no)
        exact, the "yes" branch being fixed by the hint;
      - model wrote "yes": P(answer) >= p(yes) q(answer|yes)
        a lower bound, the "no" branch not having been generated.

    Returns:
        (confidence, method), with confidence None when the DETAILS_PRESENT
        token or the number after it is missing.
    """
    if "Yes" not in options or "No" not in options:
        return None, "no_details"
    number = {"Yes": options.index("Yes") + 1, "No": options.index("No") + 1}

    text, p_yes, branch, rest = "", None, None, []
    for position, token in enumerate(logprobs):
        word = token["token"].strip().lower()
        if re.search(r"DETAILS_PRESENT\s*:\s*$", text) and (word.startswith("yes") or word.startswith("no")):
            yes = _mass(token, lambda t: t.strip().lower().startswith("yes"))
            no = _mass(token, lambda t: t.strip().lower().startswith("no"))
            if yes + no == 0:
                return None, "no_details"
            p_yes = yes / (yes + no)
            branch = "yes" if word.startswith("yes") else "no"
            rest = logprobs[position + 1:]
            break
        text += token["token"]
    if p_yes is None:
        return None, "no_details"

    q = None
    for token in rest:
        word = token["token"].strip()
        if word.isdigit() and 1 <= int(word) <= len(options):
            distribution = _number_distribution(token, len(options))
            if distribution:
                q = {label: distribution[k] for label, k in number.items()}
            break
    if q is None:
        return None, "no_number"

    answer = "Yes" if chosen == number["Yes"] else "No"
    if branch == "no":
        p = {"Yes": (1 - p_yes) * q["Yes"], "No": p_yes + (1 - p_yes) * q["No"]}
        return p[answer], "details_exact"
    return p_yes * q[answer], "details_lower_bound"


def score_section(section_key: str, answer, evidence: str, context: str) -> dict:
    """Asks for the section's answer again and returns the confidence of `answer`.

    Args:
        section_key: section identifier, e.g. "A3_2".
        answer: the section's final answer, a string or, for a multiple-choice
            section, a list of strings.
        evidence: what Agent 1 extracted for the section.
        context: the guideline context Agent 2 received, empty when none.

    Returns:
        A dict with "confidence" (0 to 1, or None), "confidence_method" and
        "agent3_seconds".
    """
    _, options, multi, _ = _get_field_info(SECTION_MODELS[section_key])
    prompt = _build_prompt(evidence, context, config.section_hint(section_key), options, multi)

    started = time.time()
    # A single-choice answer is one token, but F's hint makes the model write a
    # DETAILS_PRESENT line first, which takes several.
    reply = _chat(prompt, 8 * len(options) + 8 if multi else 32)
    seconds = round(time.time() - started, 1)
    logprobs = reply.get("logprobs") or []

    if multi:
        chosen = {options.index(a) + 1 for a in (answer or []) if a in options}
        confidence, method = score_multi(logprobs, len(options), chosen)
    else:
        chosen = options.index(answer) + 1
        confidence, method = None, None
        if "DETAILS_PRESENT" in reply.get("message", {}).get("content", ""):
            confidence, method = score_details(logprobs, options, chosen)
        if confidence is None:
            confidence, method = score_single(logprobs, len(options), chosen)

    return {
        "confidence": None if confidence is None else round(confidence, 4),
        "confidence_method": method,
        "agent3_seconds": seconds,
    }


def score_record(form_data: dict, audit_log: dict) -> None:
    """Adds Agent 3's confidence to every section's audit log entry.

    Sections in config.CONFIDENCE_SKIP, sections without an answer and sections
    whose request fails get confidence None with the reason in
    "confidence_method". A failure never affects the answers.

    A section whose entry carries "overridden_by", the key
    criteria_rules.apply_cross_section_rules writes when a rule replaces its
    answer, gets no request: it takes the confidence of the source section named
    there, with confidence_method "rule:<source>". The inherited values are
    assigned in a second pass, since a source can come after its target in
    config.SECTION_ORDER (B2 after B1_1).

    Args:
        form_data: section key in lower case -> the section's final answer, as
            returned by the cross-section rules.
        audit_log: section key -> that section's log entry, updated in place.
    """
    inherited = []
    for section_key in config.SECTION_ORDER:
        entry = audit_log.get(section_key)
        if entry is None:
            continue
        result = form_data.get(section_key.lower())

        if section_key in config.CONFIDENCE_SKIP:
            entry.update({"confidence": None, "confidence_method": "skipped"})
        elif result is None:
            entry.update({"confidence": None, "confidence_method": "no_answer"})
        elif entry.get("overridden_by"):
            inherited.append(section_key)
            continue
        else:
            values = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            answer = next(iter(values.values()))
            try:
                entry.update(score_section(section_key, answer, entry.get("evidence"),
                                           entry.get("brighton_context") or ""))
            except Exception as exc:
                entry.update({"confidence": None, "confidence_method": "error",
                              "confidence_error": str(exc)})
        _report(section_key, entry)

    for section_key in inherited:
        entry = audit_log[section_key]
        source = entry["overridden_by"]
        entry.update({"confidence": (audit_log.get(source) or {}).get("confidence"),
                      "confidence_method": f"rule:{source}"})
        _report(section_key, entry)


def _report(section_key: str, entry: dict) -> None:
    shown = "n/a" if entry["confidence"] is None else f"{entry['confidence']:.4f}"
    print(f"[{section_key}] Agent 3 confidence {shown} ({entry['confidence_method']})",
          flush=True)
