"""
Deterministic safety nets applied on top of the two LLM agents' output:
per-section keyword gates, the section-F details gate, and cross-section
dependency rules.

"""

import re

import config


def apply_keyword_gate(section_key: str, section_result, evidence: str, reasoning_text: str):
    """
    See config.SECTION_KEYWORD_GATES.

    For sections gated on one specific procedure (autopsy, surgery), the LLM
    can hallucinate a positive answer even when that procedure is never
    mentioned. If Agent 2 answered positively but none of the section's
    trigger keywords appear (case-insensitively) in Agent 1's evidence, the
    answer is deterministically reverted to the section's negative default.

    Returns (section_result, reasoning_text) -- section_result unchanged
    unless the gate triggers, in which case it is rebuilt via the model's
    own constructor (not setattr) so it still passes Pydantic validation,
    and reasoning_text gets a "[SYSTEM OVERRIDE]" note appended.
    """
    # Only A1/A2 have a gate defined; every other section skips this block.
    gate_info = config.SECTION_KEYWORD_GATES.get(section_key)
    if not gate_info:
        return section_result, reasoning_text

    # Generic single-field lookup: works for both single-choice (str) and
    # multi-select (list) section schemas without a per-section if/elif.
    field_name = list(type(section_result).model_fields.keys())[0]
    llm_chosen_answer = getattr(section_result, field_name)

    # "Positive" means Agent 2 picked something other than the section's
    # negative default -- i.e. it's claiming the procedure DID happen.
    if isinstance(llm_chosen_answer, list):
        is_positive = any(ans != gate_info["default_option_text"] for ans in llm_chosen_answer)
    else:
        is_positive = (llm_chosen_answer != gate_info["default_option_text"])

    if is_positive:
        # Case-insensitive substring check: does ANY expected keyword show
        # up anywhere in Agent 1's evidence for this section?
        evidence_lower = (evidence or "").lower()
        has_keyword = any(kw.lower() in evidence_lower for kw in gate_info["keywords"])

        if not has_keyword:
            print(
                f"[{section_key}] SOFT GATE TRIGGERED: LLM hallucinated a positive answer. "
                f"Reverting to default.",
                flush=True,
            )
            # Rebuilt via the model's own constructor (not setattr) so the
            # forced value still passes Pydantic validation.
            section_result = type(section_result)(**{field_name: gate_info["default_option_text"]})
            # Appended (not replacing) reasoning_text, so the audit log keeps
            # both the model's original reasoning and the override note.
            reasoning_text += (
                f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
                f"but no triggering keywords {gate_info['keywords']} were found in the evidence. "
                f"Answer was automatically reverted to the negative default."
            )

    return section_result, reasoning_text


def apply_details_gate(section_key: str, section_result, reasoning_text: str):
    """
    Section F only ("reported by specialist, without details"): a real bug
    was traced where Agent 2's own reasoning correctly concluded that
    specific clinical details WERE present in the evidence, but the
    FINAL_OPTION/FINAL_ANSWER lines still answered 'Yes' (reported WITHOUT
    details) anyway -- a self-contradiction between the reasoning prose and
    the final lines that the FINAL_OPTION-vs-FINAL_ANSWER cross-check in
    agents.evaluate_section cannot catch, since those two lines agreed with
    each other (both wrong in the same direction).

    Rather than trusting the model's own Yes/No mapping for F -- the step
    where it was observed to flip -- SECTION_HINTS["F"] (config.py) asks the
    model for an explicit 'DETAILS_PRESENT: yes/no' line, a factual judgment
    it has reliably reported correctly. This function derives F's answer
    MECHANICALLY from that line instead: the model still does the actual
    clinical-language judgment (is a specific finding present?), only the
    final Yes/No label -- not the underlying evidence -- is decided here.

    Returns (section_result, reasoning_text), unchanged if section_key is
    not "F" or the DETAILS_PRESENT line is missing/unparseable (e.g. the
    model skipped it despite the instruction -- degrades gracefully rather
    than failing the evaluation).
    """
    if section_key != "F":
        return section_result, reasoning_text

    match = re.search(r"DETAILS_PRESENT:\s*(yes|no)", reasoning_text, re.IGNORECASE)
    if not match:
        return section_result, reasoning_text

    details_present = match.group(1).lower() == "yes"
    # F's schema: "Yes" means reported WITHOUT details, "No" means reported
    # WITH details (or not reported at all) -- see models.F_ReportedBySpecialist.
    correct_answer = "No" if details_present else "Yes"

    field_name = list(type(section_result).model_fields.keys())[0]
    llm_chosen_answer = getattr(section_result, field_name)

    if llm_chosen_answer != correct_answer:
        print(
            f"[F] DETAILS GATE TRIGGERED: model's own DETAILS_PRESENT="
            f"{'yes' if details_present else 'no'} implies '{correct_answer}', but "
            f"FINAL_ANSWER was '{llm_chosen_answer}'. Overriding.",
            flush=True,
        )
        # Rebuilt via the model's own constructor (not setattr), same
        # reasoning as apply_keyword_gate: keeps Pydantic validation.
        section_result = type(section_result)(**{field_name: correct_answer})
        reasoning_text += (
            f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
            f"which contradicted its own DETAILS_PRESENT={'yes' if details_present else 'no'} "
            f"judgment. Answer was automatically corrected to '{correct_answer}'."
        )

    return section_result, reasoning_text


def apply_cross_section_rules(form_data: dict, audit_log: dict) -> dict:
    """
    See config.CROSS_SECTION_RULES.

    Evaluated once, after every section has been independently filled in
    (regardless of which execution mode produced form_data): if
    rule["if_section"]'s answer has any value other than rule["none_option"],
    force rule["then_section"]'s answer to rule["forced_value"], and append
    an explanatory note to that section's audit log entry.

    Mutates and returns form_data.
    """
    for rule in config.CROSS_SECTION_RULES:
        # Both sections must have been filled in successfully; skip the
        # rule entirely if either one failed during its own evaluation.
        if_result = form_data.get(rule["if_section"])
        then_result = form_data.get(rule["then_section"])

        if if_result is None or then_result is None:
            continue

        # Generic single-field lookup, same idiom as apply_keyword_gate.
        if_field = list(type(if_result).model_fields.keys())[0]
        then_field = list(type(then_result).model_fields.keys())[0]

        # Normalize to a list so single-choice and multi-select "if"
        # sections can be checked with the same any(...) expression below.
        if_answers = getattr(if_result, if_field)
        if not isinstance(if_answers, list):
            if_answers = [if_answers]

        # True if the "if" section reported at least one real finding
        # (anything other than its "none/unknown" option).
        has_non_default = any(ans != rule["none_option"] for ans in if_answers)

        if has_non_default:
            current_value = getattr(then_result, then_field)
            # Only override if the "then" section doesn't already match --
            # avoids redundant log noise when Agent 2 already agreed on its own.
            if current_value != rule["forced_value"]:
                print(
                    f"[CROSS-SECTION RULE] '{rule['if_section']}' triggered. "
                    f"Forcing '{rule['then_section']}' to '{rule['forced_value']}'.",
                    flush=True,
                )
                # Rebuilt via the model's own constructor (not setattr), same
                # reasoning as apply_keyword_gate: keeps Pydantic validation.
                form_data[rule["then_section"]] = type(then_result)(
                    **{then_field: rule["forced_value"]}
                )
                # Note appended to the audit log entry of the OVERRIDDEN
                # section, so the override is traceable without re-running
                # the pipeline.
                audit_key = rule["audit_key"]
                if audit_key in audit_log:
                    audit_log[audit_key]["reasoning"] = (
                        audit_log[audit_key].get("reasoning", "")
                        + f"\n\n[SYSTEM OVERRIDE]: {rule['override_message']}"
                    )

    return form_data
