"""
Deterministic safety nets applied on top of the two LLM agents' output:
per-section keyword gates, the section-F details gate, the B2 absent-pulses
gate, and cross-section dependency rules.

Every function here is a pure post-processing step: it takes what Agent 2
answered and either returns it unchanged or replaces it with a value derived
mechanically from the evidence or from another section's answer. Overrides are
always recorded in the reasoning text (or the audit log) with a
"[SYSTEM OVERRIDE]" note, so a forced answer is never indistinguishable from
one the model produced on its own.
"""

import re

import config


def apply_keyword_gate(section_key: str, section_result, evidence: str, reasoning_text: str):
    """Reverts a hallucinated positive answer when the evidence never mentions
    the procedure the section asks about.

    Applies only to the sections listed in config.SECTION_KEYWORD_GATES (A1,
    A2, X), each of which asks about one specific procedure or finding. If
    Agent 2 answered with anything other than the section's negative default
    but none of the section's trigger keywords appear in Agent 1's evidence,
    the answer is forced back to that default.

    One-directional by design: it can only remove an unsupported positive, not
    add a missing one. Keyword presence alone does not imply a positive answer,
    since the evidence could equally be negating the procedure.

    Args:
        section_key: section identifier, e.g. "A1".
        section_result: the section's Pydantic model instance.
        evidence: the text Agent 1 extracted for this section.
        reasoning_text: Agent 2's full response.

    Returns:
        (section_result, reasoning_text), unchanged unless the gate fires.
    """
    gate_info = config.SECTION_KEYWORD_GATES.get(section_key)
    if not gate_info:
        return section_result, reasoning_text

    # Single-field lookup, so this works for both single-choice (str) and
    # multi-select (list) schemas without a per-section branch.
    field_name = list(type(section_result).model_fields.keys())[0]
    llm_chosen_answer = getattr(section_result, field_name)

    # "Positive" = anything other than the section's negative default, i.e.
    # Agent 2 is claiming the procedure did happen.
    if isinstance(llm_chosen_answer, list):
        is_positive = any(ans != gate_info["default_option_text"] for ans in llm_chosen_answer)
    else:
        is_positive = (llm_chosen_answer != gate_info["default_option_text"])

    if is_positive:
        evidence_lower = (evidence or "").lower()
        has_keyword = any(kw.lower() in evidence_lower for kw in gate_info["keywords"])

        if not has_keyword:
            print(
                f"[{section_key}] SOFT GATE TRIGGERED: LLM hallucinated a positive answer. "
                f"Reverting to default.",
                flush=True,
            )
            # Rebuilt through the model's constructor rather than setattr, so
            # the forced value is re-validated against the schema.
            section_result = type(section_result)(**{field_name: gate_info["default_option_text"]})
            # Appended, not replaced: the audit log keeps the model's original
            # reasoning alongside the override note.
            reasoning_text += (
                f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
                f"but no triggering keywords {gate_info['keywords']} were found in the evidence. "
                f"Answer was automatically reverted to the negative default."
            )

    return section_result, reasoning_text


def apply_details_gate(section_key: str, section_result, reasoning_text: str):
    """Derives section F's Yes/No answer from the model's own DETAILS_PRESENT
    line instead of from the answer it selected.

    F asks whether the diagnosis was reported WITHOUT details, so "Yes" means
    no supporting detail was given and "No" means details were present (or the
    diagnosis was not reported at all). That inversion is where Agent 2 was
    observed to contradict itself: its prose could correctly identify specific
    findings and its FINAL_OPTION/FINAL_ANSWER lines still answer "Yes". Both
    lines agreeing with each other, the cross-check in agents.evaluate_section
    cannot catch it.

    config.SECTION_HINTS["F"] therefore asks for an explicit
    "DETAILS_PRESENT: yes/no" line -- a factual judgment, not a mapping onto
    the schema's wording -- and this function does the mapping mechanically.
    The clinical judgment stays with the model; only the label is decided here.

    Args:
        section_key: section identifier; anything other than "F" is a no-op.
        section_result: the section's Pydantic model instance.
        reasoning_text: Agent 2's full response, which should carry the
            DETAILS_PRESENT line.

    Returns:
        (section_result, reasoning_text), unchanged for other sections and
        when the DETAILS_PRESENT line is absent or unparseable -- a missing
        line degrades to "trust the model" rather than failing the section.
    """
    if section_key != "F":
        return section_result, reasoning_text

    match = re.search(r"DETAILS_PRESENT:\s*(yes|no)", reasoning_text, re.IGNORECASE)
    if not match:
        return section_result, reasoning_text

    details_present = match.group(1).lower() == "yes"
    # See models.F_ReportedBySpecialist: the schema's "Yes"/"No" are inverted
    # with respect to the presence of details.
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
        section_result = type(section_result)(**{field_name: correct_answer})
        reasoning_text += (
            f"\n\n[SYSTEM OVERRIDE]: The LLM originally selected '{llm_chosen_answer}', "
            f"which contradicted its own DETAILS_PRESENT={'yes' if details_present else 'no'} "
            f"judgment. Answer was automatically corrected to '{correct_answer}'."
        )

    return section_result, reasoning_text


def apply_absent_pulses_gate(section_key: str, section_result, evidence: str, reasoning_text: str):
    """Drops B2's "Absent pulses in legs or arms" when the evidence contains no
    pulse examination at all.

    The model repeatedly selects this option on the strength of imaging
    language alone, reasoning that absent blood flow on a Doppler study implies
    absent pulses. The two are different findings: one is a vascular imaging
    result, the other a physical examination.

    Scoped to this single section/option pair rather than generalised to a
    "was this option justified" check, because its distinguishing evidence is
    close to unambiguous -- the words polso/polsi/pulse either appear or they
    do not. B2's other options and A3_2's imaging modalities vary too much in
    phrasing for a short keyword list to be safe.

    Args:
        section_key: section identifier; anything other than "B2" is a no-op.
        section_result: the section's Pydantic model instance.
        evidence: the text Agent 1 extracted for this section.
        reasoning_text: Agent 2's full response.

    Returns:
        (section_result, reasoning_text). Only the offending option is
        removed; any other options selected in B2 are left untouched.
    """
    if section_key != "B2":
        return section_result, reasoning_text

    target = "Absent pulses in legs or arms"
    field_name = list(type(section_result).model_fields.keys())[0]
    selected = getattr(section_result, field_name)

    if target not in selected:
        return section_result, reasoning_text

    evidence_lower = (evidence or "").lower()
    has_pulse_keyword = any(kw in evidence_lower for kw in ("polso", "polsi", "pulse"))

    if not has_pulse_keyword:
        print(
            f"[B2] ABSENT PULSES GATE TRIGGERED: '{target}' selected but no pulse-exam "
            f"keyword ('polso'/'polsi'/'pulse') found in the evidence -- dropping it.",
            flush=True,
        )
        new_selected = [s for s in selected if s != target]
        section_result = type(section_result)(**{field_name: new_selected})
        reasoning_text += (
            f"\n\n[SYSTEM OVERRIDE]: Removed '{target}' from the selection -- no "
            f"pulse-exam keyword found in the evidence; absent flow on an imaging "
            f"study alone is not sufficient justification for this option."
        )

    return section_result, reasoning_text


def apply_section_gates(section_key: str, section_result, evidence: str, reasoning_text: str):
    """Applies every enabled per-section gate, in order.

    Single entry point used by all execution modes, so a section's answer goes
    through the same post-processing whichever way the evidence was gathered.
    Each gate is skipped when switched off in config.SECTION_GATES_ENABLED,
    which is what makes an ablation a config change rather than a code change.

    Args:
        section_key: section identifier, e.g. "B2".
        section_result: the section's Pydantic model instance.
        evidence: the text Agent 1 extracted for this section.
        reasoning_text: Agent 2's full response.

    Returns:
        (section_result, reasoning_text) after the enabled gates. With all of
        them disabled this is the identity function, i.e. the model's raw
        answer is kept.
    """
    if config.SECTION_GATES_ENABLED.get("keyword", True):
        section_result, reasoning_text = apply_keyword_gate(
            section_key, section_result, evidence, reasoning_text
        )
    if config.SECTION_GATES_ENABLED.get("details", True):
        section_result, reasoning_text = apply_details_gate(
            section_key, section_result, reasoning_text
        )
    if config.SECTION_GATES_ENABLED.get("absent_pulses", True):
        section_result, reasoning_text = apply_absent_pulses_gate(
            section_key, section_result, evidence, reasoning_text
        )
    return section_result, reasoning_text


def apply_cross_section_rules(form_data: dict, audit_log: dict) -> dict:
    """Enforces the questionnaire's structural dependencies between sections.

    Runs once after every section has been filled in independently, regardless
    of which config.EXTRACTOR_MODE produced form_data, so all execution modes
    share one implementation. Each rule in config.CROSS_SECTION_RULES fires in
    one of two ways:

      - "none_option": when the source section reports any real finding, i.e.
        any value other than that option (a finding in B2 implies B1.1 is
        positive).
      - "trigger_value": when the source section's answer equals that value
        (A3.1 reporting no imaging implies A3.2 cannot list any study).

    Either way the target section is forced to the rule's "forced_value".

    Args:
        form_data: section key (lowercase) -> Pydantic instance or None.
        audit_log: section key (original casing) -> per-section log dict;
            overridden sections get a "[SYSTEM OVERRIDE]" note appended to
            their reasoning, so the override is traceable without re-running.

    Returns:
        form_data, mutated in place.
    """
    for rule in config.CROSS_SECTION_RULES:
        # Skip the rule entirely if either section failed its own evaluation:
        # a forced answer derived from a missing one would be unfounded.
        if_result = form_data.get(rule["if_section"])
        then_result = form_data.get(rule["then_section"])

        if if_result is None or then_result is None:
            continue

        if_field = list(type(if_result).model_fields.keys())[0]
        then_field = list(type(then_result).model_fields.keys())[0]

        # Normalized to a list so single-choice and multi-select source
        # sections share the same any(...) test below.
        if_answers = getattr(if_result, if_field)
        if not isinstance(if_answers, list):
            if_answers = [if_answers]

        if "trigger_value" in rule:
            should_trigger = any(ans == rule["trigger_value"] for ans in if_answers)
        else:
            should_trigger = any(ans != rule["none_option"] for ans in if_answers)

        if should_trigger:
            current_value = getattr(then_result, then_field)
            # Only override a value that actually differs, to keep the log
            # free of entries where Agent 2 had already agreed.
            if current_value != rule["forced_value"]:
                print(
                    f"[CROSS-SECTION RULE] '{rule['if_section']}' triggered. "
                    f"Forcing '{rule['then_section']}' to '{rule['forced_value']}'.",
                    flush=True,
                )
                form_data[rule["then_section"]] = type(then_result)(
                    **{then_field: rule["forced_value"]}
                )
                audit_key = rule["audit_key"]
                if audit_key in audit_log:
                    audit_log[audit_key]["reasoning"] = (
                        audit_log[audit_key].get("reasoning", "")
                        + f"\n\n[SYSTEM OVERRIDE]: {rule['override_message']}"
                    )

    return form_data
