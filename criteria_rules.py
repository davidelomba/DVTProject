"""
Cross-section dependency rules, applied on top of Agent 2's answers once every
section has been filled in.

No function here calls a model. A rule replaces a section's answer with a value
derived from another section's answer, and edits form_data and audit_log in
place. Each override carries a "[SYSTEM OVERRIDE]" note in the reasoning text,
so a forced answer is never indistinguishable from one the model produced on
its own.
"""

import config
from models import SECTION_MODELS


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
            their reasoning and an "overridden_by" key naming the source
            section, so the override is traceable without re-running.

    Returns:
        form_data, mutated in place.
    """
    for rule in config.CROSS_SECTION_RULES:
        # Only the source section has to be present: a forced answer derived
        # from a missing one would be unfounded. The target may be None, since
        # forced_value comes from the rule and not from what the target holds.
        if_result = form_data.get(rule["if_section"])
        if if_result is None:
            continue

        then_result = form_data.get(rule["then_section"])
        then_model = (type(then_result) if then_result is not None
                      else SECTION_MODELS[rule["then_section"].upper()])

        if_field = list(type(if_result).model_fields.keys())[0]
        then_field = list(then_model.model_fields.keys())[0]

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
            current_value = getattr(then_result, then_field) if then_result is not None else None
            # Only override a value that actually differs, to keep the log
            # free of entries where Agent 2 had already agreed. A section left
            # None by a failed evaluation is always filled.
            if then_result is None or current_value != rule["forced_value"]:
                print(
                    f"[CROSS-SECTION RULE] '{rule['if_section']}' triggered. "
                    f"Forcing '{rule['then_section']}' to '{rule['forced_value']}'.",
                    flush=True,
                )
                form_data[rule["then_section"]] = then_model(
                    **{then_field: rule["forced_value"]}
                )
                audit_key = rule["audit_key"]
                if audit_key in audit_log:
                    audit_log[audit_key]["reasoning"] = (
                        audit_log[audit_key].get("reasoning", "")
                        + f"\n\n[SYSTEM OVERRIDE]: {rule['override_message']}"
                    )
                    audit_log[audit_key]["overridden_by"] = rule["if_section"].upper()

    return form_data
