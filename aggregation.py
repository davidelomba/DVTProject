"""
Serialisation of the filled-in criteria form.
"""

from pydantic import BaseModel

import config

# Key under which the result file holds Agent 3's verdicts. The leading
# underscore keeps it apart from the section keys, which are what
# evaluate_predictions, compare_runs and export_redcap_csv read
VERIFIER_KEY = "_verifier"


def form_to_json_summary(form: BaseModel, audit_log: dict = None) -> dict:
    """Serialises the filled-in criteria form into a dictionary.

    Args:
        form: the models.DVT_CriteriaForm returned by pipeline.run_pipeline.
        audit_log: the audit log returned with it. When Agent 3 has run, its
            verdicts are copied from here into the summary, so they survive in
            the result file when the audit log is not kept.

    Returns:
        The form as nested dicts. Sections left None by a failed evaluation are
        dropped rather than written as nulls, so a missing answer is absent
        instead of looking like an answer of "none". When Agent 3 has run,
        VERIFIER_KEY maps each section to {variant: {"agree", "p_agree"}}.
    """

    summary = form.model_dump(exclude_none=True)
    if audit_log:
        verdicts = verifier_verdicts(audit_log)
        if verdicts:
            summary[VERIFIER_KEY] = verdicts
    return summary


def verifier_verdicts(audit_log: dict) -> dict:
    """Agent 3's verdicts in an audit log, the form the result file keeps.

    Args:
        audit_log: section key -> that section's log entry.

    Returns:
        Section key -> {variant: {"agree", "p_agree"}}, for the sections of
        config.SECTION_ORDER that carry a "verifier" entry; empty when none does.
    """

    return {
        section: {
            variant: {"agree": outcome.get("agree"), "p_agree": outcome.get("p_agree")}
            for variant, outcome in audit_log[section]["verifier"].items()
        }
        for section in config.SECTION_ORDER
        if isinstance(audit_log.get(section), dict) and "verifier" in audit_log[section]
    }
