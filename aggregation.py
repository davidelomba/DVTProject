"""
Serialisation of the filled-in criteria form.
"""

from pydantic import BaseModel

import config

# Key under which the result file holds Agent 3's confidence values. The
# leading underscore keeps it apart from the section keys, which are what
# evaluate_predictions, compare_runs and export_redcap_csv read.
CONFIDENCE_KEY = "_confidence"


def form_to_json_summary(form: BaseModel, audit_log: dict = None) -> dict:
    """Serialises the filled-in criteria form into a dictionary.

    Args:
        form: the models.DVT_CriteriaForm returned by pipeline.run_pipeline.
        audit_log: the audit log returned with it. When Agent 3 has run, its
            confidence values are copied from here into the summary, so they
            survive in the result file when the audit log is not kept.

    Returns:
        The form as nested dicts. Sections left None by a failed evaluation are
        dropped rather than written as nulls, so a missing answer is absent
        instead of looking like an answer of "none". When Agent 3 has run,
        CONFIDENCE_KEY maps each section to its confidence, None included for
        the sections it did not score.
    """

    summary = form.model_dump(exclude_none=True)
    if audit_log:
        confidence = {
            section: audit_log[section]["confidence"]
            for section in config.SECTION_ORDER
            if isinstance(audit_log.get(section), dict) and "confidence" in audit_log[section]
        }
        if confidence:
            summary[CONFIDENCE_KEY] = confidence
    return summary
