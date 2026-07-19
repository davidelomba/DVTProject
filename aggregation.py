"""
Final output of the pipeline: the JSON with all checkboxes filled in.

Note: the overall Level of Certainty (LOC) is NOT computed -- the project's
goal is only to fill in the questionnaire from the clinical record, not to
produce the final diagnostic classification. This module therefore just
serializes the DVT_CriteriaForm populated by the pipeline.
"""

from models import DVT_CriteriaForm


def form_to_json_summary(form: DVT_CriteriaForm) -> dict:
    """Serializes the filled-in form (only the fields that were actually populated)."""
    return form.model_dump(exclude_none=True)
