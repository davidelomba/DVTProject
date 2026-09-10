"""
Serialisation of the filled-in criteria form.
"""

from pydantic import BaseModel


def form_to_json_summary(form: BaseModel) -> dict:
    """Serialises the filled-in criteria form into a dictionary.

    Args:
        form: the models.DVT_CriteriaForm returned by pipeline.run_pipeline.

    Returns:
        The form as nested dicts. Sections left None by a failed evaluation are
        dropped rather than written as nulls, so a missing answer is absent
        instead of looking like an answer of "none".
    """

    return form.model_dump(exclude_none=True)