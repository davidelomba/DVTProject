"""
Final output of the pipeline: the JSON representation of the evaluated criteria form.
"""

from pydantic import BaseModel

def form_to_json_summary(form: BaseModel) -> dict:
    """
    Serializes the filled-in criteria form into a dictionary.
    """
    return form.model_dump(exclude_none=True)