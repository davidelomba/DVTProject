"""
Reads a section's field, options and heading from its Pydantic schema.

The one place where the section schemas are introspected, shared by the
pipeline (agents, confidence) and by the evaluation tools (evaluate_predictions,
export_redcap_csv, export_prompts). It imports nothing but the standard library,
so the evaluation tools stay runnable without langchain or Ollama, and it takes
the schema class as an argument, so it serves whichever module is installed as
`models`.
"""

from typing import get_args, get_origin


def field_info(section_model):
    """Introspects a section's schema to find its field, options and heading.

    Args:
        section_model: a Pydantic class from models.SECTION_MODELS.

    Returns:
        (field_name, valid_options, is_multi_select, description). Multi-select
        sections are typed List[Literal[...]], single-choice ones Literal[...]
        directly. valid_options is in schema order, which numbers the options
        in the prompts and gives them their REDCap codes. description is the
        field's own description in models.py, the section heading as the
        printed questionnaire words it, or "" when the field carries none.
    """

    # Every section schema has exactly one field, read generically rather than
    # hardcoding "answer" vs "studies" vs "symptoms".
    field_name = next(iter(section_model.model_fields.keys()))
    field = section_model.model_fields[field_name]
    annotation = field.annotation
    description = field.description or ""

    if get_origin(annotation) is list:
        inner = get_args(annotation)[0]
        return field_name, list(get_args(inner)), True, description

    # Single-choice fields are typed Literal[...] directly.
    return field_name, list(get_args(annotation)), False, description
