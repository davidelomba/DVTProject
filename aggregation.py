"""
Output finale della pipeline: il JSON con tutte le crocette compilate.

Nota: non viene calcolato il Level of Certainty (LOC) complessivo -- l'obiettivo
del progetto e' solo la compilazione del questionario a partire dalla cartella
clinica, non la classificazione diagnostica finale. Questo modulo si limita quindi
a serializzare il DVT_CriteriaForm popolato dalla pipeline.
"""

from models import DVT_CriteriaForm


def form_to_json_summary(form: DVT_CriteriaForm) -> dict:
    """Serializza il form compilato (solo i campi effettivamente popolati)."""
    return form.model_dump(exclude_none=True)
