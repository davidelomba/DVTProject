"""
Installs the myocarditis schema as the `models` module.

pipeline, agentic_graph, criteria_rules, confidence and evaluate_predictions
bind models.SECTION_MODELS at import time, so models_myo has to be registered
under that name before any of them is imported. Imported for its side effect,
before any project module; it also puts the project root on sys.path.
"""

import sys

import project_path  # noqa: F401
import models_myo

sys.modules["models"] = models_myo
