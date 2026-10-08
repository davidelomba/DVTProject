"""
Scores a myocarditis run with evaluate_predictions.

Same scoring code as the DVT runs: evaluate_predictions introspects the schema
rather than repeating the options, so pointing it at models_myo is all a second
domain needs. myo_schema is imported first, because evaluate_predictions binds
the schema at import time.

Usage:
    python myo/evaluate_myo.py ./myo/output_myo
    python myo/evaluate_myo.py ./myo/output_myo --no-matrices
"""

from pathlib import Path

import myo_schema  # noqa: F401  # isort: skip -- must precede the project imports

import config
import evaluate_predictions

MYO_DIR = Path(__file__).resolve().parent

config.SECTION_ORDER = ["E", "F"]

# The records and their reference answers live beside this file, not in
# data/synthetic_records.
evaluate_predictions.DEFAULT_GROUND_TRUTH_DIR = MYO_DIR / "data" / "records"
evaluate_predictions.DEFAULT_PREDICTIONS_DIR = MYO_DIR / "output_myo"


if __name__ == "__main__":
    evaluate_predictions.main()
