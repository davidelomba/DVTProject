"""
Scores a myocarditis run with evaluate_predictions.

Same scoring code as the DVT runs: evaluate_predictions introspects the schema
rather than repeating the options, so pointing it at models_myo is all a second
domain needs. The shim has to run before the import, because the module binds
the schema at import time.

Usage:
    python myo/evaluate_myo.py ./myo/output_myo
    python myo/evaluate_myo.py ./myo/output_myo --no-matrices
"""

import sys
from pathlib import Path

MYO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MYO_DIR))
sys.path.insert(0, str(MYO_DIR.parent))

import models_myo                                    # noqa: E402
sys.modules["models"] = models_myo

import config                                        # noqa: E402
config.SECTION_ORDER = ["E", "F"]

import evaluate_predictions                          # noqa: E402

# The records and their reference answers live beside this file, not in
# data/synthetic_records.
evaluate_predictions.DEFAULT_GROUND_TRUTH_DIR = MYO_DIR / "data" / "records"
evaluate_predictions.DEFAULT_PREDICTIONS_DIR = MYO_DIR / "output_myo"


if __name__ == "__main__":
    evaluate_predictions.main()
