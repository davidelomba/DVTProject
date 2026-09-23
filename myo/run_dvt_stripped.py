"""
Runs the DVT corpus with the components a second questionnaire cannot inherit.

The myocarditis arm has no hints, no gates, no cross-section rules and no
guideline context, because all four are written for the DVT form or wait on a
paper that has not arrived. Comparing it with the reference would therefore
measure the transfer plus the removal of those four. This run removes the same
four from the DVT corpus, so the two arms differ in the questionnaire and not in
the configuration.

The stripping happens in memory, through the same mechanism run_myo uses, so the
two arms are stripped identically by construction rather than by two separate
edits. config.py is not touched, and the audit log records the result under
_run_config as it does for any run.

One asymmetry stays and cannot be removed: the ten DVT section queries were
rewritten and measured over the life of the project, while the two myocarditis
ones were written once. The comparison favours the DVT arm by that much.

Usage:
    python myo/run_dvt_stripped.py --output-dir ./output_stripped
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config                                        # noqa: E402


def strip():
    """Removes what a second questionnaire cannot inherit."""

    config.SECTION_HINTS_ENABLED = False
    config.SECTION_HINTS_DISABLED = set()
    config.BRIGHTON_CONTEXT_ENABLED = False
    config.GUIDELINE_ANCHORS_ENABLED = False
    config.SECTION_DESCRIPTIONS_ENABLED = False
    config.CROSS_SECTION_RULES = []
    config.SECTION_GATES_ENABLED = {
        key: False for key in config.SECTION_GATES_ENABLED
    }


if __name__ == "__main__":
    strip()
    import run_synthetic_records

    print(f"hints {config.SECTION_HINTS_ENABLED}, guideline context "
          f"{config.BRIGHTON_CONTEXT_ENABLED}, gates {config.SECTION_GATES_ENABLED}, "
          f"cross-section rules {len(config.CROSS_SECTION_RULES)}\n", flush=True)
    run_synthetic_records.main()
