"""
Compares pipeline output JSON files against the synthetic ground-truth JSONs
in data/synthetic_records/ (see generate_synthetic_records.py), and reports
per-section accuracy plus TP/TN/FP/FN counts.

COMPARISON ONLY: this does NOT run the pipeline itself. Run main.py (or your
own driver) once per synthetic record beforehand -- each run's output is
matched to its ground truth by the "record_id" field INSIDE the JSON, not by
filename, so the existing timestamped filenames from main.py
(<record_id>_<timestamp>.json) work without renaming. If a record_id has
multiple output files (re-runs), the most recent one wins (files are sorted
by name before loading, and main.py's timestamp format sorts chronologically).

METRICS, per section AND aggregated overall:
  - Exact-match accuracy: fraction of compared records where the predicted
    value for that section equals the ground-truth value exactly (same set
    for multi-select sections, same string for single-choice sections).
  - TP/TN/FP/FN: computed at the OPTION level, not the section level -- for
    every valid Literal option of a section (see models.SECTION_MODELS),
    checks whether it was selected in the prediction vs the ground truth.
    This treats every section uniformly (a single-choice section is just the
    "exactly 1 of N options selected" special case), and is what makes
    TP/TN/FP/FN meaningful for multi-select sections (A3_2, B1_2, B2), where
    a partially-right answer isn't simply "correct" or "wrong".

Records/sections with a missing prediction (no output JSON with a matching
record_id, or the pipeline left that section as None after a failed attempt)
are reported separately as "missing" and excluded from accuracy/TP-TN-FP-FN,
rather than being silently counted as wrong.

Deliberately standalone: only imports models.py (pydantic + typing), not
agents.py/pipeline.py, so this script has no langchain/Ollama dependency and
runs instantly regardless of whether the pipeline itself is set up.

Usage:
    python evaluate_synthetic_records.py [predictions_dir]
    predictions_dir defaults to ./output

Output: printed report + a JSON file saved to
        synthetic_records_eval_<timestamp>.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import get_args, get_origin

from models import SECTION_MODELS

GROUND_TRUTH_DIR = Path(__file__).parent / "data" / "synthetic_records"


def _field_info(section_name: str):
    """(field_name, valid_options, is_multi_select) for a section. Re-
    implemented here standalone (same convention as agents._get_field_info)
    rather than imported, to keep this script free of agents.py's langchain
    dependency -- it only needs to introspect models.py's Pydantic schema."""
    model = SECTION_MODELS[section_name]
    field_name = next(iter(model.model_fields.keys()))
    annotation = model.model_fields[field_name].annotation
    if get_origin(annotation) is list:
        inner = get_args(annotation)[0]
        return field_name, list(get_args(inner)), True
    return field_name, list(get_args(annotation)), False


def load_ground_truth() -> dict:
    """record_id -> ground truth dict, read from every
    data/synthetic_records/*_ground_truth.json file."""
    gt = {}
    for path in sorted(GROUND_TRUTH_DIR.glob("*_ground_truth.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gt[data["record_id"]] = data
    return gt


def load_predictions(predictions_dir: Path) -> dict:
    """record_id -> prediction dict, read from every *.json file in
    predictions_dir EXCEPT *_audit_log.json (not a form output). Files are
    processed in sorted (chronological, given main.py's timestamp naming)
    order, so a later re-run of the same record_id overwrites an earlier one."""
    preds = {}
    if not predictions_dir.is_dir():
        return preds
    for path in sorted(predictions_dir.glob("*.json")):
        if path.name.endswith("_audit_log.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        record_id = data.get("record_id")
        if record_id:
            preds[record_id] = data
    return preds


def _selected_set(value):
    """Normalizes a section's value (a dict holding one field, either a
    string or a list) into a set of selected option strings, so single-
    choice and multi-select sections can be compared identically. Returns
    None if the section itself is missing (pipeline left it None)."""
    if value is None:
        return None
    inner = next(iter(value.values()))
    return set(inner) if isinstance(inner, list) else {inner}


def evaluate(ground_truth: dict, predictions: dict) -> dict:
    section_names = list(SECTION_MODELS.keys())
    report = {
        "records_compared": 0,
        "records_missing_prediction": [],
        "sections": {
            name: {"compared": 0, "missing": 0, "exact_matches": 0,
                   "tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for name in section_names
        },
    }

    for record_id, gt_record in ground_truth.items():
        pred_record = predictions.get(record_id)
        if pred_record is None:
            report["records_missing_prediction"].append(record_id)
            continue
        report["records_compared"] += 1

        for section_name in section_names:
            _, valid_options, _ = _field_info(section_name)
            gt_value = gt_record.get(section_name.lower())
            pred_value = pred_record.get(section_name.lower())
            sec = report["sections"][section_name]

            if pred_value is None:
                sec["missing"] += 1
                continue

            gt_set = _selected_set(gt_value)
            pred_set = _selected_set(pred_value)
            sec["compared"] += 1

            if gt_set == pred_set:
                sec["exact_matches"] += 1

            for option in valid_options:
                is_true = option in gt_set
                is_pred = option in pred_set
                if is_true and is_pred:
                    sec["tp"] += 1
                elif not is_true and not is_pred:
                    sec["tn"] += 1
                elif is_pred and not is_true:
                    sec["fp"] += 1
                else:
                    sec["fn"] += 1

    # Derived per-section accuracy + overall totals across all sections.
    overall = {"compared": 0, "exact_matches": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for sec in report["sections"].values():
        sec["accuracy"] = sec["exact_matches"] / sec["compared"] if sec["compared"] else None
        for k in overall:
            overall[k] += sec[k]
    overall["accuracy"] = overall["exact_matches"] / overall["compared"] if overall["compared"] else None
    report["overall"] = overall

    return report


def print_report(report: dict):
    print(f"\nRecords compared: {report['records_compared']}")
    missing = report["records_missing_prediction"]
    if missing:
        print(f"Records with NO prediction found ({len(missing)}): {', '.join(missing)}")

    header = f"{'Section':<8} {'Acc':>7} {'Compared':>9} {'Missing':>8} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
    print("\n" + header)
    print("-" * len(header))
    for name, sec in report["sections"].items():
        acc = f"{sec['accuracy']*100:.1f}%" if sec["accuracy"] is not None else "n/a"
        print(f"{name:<8} {acc:>7} {sec['compared']:>9} {sec['missing']:>8} "
              f"{sec['tp']:>5} {sec['tn']:>5} {sec['fp']:>5} {sec['fn']:>5}")

    o = report["overall"]
    acc = f"{o['accuracy']*100:.1f}%" if o["accuracy"] is not None else "n/a"
    print("-" * len(header))
    print(f"{'TOTAL':<8} {acc:>7} {o['compared']:>9} {'':>8} "
          f"{o['tp']:>5} {o['tn']:>5} {o['fp']:>5} {o['fn']:>5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_dir", nargs="?", default=str(Path(__file__).parent / "output"),
                         help="Directory of pipeline output JSON files (default: ./output, "
                              "resolved relative to this script's own location)")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    if not ground_truth:
        sys.exit(f"No ground truth files found in {GROUND_TRUTH_DIR} -- run generate_synthetic_records.py first.")

    predictions = load_predictions(Path(args.predictions_dir))
    if not predictions:
        print(f"[WARNING] No prediction JSON files found in {args.predictions_dir} -- "
              f"every record will be reported as missing.", flush=True)

    report = evaluate(ground_truth, predictions)
    print_report(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"synthetic_records_eval_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
