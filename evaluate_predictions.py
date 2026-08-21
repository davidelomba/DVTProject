"""
Scores a pipeline run against a set of reference answers.

Does NOT run the pipeline: the records must already have been processed. Both
the predictions and the reference answers are read from directories given on
the command line, so the same script serves the synthetic corpus and any other
annotated set -- real records included, once reference answers exist for them.

Predictions are matched to their reference by the "record_id" field inside the
JSON rather than by filename, so timestamped outputs need no renaming; when a
record has several files, the most recent one wins.

METRICS, per section and overall:
  - Exact-match accuracy: the predicted answer equals the reference.
  - TP/TN/FP/FN, precision, recall and F1, counted per OPTION rather than per
    section: every option of every section is scored as selected or not, in
    the prediction and in the reference. Without this, a multi-select section
    answered half right would count as simply wrong.
  - Confusion matrix, for single-choice sections only: which reference option
    was answered with which predicted option. It shows WHICH options get
    mistaken for each other, something the aggregate numbers hide. Not produced
    for multi-select sections, where a prediction is a set rather than a class
    and the matrix is not defined.

Precision, recall and F1 deliberately ignore true negatives -- the options
correctly left unselected. They are the majority of every count, since most
options do not apply to most records, so any metric including them is dominated
by the easy cases and reads far higher than the real performance.

A section the pipeline left as None, or a record with no output at all, is
reported as "missing" and left out of the metrics instead of being counted as
an error.

Needs scikit-learn (metrics and confusion matrix) and models.py, but not
langchain or Ollama: it runs without the pipeline's own stack installed.

Usage:
    python evaluate_predictions.py [predictions_dir] [--ground-truth DIR]
    defaults: ./output and ./data/synthetic_records

Writes evaluation_<timestamp>.json next to the printed report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import get_args, get_origin

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_recall_fscore_support

from models import SECTION_MODELS

DEFAULT_GROUND_TRUTH_DIR = Path(__file__).parent / "data" / "synthetic_records"
DEFAULT_PREDICTIONS_DIR = Path(__file__).parent / "output"


def _field_info(section_name: str):
    """(field_name, valid_options, is_multi_select) for a section.

    Re-implemented here rather than imported from agents.py, which would pull
    in langchain; it only needs to introspect models.py's Pydantic schema.
    """
    model = SECTION_MODELS[section_name]
    field_name = next(iter(model.model_fields.keys()))
    annotation = model.model_fields[field_name].annotation
    if get_origin(annotation) is list:
        inner = get_args(annotation)[0]
        return field_name, list(get_args(inner)), True
    return field_name, list(get_args(annotation)), False


def load_ground_truth(directory: Path) -> dict:
    """record_id -> reference answers, from every *_ground_truth.json in a
    directory."""
    gt = {}
    for path in sorted(directory.glob("*_ground_truth.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gt[data["record_id"]] = data
    return gt


def load_predictions(predictions_dir: Path) -> dict:
    """record_id -> prediction dict, read from every *.json file in
    predictions_dir EXCEPT *_audit_log.json (not a form output). Files are
    processed in sorted order, which is chronological given the timestamped
    filenames, so a later re-run of the same record_id overwrites an earlier
    one."""
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
    """Normalizes a section's value (a dict holding one field, either a string
    or a list) into a set of selected option strings, so single-choice and
    multi-select sections can be compared identically. None if the section
    itself is missing (pipeline left it None)."""
    if value is None:
        return None
    inner = next(iter(value.values()))
    return set(inner) if isinstance(inner, list) else {inner}


def _binary_rows(pairs, options):
    """Turns (reference set, predicted set) pairs into two 0/1 matrices.

    One column per option, in schema order, which is the shape scikit-learn's
    multilabel helpers expect and the reason single-choice and multi-select
    sections can share the same scoring path.

    Args:
        pairs: list of (gt_set, pred_set) for one section.
        options: the section's options, in schema order.

    Returns:
        (y_true, y_pred), each a list of 0/1 lists.
    """
    y_true = [[int(opt in gt) for opt in options] for gt, _ in pairs]
    y_pred = [[int(opt in pred) for opt in options] for _, pred in pairs]
    return y_true, y_pred


def _score_section(pairs, options, is_multi_select) -> dict:
    """Computes every metric for one section from its collected answer pairs.

    Args:
        pairs: list of (gt_set, pred_set), one per compared record.
        options: the section's options, in schema order.
        is_multi_select: whether several options may apply, which decides
            whether a confusion matrix is meaningful.

    Returns:
        A dict of counts and metrics, with "confusion_matrix" present only for
        single-choice sections.
    """
    y_true, y_pred = _binary_rows(pairs, options)

    # One 2x2 table per option; summing them gives the section's totals.
    per_option = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(len(options))))
    tn, fp, fn, tp = (int(per_option[:, i, j].sum()) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))

    # "micro" pools the options before dividing, so the result matches the
    # tp/fp/fn totals above rather than averaging per-option rates.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    scored = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }

    if not is_multi_select:
        # Every answer is exactly one option, so it can be treated as a class.
        labels = list(range(len(options)))
        true_idx = [options.index(next(iter(gt))) for gt, _ in pairs]
        pred_idx = [options.index(next(iter(pred))) for _, pred in pairs]
        scored["confusion_matrix"] = confusion_matrix(true_idx, pred_idx, labels=labels).tolist()

    return scored


def evaluate(ground_truth: dict, predictions: dict) -> dict:
    """Scores predictions against reference answers, section by section.

    Args:
        ground_truth: record_id -> reference answers.
        predictions: record_id -> pipeline output.

    Returns:
        A report dict with per-section and overall metrics, plus the lists of
        records and sections that had no prediction to score.
    """
    section_names = list(SECTION_MODELS.keys())
    report = {
        "records_compared": 0,
        "records_missing_prediction": [],
        "sections": {},
    }

    # Answers are collected first and scored afterwards, so every metric for a
    # section is computed once, on the whole set, by scikit-learn.
    collected = {name: [] for name in section_names}
    missing = {name: 0 for name in section_names}

    for record_id, gt_record in ground_truth.items():
        pred_record = predictions.get(record_id)
        if pred_record is None:
            report["records_missing_prediction"].append(record_id)
            continue
        report["records_compared"] += 1

        for section_name in section_names:
            gt_value = gt_record.get(section_name.lower())
            pred_value = pred_record.get(section_name.lower())
            if pred_value is None:
                missing[section_name] += 1
                continue
            collected[section_name].append((_selected_set(gt_value), _selected_set(pred_value)))

    overall = {"compared": 0, "exact_matches": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for section_name in section_names:
        _, options, is_multi_select = _field_info(section_name)
        pairs = collected[section_name]
        sec = {
            "compared": len(pairs),
            "missing": missing[section_name],
            "exact_matches": sum(1 for gt, pred in pairs if gt == pred),
            "options": list(options),
        }
        sec["accuracy"] = sec["exact_matches"] / len(pairs) if pairs else None

        if pairs:
            sec.update(_score_section(pairs, options, is_multi_select))
        else:
            sec.update({"tp": 0, "tn": 0, "fp": 0, "fn": 0,
                        "precision": None, "recall": None, "f1": None})

        report["sections"][section_name] = sec
        for key in overall:
            overall[key] += sec[key]

    overall["accuracy"] = overall["exact_matches"] / overall["compared"] if overall["compared"] else None
    # Recomputed from the pooled totals rather than averaged across sections,
    # so a section with more options does not weigh the same as a smaller one.
    tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
    overall["precision"] = tp / (tp + fp) if (tp + fp) else None
    overall["recall"] = tp / (tp + fn) if (tp + fn) else None
    p, r = overall["precision"], overall["recall"]
    overall["f1"] = 2 * p * r / (p + r) if (p and r) else None
    report["overall"] = overall

    return report


def _pct(value):
    """A ratio as a percentage, or n/a when it is undefined."""
    return f"{value*100:.1f}%" if value is not None else "n/a"


def print_report(report: dict, show_matrices: bool = True):
    """Prints the report from evaluate() as a per-section table.

    Args:
        report: as returned by evaluate().
        show_matrices: also print one confusion matrix per single-choice
            section, under the table.
    """
    print(f"\nRecords compared: {report['records_compared']}")
    missing = report["records_missing_prediction"]
    if missing:
        print(f"Records with NO prediction found ({len(missing)}): {', '.join(missing)}")

    header = (f"{'Section':<8} {'Acc':>7} {'Compared':>9} {'Missing':>8} "
              f"{'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("\n" + header)
    print("-" * len(header))
    for name, sec in report["sections"].items():
        print(f"{name:<8} {_pct(sec['accuracy']):>7} {sec['compared']:>9} {sec['missing']:>8} "
              f"{sec['tp']:>5} {sec['tn']:>5} {sec['fp']:>5} {sec['fn']:>5} "
              f"{_pct(sec['precision']):>7} {_pct(sec['recall']):>7} {_pct(sec['f1']):>7}")

    o = report["overall"]
    print("-" * len(header))
    print(f"{'TOTAL':<8} {_pct(o['accuracy']):>7} {o['compared']:>9} {'':>8} "
          f"{o['tp']:>5} {o['tn']:>5} {o['fp']:>5} {o['fn']:>5} "
          f"{_pct(o['precision']):>7} {_pct(o['recall']):>7} {_pct(o['f1']):>7}")

    if show_matrices:
        print_confusion_matrices(report)


def print_confusion_matrices(report: dict):
    """Prints one reference-vs-predicted matrix per single-choice section."""
    for name, sec in report["sections"].items():
        matrix = sec.get("confusion_matrix")
        if not matrix:
            continue
        options = sec["options"]
        print(f"\n{name} -- rows: reference, columns: predicted")
        for i, option in enumerate(options, start=1):
            print(f"   {i}. {option[:70]}")
        print("        " + "".join(f"{j:>6}" for j in range(1, len(options) + 1)))
        for i, row in enumerate(matrix, start=1):
            print(f"   {i:<5} " + "".join(f"{count:>6}" for count in row))


def main():
    """Compares one run's output against reference answers and saves a report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_dir", nargs="?", default=str(DEFAULT_PREDICTIONS_DIR),
                        help="directory of pipeline output JSON files (default: ./output)")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH_DIR),
                        help="directory of *_ground_truth.json reference files "
                             "(default: ./data/synthetic_records)")
    parser.add_argument("--no-matrices", action="store_true",
                        help="skip the per-section confusion matrices")
    args = parser.parse_args()

    ground_truth_dir = Path(args.ground_truth)
    ground_truth = load_ground_truth(ground_truth_dir)
    if not ground_truth:
        sys.exit(f"No *_ground_truth.json files found in {ground_truth_dir}")

    predictions = load_predictions(Path(args.predictions_dir))
    if not predictions:
        print(f"[WARNING] No prediction JSON files found in {args.predictions_dir} -- "
              f"every record will be reported as missing.", flush=True)

    report = evaluate(ground_truth, predictions)
    report["ground_truth_dir"] = str(ground_truth_dir)
    report["predictions_dir"] = str(Path(args.predictions_dir))
    print_report(report, show_matrices=not args.no_matrices)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"evaluation_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
