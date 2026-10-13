"""
Agreement between repeated runs of the same configuration.

Given two or more run directories produced with identical settings (at a
temperature above 0, so that they can differ), every section of every record
has one answer per run. The runs play the part of raters and the records the
part of subjects, as in the inter-rater analysis of the VAC4EU validation
pipeline. Reported per section and overall:

  - Fleiss' kappa across the runs, the answer set being the category (the same
    exact-match convention evaluate_predictions uses), unweighted;
  - Gwet's AC1 across the runs, steadier than kappa when one answer dominates.
    Its number of categories is the number of distinct answers observed in the
    section, since a multiple-choice section has no fixed list of categories;
  - unanimous: the share of records on which every run gave the same answer;
  - the accuracy of each run against the ground truth, and that of the
    majority vote (an answer given by more than half of the runs; a record
    with no such answer counts as wrong and is counted apart).

Agreement as a confidence signal: each run's answer gets the share of runs
that gave the same answer (1/3, 2/3 or 1 with three runs). The AUROC of that
share between right and wrong answers says whether a stable answer is more
often right. When the runs carry Agent 3's verifier, the AUROC of P(YES) on the
same answers is reported beside it, per input variant.

A section a run left without an answer counts as the category "<missing>" and
as wrong.

Usage:
    python agreement_runs.py ./myo/output_r1 ./myo/output_r2 ./myo/output_r3 \\
        --ground-truth ./myo/data/records
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluate_predictions import (
    DEFAULT_GROUND_TRUTH_DIR,
    REPORTS_DIR,
    _selected_set,
    load_ground_truth,
)
from evaluate_verifier import auroc, load_runs

MISSING = "<missing>"


def _label(value) -> str:
    """The answer of one section as a category label: its sorted option set."""

    selected = _selected_set(value)
    return MISSING if selected is None else repr(sorted(selected))


def fleiss_kappa(table: list) -> float:
    """Fleiss' kappa from per-subject category counts.

    Args:
        table: one Counter per subject, category -> number of raters, every
            subject rated by the same number of raters.

    Returns:
        The kappa, or None when the expected agreement is 1 (every rating in
        one category), where it is undefined.
    """

    n = sum(table[0].values())
    total = len(table) * n
    categories = {c for row in table for c in row}
    p = {c: sum(row[c] for row in table) / total for c in categories}
    observed = sum((sum(v * v for v in row.values()) - n) / (n * (n - 1)) for row in table) / len(table)
    expected = sum(v * v for v in p.values())
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def gwet_ac1(table: list):
    """Gwet's AC1 from per-subject category counts.

    The chance agreement is the sum over categories of p(1 - p), divided by
    the number of categories minus one, the categories being those observed.

    Returns:
        The AC1; 1.0 when a single category was observed, since every rating
        then agrees.
    """

    n = sum(table[0].values())
    total = len(table) * n
    categories = {c for row in table for c in row}
    if len(categories) == 1:
        return 1.0
    p = {c: sum(row[c] for row in table) / total for c in categories}
    observed = sum((sum(v * v for v in row.values()) - n) / (n * (n - 1)) for row in table) / len(table)
    expected = sum(v * (1 - v) for v in p.values()) / (len(categories) - 1)
    return (observed - expected) / (1 - expected)


def collect(run_dirs: list, ground_truth: dict) -> tuple:
    """Answers, correctness and verifier scores for the records all runs share.

    Returns:
        (rows, sections): rows holds one dict per (record, section) with the
        label of every run, the reference label and, per run, the verifier
        results; sections lists the section keys in ground-truth order.
    """

    runs = [load_runs(Path(d)) for d in run_dirs]
    records = sorted(set(ground_truth).intersection(*[set(r) for r in runs]))
    if not records:
        return [], []
    sections = [k for k in ground_truth[records[0]] if k != "record_id" and not k.startswith("_")]

    rows = []
    for record_id in records:
        truth = ground_truth[record_id]
        for key in sections:
            labels, verdicts = [], []
            for run in runs:
                prediction, audit = run[record_id]
                labels.append(_label(prediction.get(key)))
                entry = next((v for k, v in audit.items() if k.lower() == key and isinstance(v, dict)), {})
                verdicts.append(entry.get("verifier") or {})
            rows.append({"record_id": record_id, "section": key.upper(),
                         "labels": labels, "truth": _label(truth[key]),
                         "verifier": verdicts})
    return rows, [s.upper() for s in sections]


def summarise(rows: list) -> dict:
    """The agreement and accuracy figures of the module docstring for some rows."""

    n_runs = len(rows[0]["labels"])
    table = [Counter(r["labels"]) for r in rows]

    majority_right, no_majority = 0, 0
    for r, counts in zip(rows, table):
        label, count = counts.most_common(1)[0]
        if count * 2 > n_runs:
            majority_right += label == r["truth"]
        else:
            no_majority += 1

    unanimous = [r for r, counts in zip(rows, table) if len(counts) == 1]
    split = [r for r, counts in zip(rows, table) if len(counts) > 1]

    def _accuracy(subset):
        answers = [lab == r["truth"] for r in subset for lab in r["labels"]]
        return sum(answers) / len(answers) if answers else None

    # One entry per run's answer: its share of agreeing runs and whether it is right
    share_right, share_wrong = [], []
    p_yes = {}
    for r, counts in zip(rows, table):
        for run_index, label in enumerate(r["labels"]):
            share = counts[label] / n_runs
            right = label == r["truth"]
            (share_right if right else share_wrong).append(share)
            for variant, outcome in r["verifier"][run_index].items():
                if outcome.get("p_agree") is not None:
                    p_yes.setdefault(variant, ([], []))[0 if right else 1].append(outcome["p_agree"])

    return {
        "records": len(rows),
        "fleiss_kappa": fleiss_kappa(table),
        "gwet_ac1": gwet_ac1(table),
        "unanimous": len(unanimous) / len(rows),
        "accuracy_per_run": [sum(r["labels"][i] == r["truth"] for r in rows) / len(rows)
                             for i in range(n_runs)],
        "accuracy_majority": majority_right / len(rows),
        "no_majority": no_majority,
        "accuracy_when_unanimous": _accuracy(unanimous),
        "accuracy_when_split": _accuracy(split),
        "auroc_agreement_share": auroc(share_right, share_wrong),
        "auroc_verifier": {v: auroc(right, wrong) for v, (right, wrong) in sorted(p_yes.items())},
    }


def evaluate(rows: list, sections: list) -> dict:
    report = {"overall": summarise(rows), "per_section": {}}
    for section in sections:
        subset = [r for r in rows if r["section"] == section]
        if subset:
            report["per_section"][section] = summarise(subset)
    report["split_records"] = [
        {"record_id": r["record_id"], "section": r["section"],
         "answers": r["labels"], "truth": r["truth"]}
        for r in rows if len(set(r["labels"])) > 1
    ]
    return report


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(report: dict, run_dirs: list) -> None:
    print(f"runs: {', '.join(str(d) for d in run_dirs)}")
    header = (f"{'section':<8} {'kappa':>6} {'AC1':>6} {'unan.':>6} "
              f"{'acc/run':>17} {'major.':>7} {'AUROC agr.':>10}")
    print(header)
    print("-" * len(header))
    for name, s in [*report["per_section"].items(), ("ALL", report["overall"])]:
        per_run = " ".join(_fmt(a, 2) for a in s["accuracy_per_run"])
        print(f"{name:<8} {_fmt(s['fleiss_kappa']):>6} {_fmt(s['gwet_ac1']):>6} "
              f"{_fmt(s['unanimous'], 2):>6} {per_run:>17} {_fmt(s['accuracy_majority'], 2):>7} "
              f"{_fmt(s['auroc_agreement_share']):>10}")
    o = report["overall"]
    print(f"\naccuracy when unanimous {_fmt(o['accuracy_when_unanimous'])}, "
          f"when split {_fmt(o['accuracy_when_split'])}; records without a majority "
          f"{o['no_majority']}")
    for variant, value in o["auroc_verifier"].items():
        print(f"AUROC verifier P(YES), variant {variant}: {_fmt(value)} (same answers)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="two or more run directories")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH_DIR))
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are needed")

    rows, sections = collect(args.run_dirs, load_ground_truth(Path(args.ground_truth)))
    if not rows:
        print("No record is shared by all the runs and the ground truth.")
        return
    report = evaluate(rows, sections)
    report["runs"] = [str(d) for d in args.run_dirs]
    print_report(report, args.run_dirs)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"agreement_{Path(args.run_dirs[0]).name}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
