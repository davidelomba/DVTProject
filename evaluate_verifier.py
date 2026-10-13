"""
Scores Agent 3's verifier against the ground truth.

For every section that has a verifier result, the section's answer is right or
wrong against the ground truth, and the verifier agreed or not with a P(YES).
Reported for each input variant, overall and per section:

  - AUROC of P(YES) between right and wrong answers: the probability that a
    random right answer gets a higher P(YES) than a random wrong one, ties
    counting one half;
  - approved errors: wrong answers the verifier agreed with, over the wrong
    answers;
  - rejected right answers: right answers it disagreed with, over the right
    answers.

Sections whose verdict a cross-section rule copied (method "rule:<source>")
are counted, since that verdict is the one the form carries; sections without
a verdict are listed apart.

Usage:
    python evaluate_verifier.py ./output_verifier
    python evaluate_verifier.py ./myo/output_myo_verifier --ground-truth ./myo/data/records
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from evaluate_predictions import (
    DEFAULT_GROUND_TRUTH_DIR,
    REPORTS_DIR,
    _selected_set,
    load_ground_truth,
)


def load_runs(run_dir: Path) -> dict:
    """record_id -> (prediction, audit log), the newest file per record."""

    runs = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.name.endswith("_audit_log.json"):
            continue
        audit_path = path.with_name(path.stem + "_audit_log.json")
        if not audit_path.exists():
            continue
        prediction = json.loads(path.read_text(encoding="utf-8"))
        runs[prediction["record_id"]] = (
            prediction, json.loads(audit_path.read_text(encoding="utf-8")))
    return runs


def auroc(right: list, wrong: list):
    """P(score of a right answer > score of a wrong one), ties counting 1/2.

    None when either group is empty.
    """

    if not right or not wrong:
        return None
    wins = sum(1.0 if r > w else 0.5 if r == w else 0.0 for r in right for w in wrong)
    return wins / (len(right) * len(wrong))


def collect(runs: dict, ground_truth: dict) -> list:
    """One row per (record, section, variant) with a verifier result."""

    rows = []
    for record_id, (prediction, audit) in runs.items():
        truth = ground_truth.get(record_id)
        if truth is None:
            continue
        for section_key, entry in audit.items():
            if section_key.startswith("_") or not isinstance(entry, dict):
                continue
            key = section_key.lower()
            if key not in truth:
                continue
            correct = _selected_set(prediction.get(key)) == _selected_set(truth[key])
            for variant, outcome in (entry.get("verifier") or {}).items():
                rows.append({
                    "record_id": record_id,
                    "section": section_key,
                    "variant": variant,
                    "correct": correct,
                    "agree": outcome.get("agree"),
                    "p_agree": outcome.get("p_agree"),
                    "method": outcome.get("method"),
                    "overflow": outcome.get("overflow", False),
                })
    return rows


def summarise(rows: list) -> dict:
    """The metrics of the module docstring for one group of rows."""

    scored = [r for r in rows if r["agree"] is not None]
    right = [r for r in scored if r["correct"]]
    wrong = [r for r in scored if not r["correct"]]

    return {
        "sections": len(rows),
        "without_verdict": len(rows) - len(scored),
        "overflow": sum(1 for r in rows if r["overflow"]),
        "right": len(right),
        "wrong": len(wrong),
        "auroc_verifier": auroc([r["p_agree"] for r in right], [r["p_agree"] for r in wrong]),
        "approved_errors": sum(1 for r in wrong if r["agree"]),
        "rejected_right": sum(1 for r in right if not r["agree"]),
    }


def evaluate(rows: list) -> dict:
    report = {}
    for variant in sorted({r["variant"] for r in rows}):
        variant_rows = [r for r in rows if r["variant"] == variant]
        report[variant] = {
            "overall": summarise(variant_rows),
            "per_section": {s: summarise([r for r in variant_rows if r["section"] == s])
                            for s in sorted({r["section"] for r in variant_rows})},
            "errors": [{k: r[k] for k in ("record_id", "section", "agree", "p_agree")}
                       for r in variant_rows if not r["correct"]],
        }
    return report


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    for variant, data in report.items():
        o = data["overall"]
        print(f"\n=== variant: {variant} ===")
        print(f"sections {o['sections']}  right {o['right']}  wrong {o['wrong']}  "
              f"without verdict {o['without_verdict']}  overflow {o['overflow']}")
        print(f"AUROC P(YES) {_fmt(o['auroc_verifier'])}")
        print(f"approved errors {o['approved_errors']}/{o['wrong']}   "
              f"rejected right answers {o['rejected_right']}/{o['right']}")
        print(f"\n{'section':<8} {'right':>5} {'wrong':>5} {'AUROC':>6} {'appr.err':>8} {'rej.right':>9}")
        for section, s in data["per_section"].items():
            print(f"{section:<8} {s['right']:>5} {s['wrong']:>5} {_fmt(s['auroc_verifier'], 2):>6} "
                  f"{s['approved_errors']:>8} {s['rejected_right']:>9}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="directory of a run produced with the verifier on")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = collect(load_runs(run_dir), load_ground_truth(Path(args.ground_truth)))
    if not rows:
        print("No verifier results found.")
        return
    report = evaluate(rows)
    print_report(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"verifier_{run_dir.name}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
