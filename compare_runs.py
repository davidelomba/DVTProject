"""
Run-to-run stability check: compares TWO pipeline runs over the same records
and reports how many answers changed between them.

WHY THIS EXISTS: config.LLM_TEMPERATURE is 0.0, so the pipeline is nominally
deterministic -- but Ollama does not guarantee bit-identical generations across
calls (GPU non-determinism, batching, KV-cache reuse), so "temperature 0" is
not by itself evidence that a measured accuracy is reproducible. Every metric
produced by evaluate_synthetic_records.py currently comes from a SINGLE run.
This script quantifies the noise floor underneath those metrics: if two
identical runs already disagree on N% of sections, then any accuracy
difference smaller than N% between two configurations is not a result.

NOT the same thing as evaluate_synthetic_records.py, which compares ONE run
against the hand-authored ground truth. This compares two runs against EACH
OTHER (and, additionally, reports each run's own accuracy so a stability
difference can be read next to any accuracy difference).

USAGE
    # Two runs whose output files sit in the same directory (the normal case:
    # main.py / run_synthetic_records.py timestamp every file, so nothing is
    # overwritten). For each record, the two most recent files are compared.
    python compare_runs.py
    python compare_runs.py ./output

    # Two runs kept in separate directories; the most recent file per record
    # is taken from each.
    python compare_runs.py ./output_run1 ./output_run2

HOW TO PRODUCE THE TWO RUNS: execute run_synthetic_records.py twice, changing
NOTHING in between -- same config.py, same model tags, same records. Any edit
between the two runs makes the comparison measure that edit instead of the
noise floor.

Deliberately standalone: imports only models.py (pydantic + typing), like
evaluate_synthetic_records.py, so it has no langchain/Ollama dependency.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import get_args, get_origin

from models import SECTION_MODELS

GROUND_TRUTH_DIR = Path(__file__).parent / "data" / "synthetic_records"

# main.py and run_synthetic_records.py both name outputs
# "<record_id>_<YYYYMMDD>_<HHMMSS>.json". record_id itself contains
# underscores and digits, so the timestamp is anchored to the END of the name
# rather than split on the first underscore.
_OUTPUT_NAME_RE = re.compile(r"^(?P<record_id>.+)_(?P<timestamp>\d{8}_\d{6})\.json$")


def _selected_set(value):
    """Normalizes a section's value (a dict holding one field, either a string
    or a list) into a set of selected option strings, so single-choice and
    multi-select sections compare identically. None if the section is missing
    (the pipeline left it None after a failed evaluation)."""
    if value is None:
        return None
    inner = next(iter(value.values()))
    return set(inner) if isinstance(inner, list) else {inner}


def load_run_files(directory: Path) -> dict:
    """record_id -> [(timestamp, parsed_json), ...] sorted oldest-first.

    Skips *_audit_log.json (not a form output) and any file whose name doesn't
    carry a parseable timestamp, so a hand-renamed or unrelated JSON sitting in
    the directory can't be silently mistaken for a run.
    """
    runs = defaultdict(list)
    if not directory.is_dir():
        sys.exit(f"Not a directory: {directory}")

    for path in sorted(directory.glob("*.json")):
        if path.name.endswith("_audit_log.json"):
            continue
        match = _OUTPUT_NAME_RE.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        record_id = data.get("record_id") or match.group("record_id")
        runs[record_id].append((match.group("timestamp"), data, path.name))

    for record_id in runs:
        runs[record_id].sort(key=lambda t: t[0])
    return runs


def pair_up(dir_a: Path, dir_b: Path = None) -> dict:
    """record_id -> (older, newer) prediction pair, each (timestamp, data, filename).

    One directory: the two most recent files for that record (the two runs are
    assumed to be consecutive, which is what running the driver twice produces).
    Two directories: the most recent file for that record from each.
    """
    if dir_b is None:
        runs = load_run_files(dir_a)
        pairs = {}
        for record_id, entries in runs.items():
            if len(entries) >= 2:
                pairs[record_id] = (entries[-2], entries[-1])
        return pairs

    runs_a, runs_b = load_run_files(dir_a), load_run_files(dir_b)
    pairs = {}
    for record_id in set(runs_a) & set(runs_b):
        pairs[record_id] = (runs_a[record_id][-1], runs_b[record_id][-1])
    return pairs


def load_ground_truth() -> dict:
    """record_id -> ground truth dict, for the per-run accuracy columns."""
    gt = {}
    for path in sorted(GROUND_TRUTH_DIR.glob("*_ground_truth.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gt[data["record_id"]] = data
    return gt


def compare(pairs: dict, ground_truth: dict) -> dict:
    section_names = list(SECTION_MODELS.keys())
    report = {
        "records_compared": len(pairs),
        "sections": {
            name: {"compared": 0, "changed": 0, "correct_a": 0, "correct_b": 0}
            for name in section_names
        },
        "differences": [],
    }

    for record_id, ((ts_a, data_a, file_a), (ts_b, data_b, file_b)) in sorted(pairs.items()):
        gt_record = ground_truth.get(record_id)

        for name in section_names:
            key = name.lower()
            set_a = _selected_set(data_a.get(key))
            set_b = _selected_set(data_b.get(key))
            sec = report["sections"][name]

            # A section missing from BOTH runs is consistently missing, which
            # is itself stable -- but there's no answer to compare, so it's
            # excluded rather than counted as "unchanged" (that would inflate
            # the stability figure with sections that never produced anything).
            if set_a is None and set_b is None:
                continue

            sec["compared"] += 1
            if set_a != set_b:
                sec["changed"] += 1
                report["differences"].append({
                    "record_id": record_id,
                    "section": name,
                    "run_a": sorted(set_a) if set_a is not None else None,
                    "run_b": sorted(set_b) if set_b is not None else None,
                })

            if gt_record is not None:
                gt_set = _selected_set(gt_record.get(key))
                if set_a == gt_set:
                    sec["correct_a"] += 1
                if set_b == gt_set:
                    sec["correct_b"] += 1

    overall = {"compared": 0, "changed": 0, "correct_a": 0, "correct_b": 0}
    for sec in report["sections"].values():
        sec["stability"] = (
            (sec["compared"] - sec["changed"]) / sec["compared"] if sec["compared"] else None
        )
        for k in overall:
            overall[k] += sec[k]
    overall["stability"] = (
        (overall["compared"] - overall["changed"]) / overall["compared"]
        if overall["compared"] else None
    )
    report["overall"] = overall
    return report


def print_report(report: dict):
    print(f"\nRecords compared across both runs: {report['records_compared']}")

    header = f"{'Section':<8} {'Stability':>10} {'Compared':>9} {'Changed':>8} {'AccRunA':>8} {'AccRunB':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, sec in report["sections"].items():
        n = sec["compared"]
        stab = f"{sec['stability']*100:.1f}%" if sec["stability"] is not None else "n/a"
        acc_a = f"{sec['correct_a']/n*100:.1f}%" if n else "n/a"
        acc_b = f"{sec['correct_b']/n*100:.1f}%" if n else "n/a"
        print(f"{name:<8} {stab:>10} {n:>9} {sec['changed']:>8} {acc_a:>8} {acc_b:>8}")

    o = report["overall"]
    n = o["compared"]
    stab = f"{o['stability']*100:.1f}%" if o["stability"] is not None else "n/a"
    acc_a = f"{o['correct_a']/n*100:.1f}%" if n else "n/a"
    acc_b = f"{o['correct_b']/n*100:.1f}%" if n else "n/a"
    print("-" * len(header))
    print(f"{'TOTAL':<8} {stab:>10} {n:>9} {o['changed']:>8} {acc_a:>8} {acc_b:>8}")

    if report["differences"]:
        print(f"\nSections that changed between the two runs ({len(report['differences'])}):")
        for d in report["differences"]:
            print(f"  {d['record_id']} [{d['section']}]")
            print(f"      run A: {d['run_a']}")
            print(f"      run B: {d['run_b']}")
    else:
        print("\nNo section changed between the two runs: output was fully reproducible.")

    # The headline number for the thesis: any accuracy gap smaller than this
    # can't be distinguished from run-to-run noise.
    if o["compared"]:
        noise = o["changed"] / o["compared"] * 100
        print(
            f"\nNoise floor: {noise:.1f}% of sections changed between two identical runs. "
            f"Accuracy differences below this are not interpretable as real effects."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dirs", nargs="*", default=None,
        help="One directory (compares the two most recent files per record) or "
             "two directories (compares the most recent file per record from each). "
             "Defaults to ./output, resolved relative to this script's own location.",
    )
    args = parser.parse_args()

    dirs = args.dirs or [str(Path(__file__).parent / "output")]
    if len(dirs) > 2:
        sys.exit("Pass at most two directories.")

    dir_a = Path(dirs[0])
    dir_b = Path(dirs[1]) if len(dirs) == 2 else None

    pairs = pair_up(dir_a, dir_b)
    if not pairs:
        sys.exit(
            "No record had two runs to compare.\n"
            "Run run_synthetic_records.py twice without changing anything in between, "
            "or pass two directories explicitly."
        )

    report = compare(pairs, load_ground_truth())
    print_report(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"run_stability_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
