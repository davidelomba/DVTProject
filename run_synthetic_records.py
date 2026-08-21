"""
Runs the extraction pipeline once for every synthetic clinical record in
data/synthetic_records/ (see generate_synthetic_records.py), saving output +
audit log to ./output/ in the same format/naming convention main.py uses
for a single record -- so evaluate_predictions.py can pick up the
results directly with its default predictions directory (./output).

Uses whichever config.EXTRACTOR_MODE is currently set in config.py, exactly
like main.py -- to compare modes, run this script once per mode (edit
config.py, rerun; timestamped filenames mean nothing gets overwritten, so
results from different modes/runs can coexist in ./output).

record_id is derived from each file's name (without .txt). This is exactly
the record_id generate_synthetic_records.py already baked into the matching
*_ground_truth.json, which is what lets evaluate_predictions.py match
predictions back to ground truth later -- no manual bookkeeping needed.

One record failing does not stop the batch: the error is printed and the
script moves on to the next record, same resilience pattern as pipeline.py's
own per-section loop.

Note: this calls pipeline.run_pipeline() once per record, so the per-record
setup is repeated every time: the Brighton PDF is re-parsed and the EHR vector
store rebuilt from scratch (the Brighton store is reloaded from disk, not
re-embedded). Inherited from pipeline.py's current design, not changed here.

Usage:
    python run_synthetic_records.py
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import config
from pipeline import run_pipeline
from aggregation import form_to_json_summary

RECORDS_DIR = Path(__file__).parent / "data" / "synthetic_records"
BRIGHTON_PDF_PATH = str(Path(__file__).parent / "data" / "reference" / "1-s2.0-S0264410X22010854-main.pdf")
OUTPUT_DIR = Path(__file__).parent / "output"


def run_one(record_id: str, record_path: Path) -> Path:
    """Runs the pipeline on one record and writes its two output files.

    Args:
        record_id: derived from the file name; must match the record_id in the
            corresponding *_ground_truth.json for the evaluation to pair them.
        record_path: the .txt clinical record.

    Returns:
        Path of the form JSON that was written.
    """
    form, audit_log = run_pipeline(record_id, str(record_path), BRIGHTON_PDF_PATH)
    summary = form_to_json_summary(form)

    # Same shared-timestamp-per-run convention as main.py, so a JSON and its
    # audit log always pair up and re-running never overwrites a prior run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{record_id}_{timestamp}.json"
    audit_path = OUTPUT_DIR / f"{record_id}_{timestamp}_audit_log.json"

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_log, indent=2, ensure_ascii=False), encoding="utf-8")

    return output_path


def main():
    """Runs every synthetic record through the pipeline, in file-name order.

    One record failing does not stop the batch: the error is reported and the
    run continues.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    record_paths = sorted(RECORDS_DIR.glob("*.txt"))

    if not record_paths:
        print(f"No .txt records found in {RECORDS_DIR} -- run generate_synthetic_records.py first.", flush=True)
        return

    print(f"Found {len(record_paths)} synthetic records. Running pipeline "
          f"(EXTRACTOR_MODE={config.EXTRACTOR_MODE!r})...\n", flush=True)

    succeeded, failed, durations = [], [], []
    batch_start = time.time()
    for i, record_path in enumerate(record_paths, start=1):
        record_id = record_path.stem

        # Non-blocking sanity check: flags a stray .txt with no matching
        # ground truth, which would silently go unscored later.
        gt_path = RECORDS_DIR / f"{record_id}_ground_truth.json"
        if not gt_path.exists():
            print(f"[{i}/{len(record_paths)}] {record_id}: [WARNING] no matching "
                  f"{gt_path.name} found -- evaluate_predictions.py won't "
                  f"be able to score this record.", flush=True)

        print(f"[{i}/{len(record_paths)}] {record_id} ...", flush=True)
        t0 = time.time()
        try:
            output_path = run_one(record_id, record_path)
            succeeded.append(record_id)
            elapsed = time.time() - t0
            durations.append(elapsed)
            # Remaining time estimated from the mean so far rather than the
            # last record: per-record duration varies with how many tool calls
            # the agentic extractor decides to make.
            remaining = (len(record_paths) - i) * (sum(durations) / len(durations))
            print(f"  -> saved {output_path.name} in {elapsed / 60:.1f} min "
                  f"(ETA {remaining / 60:.0f} min for the remaining "
                  f"{len(record_paths) - i})", flush=True)
        except Exception as exc:
            failed.append(record_id)
            print(f"  -> FAILED after {(time.time() - t0) / 60:.1f} min: {exc}", flush=True)
            traceback.print_exc()

    total = time.time() - batch_start
    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed "
          f"in {total / 60:.0f} min total.", flush=True)
    if durations:
        print(f"Per record: mean {sum(durations) / len(durations) / 60:.1f} min, "
              f"min {min(durations) / 60:.1f}, max {max(durations) / 60:.1f}.", flush=True)
    if failed:
        print(f"Failed records: {', '.join(failed)}", flush=True)
    print(f"\nNow run: python evaluate_predictions.py", flush=True)


if __name__ == "__main__":
    main()
