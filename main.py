"""
Entry point for the clinical extraction pipeline.

Usage:
    python main.py

Edit the three variables below to point to the actual clinical record,
the reference guidelines PDF and the desired patient/record identifier.
"""

import json
import os
from datetime import datetime

from pipeline import run_pipeline
from aggregation import form_to_json_summary


def main():
    # Replace these with the actual locations of your files.
    record_id = "PATIENT_001"
    patient_ehr_path = "./data/patient_001.txt"                                   # Plain .txt clinical record
    brighton_pdf_path = "./data/reference/1-s2.0-S0264410X22010854-main.pdf"      # Brighton guidelines PDF

    # Run the extraction and evaluation pipeline
    form, audit_log = run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)

    # Serialize the populated schema
    summary = form_to_json_summary(form)

    print("\n--- Filled-in JSON (checkboxes) ---")
    print(json.dumps(summary, indent=2))

    # Ensure output directory exists
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp shared by both files of this run, so a JSON and its matching
    # audit log can always be paired up, and re-running never overwrites a
    # previous run's output.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the final structured output
    output_path = os.path.join(output_dir, f"{record_id}_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save the Audit log.
    # Contains extracted evidence, guideline context and full model reasoning for
    # every section (including failed ones). Kept as a separate file (not merged
    # into the clean output JSON) so it doesn't need to be shared downstream,
    # but remains available whenever a specific answer needs manual verification
    # without re-running the pipeline.
    audit_path = os.path.join(output_dir, f"{record_id}_{timestamp}_audit_log.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {os.path.abspath(output_path)}")
    print(f"Audit log saved to: {os.path.abspath(audit_path)}")


if __name__ == "__main__":
    main()