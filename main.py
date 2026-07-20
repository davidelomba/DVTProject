"""
Entry point for the DVT pipeline.

Usage:
    python main.py

Edit the three variables below to point to the actual clinical record,
Brighton PDF, and the desired patient/record identifier.
Run test_step0.py first to verify the model is behaving reliably before
processing real patient data.
"""

import json
import os

from pipeline import run_pipeline
from aggregation import form_to_json_summary


def main():
    # Replace these with the actual locations of your files.
    record_id = "PATIENT_001"
    patient_ehr_path = "./patient_001.txt"                        # plain .txt clinical record
    brighton_pdf_path = "./1-s2.0-S0264410X22010854-main.pdf"    # Brighton paper PDF

    form, audit_log = run_pipeline(record_id, patient_ehr_path, brighton_pdf_path)

    summary = form_to_json_summary(form)

    print("\n--- Filled-in JSON (checkboxes) ---")
    print(json.dumps(summary, indent=2))

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{record_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Audit log: evidence, Brighton context, and full model reasoning for
    # every section, including failed ones. Kept as a separate file (not
    # merged into the clean output JSON) so it doesn't need to be shared
    # downstream, but is available whenever a specific answer needs to be
    # checked without re-running the pipeline.
    audit_path = os.path.join(output_dir, f"{record_id}_audit_log.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {os.path.abspath(output_path)}")
    print(f"Audit log saved to: {os.path.abspath(audit_path)}")


if __name__ == "__main__":
    main()
