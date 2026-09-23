"""
Turns MYO_dummy_cases.xlsx into the record and ground-truth files the pipeline
and evaluate_predictions.py read.

One .txt per row holding the text of the dummy case, and one
<record_id>_ground_truth.json beside it holding the two answers, in the shape
models_myo produces. record_id is Case_id and Variation_id joined, so the rows
of one base case stay distinguishable.

The spreadsheet's answers are written by hand and abbreviate the questionnaire's
option text, so OPTION_ALIASES maps each abbreviation to the string in
models_myo. An answer that matches nothing stops the conversion rather than
being dropped: a silently unmapped option would score as an error later.

Usage:
    python myo/convert_cases.py path/to/MYO_dummy_cases.xlsx
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models_myo import SECTION_MODELS  # noqa: E402


RECORDS_DIR = Path(__file__).resolve().parent / "data" / "records"

# Spreadsheet wording -> the option as models_myo spells it. The keys are
# compared after collapsing whitespace and lowercasing.
OPTION_ALIASES = {
    "E": {
        "paroxysmal or sustained atrial or ventricular arrhythmias":
            "Paroxysmal or sustained atrial or ventricular arrhythmias (premature "
            "atrial or ventricular beats, and/or supraventricular or ventricular "
            "tachycardia, interventricular conduction delay, abnormal Q waves, low "
            "voltages): :e.g (Sinus tachycardia, SVT, atrial fibrillation, PVCs, VT, VF)",
        "st-segment or t-wave abnormalities":
            "ST, segment or T, wave abnormalities (elevation or inversion)",
    },
    "F": {
        "global systolic or diastolic function depression/abnormality":
            "Global systolic or diastolic function depression/abnormality - "
            "(*Decreased longitudinal and circumferential strain and strain rates on "
            "tissue Doppler)",
        "new focal or diffuse left or right ventricular function abnormalities":
            "New focal or diffuse left or right ventricular function abnormalities "
            "(e.g. decreased ejection fraction)",
    },
}

# Spreadsheet column -> section key.
ANSWER_COLUMNS = {"E": "E_ECG", "F": "F_echocardiogram"}


def _valid_options(section_key: str) -> list:
    """The options models_myo accepts for one section, in schema order."""

    from typing import get_args
    model = SECTION_MODELS[section_key]
    annotation = model.model_fields[next(iter(model.model_fields))].annotation
    return list(get_args(get_args(annotation)[0]))


def _collapse(text: str) -> str:
    """Whitespace collapsed and lowercased, for comparing written answers."""

    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def parse_answer(section_key: str, raw: str) -> list:
    """The options one spreadsheet cell names.

    Args:
        section_key: E or F.
        raw: the cell, one or more options joined by +.

    Returns:
        The matching strings from models_myo, in schema order.

    Raises:
        ValueError: when a piece matches no option, so the mapping is fixed
            once rather than losing an answer.
    """

    options = _valid_options(section_key)
    by_collapsed = {_collapse(opt): opt for opt in options}
    aliases = {k: v for k, v in OPTION_ALIASES[section_key].items()}

    chosen = []
    for piece in _collapse(raw).split("+"):
        piece = piece.strip()
        if not piece:
            continue
        if piece in by_collapsed:
            chosen.append(by_collapsed[piece])
        elif piece in aliases:
            chosen.append(aliases[piece])
        else:
            raise ValueError(f"{section_key}: no option matches {piece!r}")
    return [opt for opt in options if opt in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="MYO_dummy_cases.xlsx")
    parser.add_argument("--output-dir", type=Path, default=RECORDS_DIR)
    args = parser.parse_args()

    import openpyxl
    sheet = openpyxl.load_workbook(args.workbook)["Cases"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    header = rows[0]
    column = {name: index for index, name in enumerate(header) if name}
    cases = [row for row in rows[1:] if row[column["Case_id"]] is not None]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in cases:
        case_id = str(row[column["Case_id"]]).strip()
        variation = str(row[column["Variation_id"]]).strip().replace(".", "_")
        record_id = f"{case_id}_{variation}"
        text = str(row[column["Text of the dummy case"]] or "").strip()

        answers = {"record_id": record_id}
        for section_key, column_name in ANSWER_COLUMNS.items():
            selected = parse_answer(section_key, row[column[column_name]])
            field = next(iter(SECTION_MODELS[section_key].model_fields))
            answers[section_key.lower()] = {field: selected}

        (args.output_dir / f"{record_id}.txt").write_text(text, encoding="utf-8")
        with open(args.output_dir / f"{record_id}_ground_truth.json", "w",
                  encoding="utf-8") as handle:
            json.dump(answers, handle, indent=2, ensure_ascii=False)
        written += 1
        print(f"{record_id:12s} {len(text):5d} caratteri   "
              f"E {len(answers['e']['findings'])}   F {len(answers['f']['findings'])}")

    print(f"\n{written} record scritti in {args.output_dir}")


if __name__ == "__main__":
    main()
