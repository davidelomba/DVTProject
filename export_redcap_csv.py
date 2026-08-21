"""
Converts the pipeline's questionnaire output into a CSV that REDCap can import,
so the Level of Certainty (LOC) can be computed by the REDCap project itself.

Writes the format REDCap's Data Import Tool expects: variable names as headers,
numeric codes as values, and checkboxes as <field>___<code> holding 0 or 1.

WHAT IS AND IS NOT WRITTEN
  - The ten criteria the pipeline answers, as numeric codes read from each
    section's schema (see _option_code).
  - Free-text and date fields, as empty columns.
  - Nothing for REDCap's calculated fields (sum_*, calc_*, text_*, colors_*,
    level*), which the LOC computation produces from the imported answers.

A blank cell clears that field on import; --skip-empty-fields writes only the
criteria columns.

USAGE
    python export_redcap_csv.py                                  # ./output -> ./redcap_import.csv
    python export_redcap_csv.py ./output ./redcap_import.csv     # explicit paths
    python export_redcap_csv.py --skip-empty-fields              # criteria columns only

Standalone by design: imports models.py only, so it needs no langchain or
Ollama installation.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import get_args, get_origin

from models import SECTION_MODELS

# Output naming convention: "<record_id>_<YYYYMMDD>_
# <HHMMSS>.json", with the timestamp anchored to the END because record_id
# itself contains underscores and digits.
_OUTPUT_NAME_RE = re.compile(r"^(?P<record_id>.+)_(?P<timestamp>\d{8}_\d{6})\.json$")

# REDCap's per-form status field: 0 Incomplete, 1 Unverified, 2 Complete.
# Set to Complete so the imported rows are treated as finished data entry
# rather than drafts.
FORM_COMPLETE_VALUE = "2"

# How each questionnaire section maps onto REDCap fields, in the order the
# columns appear in the project's own export.
#
# "radio"    -> one column, holding the option's 1-based position.
# "checkbox" -> one column per option, named <field>___<position>, holding 1
#               when that option was selected and 0 when it was not.
# "yesno"    -> one column, but coded 1/0 rather than 1/2. Section F is the
#               only one of these, and the only place where the code does NOT
#               follow the option's position in the schema.
SECTION_FIELDS = {
    "A1": ("criteria_a1", "radio"),
    "A2": ("criteria_a2", "radio"),
    "A3_1": ("criteria_a3_1", "radio"),
    "A3_2": ("criteria_a3_2", "checkbox"),
    "B1_1": ("criteria_b1_1", "radio"),
    "B1_2": ("criteria_b1_2", "checkbox"),
    "B2": ("criteria_b2", "checkbox"),
    "C": ("criteria_c", "radio"),
    "F": ("criteria_f", "yesno"),
    "X": ("criteria_x", "radio"),
}

# Answers of the yesno section, mapped to REDCap's own coding.
_YESNO_CODES = {"Yes": "1", "No": "0"}

# Fields belonging to the form but not produced by the pipeline: free text,
# dates and the registration data entered by hand. Written as empty columns so
# the CSV mirrors the form's structure, unless --skip-empty-fields is given.
# Positioned by the key they follow in the export, so the column order matches
# what REDCap itself produces.
UNFILLED_FIELDS_BEFORE = {
    "criteria_a1": [
        "vac4eu_id", "illness_tcd", "thrombocytopenia_tcd", "illness_hospital",
        "hospital_tcd", "illness_admittingdiag", "illness_dischargediag",
    ],
    "criteria_a2": ["a1_autopsy_tcd", "a1_description"],
    "criteria_a3_1": ["a2_description"],
    "criteria_b1_1": ["a3_2_other", "criteria_a3_3"],
    "criteria_f": ["ddimer_hm_value", "c_test_limitnormal"],
}
UNFILLED_FIELDS_AFTER = {
    "criteria_x": ["x_description"],
}


def _section_options(section_key: str) -> list[str]:
    """The section's options, in the order its schema declares them.

    That order is what gives each option its REDCap code, so reading it from
    the schema rather than repeating it here means a change to models.py cannot
    silently produce a CSV whose codes point at the wrong options.

    Args:
        section_key: a key of SECTION_MODELS, e.g. "A3_2".

    Returns:
        The option strings, in schema order.
    """

    field = next(iter(SECTION_MODELS[section_key].model_fields.values()))
    annotation = field.annotation

    # Multi-select sections are list[Literal[...]], single-choice ones are a
    # bare Literal[...]; unwrap the list before reading the literal's values.
    if get_origin(annotation) is list:
        annotation = get_args(annotation)[0]
    return [value for value in get_args(annotation) if isinstance(value, str)]


def _option_code(section_key: str, answer: str) -> str:
    """The REDCap code for one selected option.

    Args:
        section_key: a key of SECTION_MODELS.
        answer: the option text, exactly as the schema spells it.

    Returns:
        The option's 1-based position, as a string.

    Raises:
        ValueError: if the answer is not one of the section's options, which
            means the results file and models.py disagree.
    """

    options = _section_options(section_key)
    try:
        return str(options.index(answer) + 1)
    except ValueError:
        raise ValueError(
            f"{section_key}: {answer!r} is not one of this section's options "
            f"({options})"
        ) from None


def build_column_order(skip_empty_fields: bool = False) -> list[str]:
    """The CSV's columns, in the order REDCap's own raw export uses.

    Args:
        skip_empty_fields: omit the free-text/date columns the pipeline never
            fills, leaving only record_id, the criteria and the form status.

    Returns:
        The column names.
    """

    columns = ["record_id"]

    for section_key, (field, kind) in SECTION_FIELDS.items():
        if not skip_empty_fields:
            columns.extend(UNFILLED_FIELDS_BEFORE.get(field, []))

        if kind == "checkbox":
            columns.extend(
                f"{field}___{i}" for i in range(1, len(_section_options(section_key)) + 1)
            )
        else:
            columns.append(field)

        if not skip_empty_fields:
            columns.extend(UNFILLED_FIELDS_AFTER.get(field, []))

    columns.append("criteria_form_complete")
    return columns


def row_from_result(result: dict, skip_empty_fields: bool = False) -> dict:
    """Converts one pipeline result file into one REDCap row.

    Args:
        result: the parsed contents of a <record_id>_<timestamp>.json file.
        skip_empty_fields: passed through to build_column_order.

    Returns:
        A dict keyed by column name. A section the pipeline left unanswered
        (None, after a failed evaluation) yields empty cells rather than a
        default, so a missing answer stays visibly missing instead of being
        silently entered as a real one.
    """

    row = {column: "" for column in build_column_order(skip_empty_fields)}
    row["record_id"] = result.get("record_id", "")
    row["criteria_form_complete"] = FORM_COMPLETE_VALUE

    for section_key, (field, kind) in SECTION_FIELDS.items():
        section = result.get(section_key.lower())
        options = _section_options(section_key)

        if section is None:

            # Checkbox columns stay empty too: writing 0 everywhere would
            # assert that every option was considered and rejected.
            continue

        value = next(iter(section.values()))

        if kind == "checkbox":

            # Verify that the pipeline's answer is a subset of the schema's options
            selected = set(value)
            unknown = selected - set(options)
            if unknown:
                raise ValueError(f"{section_key}: unknown option(s) {sorted(unknown)}")
            
            # Every box is written explicitly, including the unticked ones:
            # REDCap reads a blank checkbox cell as "leave unchanged", so an
            # answer of "none of these" has to be sent as an actual row of 0s.
            for i, option in enumerate(options, start=1):
                row[f"{field}___{i}"] = "1" if option in selected else "0"
        elif kind == "yesno":
            if value not in _YESNO_CODES:
                raise ValueError(f"{section_key}: {value!r} is not Yes or No")
            row[field] = _YESNO_CODES[value]
        else:
            row[field] = _option_code(section_key, value)

    return row


def load_latest_results(directory: Path) -> list[dict]:
    """Reads the most recent result file for each record in a directory.

    Re-running the pipeline timestamps every output rather than overwriting it,
    so a directory normally holds several runs; only the newest answers per
    record belong in an import.

    Args:
        directory: where the pipeline wrote its results.

    Returns:
        The parsed result dicts, sorted by record_id.
    """

    if not directory.is_dir():
        sys.exit(f"Not a directory: {directory}")

    newest = defaultdict(list)
    for path in sorted(directory.glob("*.json")):

        # Audit logs share the prefix but are not form outputs; files without a
        # parseable timestamp are skipped rather than guessed at.
        if path.name.endswith("_audit_log.json"):
            continue
        match = _OUTPUT_NAME_RE.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        record_id = data.get("record_id") or match.group("record_id")
        newest[record_id].append((match.group("timestamp"), data))

    return [max(entries, key=lambda t: t[0])[1] for _, entries in sorted(newest.items())]


def write_csv(results: list[dict], destination: Path, skip_empty_fields: bool = False) -> None:
    """Writes the REDCap import file.

    Args:
        results: parsed pipeline result dicts.
        destination: path of the CSV to write.
        skip_empty_fields: omit the columns the pipeline never fills.

    Raises:
        ValueError: propagated from row_from_result if a result file holds an
            answer that is not in the schema; nothing is written in that case,
            so a partially converted file cannot be imported by mistake.
    """
    
    columns = build_column_order(skip_empty_fields)
    rows = [row_from_result(result, skip_empty_fields) for result in results]

    # newline="" is required by csv on Windows, otherwise every row is followed
    # by a blank one. utf-8-sig matches the BOM REDCap writes in its own
    # exports, so the file opens correctly in Excel as well.
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Command-line entry point; see the module docstring for usage."""
    parser = argparse.ArgumentParser(
        description="Convert pipeline results into a REDCap import CSV.",
    )
    parser.add_argument(
        "input_dir", nargs="?", default="output", type=Path,
        help="directory holding the pipeline's result files (default: output)",
    )
    parser.add_argument(
        "destination", nargs="?", default="redcap_import.csv", type=Path,
        help="CSV to write (default: redcap_import.csv)",
    )
    parser.add_argument(
        "--skip-empty-fields", action="store_true",
        help="write only record_id, the criteria and the form status, omitting "
             "the free-text and date columns the pipeline does not fill",
    )
    args = parser.parse_args()

    results = load_latest_results(args.input_dir)
    if not results:
        sys.exit(f"No pipeline result files found in {args.input_dir}")

    write_csv(results, args.destination, args.skip_empty_fields)

    columns = build_column_order(args.skip_empty_fields)
    print(f"Wrote {len(results)} record(s) x {len(columns)} column(s) to {args.destination}")
    incomplete = [
        r.get("record_id", "?") for r in results
        if any(r.get(key.lower()) is None for key in SECTION_FIELDS)
    ]
    if incomplete:
        print(
            f"WARNING: {len(incomplete)} record(s) have at least one unanswered "
            f"section, left blank in the CSV: {', '.join(incomplete)}"
        )


if __name__ == "__main__":
    main()
