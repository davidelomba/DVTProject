"""
Writes the myocarditis prompts, queries and hints with export_prompts.py.

The texts are the ones a myocarditis run sends: myo_schema installs the schema
first and run_myo.apply_domain replaces the prompts, the queries, the hints and
the search tool's description before export_prompts reads them. Same files and
same manifest as the DVT export; answer_single.txt is absent, since every
section of this questionnaire is multiple-choice, and queries.tex is a
longtable with its caption and label tab:myo_queries.

Usage:
    python myo/export_prompts_myo.py
    python myo/export_prompts_myo.py --output-dir ../Relazione/TESILATEX/prompts_myo
"""

from pathlib import Path

import myo_schema  # noqa: F401  # isort: skip -- must precede the project imports

import export_prompts
import run_myo

export_prompts.DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "prompts_export"

# E's query is longer than a page at a 9cm column, so the table breaks across
# pages and carries its own caption.
export_prompts.QUERY_TABLE = "longtable"
export_prompts.QUERY_WIDTH = "0.82\\linewidth"
export_prompts.QUERY_CAPTION = "Le cinque query di sezione della miocardite."
export_prompts.QUERY_LABEL = "tab:myo_queries"


if __name__ == "__main__":
    run_myo.apply_domain()
    export_prompts.main()
