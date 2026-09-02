# DVTProject

A Python pipeline that
fills a diagnostic questionnaire for deep vein thrombosis
from free-text clinical records, using two local LLM agents through Ollama.

Ten closed-answer sections (A1, A2, A3_1, A3_2, B1_1, B1_2, B2, C, F, X). The
output is not a diagnosis: REDCap computes the Level of Certainty from the
answers.

## Working rules

**Ask before changing anything.** Propose the edit, wait for approval, then
apply it. This holds for code, comments, docstrings and configuration alike.

**Commit messages**: English, short subject line, no double quotes anywhere in
the message. Body explains why, not just what.

**Report measurements, not impressions.** Claims about the pipeline's behaviour
should be checked against the audit logs or the evaluation before being stated.
When something is unverified, say so.

## Architecture

- `pipeline.py` orchestrates one record; `agents.py` holds both agents.
- Agent 1 (extractor) copies evidence verbatim. Agent 2 (evaluator) answers with
  `FINAL_OPTION` / `FINAL_ANSWER` lines.
- `config.EXTRACTOR_MODE` selects `full_text`, `rag` (baselines) or
  `agentic_graph` (reference mode, a LangGraph state machine).
- `models.py` holds the Pydantic schema and is the single source of truth for
  section options and their order. Other modules introspect it rather than
  repeating the options.
- Deterministic post-processing lives in `criteria_rules.py`.

In `agentic_graph` mode the extractor returns the raw retriever chunks
(`intermediate_steps`), not its own final answer, so it chooses search queries
but performs no evidence selection.

## Domain constraints

- **Do not touch `SECTION_KEYWORD_GATES["X"]`** until the clinicians supply the
  complete list of alternative diagnoses.
- **Do not add schema-level "none of the above" options** for A3_2 or B1_2: the
  printed questionnaire does not have them.
- **Cross-section rules are always on**, deliberately outside the
  `config.SECTION_GATES_ENABLED` ablation switches: they encode the form's
  structure, not a workaround for a model weakness.
- **Keep the pipeline language-agnostic.** Do not hardcode Italian-only queries
  or logic; rely on the multilingual embedding model. The bilingual stems in
  `SECTION_KEYWORD_GATES` and the Italian synthetic corpus are accepted
  exceptions.

## Documentation standards

Docstrings and comments describe **what objectively exists now**. No hypotheticals
("if you later switch to PDF..."), no roads not taken, no references to previous
behaviour or to project history. Numbers quoted in prose must match the code.

## Commands

```bash
python main.py                                    # one record, paths edited by hand
python run_synthetic_records.py                   # all 30 records, about 6 hours
python run_synthetic_records.py --only SYN_02     # a subset, about 12 min each
python evaluate_predictions.py                    # score ./output against the corpus
python generate_synthetic_records.py --check      # fidelity audit, no LLM call
python export_redcap_csv.py                       # results -> REDCap import CSV
```

A partial run is not a run: `evaluate_predictions` keeps the newest file per
record, so scoring after `--only` mixes runs. Fine for a targeted check, not a
number to report.

## What the measurements say

Run of 2026-08-22, 30 records: micro accuracy 85.3%, macro kappa 0.597.

- **Generation noise is settled.** `compare_runs.py` on 2026-08-22 found 1
  section changed out of 300 between two identical runs (stability 99.7%), so a
  difference between runs is attributable to whatever was edited between them.
- **Sampling uncertainty is not.** With 30 records the 95% interval on a
  section's accuracy is 15 to 18 points wide. That, not generation noise, is
  what limits how small a difference can be read.
- **F and A2 have Cohen kappa of exactly 0.000** at 93.3% and 86.7% accuracy:
  they answer the majority class everywhere and miss every minority case. Their
  accuracy equals their majority baseline. F's two positive records are
  SYN_17 and SYN_29, the only two scenarios written to test it, and both fail.
- Read the metrics in this order: majority baseline and gain, then kappa, then
  accuracy with its interval. Accuracy alone ranked F above A3_2, which has
  half the accuracy and does far more work (gain +20.0 against +0.0).

## Open items

- **The Level of Certainty is never measured.** REDCap computes it from the
  answers and it is the only number a clinician acts on, but every metric here
  is section-level. Getting it needs the REDCap Data Dictionary, or one import
  and export cycle through the project itself. It is ordinal, so weighted kappa
  rather than plain kappa.
- **The corpus cannot measure some sections.** F has 2 positive records out of
  30, A2 has 4, X has 2. No model can be evaluated on those counts; the fix is
  more scenarios, not a better prompt.
- September 2026, once a 16 GB GPU is available: re-run the gate ablation with a
  larger model, starting with F's details gate, which fires on 23 of 30 records
  and may be doing nothing but reproducing the majority answer. Also test
  whether the TRANSCRIPTION RULE in `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` still does
  anything (it governs a string the code discards).
- Known weak sections: A3_2 (0.533) and B2 (0.600). B2 over-selects, with
  precision 78.7% against recall 96.0%; its two overlapping options are a
  wording problem, not only a model problem.
