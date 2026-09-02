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

## Open items

- `compare_runs.py` has never been run. Every metric so far comes from a single
  run, so there is no noise floor and small differences are not interpretable.
  This blocks the interpretation of everything else.
- September 2026, once a 16 GB GPU is available: re-run the gate ablation with a
  larger model, and test whether the TRANSCRIPTION RULE in
  `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` still does anything (it governs a string the
  code discards).
- Known weak sections: A3_2 (0.533) and B2 (0.600). B2 over-selects; its two
  overlapping options are a wording problem, not only a model problem.
