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

- **`SECTION_KEYWORD_GATES["X"]` follows Table 2 of the Brighton paper**, the
  DVT row. Add a term only when that row names it; the clinicians have still to
  confirm Table 2 is the intended source.
- **Do not add schema-level "none of the above" options** for A3_2 or B1_2: the
  printed questionnaire does not have them.
- **B1.1 and B1.2 record the presumed diagnosis of a specific syndrome**, DVT of
  lower or upper limbs in Brighton Table 3. The non-specific extremity signs
  (swelling, pain, redness, warmth, absent pulses) are the other branch of that
  table and belong to B2. A syndrome ruled out by imaging leaves B1.2 empty; a
  reported diagnosis with no documented symptom still fills it.
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
python run_synthetic_records.py                   # all 30 records
python run_synthetic_records.py --only SYN_02     # a subset
python evaluate_predictions.py                    # score ./output against the corpus
python compare_runs.py                            # two runs against each other
python generate_synthetic_records.py --check      # fidelity audit, no LLM call
python export_redcap_csv.py                       # results -> REDCap import CSV
```

A record costs about 56 seconds on mari (2 x RTX 2080 Ti), so a full run is
half an hour there. The same run took 715 seconds per record on the laptop,
about six hours.

A partial run is not a run: `evaluate_predictions` keeps the newest file per
record, so scoring after `--only` mixes runs. Fine for a targeted check, not a
number to report.

## Machines

Results are not comparable across machines. The same three records, same
models, same temperature 0, gave three different sections out of thirty on
the laptop and on mari, against a generation noise floor of 1 in 300 measured
on the laptop and 0 in 30 measured on mari. Each machine is deterministic; the
two disagree with each other. Ollama 0.33.2 on the laptop against 0.32.1 on
mari is the most likely cause, GPU architecture the other candidate.

Every run's audit log records `hostname` and `ollama_version` under
`_run_config.environment`. Check them before comparing two result files.

## What the measurements say

Two evaluators on mari, 30 records each, everything else identical:

```
llama3:8b     micro 85.0%   macro kappa 0.628
qwen3.6:27b   micro 96.7%   macro kappa 0.906
```

- **The model was the binding constraint, not the prompts.** A3_2 went from
  46.7% to 83.3% and B2 from 63.3% to 96.7%, with non-overlapping confidence
  intervals. F had kappa 0.000 on every run since August, answering the majority
  class everywhere; it now answers both of its positive records correctly.
  Text-vs-number answer conflicts went from 27 to 0.
- **Sampling uncertainty binds.** With 30 records the 95% interval on a
  section's accuracy is 10 to 18 points wide. Generation noise does not:
  two identical runs on mari changed 0 sections out of 300.
- **A3_2's residual errors are all over-selection**, precision 80.8% against
  recall 100%: the answer contains the right modality plus one more.
- Read the metrics in this order: majority baseline and gain, then kappa, then
  accuracy with its interval. Accuracy alone ranked F above A3_2 under the 8B
  model, where F gained nothing over a constant answer and A3_2 gained 13
  points.

## Open items

- **The Level of Certainty is never measured.** REDCap computes it from the
  answers and it is the only number a clinician acts on, but every metric here
  is section-level. Getting it needs the REDCap Data Dictionary, or one import
  and export cycle through the project itself. It is ordinal, so weighted kappa
  rather than plain kappa.
- **Some sections rest on very few records.** F has 2 positive records out of
  30, A2 has 4, X has 2. qwen3.6:27b answers them correctly, but two cases
  cannot establish that a section works; the fix is more scenarios.
- **Questions for the clinicians.** Whether Table 2 of the Brighton paper is the
  intended source for X's alternative diagnoses; which procedures count as A2's
  "other procedure done that confirmed presence of DVT", given that Brighton
  asks for a procedure that confirms a thrombus; whether B2 option 4 applies
  when only calf pain is documented.
- **The gates lose their purpose under qwen3.6:27b.** F's details gate fired on
  23 of 30 records with the 8B model and on none with the 27B: the model's
  answer already matches its own DETAILS_PRESENT line. The keyword gate fired
  6 times, correcting 4 and breaking 2. A run with
  `SECTION_GATES_ENABLED` all False would measure what they are still worth.
- Test whether the TRANSCRIPTION RULE in `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` does
  anything: it governs a string the code discards.
- Weakest section left: A3_2 at 83.3%, whose five errors are all one modality
  too many. Its two ultrasound options are a wording problem inherited from the
  Brighton table, which bundles them as one modality.
