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

On the extended 40-record corpus, qwen3.6:27b scores micro 99.2%, macro kappa
0.974, three wrong sections out of 400. Seven sections of ten are at 100%; A2,
B1.1 and F each miss one record. The ten scenarios added in September score at
the same rate as the original thirty, so the sections that had been unmeasurable
hold up.

Two of the three residual errors wait on the clinicians rather than on the
pipeline. SYN_03 reads a CT venography as A2's "other procedure", which is the
open A2 question; SYN_10 reads an uncharacterised leg discomfort as a B1.1
symptom, citing the guideline's own list of non-specific signs, which is the
open B1.1 question. The third, SYN_10's F, is a model error: the record reports
no diagnosis at all and the model answered the counterfactual, reasoning that a
diagnosis based on this text would be a bare conclusion. The F hint now states
the precondition that the criterion needs a reported diagnosis.

- **The model was the binding constraint, not the prompts.** A3_2 went from
  46.7% to 83.3% and B2 from 63.3% to 96.7%, with non-overlapping confidence
  intervals. F had kappa 0.000 on every run since August, answering the majority
  class everywhere; it now answers both of its positive records correctly.
  Text-vs-number answer conflicts went from 27 to 0.
- **Sampling uncertainty binds.** With 30 records the 95% interval on a
  section's accuracy is 10 to 18 points wide. Generation noise does not:
  two identical runs on mari changed 0 sections out of 300.
- **A hint can carry a criterion rather than a rule.** Rewriting B1.1's hint
  around the clinicians' epistemic distinction changed 2 sections out of 400
  between two otherwise identical runs, both the intended ones. The model then
  reconstructed the distinction in its own words on both, writing that the
  document notes the absence of reported symptoms rather than documenting their
  clinical absence in the patient.
- **The hints help and hurt, section by section.** Dropping all of them takes
  micro accuracy from 97.2% to 91.5%, but the total hides opposite effects: F
  loses 53.9 points and inverts, kappa -0.324, since without the instruction the
  model reads "reported without details" the intuitive way; B1.1 loses 12.5;
  A3.2 gains 5.3 and B2 gains 2.5, both hints having been written against
  llama3:8b failures; A1, A2, A3.1, C and X do not move, so about 3300 of the
  7182 injected characters do nothing. Audit them one at a time, not as a block.
- **The details gate contributes nothing to F.** With hints on it produces zero
  overrides on 40 records, so F's 97.5% is the hint alone.
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
- **A2 and X still rest on few records**, 4 and 6 of 40. F went from 2 positives
  to 7 with the September expansion and now scores kappa 0.918.
- **Questions for the clinicians.**
  - Does X mean an alternative diagnosis for the acute illness in general, or
    one of the Table 2 conditions that mimic a DVT? The model reads it broadly
    and the keyword gate overrides it on three records; the answer decides
    whether that gate saves three answers or destroys three.
  - Which procedures count as A2's "other procedure done that confirmed
    presence of DVT", given that Brighton asks for a procedure that confirms a
    thrombus?
  - Does B2 option 4 apply when only calf pain is documented?
  - Is Table 2 of the Brighton paper the intended source for X's list?
  - Does a vague, uncharacterised symptom, "a generic discomfort in the leg"
    with no site or intensity, count as B1.1's "at least one symptom or sign
    was reported", or does the section stay unknown? SYN_10 turns on this: the
    ground truth says unknown, the model reads the discomfort as a symptom.

## Settled by the clinicians

Answers received 2026-09-07. They close three of the questions above.

- **B1.1, second option against third.** The distinction is epistemic and is
  about the patient, not the document. The second option means the record tells
  you the patient did not have signs or symptoms; the third means the record
  leaves you not knowing. A note stating that it contains no symptom section is
  the third: the absence of a report is not a report of absence. The ground
  truth was right on SYN_32 and SYN_35 and the model wrong, and the B1.1 hint
  now carries the reason rather than the rule alone.
- **B2 implies B1.1.** Selecting any of B2's first four options implies B1.1's
  first option. This is the cross-section rule already in
  `config.CROSS_SECTION_RULES`, now confirmed from outside the project.
- **C's normal range.** Use the laboratory's own reference range when the record
  gives one, otherwise 500 ng/mL. This is what the C hint already says, and C
  has scored 100% on every run under qwen3.6:27b.
- **The gates lose their purpose under qwen3.6:27b.** Reconstructed from the
  audit logs: all gates on 290/300, no gates at all 288/300. The details gate
  fired on 23 of 30 records with the 8B and on none with the 27B. All the
  remaining value sits in the X keyword gate, worth 3 sections, and those are
  the three records the clinicians' answer above decides.
- Test whether the TRANSCRIPTION RULE in `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` does
  anything: it governs a string the code discards.
- **Three wrong sections in 400 is past what 40 records can resolve.** Each
  record is worth 0.25 points and the 95% interval on the total is [98, 100],
  so a one-section change is not a measurable difference. What is still worth
  reading is which category an error falls into, not the total.
