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
the message. The body states what changed, not why.

**Report measurements, not impressions.** Claims about the pipeline's behaviour
should be checked against the audit logs or the evaluation before being stated.
When something is unverified, say so.

## Architecture

- `pipeline.py` orchestrates one record; `agents.py` holds both agents.
- Agent 1 (extractor) copies evidence verbatim. Agent 2 (evaluator) answers with
  `FINAL_OPTION` / `FINAL_ANSWER` lines.
- `config.EXTRACTOR_MODE` selects `full_text`, `rag` (baselines),
  `agentic_graph` (reference mode, a LangGraph state machine) or `raw_record`
  (no Agent 1 at all).
- `models.py` holds the Pydantic schema and is the single source of truth for
  section options and their order. Other modules introspect it rather than
  repeating the options.
- Deterministic post-processing lives in `criteria_rules.py`.
- `docs/DOCUMENTAZIONE_CODICE.md` describes the code module by module,
  `docs/RISULTATI_SPERIMENTALI.md` holds every measurement with the
  configuration that produced it, and `docs/SCALETTA_TESI.md` the thesis
  outline. All three are in Italian and are the source the thesis is written
  from; keep them in step with this file.

In `agentic_graph` mode the extractor returns the raw retriever chunks
(`intermediate_steps`), not its own final answer, so it chooses search queries
but performs no evidence selection.

## Domain constraints

- **Criterion X asks whether any alternative diagnosis explains the acute
  illness**, not whether one of Table 2's conditions does. Section 5.2.7 states
  the criterion as no alternate etiology that could explain the clinical illness
  and calls Table 2 a list of possible etiologies, so a condition outside that
  list still counts. Table 2 reaches the evaluator through the context retrieved
  for the section.
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
python run_synthetic_records.py                   # every record in the corpus
python run_synthetic_records.py --only SYN_02     # a subset
python run_synthetic_records.py --output-dir ./output_full_text   # its own arm
python evaluate_predictions.py                    # score ./output against the corpus
python compare_runs.py                            # two runs against each other
python generate_synthetic_records.py --check      # fidelity audit, no LLM call
python export_redcap_csv.py                       # results -> REDCap import CSV
```

A record costs about 265 seconds on mari (2 x RTX 2080 Ti) under the 27B
evaluator, so a full run over the 40 records is close to three hours. The 8B
evaluator took 56 seconds a record there, and 715 on the laptop.

A partial run is not a run: `evaluate_predictions` keeps the newest file per
record, so scoring after `--only` mixes runs. Fine for a targeted check, not a
number to report. `export_redcap_csv` selects the same way, so an experimental
run left in `output/` becomes the CSV that goes to REDCap. Check
`_run_config.models.evaluator` on the newest file before either command.

Give every experimental arm its own directory with `--output-dir`, and pass that
directory to `evaluate_predictions`. `compare_runs ./output ./output_other`
reports both arms' accuracy side by side and lists the sections where they
differ, which is the comparison worth reading when only one component changed.

## The reference configuration

Everything the numbers under What the measurements say were produced with. An
experimental arm changes one line of it and puts the run in its own directory;
this is what `config.py` goes back to afterwards.

```python
LLM_MODEL_NAME           = "llama3:8b-instruct-q4_0"
EVALUATOR_LLM_MODEL_NAME = "qwen3.6:27b"
AGENTIC_LLM_MODEL_NAME   = "llama3.1:8b-instruct-q4_0"
LLM_REASONING            = False
EXTRACTOR_MODE           = "agentic_graph"
BRIGHTON_CONTEXT_ENABLED = True
SECTION_DESCRIPTIONS_ENABLED = False
SECTION_HINTS_ENABLED    = True
SECTION_HINTS_DISABLED   = {"B2"}
SECTION_GATES_ENABLED    = {"keyword": False, "details": False, "absent_pulses": True}
```

All three per-section gates are off. Only the cross-section rules remain, and
they encode the form's structure rather than a model weakness.

Check it before every launch, since an arm left in place is how a run gets
attributed to the wrong configuration:

```bash
grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|SECTION_HINTS_DISABLED)' config.py
```

`_run_config` records all of it in every audit log, so a result file can always
be traced back even when the working copy has moved on.

The ground-truth files hold the same section keys as the pipeline's output, so
`export_redcap_csv` converts them once they carry a name it parses. That is what
makes the Level of Certainty measurable:

```bash
mkdir /tmp/gt
for f in data/synthetic_records/*_ground_truth.json; do
  cp "$f" "/tmp/gt/$(basename "${f%_ground_truth.json}")_20260101_000000.json"
done
python export_redcap_csv.py /tmp/gt gt_import.csv --skip-empty-fields
python export_redcap_csv.py output pred_import.csv --skip-empty-fields
```

`load_latest_results` reads the timestamp from the filename, hence the rename;
any value works, since there is one ground-truth file per record. Two imports
and two exports through REDCap give 40 pairs of LOC, which is ordinal, so
quadratic weighted kappa and a 5 x 5 confusion matrix rather than plain kappa.

## Machines

Results are not comparable across machines. The same three records, same
models, same temperature 0, gave three different sections out of thirty on
the laptop and on mari, against a generation noise floor of 1 in 300 measured
on the laptop and 0 in 30 measured on mari. Each machine is deterministic; the
two disagree with each other. Ollama 0.33.2 on the laptop against 0.32.1 on
mari is the most likely cause, GPU architecture the other candidate.

Every run's audit log records `hostname` and `ollama_version` under
`_run_config.environment`. Check them before comparing two result files.

`_run_config.section_hints_fingerprint` records the length and a sha256 prefix
of the hint each section actually received, plus an `all` digest of the whole
set, so two runs whose hints were rewritten between them no longer carry the
same signature. Runs before 2026-09-08 lack it and are told apart by their date.

## What the measurements say

Two evaluators on mari, 30 records each, everything else identical:

```
llama3:8b     micro 85.0%   macro kappa 0.628
qwen3.6:27b   micro 96.7%   macro kappa 0.906
```

On the extended 40-record corpus, qwen3.6:27b scores micro 99.5%, macro kappa
0.982, two wrong sections out of 400. Eight sections of ten are at 100%; A2 and
B1.1 miss one record each. The ten scenarios added in September score at the
same rate as the original thirty, so the sections that had been unmeasurable
hold up.

**Neither residual error is attributable to the model.** Both are readings of
the questionnaire that differ from the ground truth, argued from the guideline
text the model was given. SYN_03 calls a CT venography A2's "other procedure";
SYN_10 counts an uncharacterised leg discomfort as a B1.1 symptom, citing the
guideline's own list of non-specific signs. Both are open questions for the
clinicians, so the pipeline has stopped making mistakes and started disagreeing.
Only an outside authority settles the difference.

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
- **An ablation is only valid for the configuration it was run in.** The details
  gate was measured inert under qwen3.6:27b, zero overrides on 40 records. After
  the F hint gained the precondition that the criterion needs a reported
  diagnosis, the same gate fired once and reverted the one answer the hint had
  fixed: the model answered No on SYN_10 and the gate forced Yes. Its mapping,
  DETAILS_PRESENT=no implies Yes, assumes a diagnosis exists. `details` is now
  False and F is 40 of 40. Second independent instance of the same lesson: the
  first was A2, whose hint the full ablation had classified neutral because the
  keyword gate covered it, and which turned out to be worth 7.5 points once that
  gate was scoped to the thrombectomy option.
- **The hint and gate configuration is tuned to one model.** Swapping the
  evaluator for `medgemma:27b`, everything else identical and the hint
  fingerprint proving it, takes micro accuracy from 99.5% to 93.2% and macro
  kappa from 0.982 to 0.837. The errors are systematically over-selection, 31
  false positives against 8 false negatives, and they land where the
  configuration was pruned against qwen: A3.2 drops to 67.5%, nine of its
  thirteen errors being the ultrasound pairing the removed contrastive clause
  used to prevent; B2, whose hint is disabled, gains three false positives; F
  falls to 87.5% and errs in both directions despite its 1216-character hint.
  MedGemma answers both of qwen's residual errors correctly, so it is not worse
  everywhere, it is differently wrong. The prompts are compensations for one
  model's failure modes, not transferable domain knowledge.
- **A schema faithful to the paper form can cost sections under another model.**
  B1.2 has no "none of the above" option because the printed questionnaire has
  none. Under qwen that is free, B1.2 scores 100%. Under medgemma it produces
  six errors: four records answered `Lower extremity DVT` where the truth is
  empty, and two sections that failed outright with `FINAL_ANSWER: 5` for a
  two-option field, the model having reasoned correctly that DVT was ruled out
  and having nowhere to put it.
- **Seven configurations measured on the same base**, each varying one
  component. `docs/RISULTATI_SPERIMENTALI.md` section 8 has the full per-section
  table.

  ```
                                    exact     micro    macro kappa
  agentic_graph (reference)        398/400    99.5%      0.982
  agentic without the context      395/400    98.75%     0.970
  raw_record                       392/400    98.0%      0.966
  full_text + 27B extractor        380/400    95.2%      0.824
  medgemma as evaluator            371/398    93.2%      0.837
  rag                              341/400    85.3%      0.607
  full_text                        341/400    85.5%      0.586
  ```

  Not letting the 8B rewrite the evidence is worth +51, a 27B extractor instead
  of an 8B is worth +39 and still lands 12 below not extracting at all, and the
  guideline context is worth +3.
- **A weak first stage decides the score.** The four extraction modes are the
  four combinations of two choices, whether the 8B **rewrites** the evidence and
  whether Agent 2 gets the whole record or the retrieved chunks:

  ```
                          whole record        retrieved chunks
     8B rewrites          full_text  341      rag          341
     verbatim evidence    raw_record 392      agentic      398
  ```

  Removing the rewriting is worth +51 and +57; adding retrieval is worth 0 with
  it and +6 without, so the two factors interact. Letting the 8B write the
  evidence returns `NO RELEVANT EVIDENCE FOUND.` on 143 sections out of 400 in
  `full_text` and 142 in `rag`. Agent 2 keeps reasoning correctly on what it
  receives, so reading only the final answer hides the cause. The chunks column
  mixes fixed with model-chosen queries, so the +6 is not attributable to
  retrieval rather than agency. An extractor exists in every mode but
  `raw_record`; what `agentic_graph` discards is the agent's final turn, not the
  model.
- **The guideline context is worth +3, and reading only the errors predicted the
  opposite.** Turning `BRIGHTON_CONTEXT_ENABLED` off takes the reference run
  from 398 to 395, losing A3.2 (40 to 38, 2 false positives and 0 false
  negatives) and B2 (40 to 39). Without the guideline the model over-selects
  imaging modalities; with it, it does not — the opposite of medgemma, which
  cited the same guideline to add options. The prediction that the context was
  harmful came from audit logs of failures only, where it appears quoted in
  wrong answers; the sections it silently got right are invisible to that
  reading and are the majority. With the context on, A3.2 and B2 are at 40/40,
  so a reranker or a larger embedding model has no headroom to recover.
- **Choosing is not writing.** `agentic_graph` belongs in the verbatim row even
  though it has a model in the loop: `extract_evidence_agentic` returns the raw
  tool observations, not the agent's final turn, so the 8B picks queries but the
  evidence never passes through its tokens. What it still controls is coverage,
  and that is inert on this corpus: 0 sections out of 400 without evidence,
  `agent1_seconds` between 6.5 and 7.2 on every section (one tool call), and
  retrieval returning the whole record each time. Chunk-level selection is
  possible by design and has no measurable effect here, which is a property of
  records that fit in one chunk rather than of the architecture.
- **Two inputs that carry the same information are not the same input.**
  `full_text` and `rag` both score 341/400 and differ on **35 sections**:
  retrieval returns the whole record either way, but joining the chunks with a
  separator and duplicating the overlap changes what the 8B extractor writes.
  The rewriting noise is larger than the difference between the two modes.
- **A correct safety net can amplify an error.** On SYN_28 the model gets A3.1
  wrong and the cross-section rule, behaving as designed, clears A3.2: one model
  error, two wrong sections. The same rules are worth 8 sections on this corpus,
  so they stay clearly positive, but their effect is not monotone.
- **The score is not the instrument any more; the audit log is.** The run that
  added that F precondition scored 99.2% with zero sections changed out of 400,
  which reads as a change that did nothing. The log showed the model had in fact
  answered correctly and the gate had reverted it. Reconstructing a gate's
  effect from the log is exact, since gates are pure post-processing and every
  override records the pre-gate answer: it predicted F 40 of 40 and micro 99.5%,
  and the confirming run returned both, changing exactly the one section.
- Read the metrics in this order: majority baseline and gain, then kappa, then
  accuracy with its interval. Accuracy alone ranked F above A3_2 under the 8B
  model, where F gained nothing over a constant answer and A3_2 gained 13
  points.

## Open items

- **The Level of Certainty is never measured.** REDCap computes it from the
  answers and it is the only number a clinician acts on, but every metric here
  is section-level, and the LOC is not a linear function of the sections: one
  wrong answer can leave it unchanged or move it a whole level. The Commands
  section above gives the two CSVs to compare. Only two of the 40 rows differ
  from the ground truth, `criteria_a2` on SYN_03 and `criteria_b1_1` on SYN_10,
  so the whole comparison turns on those.
- **Agreement with an independent reviewer is not measurable yet.** Every number
  here is scored against a ground truth written alongside the pipeline, so the
  two are not independent and their agreement is not evidence. A validation study
  reports the concordance between two blind reviewers instead, neither of them
  treated as the truth; the SeValid myocarditis experiment reports 28 of 38, 74%.
  Measuring it here takes a clinician filling the questionnaire on a subset of
  the corpus without seeing the output or the ground truth, then agreement at
  three levels, per section, per case decision and per LOC, the last one ordinal
  and so quadratic weighted. Discordant records go to a second clinician, and
  that adjudicated standard is what gives each reviewer a sensitivity. The two
  sections left, A2 on SYN_03 and B1.1 on SYN_10, are disagreements argued from
  the guideline, and accuracy scores them as errors.
- **A2 and X still rest on few records**, 4 and 6 of 40. F went from 2 positives
  to 7 with the September expansion and now scores kappa 0.918.
- **Questions for the clinicians.** The first two decide the only two wrong
  sections left.
  - Which procedures count as A2's "other procedure done that confirmed
    presence of DVT", given that Brighton asks for a procedure that confirms a
    thrombus? SYN_03 turns on this: the model calls a CT venography one, the
    ground truth puts imaging in A3.
  - Does a vague, uncharacterised symptom, "a generic discomfort in the leg"
    with no site or intensity, count as B1.1's "at least one symptom or sign
    was reported", or does the section stay unknown? SYN_10 turns on this: the
    ground truth says unknown, the model reads the discomfort as a symptom.
  - Does X mean an alternative diagnosis for the acute illness in general, or
    one of the Table 2 conditions that mimic a DVT? Acted on already, from
    section 5.2.7 and the three records above, but worth confirming. Ask it
    concretely: a patient dies of an acute myocardial infarction and the autopsy
    excludes DVT — is X answered `An alternative diagnosis was found that
    explained the acute illness`?
  - Does B2 option 4 apply when only calf pain is documented?
  - Is Table 2 of the Brighton paper the intended source for X's list?
- **All three per-section gates are now off.** The details gate reverted a
  correct answer once the F hint gained its precondition. The absent-pulses gate
  never fires. The keyword gate was removed on 2026-09-15 together with a
  ground-truth correction: it fired 3 times in 400, always on X, on SYN_11
  (death from acute myocardial infarction, autopsy excluding DVT), SYN_16 (chest
  pain attributed to a musculoskeletal cause) and SYN_20 (death from traumatic
  haemorrhage, stated as unrelated to VTE). In all three the model answered that
  an alternative diagnosis was found and was right: section 5.2.7 of the paper
  states the criterion as *no alternate etiology that could explain the clinical
  illness* and calls Table 2 *a list of possible* etiologies. The gate and the
  ground truth agreed because both were written from the same narrow reading of
  Table 2, so their agreement was circular. Those three scenarios now carry
  `An alternative diagnosis was found`, which keeps the reference run at 398.
  Two further defects made the gate indefensible: Table 2's DVT row lists
  *Physical trauma* as a general category and the keyword list encoded only the
  specific diagnoses under it, so SYN_20 failed even on the narrow reading; and
  in `rag` the gate overrode SYN_36 and SYN_39, whose keywords ARE in the list,
  because the lossy extractor had already dropped the text containing them —
  **the gate's reliability depends on the extractor's output**.
- **Agent 2 never sees the section heading.** `_build_reasoning_prompt` sends the
  evidence, the guideline context, the hint and the numbered options, so the only
  statement of what a section covers is the wording of the options. A2 is where
  that shows: the word surgical appears only inside option 3, and on SYN_03 the
  model reads "other procedure done that confirmed presence of DVT" as covering a
  CT venography. The schema settles it, since option 3 reads "no surgical
  procedure done", so the section is surgical and imaging belongs to A3.
  `config.SECTION_DESCRIPTIONS_ENABLED` now sends each field's description from
  `models.py` above its options; it is False, the value every recorded run was
  produced with, and the arm has not been run.
- Test whether the TRANSCRIPTION RULE in `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` does
  anything: it governs a string the code discards.
- **The extractor prompt never says a negation is evidence**, while the
  evaluator prompt does. On SYN_32 the same extractor hands B1.1 the sentence
  stating the record contains no symptoms and hands F the bare diagnosis,
  because the two section queries aim at different halves. F needs to know
  whether details accompany the diagnosis and its query asks for the diagnosis.
  This penalises the baselines, not the reference mode, where F is 40 of 40.
- **Re-run the reference configuration** with `keyword: False` and the corrected
  ground truth. The reconstruction predicts 398/400 unchanged, since the gate and
  the three corrected records cancel out, but it has not been confirmed by a run.
  `generate_synthetic_records.py` rewrites the ground-truth JSONs without calling
  any model, so the corrected scenarios need that before the next evaluation.
- **Two wrong sections in 400 is past what 40 records can resolve.** Each record
  is worth 0.25 points and the 95% interval on the total is [98, 100], so a
  one-section change is not a measurable difference. What is still worth reading
  is which category an error falls into, not the total. Further tuning on this
  corpus is fitting 40 records written by their own evaluator.

## Settled by the clinicians

Received 2026-09-07, and no longer listed among the questions above.

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
