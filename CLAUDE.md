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
- Agent 3 (`confidence.py`) scores each final answer after the cross-section
  rules, when `config.CONFIDENCE_ENABLED` is set. One short request per section
  to the evaluator model, without reasoning and without Agent 2's answer, sent
  straight to Ollama's local `/api/chat` for the logprobs; the confidence is
  the probability the model puts on the form's answer. It never changes an
  answer. The value goes to the audit log and to the result file under
  `_confidence`. F is in `CONFIDENCE_SKIP`.
- `docs/DOCUMENTAZIONE_CODICE.md` describes the code module by module,
  `docs/RISULTATI_SPERIMENTALI.md` holds every measurement with the
  configuration that produced it, and `docs/SCALETTA_TESI.md` the thesis
  outline. All three are in Italian and are the source the thesis is written
  from; keep them in step with this file.

In `agentic_graph` mode the extractor returns the raw retriever chunks
(`intermediate_steps`), not its own final answer, so it chooses search queries
but performs no evidence selection.

## Domain constraints

- **Criterion X asks whether an alternative diagnosis explains the acute
  illness.** Section 5.2.7 states the criterion as no alternate etiology that
  could explain the clinical illness, names Table 2 as a list of possible
  alternate etiologies for each syndrome and its associated symptoms or signs,
  and excludes the criterion from Level 1, which rests on proven thrombus
  instead. A condition outside the table still counts, since the paper calls the
  list possible rather than complete. What the paper leaves unsettled is whether
  the alternative has to explain the symptoms that suggested thrombosis or only
  the acute illness; Table 2 is built the first way, one row per syndrome
  indexed by that syndrome's non-specific symptoms. Table 2 reaches the
  evaluator through the context retrieved for the section.
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

`evaluate_predictions` and `compare_runs` save their JSON reports in `reports/`.

Measured from the audit-log timestamps on mari (2 x RTX 2080 Ti), median per
record: the reference costs 174 seconds, so a full run over the 40 records is
1h56. What sets the cost is how many model tags a run loads. `raw_record` loads
one and costs 105, 1h10 for the corpus; the arms whose Agent 1 and Agent 2 share
a tag, the reference among them, cost 165 to 175 and run in 1h50 to 1h56; every
arm running two different tags costs 250 to 310, close to three hours for the
corpus. Under the 8B evaluator a record costs 51 seconds, 0h34 for the corpus,
and took 715 on the laptop.

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
AGENTIC_LLM_MODEL_NAME   = "qwen3.6:27b"
LLM_REASONING            = False
LLM_NUM_CTX              = 4096
EXTRACTOR_MODE           = "agentic_graph"
BRIGHTON_CONTEXT_ENABLED = True
SECTION_DESCRIPTIONS_ENABLED = False
SECTION_HINTS_ENABLED    = True
SECTION_HINTS_DISABLED   = {"B2"}
SECTION_GATES_ENABLED    = {"keyword": False, "details": False, "absent_pulses": True}
CONFIDENCE_ENABLED       = False
```

The keyword and details gates are off. `absent_pulses` is on and has never fired
on this corpus. The cross-section rules are outside the switch and encode the
form's structure rather than a model weakness.

Agent 1 and Agent 2 run the same model tag, which is why the reference costs 174
seconds a record against the 250 to 310 an arm with two tags pays. They stay two
separate settings: change `AGENTIC_LLM_MODEL_NAME` alone to vary the agent and
`EVALUATOR_LLM_MODEL_NAME` alone to vary the evaluator. `LLM_MODEL_NAME` is the
extractor of the `rag` and `full_text` baselines and the reference mode does not
load it.

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

`_run_config.section_queries_fingerprint` does the same for `SECTION_QUERIES`,
which is both the brief Agent 1 works from and the key that retrieves the
guideline context. The current set digests to `bfd9536a31fe`; the set every run
before 2026-09-20 used digests to `39a5a3504655`, and runs before 2026-09-19
lack the field.

`_run_config.guideline_anchors_fingerprint` digests the guideline passage each
section resolved to, and `guideline_anchors_enabled` says whether Agent 2 read
it. The labels are digested through their resolved text, since what a heading
resolves to depends on how the PDF extracted. On mari the set digests to
`4bc8810e170a`. Runs before 2026-09-22 lack both fields.

**The anchors resolve to different text on different machines**, which is what
the fingerprint was added for. Nine of the ten are identical between mari and a
Windows working copy; X is 2770 characters on one and 2769 on the other, a
single character in section 5.2.7 plus Table 2. The likely cause is a different
`pypdf` extracting one character differently, so the digest of an anchored run
belongs to the machine that produced it.

## What the measurements say

On the 40-record corpus, qwen3.6:27b scores micro 99.5%, macro kappa
0.982, two wrong sections out of 400. Eight sections of ten are at 100%; A2 and
B1.1 miss one record each. The ten scenarios added in September score at the
same rate as the original thirty, so the sections that had been unmeasurable
hold up.

**One residual error is a genuine mistake, the other is a reading the ground
truth disagrees with.** SYN_03
calls a CT venography A2's "other procedure"; SYN_10 counts an uncharacterised
leg discomfort as a B1.1 symptom, citing the guideline's own list of
non-specific signs. SYN_10 is not settled inside the project. **SYN_03 is:** a
CT venography is an imaging study, not a procedure recovering a thrombus, and
the ground truth is right.

**What section 4.1 settles about A2 is narrower than it looked.** It names a
*surgical or a catheterization procedure* together, so the method being
endovascular is no reason to exclude it. It does not settle SYN_12 and SYN_23,
because it defines pathologic diagnosis by the **recovery** of a thrombus, and
a catheter thrombolysis dissolves one. Handed 4.1 as its context, the model
reads it that way and answers that no such procedure was done. The ground
truth's coding of those two records rests instead on the option's own words,
*other procedure done that confirmed presence of DVT*, which the intraprocedural
venography satisfies. The two are different criteria and the questionnaire does
not say which governs, which is what the clinician question below is asking.

- **A hint's contribution changes sign with the model.** Evaluator and hints
  crossed on the 40-record corpus, four runs, same extractor, same gates, same
  machine, the two hintless arms carrying the same fingerprint `dbe3b41b6a5c`:

  ```
                       with hints        without hints
  qwen3.6:27b          398/400  99.5%    365/399  91.5%
  llama3:8b            240/400  60.0%    283/400  70.75%
  ```

  The hints are worth **+33 sections to the 27B and -43 to the 8B**. F carries
  it on one section: the same hint is worth +23 to qwen, which drops to 17 of 39
  without it, and -9 to the 8B, which rises from 7 to 16. X moves the 8B from 6
  to 31 and leaves qwen at 40 either way; A1 moves it from 14 to 29; C is the
  one hint the 8B uses well, 34 falling to 29. So the model gap is not a
  constant either: **158 sections measured with the 27B's own prompts, 82
  without any**. Close to half of what looks like model capability is prompt
  tuning.
- **The gates are worth 76 sections to the 8B and nothing to the 27B.** Turning
  `keyword` and `details` back on under the 8B, everything else at the reference,
  takes it from 240 to 316 of 400, micro 60.0% to 79.0%, macro kappa 0.373 to
  0.497. The recovery is exactly and only where the gates reach: A1 goes from 14
  to 39, X from 6 to 32, F from 7 to 32, and **the other seven sections are
  identical figure for figure**. A2 does not move either, since the keyword gate
  is scoped there to the thrombectomy option alone. The same two gates were
  measured inert under qwen, which scores 398 with them on and 398 with them off.
  They were scaffolding for one model, and removing them cost nothing only
  because the model had changed.
- **A hint and its gate were designed as a pair, and the hint alone is worse than
  no hint.** Under the 8B, on each of the three sections the gates cover, the
  hint on its own scores below the hintless run: A1 14 against 29, X 6 against
  31, F 7 against 16. Add the gate and they reach 39, 32 and 32. F is coupled by
  construction, since its hint is what asks for the DETAILS_PRESENT line
  `apply_details_gate` reads, and `config.SECTION_HINTS_ENABLED` False therefore
  disables that gate too. A1 and X are not: there the keyword gate reads the
  evidence text and knows nothing of the hint, and simply reverts what the misled
  model answered. Either way the component that carries the section is the gate,
  and the hint measured beside it looks useful only because the gate is hiding
  its cost.
- **Sampling uncertainty binds.** With 30 records the 95% interval on a
  section's accuracy is 10 to 18 points wide. Generation noise does not:
  two identical runs on mari changed 0 sections out of 300.
- **A hint can carry a criterion rather than a rule.** Rewriting B1.1's hint
  around the clinicians' epistemic distinction changed 2 sections out of 400
  between two otherwise identical runs, both the intended ones. The model then
  reconstructed the distinction in its own words on both, writing that the
  document notes the absence of reported symptoms rather than documenting their
  clinical absence in the patient.
- **The hints help and hurt, section by section.** Dropping all of them under
  qwen takes the reference from 398 to 365, micro 99.5% to 91.5%, and the total
  hides opposite effects. F carries most of it, 40 falling to 17 of 39, one
  section failing outright because without the hint nothing asks for the
  DETAILS_PRESENT line the answer is derived from. B1.1 loses 7, A3.2 loses 3,
  and A1, A2, A3.1, B1_2, B2, C and X do not move at all, so 1199 of the 3071
  injected characters do nothing under this model. Audit them one at a time, not
  as a block. An earlier ablation recorded A3.2 gaining 5.3 and B2 gaining 2.5,
  and neither reproduces, because neither hint is the same object any more. B2's
  is in `SECTION_HINTS_DISABLED`, which that measurement is what caused, so
  removing the rest cannot touch it. A3.2's was 1262 characters then and is 143
  now, the contrastive clause having been cut in between: the two ablations
  measured two different hints.
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
- **The guideline anchors cost 7 sections, and reproduce the section-headings
  arm almost exactly.** `GUIDELINE_ANCHORS_ENABLED` True, everything else at the
  reference and the hint and query fingerprints proving it, takes 398 to
  **391/400**, micro 99.5% to 97.75%, macro kappa 0.982 to 0.935. The
  falsification held where it was set: C and X do not move, since their anchor
  carries the text they already received. Everything else failed in the
  informative direction. **A1 and B1.1 do not improve**, so section 4.1 reaching
  A1 and A2 was not what those sections lacked; A2 gains SYN_03 and loses SYN_12
  and SYN_23, A3_2 loses 3, and A3_1, B1_2 and B2 lose one each.

  Seven of the nine sections that change answer are **the same sections that
  changed under `SECTION_DESCRIPTIONS_ENABLED`**: SYN_03's A2 gained, SYN_12 and
  SYN_23 lost on A2 and on A3_2, SYN_07 lost `Calf pain or tenderness` on B2 and
  SYN_40 filled B1_2. Two interventions with opposite content — the form's
  heading, which is narrower than the paper, and the paper's own passage, which
  is wider — move the same records the same way. What they share is that both
  state authoritatively what a section is about, and where that statement
  differs from the option text the model follows the statement.

  **The A2 reasoning says why, and it revises what section 4.1 settles.** Given
  4.1, the model quotes it correctly and then excludes SYN_12: a catheter
  thrombolysis dissolves the thrombus rather than recovering it, and the
  confirmation came from an intraprocedural venography, which is imaging. That
  reading is defensible on 4.1, which defines *pathologic* diagnosis by thrombus
  recovery. The option, however, asks whether an **other procedure confirmed
  presence of DVT**, which the venography did. So 4.1 and A2's second option are
  not the same criterion, and anchoring A2 to 4.1 hands the model a stricter
  test than its own question. A3_1 on SYN_28 is the same failure: given the list
  of accepted modalities, the model rejects an impedance plethysmography and
  answers that no imaging was done, though A3.1 asks nothing about modality; the
  cross-section rule then clears A3_2, two wrong sections from one error.
  `GUIDELINE_ANCHORS_ENABLED` stays False.
- **At 200/40/3 the anchors are noise.** Against a control run in the same
  regime, produced for the purpose because the earlier 200/40/3 arms predate
  B2's query rewrite, they move 385 to 386 of 400: five sections corrected, four
  broken, and A2 on SYN_23 fails outright with no parseable answer after three
  attempts. A2 is where the model is least stable under any intervention.
- **Twenty-two configurations measured on the same base**, each varying one
  component. `docs/RISULTATI_SPERIMENTALI.md` section 8 has the full per-section
  table.

  ```
                                    exact     micro    macro kappa
  agentic_graph (reference)        398/400    99.5%      0.982
  same, 8B agent and old B2 query  398/400    99.5%      0.982
  agentic without the context      395/400    98.75%     0.970
  raw_record                       392/400    98.0%      0.966
  agentic with section headings    392/400    98.0%      0.940
  agentic with guideline anchors   391/400    97.75%     0.935
  agentic 200/40/3, anchors        386/399    96.7%      0.917
  agentic 200/40/3, control        385/400    96.25%     0.911
  agentic 200/40/3, 27B agent      383/400    95.8%      0.905
  agentic 200/40/3, new queries    383/400    95.8%      0.902
  agentic, chunks 200/40, k 3      382/400    95.5%      0.895
  full_text + 27B extractor        379/399    95.0%      0.816
  rag 200/40/3, 27B, new queries   374/399    93.7%      0.797
  medgemma as evaluator            371/398    93.2%      0.837
  rag 200/40/3, 27B extractor      369/400    92.2%      0.781
  qwen without the hints           365/399    91.5%      0.780
  rag                              341/400    85.3%      0.607
  full_text                        338/399    84.7%      0.580
  rag, chunks 200/40, k 3          337/399    84.5%      0.583
  llama3:8b evaluator, gates on    316/400    79.0%      0.497
  llama3:8b evaluator, no hints    283/400    70.75%     0.423
  llama3:8b evaluator, with hints  240/400    60.0%      0.373
  ```

  Not letting the 8B rewrite the evidence is worth +54, a 27B extractor instead
  of an 8B is worth +41 and still lands 13 below not extracting at all, and the
  guideline context is worth +3. Runs 2 to 7 carried the `keyword` gate and the
  pre-2026-09-15 ground truth; they are reported reconstructed to the current
  configuration, by undoing the gate's recorded overrides on the stored
  predictions and rescoring. On `raw_record`, `rag` and the no-context arm the
  gate and the ground-truth correction cancel and the published figure stands;
  `full_text` and the 27B-extractor arm lose one section each on X, where the
  lossy extractor had dropped the text the gate keyed on, so there is no
  override to undo.
- **A weak first stage decides the score.** The four extraction modes are the
  four combinations of two choices, whether the 8B **rewrites** the evidence and
  whether Agent 2 gets the whole record or the retrieved chunks:

  ```
                          whole record        retrieved chunks
     8B rewrites          full_text  338      rag          341
     verbatim evidence    raw_record 392      agentic      398
  ```

  Removing the rewriting is worth +54 and +57; adding retrieval is worth +3 with
  it and +6 without, so the two factors interact. Letting the 8B write the
  evidence returns `NO RELEVANT EVIDENCE FOUND.` on 143 sections out of 400 in
  `full_text` and 142 in `rag`, and that is not where the loss is. Partition
  the sections by whether the evidence came back empty and score the reference
  on the same ones: on the empty partition the arm gets 93.0% against the
  reference's 100%, losing 10 sections; on the partition with evidence it gets
  80.1% against 99.2%, losing 49. **The rewriting costs five times more through
  what it writes than through what it omits**, and the reference sits at 99-100%
  in both partitions, so the empty sections are not the easy ones. An empty
  evidence is usually correct behaviour, since on about a third of
  section-record pairs there is nothing to extract for that criterion. Agent 2
  keeps reasoning correctly on what it receives, so reading only the final
  answer hides the cause. The chunks column
  mixes fixed with model-chosen queries, so the +6 is not attributable to
  retrieval rather than agency. All four cells were measured with retrieval
  inert; made to select it costs 16 sections, which is the bullet below. An extractor exists in every mode but
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
- **Correct information, correctly used, cost 6 sections.**
  `SECTION_DESCRIPTIONS_ENABLED` sends each section's own heading from
  `models.py` above its numbered options, the wording the printed form uses.
  Everything else identical and the hint fingerprint proving it, the reference
  run goes from 398 to 392, micro 99.5% to 98.0%, macro kappa 0.982 to 0.940.
  Six records move, one gain and seven losses. A2's heading names the section
  Surgical procedure, which corrects SYN_03, where a CT venography had been
  called an "other procedure" and is a real gain. On SYN_12 and SYN_23 the model
  refuses the ground truth's "other procedure" for a catheter thrombolysis and a
  percutaneous IVC filter placement, reasoning that interventional radiology is
  categorised apart from open surgery and citing "these Brighton criteria
  validations" for it. **That citation is invented and section 4.1 of the paper
  says the opposite**: the definitive pathologic diagnosis includes recovery of
  a thrombus by *surgical or a catheterization procedure, such as thrombectomy*,
  naming the two together. Those two sections are model errors, and they are
  what takes A2's "Other procedure" option to never being predicted, kappa 0.649
  against a 90% majority baseline. A3_2 on those same two records adds
  `Contrast venography`,
  and both records state that the intraprocedural venography confirmed the
  thrombus, which is literally what A3.2 asks; the ground truth counts that
  venography as a procedure in A2 and not as a study in A3_2, a convention that
  avoids double counting but is written down nowhere. B2 on SYN_25 adds
  `Redness, warmth, or pain in one or more extremities` where only calf pain
  and taut painful skin are documented, which is clinician question four.
  B1_2 on SYN_40 fills a section the truth leaves empty, arguable and probably
  the truth's. SYN_07 is a model error too, dropping `Calf pain or tenderness`
  because the record says diffuse tenderness rather than calf pain. Net of what
  is settled the balance is +1 and -3, so **-2**. Sections are separate
  evaluator calls, so the A3_2 change is not a consequence of the A2 answer but
  a second effect of the same edit. The switch stays False, on A2 on the merits
  as well. What the run demonstrates, in a sharper form than the score, is that
  **the form's heading is narrower than the paper the form derives from**: the
  model read the heading correctly and became less faithful to the primary
  source. Worth remeasuring once a clinician-written reference exists, since the
  three coding disagreements left are still scored against a ground truth that
  is one of the parties to them.
- **Choosing is not writing.** `agentic_graph` belongs in the verbatim row even
  though it has a model in the loop: `extract_evidence_agentic` returns the raw
  tool observations, not the agent's final turn, so the 8B picks queries but the
  evidence never passes through its tokens. What it still controls is coverage,
  and that is inert on this corpus: 0 sections out of 400 without evidence,
  `agent1_seconds` between 6.5 and 7.2 on every section (one tool call), and
  retrieval returning the whole record each time. It returns the whole record
  because `EHR_RETRIEVER_K` is 5 while a record of this corpus (317 to 1185
  characters, median 970) splits into one or two chunks, so the tool hands back
  all of them and there is nothing to select. Five
  chunks of 800 characters overlapping by 150 cover about 3400 characters, which
  is the length a record has to exceed before retrieval starts choosing at all.
  What does vary by section is the order: the tool joins the two chunks by
  relevance to the query the agent wrote, so the same halves reach Agent 2
  differently arranged. Checked on two records, identical across the four
  sections read on SYN_01 and reversed between A1 and X on SYN_23.
- **Made to select, retrieval costs 16 sections, and costs them where the answer
  needs the whole record.** `EHR_CHUNK_SIZE` 200, `EHR_CHUNK_OVERLAP` 40 and
  `EHR_RETRIEVER_K` 3, everything else at the reference and the hint fingerprint
  `ab63b8e5f5af` proving it, take `agentic_graph` from 398 to 382 of 400, micro
  99.5% to 95.5%, macro kappa 0.982 to 0.895. The evidence reaching Agent 2 goes
  from **100% of the record on all 400 sections to 66.4%**, so this is the first
  run in which retrieval selects at all. The loss lands on the sections that
  need exhaustive coverage: the three multi-select sections carry nine of the
  sixteen (A3_2 -4, B2 -4, B1_2 -1), and X loses 4 in one direction, flipping to
  `No alternative diagnosis` on SYN_11, SYN_37 and SYN_38 because a criterion of
  exclusion cannot be settled on two thirds of the text. A1, B1_1, C and F do
  not move, C and F staying at 40 of 40: their answer rests on one distinctive
  token, which retrieval finds reliably. Under `rag` the same change moves the
  score from 341/400 to 337/399 while **55 sections change answer**, because
  coverage stays at 12.5% against 12.1%. That average is dominated by the
  sections where the extractor returns nothing; where it returns something it
  produces about 146 characters out of 944, one or two sentences, which is one
  section's worth. **The size of the evidence is set by the task and not by the
  input**, so changing retrieval does not change how much leaves the extractor,
  only whether the fact it needed was inside what went in. Empty evidence falls
  from 145 to 120 with no gain. What the arm does
  not show is that retrieval hurts on real documents: shrinking the window
  discards text that would have fit in the prompt anyway, so it reproduces the
  selection pressure without its cause.
- **The rewriting cost a bigger extractor cannot remove is 13 sections, in both
  retrieval regimes.** Repeating the small-chunk `rag` arm with `qwen3.6:27b` as
  extractor instead of the 8B, one variable and the same fingerprint, takes it
  from 337/399 to **369/400**, micro 84.5% to 92.2%, macro kappa 0.583 to 0.781,
  so a 27B extractor is worth +32 there against +41 at 800-character chunks. It
  still lands 13 below `agentic_graph` in the same regime (382), and at 800
  `full_text` + 27B lands 13 below `raw_record` (379 against 392). The 200-chunk
  comparison is not a clean isolation, since `agentic` writes its own queries
  while `rag` uses the fixed section query. **Extractor fidelity is a model
  property**: on the identical input the 27B copies 96% of fragments verbatim
  with 0 preambles against the 8B's 71% and 20, though the same 27B drops to 72%
  when handed the whole record at 800, so fidelity depends on the regime too.
  The 27B also returns **more** empty evidence, 144 against 120, while scoring
  32 sections higher, which is the sharpest confirmation that an empty evidence
  is usually correct behaviour rather than the damage. The partition holds:
  against `agentic` in the same regime the 8B loses 36 sections where evidence
  exists and 8 where it does not, the 27B 11 and 2. F does not move, 31 and 32
  with kappa -0.046 against an 82.5% baseline, because the absence of detail it
  asks about is a property of the record that no fragment carries.
- **A 27B agent buys one section and saves an hour, and that settles the model
  swapping question.** Replacing `AGENTIC_LLM_MODEL_NAME` with `qwen3.6:27b` in
  the 200/40/3 arm, so Agent 1 and Agent 2 share a tag, takes it from 382 to
  383 of 400. **Exactly one section changes**, X from 36 to 37, and the other
  nine are identical figure for figure: in a regime where the query decides
  which three chunks of six arrive, a three times larger brief-writer is worth
  one section in 400. The timing is the result. `agent1` **falls** from 70 to 64
  seconds a record with the larger model, `agent2` from 192 to 91, close to
  `raw_record`'s 89 with no second model at all, and the run takes 1h54 against
  3h02. A bigger model being faster rules out generation cost: what the slow
  arms pay is swapping two distinct models on the GPU twice a section. Practical
  consequence: `llama3.1:8b` holds that slot because a tool-calling model was
  needed, not because it was fast, and the 27B is **1.6 times quicker**. Decide
  that swap on time, since one section in 400 is not a measurable difference.
- **Of three under-covering queries the audit found, only one was worth
  fixing.** Rewriting A2, A3_2 and B2 — fingerprint `a71fbd4109ec` — and
  measuring both modes at 200/40/3: `rag` with the 27B extractor goes 369/400 to
  374/399, `agentic_graph` 382 to 383. The totals hide three different effects.
  **B2 works and works as predicted**: +5 in `rag` and +2 in `agentic`, naming
  arms and extremities recovers the two upper-extremity records answered `None
  of the above` (SYN_03, SYN_22) and the absent-pulses one (SYN_18), and drops
  two false positives (SYN_07, SYN_25); in `rag` a sixth section follows
  indirectly, B1.1 on SYN_03 turning positive because the cross-section rule
  reads the symptom B2 now carries. **A3_2 does not**: 0 in `rag`, where it
  trades the disputed `Contrast venography` on SYN_12 and SYN_23 for the
  compression-versus-Doppler pairing on SYN_04 and SYN_26, and -2 in `agentic`
  for the same pairing — **the failure mode the removed A3.2 contrastive clause
  used to prevent, back through the query instead of the hint**. **A2 is
  unstable**: +1 in `agentic` but -1 in `rag`, and on SYN_12 naming
  catheter-directed intervention pushes the model onto `Thrombectomy` in one
  mode and produces the run's only unparseable section in the other. Naming an
  `other` branch's instances does not make that branch reachable; it can push
  the model onto a different positive option.
- **The reference moved to a 27B agent and B2's rewritten query, and no answer
  changed.** The configuration in effect since 2026-09-20 runs `qwen3.6:27b` as
  Agent 1 and carries `section_queries_fingerprint` `bfd9536a31fe`, B2's query
  alone rewritten. It scores 398/400, micro 99.5%, macro kappa 0.982, with
  **zero sections changed out of 400** against the arm it replaced. Not an
  equivalent total: the same answers, record by record. Both edits were
  predicted inert from the same mechanism, that with `k` 5 and records splitting
  into one or two chunks retrieval returns everything, so neither the query nor
  the agent's brief can change what Agent 2 sees. The run costs 174 seconds a
  record against 270, **1h56 against 3h00**, and `agent2` lands on 89 seconds,
  the same figure `raw_record` posts with no second model at all — a third
  independent confirmation of the swapping cost, and the first at the reference
  parameters. **Every delta measured against the old reference therefore stays
  valid without recomputation**, since the answers it was computed from are the
  same ones.
- **The accuracy-time frontier has two points, and neither rewrites the
  evidence.** Comparing only the arms that vary the evidence path, with the
  timings measured from the audit-log timestamps: `agentic_graph` 800/5 is the
  most accurate at 398 and 270 seconds a record, `raw_record` the fastest at 392
  and 105. **Everything else is dominated by `raw_record` on both axes**,
  including the two 27B-extractor arms, which at 166 seconds are still slower
  and 13 or 23 sections lower. Six sections is 1.5 points and the two Wilson
  intervals overlap, so on this corpus `raw_record` is a defensible engineering
  choice. `agent2` measures the same work in every arm yet costs 89 seconds with
  no extractor, 143 when the extractor is the same model as the evaluator and
  about 195 when it is a different one, while `raw_record` hands Agent 2 the
  longest prompt of all; the ordering is consistent with the cost of swapping
  models on the GPU, which is **not isolated**. The ranking is conditional on
  the testbed: `raw_record` sits on the frontier because records run 317 to 1185
  characters, and on real documentation that row may not exist at all.
- **Two inputs that carry the same information are not the same input.**
  `full_text` and `rag` score 338 and 341 and differ on **35 sections**:
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
- **The architecture transfers; the prompt tuning is what does not.** Two
  sections of the Brighton myocarditis questionnaire, 17 dummy cases with a
  ground truth written by two authors outside the project, run in
  `agentic_graph` with no hints, no gates, no cross-section rules and no
  guideline context: only the schema, the two section queries and the disease
  terms in the prompts differ from the DVT arm. The matched control is the DVT
  corpus stripped of the same four components, **353/399, micro 88.47%, macro
  kappa 0.725**, against a prediction of 354 obtained by adding the components'
  measured effects, so on this configuration they are close to additive.

  **E scores 17 of 17 in every arm, and no intervention changes a single
  answer** — not the guideline, not the context window. On an unseen
  questionnaire, an unseen schema and two queries written once and blind. F
  scores 3 to 5 of 17, and all thirteen errors of the first arm turn on one
  distinction, `Segmental wall motion abnormalities` against `Global systolic or
  diastolic function depression`: one disagreement repeated, seven of the
  thirteen being seven variants of one base case. Two read in full are
  defensible for the model; on one the ground truth answers `No abnormalities
  seen on ECHO` while the record describes hypokinesia and a borderline ejection
  fraction. The authors' own expected-ambiguity column does not predict the
  errors either: 5 of 8 `low` are wrong.
- **The guideline context is worth +2 on that corpus, and the DVT mechanism does
  not reproduce.** Prediction before the run: F rises from 4 into the 4 to 7
  band by correcting the additions-only records and leaving the five swaps
  alone. The number held, the mechanism did not. False positives go 16 to 17 and
  the options selected go 31 to 34 against a truth of 27, so the context makes
  the model select **more**, the opposite of DVT, where turning it off costs two
  false positives on A3.2. **Whether the guideline curbs over-selection depends
  on the model and the corpus, not on the mechanism.** Four of the five swaps
  stay swaps and one becomes an outright failure: the paper lists the five
  echocardiogram findings without defining them, word for word F's option list.
- **The context window was not truncating, and is not neutral either.** Widening
  it from 4096 to 8192 changes 2 sections of 34, both on F, and F falls from 4
  to 3: nothing was being cut, so the DVT arms run at Ollama's default are not
  suspect. Two sections change all the same at temperature 0, so `num_ctx`
  perturbs generation without truncating. `LLM_NUM_CTX` is now set in
  `config.py` at 4096, the value already in force, and recorded under
  `_run_config.generation`. At 8192 one section per arm fails outright, always
  on F, with no parseable answer after three attempts; at 4096 none does.
- **Agent 3's confidence is reliable only when it is very high.** Measured by
  two probe scripts on stored answers, not yet by the pipeline itself. The
  logprob of the `FINAL_ANSWER` number after reasoning is saturated, above
  0.99998 on three replayed sections including SYN_10, which is wrong; hence
  the separate unreasoned request. On the stripped arm (46 errors) the AUROC is
  0.824 against 0.509 for `agent2_seconds`, but comparing only answers of the
  same section it is 0.516: the signal mostly tells hard sections from easy
  ones. Above 0.99 sit 242 of 400 reference sections with no error and 216 of
  399 stripped ones with 2; below it, 44 of the 46 errors, but a low value is
  not a probability, since under 0.5 Agent 2 is still right 78% of the time. F
  fails both ways, inverted without hints and false alarms with them, because
  the unreasoned request misreads the inversion and the diagnosis condition.
  SYN_03 A2 is fifth lowest of 400, yet SYN_13 A2, the same error type, scores
  0.993 on the stripped arm. 1.95 seconds a section.
- Read the metrics in this order: majority baseline and gain, then kappa, then
  accuracy with its interval. Accuracy alone ranked F above A3_2 under the 8B
  model, where F gained nothing over a constant answer and A3_2 gained 13
  points.

## Open items

- **The guideline retrieval selects hard and is keyed on the wrong string. The
  anchors that replace it are measured and cost 7 sections.** The paper is 48,467 characters
  in 76 chunks and `BRIGHTON_RETRIEVER_K` is 5, so each section reads 6.6% of
  it, chosen by that section's `SECTION_QUERIES` entry — a string written to
  find findings in a clinical record. The two retrieval paths are therefore in
  opposite regimes: on the record, retrieval returns everything and only the
  order varies; on the paper, it discards 93%. Measured on the 40-record
  corpus, section 4.1, which states the criterion A1 and A2 rest on, reaches
  Agent 2 on **0 records out of 40**, while A2's context opens on the paper's
  preamble. X, C and A3_2 do receive their passage on all 40.
  `config.GUIDELINE_ANCHORS` names the passage per section instead of searching
  for it, and `GUIDELINE_ANCHORS_ENABLED` stays False: the arm scores 391
  against 398. **What the measurement leaves open is not whether the anchors
  help but what to anchor to.** Every loss comes from the same place, a passage
  that states a criterion the section's options do not ask about, so an anchor
  is only as good as the match between the paper's definition and the
  questionnaire's wording — and on A2, A3_1 and B2 that match is exactly what
  the clinician questions are about. Anchoring to Table 3, the case definition
  the questionnaire follows, is the variant not yet tried on A2 and A3_1.
  Text resolves with hyphenated line breaks
  (`Dop-\npler`) in both paths, which de-hyphenating would change for both at
  once and is therefore its own experiment.
- **The Level of Certainty is left to REDCap, and measuring it here is out of
  scope.** Table 3 of the paper is the calculation. Level 1 is reached through
  any one of pathology, a procedure confirming a thrombus, or a confirmatory
  imaging study, and the table notes that the classification is independent of
  clinical findings; Levels 2 and 3 rest on the clinical presentation and both
  require no alternative diagnosis; the D-dimer separates Level 2 from Level 3.
  Read against the questionnaire, which the section lettering follows closely
  enough that the correspondence looks intended rather than measured, a record
  answered positively in any A section takes its level from that section alone,
  and the answers to B, C, F and X never enter. That is why a section-level
  score and the LOC are not the same measurement. Two REDCap round trips would
  add little to it, since only two of the 40 rows differ from the ground truth,
  `criteria_a2` on SYN_03 and `criteria_b1_1` on SYN_10. The claim worth making
  is structural: count the ground-truth records carrying a positive A criterion,
  and on those the errors left in B and X cannot propagate. What the pipeline
  owes the study is a correct CSV, and that is verified, with the export read
  back and its codes reconverted, 300 sections of 300 identical to the source
  JSON.
- **What F of the myocarditis questionnaire needs is a clinician, not a tool.**
  Thirteen errors on one coding distinction, untouched by the guideline and by
  the context window, and the paper does not define the two options it
  separates. The question to ask is how a systolic function reported as *ai
  limiti inferiori della norma* should be coded, and whether C0001_0_0's
  echocardiogram is to be read as normal when the record describes hypokinesia.
  Third time the project reaches the same conclusion from a different direction,
  after A2 and SYN_10.
- **Agreement with an independent reviewer is not measurable yet on DVT.** Every
  number on that corpus is scored against a ground truth written alongside the
  pipeline, so the two are not independent and their agreement is not evidence.
  The myocarditis arm is the exception and the reason it was run. A validation study
  reports the concordance between two blind reviewers instead, neither of them
  treated as the truth; the SeValid myocarditis experiment reports 28 of 38, 74%.
  Measuring it here takes a clinician filling the questionnaire on a subset of
  the corpus without seeing the output or the ground truth, then agreement at
  three levels, per section, per case decision and per LOC, the last one ordinal
  and so quadratic weighted. Discordant records go to a second clinician, and
  that adjudicated standard is what gives each reviewer a sensitivity. Of the
  two sections left, A2 on SYN_03 is a genuine model error and B1.1 on SYN_10 a
  disagreement argued from the guideline; accuracy scores both as errors.
- **A2 still rests on few records**, 4 of 40. X went from 6 positives to 9 with
  the ground-truth correction of 2026-09-15. F went from 2 positives to 7 with
  the September expansion and now scores kappa 0.918.
- **Questions for the clinicians.** The first three each decide how a record is
  coded; the rest are conventions.
  - **A2's second option and section 4.1 are two different criteria, and
    SYN_12 and SYN_23 turn on which one governs.** Section 4.1 defines a
    pathologic diagnosis as the recovery of a thrombus by a surgical or a
    catheterization procedure; the option asks whether an *other procedure done
    confirmed presence of DVT*. A catheter thrombolysis recovers nothing, so 4.1
    excludes it, while the intraprocedural venography does confirm the thrombus,
    so the option's words include it. The ground truth follows the option. The
    model follows whichever of the two it is shown: given the section heading it
    excludes both records, and given 4.1 it excludes them for a different
    reason. Which criterion the questionnaire intends is the question, and the
    heading's word Surgical is a third reading again.
  - Does a vague, uncharacterised symptom, "a generic discomfort in the leg"
    with no site or intensity, count as B1.1's "at least one symptom or sign
    was reported", or does the section stay unknown? SYN_10 turns on this: the
    ground truth says unknown, the model reads the discomfort as a symptom.
  - Must X's alternative etiology explain the symptoms that suggested
    thrombosis, or is any diagnosis that explains the acute illness enough?
    Acted on already in the second sense, and SYN_11 and SYN_16 rest on it.
    Ask it concretely: a patient dies of an acute myocardial infarction, the
    autopsy excludes DVT and no leg symptom was ever reported — is X answered
    `An alternative diagnosis was found that explained the acute illness`, or
    does the absence of a DVT presentation leave nothing for an alternative to
    displace? Table 2 is organised the first way, one row per syndrome indexed
    by that syndrome's own non-specific symptoms, and an infarction explains no
    calf pain.
  - Does a venography performed during an interventional procedure, which the
    record says confirmed the thrombus, belong in A3.2 among the studies that
    confirmed DVT, or only in A2 as the procedure? SYN_12 and SYN_23 turn on
    it: the ground truth counts it in A2 and not in A3_2, which avoids counting
    one event twice but is a convention the form does not state.
  - Does B2 option 4 apply when only calf pain is documented?
  - Does B2's "Absent pulses in legs or arms" need a pulse examination, or does
    absent flow on a Doppler study count?
  - Is A3_2 or B1_2 meant to be left empty when no option applies, the printed
    form offering no "none of the above" for either? SYN_40 is the concrete
    case: symptoms are reported so B1.1 is positive, but the ultrasound report
    is not yet available and no specialist diagnosis is on file, and the ground
    truth leaves B1.2 empty. Is a reported syndrome with no specific type the
    intended combination?
- **Two of the three per-section gates are now off.** The details gate reverted a
  correct answer once the F hint gained its precondition. The absent-pulses gate
  is still on and has never fired. The keyword gate was removed on 2026-09-15 together with a
  ground-truth correction: it fired 3 times in 400, always on X, on SYN_11
  (death from acute myocardial infarction, autopsy excluding DVT), SYN_16 (chest
  pain attributed to a musculoskeletal cause) and SYN_20 (death from traumatic
  haemorrhage, stated as unrelated to VTE). In all three the model answered that
  an alternative diagnosis was found and was right: section 5.2.7 of the paper
  states the criterion as *no alternate etiology that could explain the clinical
  illness* and calls Table 2 *a list of possible* etiologies. The gate and the
  ground truth agreed because both were written from the same narrow reading of
  Table 2, so their agreement was circular. Only SYN_20 falls in Table 2's DVT
  row, under *Physical trauma*; SYN_11 and SYN_16 match rows written for
  pulmonary embolism, so those two rest on the reading that the alternative need
  not explain the DVT symptoms. Those three scenarios now carry
  `An alternative diagnosis was found`, which keeps the reference run at 398.
  Two further defects made the gate indefensible: Table 2's DVT row lists
  *Physical trauma* as a general category and the keyword list encoded only the
  specific diagnoses under it, so SYN_20 failed even on the narrow reading; and
  in `rag` the gate overrode SYN_36 and SYN_39, whose keywords ARE in the list,
  because the lossy extractor had already dropped the text containing them —
  **the gate's reliability depends on the extractor's output**.
- **Three section queries name only part of their section's options.** Audited
  by comparing each query in `SECTION_QUERIES` against the options of its
  section. A2's query is `thrombectomy, surgical procedure related to DVT` and
  covers one of the two positive branches: `Other procedure done that confirmed
  presence of DVT` shares no term with it, and that is the branch the ground
  truth uses for the percutaneous IVC filter on SYN_12 and SYN_23 and the CT
  venography on SYN_03. A3_2's `Other` option has no vocabulary in its query
  either, and SYN_28 is a plethysmography, which matches none of the modalities
  named. B2's query lists leg findings only, while option 3 reads `legs or arms`
  and option 4 `one or more extremities`; SYN_03 is an upper-extremity case. The
  queries were written from the finding expected rather than from the option
  list, which works wherever the finding has a name and fails on every `other`
  branch. A1, A3_1, C and B1_1 only lack verbs and are fine, and X omits
  `diagnosis` and `acute illness` deliberately, being aimed at Table 2.
  **Not measurable at the reference parameters**: retrieval returns the whole
  record whatever the query asks, so rewriting one would change the guideline
  context and nothing else. Measured in the 200-chunk regime, where it does
  show: B2's rewrite is worth +5 and +2, A3_2's is 0 and -2, A2's is -1 and +1,
  so **only B2's gap was a defect**. B2's rewrite is the one now in
  `SECTION_QUERIES`; A2's and A3_2's were not adopted.
- **`print_report` raises when nothing was compared.** `_macro` returns None on
  an empty section set, and the gain column computes `macro_accuracy -
  macro_majority_baseline` without guarding it, so scoring a directory that
  matches no record ends in a TypeError instead of an empty report. `_pct` and
  `_num` already print `n/a`; the subtraction is the one place that does not.
- **The audit log does not record which guideline was read.** `_run_config`
  carries `brighton_context_enabled` and `guideline_anchors_fingerprint` but not
  the paper, and there are two: the DVT case definition and the myocarditis one,
  indexed into two stores. A `guideline_source` field holding the PDF name and a
  digest of the extracted text would make the context traceable the way the
  hints and the queries already are.
- **Agent 3 has not run inside the pipeline.** The probes sent the same prompt
  as `confidence.py` but scored the answer before the cross-section rules. On
  the stripped arm the rules and the context are off, so its figures hold for
  `confidence.py` as they are. The reference needs a confirming run in its own
  directory with `CONFIDENCE_ENABLED = True`; `compare_runs` against it must
  show zero changed answers. The 0.99 threshold still needs an arm with hints
  on and more errors, such as 200/40/3.
- Test whether the TRANSCRIPTION RULE in `AGENTIC_EXTRACTOR_SYSTEM_PROMPT` does
  anything: it governs a string the code discards.
- **The extractor prompt never says a negation is evidence**, while the
  evaluator prompt does. On SYN_32 the same extractor hands B1.1 the sentence
  stating the record contains no symptoms and hands F the bare diagnosis,
  because the two section queries aim at different halves. F needs to know
  whether details accompany the diagnosis and its query asks for the diagnosis.
  This penalises the baselines, not the reference mode, where F is 40 of 40.
- **The audit log cannot be retained on real records.** It holds the evidence
  verbatim and the reasoning that quotes it, so on clinical documentation it is a
  copy of patient data in a JSON file. Everything the project measures rests on
  it, which makes the diagnosability a property of the testbed rather than of the
  deployed system. What survives without reproducing the record is the chosen
  answer, `answer_conflict`, the per-agent timings, whether the evidence came
  back empty, and `_run_config`; those keep provenance and the reliability
  flags, and lose the ability to reconstruct why an answer is wrong. The reduced
  form has not been designed.
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
