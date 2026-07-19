# Corrected Action Plan — Automated Compilation of the DVT Form (SEVALID_DVT)

## 0. Differences from the original plan

The original plan (Steps 1-5) was structurally sound but covered only
Criteria B and C with Pydantic, assumed reliable agentic tool-calling on a
quantized 8B model with no preliminary check, and stopped at producing JSON
per section without addressing final aggregation. The corrections below
integrate these points. (Update: the LOC aggregation point below was later
dropped from scope — see section 6.)

## 1. Environment setup (unchanged, plus one extra test)

Same stack: Ollama + Llama3-OpenBioLLM-8B or BioMistral (GGUF Q4, ~5.5 GB RAM),
`HuggingFaceEmbeddings` with `BAAI/bge-small-en-v1.5` (~130 MB), Chroma as the
vector store.

**Added — mandatory Step 0 before building the pipeline:**
verify that LangChain's `.with_structured_output()` works reliably with the
chosen model via Ollama. Many fine-tuned clinical models (OpenBioLLM,
BioMistral) are not trained for structured function-calling: if the test
fails, the fallback is to force JSON output via prompting + Pydantic
parsing/validation with retry, instead of relying on native tool-calling.
See `agents.py` for the test and fallback implementation.

## 2. Full mapping of the checkbox questions into Pydantic

The real REDCap form contains more sections than the ones covered in the
original plan. Full inventory of the choice questions (checkboxes), excluding
all free-text fields (dates, descriptions, "specify other"):

| Section | Question | Type | Options |
|---|---|---|---|
| A1 | Autopsy | single choice | 3 |
| A2 | Surgical procedure | single choice | 3 |
| A3.1 | Imaging — outcome | single choice | 3 |
| A3.2 | Imaging — which studies | multi-select | 5 (incl. "Other", free text excluded) |
| B1.1 | Symptoms reported | single choice | 3 |
| B1.2 | Type of DVT | multi-select | 2 |
| B2 | New clinical symptoms | multi-select | 5 (incl. "None...") |
| C | D-Dimer | single choice | 3 |
| F | Reported by specialist | single choice (Yes/No) | 2 |
| X | Alternative diagnosis | single choice | 2 |

All of these are now modeled in `models.py` (the original plan only covered B
and C). A **validator** was also added on B2: if the option "None of the
above..." is selected, no other option can be co-selected (logical mutual
exclusivity not guaranteed by the base Pydantic schema).

## 3. RAG system (unchanged setup, two flows)

- **Static KB**: Brighton paper (DVT synonyms) → Chroma, never changes.
- **Dynamic KB**: the patient's clinical record → chunking
  (`RecursiveCharacterTextSplitter`, chunk ~800, overlap ~150) → temporary
  Chroma store per patient.

## 4. Two-agent architecture — with a robustness fallback

**Agent 1 (Extractor):** by default uses **direct retrieval** (similarity
search + prompt with injected context), not a ReAct agent with a tool loop.
Agentic tool-calling is still available (`agents.py` implements it) but
should only be enabled if the Step 0 test confirms the model handles it
reliably — on a quantized 8B model the risk of inconsistent tool-call
parsing is real.

**Agent 2 (Evaluator):** unchanged in principle — `.with_structured_output()`
on the current section's Pydantic model, temperature 0. Added **few-shot
examples** in the prompt for handling negations ("no edema" → None), which
was already flagged as a concern in the original plan but without a concrete
countermeasure.

Both agents run at `temperature=0` (the original plan only specified this
for Agent 2).

## 5. Execution loop over sections

Extended to all 10 checkbox questions (not just A1, B, C as in the original
plan): A1 → A2 → A3.1 → A3.2 → B1.1 → B1.2 → B2 → C → F → X.

## 6. Final output: compilation only, no LOC classification (updated)

Clarification of the goal: the project requires **only filling in the
checkboxes** from the clinical record, not classifying the overall Level of
Certainty. The point raised in an earlier version of this plan (missing
Brighton combination algorithm) is therefore no longer relevant to the
project's scope: it should not be implemented.

`aggregation.py` was simplified accordingly — it just serializes the
populated `DVT_CriteriaForm` into a JSON with all checkboxes filled in. This
is the project's final output.

## 7. Validation (still relevant, not implemented in the provided code)

Before presenting results, it would help to compare against a small set of
manually annotated records (even 10-15), with at least per-criterion
accuracy and, if a second annotator is available, Cohen's kappa. This is the
step that gives a defensible quantitative figure to present to the professor,
beyond the architectural description. Not implemented in the provided code
(requires a dataset), but a natural next step once the pipeline produces
stable output.

## 8. Note on sensitive data

Clinical records are sensitive health data: make sure the local-only setup
(Ollama, no external API calls) is consistent with your university's/ethics
committee's data-handling policies, if applicable to your project.
