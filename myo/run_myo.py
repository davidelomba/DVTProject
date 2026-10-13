"""
Runs the pipeline over the myocarditis cases, in agentic_graph mode.

Same code as the DVT runs: pipeline.run_pipeline, both agents, the verbatim
evidence of agentic_graph. What differs is the schema, the section queries, the
terms naming the condition in the prompts and the temperature of Agents 1 and 2,
MYO_TEMPERATURE unless --temperature overrides it. The cross-section rules are
off, not translated: they encode the DVT form, and a myocarditis run measures
the architecture without them.

myo_schema, imported first, installs models_myo as `models`, because pipeline
and criteria_rules bind SECTION_MODELS at import time.

The guideline context is off unless --guideline asks for it, so the two arms
differ in one setting. The paper is indexed either way, because run_pipeline
builds the store before knowing whether anything will query it, and the store
lives under myo/ rather than in the shared directory: build_brighton_kb reloads
an existing index without looking at the text it was given, so one directory for
two papers would hand this arm the DVT guideline without saying so.

Usage:
    python myo/run_myo.py --output-dir ./myo/output_myo
    python myo/run_myo.py --guideline --output-dir ./myo/output_myo_guideline
    python myo/run_myo.py --only C0001
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import myo_schema  # noqa: F401  # isort: skip -- must precede the project imports

import agents
import config
import pipeline
import prompts_myo
import rag_setup
from aggregation import form_to_json_summary
from langchain_core.tools.retriever import create_retriever_tool

MYO_DIR = Path(__file__).resolve().parent

# Sections in the order of the printed questionnaire.
SECTION_ORDER = ["A1", "B", "C", "E", "F"]

# Temperature of Agents 1 and 2. Agent 3 keeps config.VERIFIER_TEMPERATURE. No
# seed is set, so each run samples independently.
MYO_TEMPERATURE = 0.5

# One query per section: the section's name and its options, every one of
# them, in the words of CriteriaForm_SEVALIDMYOCARDITIS and no others, so no
# query carries vocabulary taken from the records it is run on.
SECTION_QUERIES = {
    "A1": "cardiac symptoms: acute chest pain or pressure, palpitation, dyspnea "
          "at rest, with exercise or lying down, diaphoresis, sudden death, none "
          "of the above, unknown if present or absent",
    "B": "physical exam features: pericardial friction rub, pulsus paradoxus, "
         "distant heart sounds, none of the above, unknown if present or absent, "
         "physical exam conforming pericarditis",
    "C": "inflammatory markers: ESR elevated, CRP elevated, high sensitivity CRP "
         "elevated, D-dimer elevated, none tested or no results, within normal "
         "range, positive inflammatory marker conforming myocarditis",
    "E": "electrocardiogram: ECG not done or no results, paroxysmal or sustained "
         "atrial or ventricular arrhythmias, supraventricular or ventricular "
         "tachycardia, sinus tachycardia, atrial fibrillation, AV nodal "
         "conduction delays, intraventricular conduction defects, AV block, new "
         "bundle branch block, continuous ambulatory ECG monitoring with frequent "
         "atrial or ventricular ectopy, ST segment or T wave abnormalities, "
         "elevation or inversion, newly reduced r wave height, low voltage, "
         "abnormal q waves, premature atrial and premature ventricular "
         "contractions, diffuse concave upward ST segment elevation, ST segment "
         "depression in aVR, PR depression without reciprocal ST segment changes, "
         "nonspecific abnormalities, no abnormalities seen on ECG, ECG "
         "abnormality conforming myocarditis or pericarditis",
    "F": "echocardiogram: ECHO not done or no results, new focal or diffuse left "
         "or right ventricular function abnormalities, decreased ejection "
         "fraction, segmental wall motion abnormalities, global systolic or "
         "diastolic function depression, decreased longitudinal and "
         "circumferential strain, ventricular dilation, wall thickness change, "
         "abnormal fluid collection or pericardial inflammation, no abnormalities "
         "seen on ECHO, echocardiogram conforming myocarditis or pericarditis",
}


def apply_domain(guideline: bool = False, temperature: float = MYO_TEMPERATURE):
    """Points the shared modules at the myocarditis sections and prompts.

    Args:
        guideline: whether Agent 2 receives the context retrieved from the
            myocarditis paper.
        temperature: temperature of Agents 1 and 2.
    """

    config.SECTION_ORDER = SECTION_ORDER
    config.LLM_TEMPERATURE = temperature
    config.EXTRACTOR_MODE = "agentic_graph"
    config.BRIGHTON_CONTEXT_ENABLED = guideline
    # Its own store: build_brighton_kb reloads an index already on disk without
    # reading the text it was handed, so the two papers need two directories.
    config.BRIGHTON_KB_PERSIST_DIR = str(MYO_DIR / "vectorstores" / "chroma_brighton_myo")
    config.SECTION_DESCRIPTIONS_ENABLED = False
    config.SECTION_HINTS = prompts_myo.SECTION_HINTS
    config.SECTION_HINTS_DISABLED = set()
    config.CROSS_SECTION_RULES = []

    extractor = prompts_myo.EXTRACTOR_SYSTEM_PROMPT_TEMPLATE.format(
        no_evidence=agents.NO_EVIDENCE
    )
    agents.EXTRACTOR_SYSTEM_PROMPT = extractor
    # agents.py builds the agentic prompt by concatenation at import time, so
    # replacing the base one does not reach it.
    agents.AGENTIC_EXTRACTOR_SYSTEM_PROMPT = extractor + (
        prompts_myo.AGENTIC_EXTRACTOR_SUFFIX_TEMPLATE.format(no_evidence=agents.NO_EVIDENCE)
    )
    agents.EVALUATOR_SYSTEM_PROMPT = prompts_myo.EVALUATOR_SYSTEM_PROMPT
    agents.MULTI_SELECT_EXAMPLE = prompts_myo.MULTI_SELECT_EXAMPLE
    pipeline.SECTION_QUERIES = SECTION_QUERIES


    def make_ehr_retriever_tool(ehr_vectorstore):
        """rag_setup.make_ehr_retriever_tool with this domain's description."""
        retriever = ehr_vectorstore.as_retriever(
            search_kwargs={"k": config.EHR_RETRIEVER_K}
        )
        return create_retriever_tool(
            retriever, "search_patient_record",
            prompts_myo.RETRIEVER_TOOL_DESCRIPTION,
        )

    # pipeline imports the name, so its own binding is the one that is called;
    # rag_setup is patched too so the two modules cannot disagree.
    rag_setup.make_ehr_retriever_tool = make_ehr_retriever_tool
    pipeline.make_ehr_retriever_tool = make_ehr_retriever_tool
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=MYO_DIR / "data" / "records")
    parser.add_argument("--output-dir", type=Path, default=MYO_DIR / "output_myo")
    parser.add_argument("--only", nargs="+", metavar="ID")
    parser.add_argument(
        "--guideline", action="store_true",
        help="send Agent 2 the context retrieved from the paper; off by default",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None, metavar="TOKEN",
        help="size of the context window, overriding config.LLM_NUM_CTX. The "
             "guideline context adds about 3800 characters to every prompt, "
             "which on the longest records puts the total near 4096 tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=MYO_TEMPERATURE,
        help=f"temperature of Agents 1 and 2 (default {MYO_TEMPERATURE})",
    )
    parser.add_argument(
        "--brighton-pdf", type=Path, default=MYO_DIR / "main.pdf",
        help="the myocarditis case definition; indexed even when the context "
             "is off, and into myo/vectorstores rather than the shared store",
    )
    args = parser.parse_args()

    pipeline = apply_domain(guideline=args.guideline, temperature=args.temperature)
    if args.num_ctx is not None:
        config.LLM_NUM_CTX = args.num_ctx

    record_paths = sorted(args.records_dir.glob("*.txt"))
    if args.only:
        record_paths = [p for p in record_paths
                        if any(fragment in p.stem for fragment in args.only)]
    if not record_paths:
        print(f"No record found in {args.records_dir}.", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(record_paths)} records, sections {config.SECTION_ORDER}, "
          f"mode {config.EXTRACTOR_MODE}, agent "
          f"{config.AGENTIC_LLM_MODEL_NAME}, evaluator "
          f"{config.EVALUATOR_LLM_MODEL_NAME}\nguideline context "
          f"{config.BRIGHTON_CONTEXT_ENABLED} from {args.brighton_pdf.name}, "
          f"index in {config.BRIGHTON_KB_PERSIST_DIR}\n"
          f"context window {config.LLM_NUM_CTX} tokens, temperature "
          f"{config.LLM_TEMPERATURE}\n"
          f"-> {args.output_dir}\n",
          flush=True)

    succeeded, failed, durations = [], [], []
    for index, record_path in enumerate(record_paths, start=1):
        record_id = record_path.stem
        print(f"[{index}/{len(record_paths)}] {record_id} ...", flush=True)
        started = time.time()
        try:
            form, audit_log = pipeline.run_pipeline(
                record_id, str(record_path), str(args.brighton_pdf)
            )
            summary = form_to_json_summary(form, audit_log)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(args.output_dir / f"{record_id}_{stamp}.json", "w",
                      encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            with open(args.output_dir / f"{record_id}_{stamp}_audit_log.json", "w",
                      encoding="utf-8") as handle:
                json.dump(audit_log, handle, indent=2, ensure_ascii=False)
            elapsed = time.time() - started
            durations.append(elapsed)
            succeeded.append(record_id)
            print(f"    done in {elapsed:.0f}s", flush=True)
        except Exception:
            failed.append(record_id)
            traceback.print_exc()

    print(f"\n{len(succeeded)} succeeded, {len(failed)} failed", flush=True)
    if durations:
        durations.sort()
        print(f"median per record: {durations[len(durations) // 2]:.0f}s", flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()
