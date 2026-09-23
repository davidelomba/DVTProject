"""
Runs the pipeline over the myocarditis cases, in agentic_graph mode.

Same code as the DVT runs: pipeline.run_pipeline, both agents, the verbatim
evidence of agentic_graph. What differs is the schema, the two section queries
and the terms naming the condition in the prompts. The deterministic layer is
off, not translated: the gates and the cross-section rules encode the DVT form,
and a myocarditis run measures the architecture without them.

models_myo is installed as `models` before pipeline is imported, because
pipeline and criteria_rules bind SECTION_MODELS at import time.

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
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

MYO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MYO_DIR.parent
sys.path.insert(0, str(MYO_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import models_myo                                    # noqa: E402
sys.modules["models"] = models_myo

import config                                        # noqa: E402
import prompts_myo                                   # noqa: E402

# One query per section, written from the option list rather than from the
# finding expected: on the DVT form the queries written the other way covered
# only part of their section and failed on every `other` branch.
SECTION_QUERIES = {
    "E": "electrocardiogram ECG findings: rhythm, atrial or ventricular "
         "arrhythmia, tachycardia, atrial fibrillation, AV block, bundle branch "
         "block, conduction delay, ST segment or T wave abnormality, Q waves, "
         "low voltage, premature atrial or ventricular contractions, ambulatory "
         "monitoring, normal ECG, ECG not performed",
    "F": "echocardiogram ECHO findings: ejection fraction, left or right "
         "ventricular function, segmental wall motion, global systolic or "
         "diastolic function, ventricular dilation, wall thickness, pericardial "
         "effusion or inflammation, normal echocardiogram, ECHO not performed",
}


def apply_domain(guideline: bool = False):
    """Points the shared modules at the myocarditis sections and prompts.

    Args:
        guideline: whether Agent 2 receives the context retrieved from the
            myocarditis paper.
    """

    config.SECTION_ORDER = ["E", "F"]
    config.EXTRACTOR_MODE = "agentic_graph"
    config.BRIGHTON_CONTEXT_ENABLED = guideline
    # Its own store: build_brighton_kb reloads an index already on disk without
    # reading the text it was handed, so the two papers need two directories.
    config.BRIGHTON_KB_PERSIST_DIR = str(MYO_DIR / "vectorstores" / "chroma_brighton_myo")
    config.GUIDELINE_ANCHORS_ENABLED = False
    config.GUIDELINE_ANCHORS = {}
    config.SECTION_DESCRIPTIONS_ENABLED = False
    config.SECTION_HINTS = {}
    config.SECTION_HINTS_DISABLED = set()
    config.SECTION_KEYWORD_GATES = {}
    config.CROSS_SECTION_RULES = []
    config.SECTION_GATES_ENABLED = {
        "keyword": False, "details": False, "absent_pulses": False,
    }

    import agents
    import rag_setup
    import pipeline

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
    pipeline.SECTION_QUERIES = SECTION_QUERIES

    from langchain_core.tools.retriever import create_retriever_tool

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
        "--brighton-pdf", type=Path, default=MYO_DIR / "main.pdf",
        help="the myocarditis case definition; indexed even when the context "
             "is off, and into myo/vectorstores rather than the shared store",
    )
    args = parser.parse_args()

    pipeline = apply_domain(guideline=args.guideline)
    from aggregation import form_to_json_summary

    record_paths = sorted(args.records_dir.glob("*.txt"))
    if args.only:
        record_paths = [p for p in record_paths
                        if any(fragment in p.stem for fragment in args.only)]
    if not record_paths:
        print(f"Nessun record in {args.records_dir}.", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(record_paths)} record, sezioni {config.SECTION_ORDER}, "
          f"modalita {config.EXTRACTOR_MODE}, agente "
          f"{config.AGENTIC_LLM_MODEL_NAME}, valutatore "
          f"{config.EVALUATOR_LLM_MODEL_NAME}\ncontesto linea guida "
          f"{config.BRIGHTON_CONTEXT_ENABLED} da {args.brighton_pdf.name}, "
          f"indice in {config.BRIGHTON_KB_PERSIST_DIR}\n-> {args.output_dir}\n",
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
            summary = form_to_json_summary(form)
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
            print(f"    fatto in {elapsed:.0f}s", flush=True)
        except Exception:
            failed.append(record_id)
            traceback.print_exc()

    print(f"\n{len(succeeded)} riusciti, {len(failed)} falliti", flush=True)
    if durations:
        durations.sort()
        print(f"mediana per record {durations[len(durations) // 2]:.0f}s", flush=True)
    if failed:
        print("falliti: " + ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()
