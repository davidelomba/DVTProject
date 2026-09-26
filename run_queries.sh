#!/bin/bash
# Two agentic_graph runs with the section queries rewritten from the printed
# questionnaire (section_queries_fingerprint e0c31c90d6ce), Agent 3 on, one
# after the other, each in its own directory:
#   1. output_queries_v3         the reference configuration, chunks 800/150/5
#   2. output_queries_v3_small   EHR chunks 200/40, retriever k 3
# Before the runs, the guideline passages each query retrieves are printed.
# After each run, the calls that filled or overflowed LLM_NUM_CTX are listed
# from agent2_tokens.
# config.py is copied before the first run and restored on exit, also when a
# run fails or the script is stopped.
#
# Launch from the project root:
#   systemd-run --user --unit=dvtqueries --working-directory=$HOME/DVTProject \
#     /bin/bash $HOME/DVTProject/run_queries.sh

cd "$HOME/DVTProject" || exit 1
PY="$HOME/DVTProject/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
rm -f __pycache__/config.*.pyc __pycache__/pipeline.*.pyc

cp config.py /tmp/config.py.bak
trap 'cp /tmp/config.py.bak config.py; echo "config.py restored"' EXIT

# Pre-flight: stop before any run if the working copy is not the one expected.
$PY - <<'PYCHECK' || exit 1
import sys
import config
import pipeline
checks = {
    "EVALUATOR_LLM_MODEL_NAME": (config.EVALUATOR_LLM_MODEL_NAME, "qwen3.6:27b"),
    "AGENTIC_LLM_MODEL_NAME": (config.AGENTIC_LLM_MODEL_NAME, "qwen3.6:27b"),
    "EXTRACTOR_MODE": (config.EXTRACTOR_MODE, "agentic_graph"),
    "SECTION_DESCRIPTIONS_ENABLED": (config.SECTION_DESCRIPTIONS_ENABLED, False),
    "GUIDELINE_ANCHORS_ENABLED": (config.GUIDELINE_ANCHORS_ENABLED, False),
    "BRIGHTON_CONTEXT_ENABLED": (config.BRIGHTON_CONTEXT_ENABLED, True),
    "EHR chunks": ((config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K), (800, 150, 5)),
    "LLM_NUM_CTX": (config.LLM_NUM_CTX, 4096),
    "CONFIDENCE_ENABLED": (config.CONFIDENCE_ENABLED, True),
    "hint fingerprint": (pipeline._hint_fingerprint()["all"], "0d4a00c11404"),
    "query fingerprint": (pipeline._query_fingerprint()["all"], "e0c31c90d6ce"),
}
bad = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
print("pre-flight:", "OK" if not bad else "FAILED")
for line in bad:
    print("  " + line)
sys.exit(1 if bad else 0)
PYCHECK

echo "=== guideline passages retrieved by each query ==="
$PY - <<'PYCONTEXT'
# The first 90 characters of each of the BRIGHTON_RETRIEVER_K chunks every
# section query retrieves from the guideline, the same for all records.
import config
from pipeline import SECTION_QUERIES
from rag_setup import build_brighton_kb, get_embeddings, load_brighton_pdf_text
from run_synthetic_records import BRIGHTON_PDF_PATH

kb = build_brighton_kb(load_brighton_pdf_text(BRIGHTON_PDF_PATH), embeddings=get_embeddings())
for section_key, query in SECTION_QUERIES.items():
    print(f"== {section_key}")
    for doc in kb.similarity_search(query, k=config.BRIGHTON_RETRIEVER_K):
        print("   ", doc.page_content[:90].replace("\n", " "))
PYCONTEXT

token_check() {
$PY - "$1" <<'PYAFTER'
# Every Agent 2 call of the run, from agent2_tokens in the audit logs. A call
# whose prompt and output together reach LLM_NUM_CTX filled the window, and a
# retry logged with a shorter prompt than the attempt before it was truncated,
# since each retry only adds to the prompt.
import glob
import json
import os
import sys
import config

run_dir = sys.argv[1]
calls = over = retried = 0
for path in sorted(glob.glob(f"{run_dir}/*_audit_log.json")):
    audit = json.load(open(path))
    record = os.path.basename(path).split("_2026")[0]
    for section_key, entry in audit.items():
        if section_key.startswith("_"):
            continue
        counts = entry.get("agent2_tokens") or []
        if len(counts) > 1:
            retried += 1
            print(f"  retried: {record} {section_key}, {len(counts)} attempts")
        previous = None
        for n, count in enumerate(counts, 1):
            if count.get("prompt") is None or count.get("output") is None:
                continue
            calls += 1
            total = count["prompt"] + count["output"]
            if total >= config.LLM_NUM_CTX:
                over += 1
                print(f"  FULL {record} {section_key} attempt {n}: "
                      f"{count['prompt']} + {count['output']} = {total}")
            elif previous is not None and count["prompt"] < previous:
                over += 1
                print(f"  TRUNCATED {record} {section_key} attempt {n}: "
                      f"prompt {count['prompt']} after {previous}")
            previous = count["prompt"]
print(f"{run_dir}: {calls} calls, {retried} sections retried, "
      f"{over} calls that filled or overflowed LLM_NUM_CTX")
PYAFTER
}

echo "=== run 1: output_queries_v3 ==="
$PY run_synthetic_records.py --output-dir ./output_queries_v3
token_check output_queries_v3

echo "=== run 2: output_queries_v3_small ==="
sed -i -e 's/^EHR_CHUNK_SIZE = 800/EHR_CHUNK_SIZE = 200/' \
       -e 's/^EHR_CHUNK_OVERLAP = 150/EHR_CHUNK_OVERLAP = 40/' \
       -e 's/^EHR_RETRIEVER_K = 5/EHR_RETRIEVER_K = 3/' config.py
grep -nE '^EHR_(CHUNK_SIZE|CHUNK_OVERLAP|RETRIEVER_K)' config.py
$PY run_synthetic_records.py --output-dir ./output_queries_v3_small
token_check output_queries_v3_small
