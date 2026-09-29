#!/bin/bash
# Two agentic_graph runs with intfloat/multilingual-e5-large as the embedding
# model, on the CPU (EMBEDDING_DEVICE), section query set f7ed731f7e7d, Agent 3
# on, each in its own directory:
#   1. output_e5large_cpu         chunks 800/150/5
#   2. output_e5large_cpu_small   chunks 200/40/3
# Before them, a check embeds every record's chunks, the guideline's chunks and
# the section queries with intfloat/multilingual-e5-small on the GPU and on the
# CPU, and reports whether the chunks each query retrieves differ between the
# two devices. The e5-small runs output_queries_v5 and output_queries_v5_small
# computed their embeddings on the GPU.
# The guideline passages are printed and the e5-large store built before the
# first run, so a model that fails to load stops the script there.
# After each run, the calls that filled or overflowed LLM_NUM_CTX are listed.
# config.py is copied before the first run and restored on exit.
#
# Launch from the project root:
#   systemd-run --user --unit=dvte5cpu --working-directory=$HOME/DVTProject \
#     /bin/bash $HOME/DVTProject/run_e5large_cpu.sh

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
    "query fingerprint": (pipeline._query_fingerprint()["all"], "f7ed731f7e7d"),
    "EMBEDDING_MODEL_NAME": (config.EMBEDDING_MODEL_NAME, "intfloat/multilingual-e5-small"),
    "EMBEDDING_DEVICE": (config.EMBEDDING_DEVICE, "cpu"),
}
bad = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
print("pre-flight:", "OK" if not bad else "FAILED")
for line in bad:
    print("  " + line)
sys.exit(1 if bad else 0)
PYCHECK

passages() {
echo "=== guideline passages retrieved by each query, $(grep -m1 '^EMBEDDING_MODEL_NAME' config.py) ==="
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
}

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

SMALL_CHUNKS='s/^EHR_CHUNK_SIZE = 800/EHR_CHUNK_SIZE = 200/;s/^EHR_CHUNK_OVERLAP = 150/EHR_CHUNK_OVERLAP = 40/;s/^EHR_RETRIEVER_K = 5/EHR_RETRIEVER_K = 3/'
LARGE_MODEL='s#^EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"#EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"#'

run() {
    echo "=== run: $1 ==="
    grep -nE '^(EMBEDDING_MODEL_NAME|EHR_CHUNK_SIZE|EHR_CHUNK_OVERLAP|EHR_RETRIEVER_K)' config.py
    $PY run_synthetic_records.py --output-dir "./$1"
    token_check "$1"
}

echo "=== e5-small: GPU against CPU ==="
$PY - <<'PYDEVICE'
# The same inputs embedded on both devices: each record split at 800/150 and
# at 200/40, the guideline split as build_brighton_kb splits it, and the ten
# section queries. For every query the top k chunks are ranked by L2 distance,
# the distance Chroma uses by default, and the two devices' lists compared.
# Agent 1 writes its own search queries in agentic_graph; the section queries
# stand in for them here.
import glob
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from pipeline import SECTION_QUERIES
from rag_setup import get_embeddings, load_brighton_pdf_text
from run_synthetic_records import BRIGHTON_PDF_PATH

models = {}
for device in ("cuda", "cpu"):
    config.EMBEDDING_DEVICE = device
    models[device] = get_embeddings()

queries = list(SECTION_QUERIES.values())
texts = {"guideline": (load_brighton_pdf_text(BRIGHTON_PDF_PATH),
                       config.BRIGHTON_CHUNK_SIZE, config.BRIGHTON_CHUNK_OVERLAP,
                       config.BRIGHTON_RETRIEVER_K)}
for path in sorted(Path("data/synthetic_records").glob("*_v2.txt")):
    record = path.read_text(encoding="utf-8")
    texts[f"{path.stem} 800"] = (record, 800, 150, 5)
    texts[f"{path.stem} 200"] = (record, 200, 40, 3)

compared = differ = 0
largest = 0.0
for name, (text, size, overlap, k) in texts.items():
    chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
    ranked = {}
    for device, model in models.items():
        docs = np.array(model.embed_documents(chunks))
        qs = np.array([model.embed_query(q) for q in queries])
        ranked[device] = [list(np.argsort(np.linalg.norm(docs - q, axis=1))[:k]) for q in qs]
        ranked[device + "_vectors"] = docs
    largest = max(largest, float(np.abs(ranked["cuda_vectors"] - ranked["cpu_vectors"]).max()))
    for section, a, b in zip(SECTION_QUERIES, ranked["cuda"], ranked["cpu"]):
        compared += 1
        if a != b:
            differ += 1
            print(f"  DIFFERENT {name} {section}: GPU {a} CPU {b}")
print(f"{compared} retrievals compared, {differ} different; "
      f"largest difference in a vector component {largest:.2e}")
PYDEVICE

sed -i -e "$LARGE_MODEL" config.py
passages || { echo "e5-large failed to load, runs skipped"; exit 1; }
run output_e5large_cpu

sed -i -e "$SMALL_CHUNKS" config.py
run output_e5large_cpu_small
