#!/usr/bin/env bash
# rag a recupero scarso con estrattore 27B. Unica variabile rispetto a
# output_small_rag, che usa lo stesso regime con estrattore 8B.
# Riporta config.py al riferimento comunque vada.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py

set_num () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)[0-9]+/\1$2/" "$CFG"; }
set_str () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"$2\"/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|EHR_CHUNK_SIZE|EHR_CHUNK_OVERLAP|EHR_RETRIEVER_K)' "$CFG"
}

reference () {
  set_str LLM_MODEL_NAME "llama3:8b-instruct-q4_0"
  set_str EXTRACTOR_MODE "agentic_graph"
  set_num EHR_CHUNK_SIZE 800
  set_num EHR_CHUNK_OVERLAP 150
  set_num EHR_RETRIEVER_K 5
}
trap reference EXIT INT TERM

reference
set_str LLM_MODEL_NAME "qwen3.6:27b"
set_str EXTRACTOR_MODE "rag"
set_num EHR_CHUNK_SIZE 200
set_num EHR_CHUNK_OVERLAP 40
set_num EHR_RETRIEVER_K 3

echo "===== rag 200/40/3 estrattore 27B  inizio $(date +%F\ %H:%M:%S) ====="
show
"$PY" -c "import config, agents, pipeline"
"$PY" run_synthetic_records.py --output-dir ./output_small_rag_27b
"$PY" evaluate_predictions.py ./output_small_rag_27b
"$PY" compare_runs.py ./output_small_rag ./output_small_rag_27b
echo "===== fine $(date +%F\ %H:%M:%S) ====="

reference
echo "===== configurazione finale ====="
show
