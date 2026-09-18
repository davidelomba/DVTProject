#!/usr/bin/env bash
# Regime a recupero scarso: chunk piccoli e k basso, sulle due modalita'
# che usano l'indice. Riporta config.py al riferimento comunque vada.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py

set_num ()  { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)[0-9]+/\1$2/" "$CFG"; }
set_mode () { sed -i -E "s/^(EXTRACTOR_MODE[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"$1\"/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|EHR_CHUNK_SIZE|EHR_CHUNK_OVERLAP|EHR_RETRIEVER_K)' "$CFG"
}

reference () {
  set_num EHR_CHUNK_SIZE 800
  set_num EHR_CHUNK_OVERLAP 150
  set_num EHR_RETRIEVER_K 5
  set_mode agentic_graph
}
trap reference EXIT INT TERM

scarce () { set_num EHR_CHUNK_SIZE 200; set_num EHR_CHUNK_OVERLAP 40; set_num EHR_RETRIEVER_K 3; }

run_arm () {
  local name=$1 outdir=$2
  echo "===== $name  inizio $(date +%F\ %H:%M:%S) ====="
  show
  "$PY" -c "import config, agents, pipeline"
  "$PY" run_synthetic_records.py --output-dir "$outdir"
  "$PY" evaluate_predictions.py "$outdir"
  echo "===== $name  fine $(date +%F\ %H:%M:%S) ====="
}

# 1 - agentic_graph a recupero scarso
reference; scarce; set_mode agentic_graph
run_arm small_agentic ./output_small_agentic

# 2 - rag a recupero scarso
reference; scarce; set_mode rag
run_arm small_rag ./output_small_rag

reference
echo "===== configurazione finale ====="
show
