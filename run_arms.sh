#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py

set_eval ()  { sed -i -E "s/^(EVALUATOR_LLM_MODEL_NAME[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"$1\"/" "$CFG"; }
set_hints () { sed -i -E "s/^(SECTION_HINTS_ENABLED[[:space:]]*=[[:space:]]*)[A-Za-z]+/\1$1/" "$CFG"; }
set_desc ()  { sed -i -E "s/^(SECTION_DESCRIPTIONS_ENABLED[[:space:]]*=[[:space:]]*)[A-Za-z]+/\1$1/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|SECTION_HINTS_DISABLED)' "$CFG"
}

reference () { set_eval "qwen3.6:27b"; set_hints True; set_desc False; }
trap reference EXIT INT TERM

run_arm () {
  local name=$1 outdir=$2
  echo "===== $name  inizio $(date +%F\ %H:%M:%S) ====="
  show
  "$PY" -c "import config, agents, pipeline"
  "$PY" run_synthetic_records.py --output-dir "$outdir"
  "$PY" evaluate_predictions.py "$outdir"
  echo "===== $name  fine $(date +%F\ %H:%M:%S) ====="
}

reference; set_eval "llama3:8b-instruct-q4_0"
run_arm 8b_eval ./output_8b_evaluator

reference; set_eval "llama3:8b-instruct-q4_0"; set_hints False
run_arm 8b_nohints ./output_8b_nohints

reference; set_desc True
run_arm descriptions ./output_descriptions

reference; set_hints False
run_arm qwen_nohints ./output_qwen_nohints

reference
echo "===== configurazione finale ====="
show
