#!/usr/bin/env bash
# Valutatore 8B con keyword e details riaccesi. Riporta config.py al riferimento.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py

set_eval ()  { sed -i -E "s/^(EVALUATOR_LLM_MODEL_NAME[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"$1\"/" "$CFG"; }
set_gate ()  { sed -i -E "s/^([[:space:]]*\"$1\":[[:space:]]*)[A-Za-z]+/\1$2/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|SECTION_HINTS_DISABLED)' "$CFG"
  grep -E '^[[:space:]]*"(keyword|details|absent_pulses)":' "$CFG"
}

reference () { set_eval "qwen3.6:27b"; set_gate keyword False; set_gate details False; }
trap reference EXIT INT TERM

set_eval "llama3:8b-instruct-q4_0"
set_gate keyword True
set_gate details True

echo "===== 8b_gates  inizio $(date +%F\ %H:%M:%S) ====="
show
"$PY" -c "import config, agents, pipeline"
"$PY" run_synthetic_records.py --output-dir ./output_8b_gates
"$PY" evaluate_predictions.py ./output_8b_gates
echo "===== 8b_gates  fine $(date +%F\ %H:%M:%S) ====="

reference
echo "===== configurazione finale ====="
show
