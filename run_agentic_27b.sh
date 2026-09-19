#!/usr/bin/env bash
# agentic_graph a recupero selettivo con agente 27B. Unica variabile rispetto a
# output_small_agentic, che usa lo stesso regime con llama3.1:8b come Agente 1.
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
  set_str AGENTIC_LLM_MODEL_NAME "llama3.1:8b-instruct-q4_0"
  set_str EXTRACTOR_MODE "agentic_graph"
  set_num EHR_CHUNK_SIZE 800
  set_num EHR_CHUNK_OVERLAP 150
  set_num EHR_RETRIEVER_K 5
}
trap reference EXIT INT TERM

# Il modello dell'agente deve saper chiamare uno strumento. Verificato
# eseguendo una tool call con lo stesso stack che usa la pipeline.
"$PY" - <<'PROBE'
import sys
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def probe(query: str) -> str:
    """Cerca informazioni nel documento clinico."""
    return "ok"

try:
    reply = ChatOllama(model="qwen3.6:27b", temperature=0).bind_tools([probe]).invoke(
        "Cerca nel documento usando lo strumento probe con query: trombosi."
    )
except Exception as exc:
    sys.exit(f"!! qwen3.6:27b non utilizzabile come agente: {type(exc).__name__}: {exc}")
if not reply.tool_calls:
    sys.exit("!! qwen3.6:27b non ha emesso alcuna tool call.")
print(f"tool calling verificato: {reply.tool_calls}")
PROBE

reference
set_str AGENTIC_LLM_MODEL_NAME "qwen3.6:27b"
set_num EHR_CHUNK_SIZE 200
set_num EHR_CHUNK_OVERLAP 40
set_num EHR_RETRIEVER_K 3

echo "===== agentic 200/40/3, agente 27B  inizio $(date +%F\ %H:%M:%S) ====="
show
"$PY" -c "import config, agents, pipeline"
"$PY" run_synthetic_records.py --output-dir ./output_small_agentic_27b
"$PY" evaluate_predictions.py ./output_small_agentic_27b
"$PY" compare_runs.py ./output_small_agentic ./output_small_agentic_27b
echo "===== fine $(date +%F\ %H:%M:%S) ====="

reference
echo "===== configurazione finale ====="
show
