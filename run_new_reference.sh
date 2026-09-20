#!/usr/bin/env bash
# Riferimento candidato: parametri di riferimento con due modifiche insieme,
# la query di B2 riscritta e qwen3.6:27b come agente.
# Il confronto e contro ./output, la run di riferimento attuale a 398/400.
# Riporta config.py e pipeline.py com'erano comunque vada.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py
PIPE=pipeline.py
BAK=/tmp/pipeline.py.newref.bak

set_num () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)[0-9]+/\1$2/" "$CFG"; }
set_str () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"$2\"/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|EHR_CHUNK_SIZE|EHR_CHUNK_OVERLAP|EHR_RETRIEVER_K)' "$CFG"
  "$PY" - <<'DIGEST'
import ast, hashlib, config
src = open("pipeline.py", encoding="utf-8").read()
i = src.index("SECTION_QUERIES = {"); j = src.index("\n}", i) + 2
queries = ast.literal_eval(src[i + len("SECTION_QUERIES = "):j].strip())
d = hashlib.sha256()
for key in config.SECTION_ORDER:
    d.update(f"{key}:{queries[key]}\n".encode("utf-8"))
print(f"section_queries_fingerprint all = {d.hexdigest()[:12]}")
DIGEST
}

reference () {
  set_str LLM_MODEL_NAME "llama3:8b-instruct-q4_0"
  set_str AGENTIC_LLM_MODEL_NAME "qwen3.6:27b"
  set_str EXTRACTOR_MODE "agentic_graph"
  set_num EHR_CHUNK_SIZE 800
  set_num EHR_CHUNK_OVERLAP 150
  set_num EHR_RETRIEVER_K 5
}

restore () {
  reference
  if [ -f "$BAK" ]; then cp "$BAK" "$PIPE"; rm -f "$BAK"; fi
}
trap restore EXIT INT TERM

if [ -f "$BAK" ]; then
  echo "!! $BAK esiste gia: controlla pipeline.py prima di rilanciare."
  exit 1
fi
cp "$PIPE" "$BAK"

# Solo la query di B2. A2 e A3_2 restano quelle originali: l'ablazione le ha
# misurate neutre o dannose.
"$PY" - <<'PATCH'
import sys
OLD = '    "B2": "calf pain, swelling, oedema, redness, warmth, absent pulses",'
NEW = ('    "B2": "calf pain or tenderness, leg swelling or pitting oedema, redness, "\n'
       '          "warmth or pain in any extremity, absent pulses in legs or arms",')
src = open("pipeline.py", encoding="utf-8").read()
if src.count(OLD) != 1:
    sys.exit(f"!! attesa una occorrenza della query di B2, trovate {src.count(OLD)}")
open("pipeline.py", "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("query di B2 riscritta")
PATCH

"$PY" -c "import ast; ast.parse(open('pipeline.py', encoding='utf-8').read())"

reference
set_str AGENTIC_LLM_MODEL_NAME "qwen3.6:27b"

echo "===== riferimento candidato  inizio $(date +%F\ %H:%M:%S) ====="
show
"$PY" -c "import config, agents, pipeline"
"$PY" run_synthetic_records.py --output-dir ./output_new_reference
"$PY" evaluate_predictions.py ./output_new_reference
"$PY" compare_runs.py ./output ./output_new_reference
echo "===== fine $(date +%F\ %H:%M:%S) ====="

restore
echo "===== configurazione finale ====="
show
