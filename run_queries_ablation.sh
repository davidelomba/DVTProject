#!/usr/bin/env bash
# Ablazione delle query di sezione nel regime a recupero selettivo.
# Riscrive le tre query che l'audit ha trovato parziali - A2, A3_2, B2 - e le
# misura su due configurazioni, ciascuna a una variabile rispetto a un braccio
# gia in archivio:
#   A  rag 200/40/3, estrattore 27B      contro output_small_rag_27b  (369)
#   B  agentic_graph 200/40/3            contro output_small_agentic  (382)
# Riporta config.py e pipeline.py com'erano comunque vada.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py
PIPE=pipeline.py
BAK=/tmp/pipeline.py.queries.bak

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
  set_str AGENTIC_LLM_MODEL_NAME "llama3.1:8b-instruct-q4_0"
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

# Un backup gia presente significa che una esecuzione precedente e stata uccisa
# senza eseguire il trap, e pipeline.py potrebbe portare ancora le query nuove.
if [ -f "$BAK" ]; then
  echo "!! $BAK esiste gia: controlla pipeline.py prima di rilanciare."
  exit 1
fi
cp "$PIPE" "$BAK"

# Le tre query dell'audit. Ogni sostituzione deve corrispondere una volta sola.
"$PY" - <<'PATCH'
import sys

NEW = {
 '    "A2": "thrombectomy, surgical procedure related to DVT",':
 '    "A2": "thrombectomy; surgical, endovascular or interventional procedure "\n'
 '          "that confirmed DVT: IVC filter placement, catheter-directed "\n'
 '          "intervention, intraoperative venography",',

 '    "A3_2": "type of imaging study performed: compression ultrasonography, doppler, venography",':
 '    "A3_2": "imaging study performed to look for DVT: compression "\n'
 '            "ultrasonography, Doppler or duplex ultrasound, CT or MR "\n'
 '            "venography, contrast venography, plethysmography or any other "\n'
 '            "modality",',

 '    "B2": "calf pain, swelling, oedema, redness, warmth, absent pulses",':
 '    "B2": "calf pain or tenderness, leg swelling or pitting oedema, redness, "\n'
 '          "warmth or pain in any extremity, absent pulses in legs or arms",',
}

src = open("pipeline.py", encoding="utf-8").read()
for old, new in NEW.items():
    if src.count(old) != 1:
        sys.exit(f"!! attesa una occorrenza, trovate {src.count(old)}:\n{old}")
    src = src.replace(old, new)
open("pipeline.py", "w", encoding="utf-8").write(src)
print("query riscritte: A2, A3_2, B2")
PATCH

"$PY" -c "import ast; ast.parse(open('pipeline.py', encoding='utf-8').read())"

run_arm () {
  local name=$1 outdir=$2 baseline=$3
  echo "===== $name  inizio $(date +%F\ %H:%M:%S) ====="
  show
  "$PY" -c "import config, agents, pipeline"
  "$PY" run_synthetic_records.py --output-dir "$outdir"
  "$PY" evaluate_predictions.py "$outdir"
  "$PY" compare_runs.py "$baseline" "$outdir"
  echo "===== $name  fine $(date +%F\ %H:%M:%S) ====="
}

# --- A: rag a recupero selettivo, estrattore 27B
reference
set_str LLM_MODEL_NAME "qwen3.6:27b"
set_str EXTRACTOR_MODE "rag"
set_num EHR_CHUNK_SIZE 200
set_num EHR_CHUNK_OVERLAP 40
set_num EHR_RETRIEVER_K 3
run_arm rag_27b_queries ./output_small_rag_27b_queries ./output_small_rag_27b

# --- B: agentic a recupero selettivo, agente di riferimento
reference
set_num EHR_CHUNK_SIZE 200
set_num EHR_CHUNK_OVERLAP 40
set_num EHR_RETRIEVER_K 3
run_arm agentic_queries ./output_small_agentic_queries ./output_small_agentic

restore
echo "===== configurazione finale ====="
show
