#!/usr/bin/env bash
# Le ancore alla linea guida, in due regimi di recupero.
#
#   1  800/150/5 con le ancore          -> ./output_anchors
#   2  200/40/3  senza ancore           -> ./output_small_control
#   3  200/40/3  con le ancore          -> ./output_small_anchors
#
# La prima si confronta con ./output_new_reference, 398/400, che porta la
# configurazione attuale. La terza si confronta con la seconda e non con un
# braccio precedente: output_small_agentic_27b non porta
# section_queries_fingerprint, quindi usa l'insieme di query anteriore alla
# riscrittura di B2 e varierebbe due cose insieme.
#
# Riporta config.py alla configurazione di riferimento comunque vada.
set -euo pipefail

PROJECT="$HOME/DVTProject"
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
CFG=config.py

set_num  () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)[0-9]+/\1$2/" "$CFG"; }
set_bool () { sed -i -E "s/^($1[[:space:]]*=[[:space:]]*)(True|False)/\1$2/" "$CFG"; }

show () {
  grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|EXTRACTOR_MODE|BRIGHTON_CONTEXT_ENABLED|SECTION_DESCRIPTIONS_ENABLED|SECTION_HINTS_ENABLED|GUIDELINE_ANCHORS_ENABLED|EHR_CHUNK_SIZE|EHR_CHUNK_OVERLAP|EHR_RETRIEVER_K)' "$CFG"
}

# La provenienza che va letta prima del punteggio.
provenance () {
  "$PY" - "$1" <<'CHECK'
import glob, json, sys
files = glob.glob(sys.argv[1] + "/*_audit_log.json")
cfg = json.load(open(max(files), encoding="utf-8"))["_run_config"]
print("  ancore  ", cfg["guideline_anchors_enabled"],
      cfg["guideline_anchors_fingerprint"]["all"], "(atteso 63c410d9cd34)")
print("  query   ", cfg["section_queries_fingerprint"]["all"], "(atteso bfd9536a31fe)")
print("  hint    ", cfg["section_hints_fingerprint"]["all"])
print("  recupero", cfg["retrieval"])
print("  ambiente", cfg["environment"])
CHECK
}

reference () {
  set_bool GUIDELINE_ANCHORS_ENABLED False
  set_num  EHR_CHUNK_SIZE 800
  set_num  EHR_CHUNK_OVERLAP 150
  set_num  EHR_RETRIEVER_K 5
}
trap reference EXIT INT TERM

for d in output_anchors output_small_control output_small_anchors; do
  if [ -d "$d" ]; then
    echo "!! $d esiste gia: spostala o cancellala, due bracci in una cartella si nascondono a vicenda."
    exit 1
  fi
done

# ---------------------------------------------------------------- 1
echo "=== 1/3  800/150/5 con le ancore ==="
reference
set_bool GUIDELINE_ANCHORS_ENABLED True
show
"$PY" run_synthetic_records.py --output-dir ./output_anchors
provenance ./output_anchors
"$PY" evaluate_predictions.py ./output_anchors --no-matrices
"$PY" compare_runs.py ./output_new_reference ./output_anchors

# ---------------------------------------------------------------- 2
echo "=== 2/3  200/40/3 senza ancore, il controllo ==="
reference
set_num EHR_CHUNK_SIZE 200
set_num EHR_CHUNK_OVERLAP 40
set_num EHR_RETRIEVER_K 3
show
"$PY" run_synthetic_records.py --output-dir ./output_small_control
provenance ./output_small_control
"$PY" evaluate_predictions.py ./output_small_control --no-matrices

# ---------------------------------------------------------------- 3
echo "=== 3/3  200/40/3 con le ancore ==="
set_bool GUIDELINE_ANCHORS_ENABLED True
show
"$PY" run_synthetic_records.py --output-dir ./output_small_anchors
provenance ./output_small_anchors
"$PY" evaluate_predictions.py ./output_small_anchors --no-matrices
"$PY" compare_runs.py ./output_small_control ./output_small_anchors

echo "=== fine. config.py torna al riferimento ==="
