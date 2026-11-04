#!/usr/bin/env bash
# The rag baseline on the DVT corpus with qwen3.6:27b as extractor, at the
# reference parameters: chunks of 800 characters, overlap 150, k 5, 1024-token
# cap and 4096-token window, the hints and queries in config.py and pipeline.py,
# the verifier on. config.py is not edited: the run sets EXTRACTOR_MODE,
# LLM_MODEL_NAME, LLM_NUM_PREDICT and LLM_NUM_CTX in memory before calling
# run_synthetic_records.main. Output in ./output_rag_27b, then
# evaluate_predictions.py, evaluate_verifier.py and compare_runs.py against
# ./output_a32_ref.
#
# Optional, through environment variables:
#   EVAL_ONLY=1  skip the run and score the directory already on disk
#   PYTHON=...   the interpreter, python by default

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PYTHON="${PYTHON:-python}"
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

OUT=./output_rag_27b
EXPECTED_HINTS="a78dc29e962f"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo; echo "=== $(stamp) $*"; }

SETUP='
import config
config.EXTRACTOR_MODE = "rag"
config.LLM_MODEL_NAME = config.AGENTIC_LLM_MODEL_NAME
config.LLM_NUM_PREDICT = 1024
config.LLM_NUM_CTX = 4096
'

if [ "${EVAL_ONLY:-0}" != "1" ]; then
    say "preflight"
    echo "host $(hostname), $("$PYTHON" --version 2>&1), $(ollama --version 2>&1 | tail -1)"
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
        echo "$OUT already holds files; move it first." >&2
        exit 1
    fi
    EXPECTED_HINTS="$EXPECTED_HINTS" SETUP="$SETUP" "$PYTHON" - <<'EOF'
import os, sys
exec(os.environ["SETUP"])
import config, pipeline, verifier
hints = pipeline._hint_fingerprint()["all"]
queries = pipeline._query_fingerprint()["all"]
print(f"mode {config.EXTRACTOR_MODE}, extractor {config.LLM_MODEL_NAME}, evaluator "
      f"{config.EVALUATOR_LLM_MODEL_NAME}, verifier {verifier.verifier_settings().get('model')} "
      f"{list(config.VERIFIER_INPUTS)}, num_predict {config.LLM_NUM_PREDICT}, num_ctx "
      f"{config.LLM_NUM_CTX}, chunks {config.EHR_CHUNK_SIZE}/{config.EHR_CHUNK_OVERLAP}/"
      f"{config.EHR_RETRIEVER_K}, context {config.BRIGHTON_CONTEXT_ENABLED}, hints {hints}, "
      f"queries {queries}, temperature {config.LLM_TEMPERATURE}")
problems = []
if hints != os.environ["EXPECTED_HINTS"]:
    problems.append(f"hint fingerprint {hints}, expected {os.environ['EXPECTED_HINTS']}")
if (config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K) != (800, 150, 5):
    problems.append("chunks are not 800/150/5")
if config.LLM_TEMPERATURE != 0.0:
    problems.append(f"temperature {config.LLM_TEMPERATURE}")
if not config.VERIFIER_ENABLED or not config.BRIGHTON_CONTEXT_ENABLED:
    problems.append("verifier or guideline context off")
if problems:
    print("STOP: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
EOF

    say "run $OUT"
    SETUP="$SETUP" "$PYTHON" - --output-dir "$OUT" <<'EOF' 2>&1 | tee "logs/$(basename "$OUT").log"
import os, sys
exec(os.environ["SETUP"])
sys.argv = ["run_synthetic_records.py"] + sys.argv[1:]
import run_synthetic_records
run_synthetic_records.main()
EOF
fi

say "score $OUT"
"$PYTHON" evaluate_predictions.py "$OUT"
"$PYTHON" evaluate_verifier.py "$OUT"
"$PYTHON" compare_runs.py ./output_a32_ref "$OUT"

say "done; reports in reports/, log in logs/"
