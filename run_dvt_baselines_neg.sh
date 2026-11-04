#!/usr/bin/env bash
# The four DVT baselines with an extractor, with config.EXTRACTOR_NEGATION_RULE
# on, at the reference parameters: 1024-token cap and 4096-token window, chunks
# of 800 characters, overlap 150, k 5, temperature 0, the hints and queries in
# config.py and pipeline.py, the verifier on. config.py is not edited: each run
# sets EXTRACTOR_MODE, LLM_MODEL_NAME, LLM_NUM_PREDICT, LLM_NUM_CTX and
# EXTRACTOR_NEGATION_RULE in memory before calling run_synthetic_records.main.
#
#   full_text_8b    full_text, extractor llama3:8b-instruct-q4_0
#   rag_8b          rag,       extractor llama3:8b-instruct-q4_0
#   full_text_27b   full_text, extractor Agent 1's tag (qwen3.6:27b)
#   rag_27b         rag,       extractor Agent 1's tag (qwen3.6:27b)
#
# Output in ./output_neg_<arm>, one after the other, then evaluate_predictions.py
# and evaluate_verifier.py on each run and compare_runs.py against
# ./output_a32_ref.
#
# Optional, through environment variables:
#   ARMS="..."   the arms to run, all four by default
#   EVAL_ONLY=1  skip the runs and score the directories already on disk
#   PYTHON=...   the interpreter, python by default

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PYTHON="${PYTHON:-python}"
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

EXPECTED_HINTS="a78dc29e962f"
SMALL_EXTRACTOR="llama3:8b-instruct-q4_0"
AGENT_TAG="$(grep -E '^AGENTIC_LLM_MODEL_NAME' config.py | sed -E 's/^[^"]*"([^"]*)".*/\1/')"
EVALUATOR_TAG="$(grep -E '^EVALUATOR_LLM_MODEL_NAME' config.py | sed -E 's/^[^"]*"([^"]*)".*/\1/')"
ARMS="${ARMS:-full_text_8b rag_8b full_text_27b rag_27b}"

arm_spec() {
    case "$1" in
        full_text_8b)  echo "full_text $SMALL_EXTRACTOR" ;;
        rag_8b)        echo "rag $SMALL_EXTRACTOR" ;;
        full_text_27b) echo "full_text $AGENT_TAG" ;;
        rag_27b)       echo "rag $AGENT_TAG" ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo; echo "=== $(stamp) $*"; }

# Python statements that set one arm's configuration in memory
setup_of() {
    local spec mode extractor
    spec="$(arm_spec "$1")"
    mode="${spec%% *}"
    extractor="${spec#* }"
    printf 'import config\nconfig.EXTRACTOR_MODE = "%s"\nconfig.LLM_MODEL_NAME = "%s"\nconfig.LLM_NUM_PREDICT = 1024\nconfig.LLM_NUM_CTX = 4096\nconfig.EXTRACTOR_NEGATION_RULE = True\n' "$mode" "$extractor"
}

if [ "${EVAL_ONLY:-0}" != "1" ]; then
    say "preflight"
    echo "host $(hostname), $("$PYTHON" --version 2>&1), $(ollama --version 2>&1 | tail -1)"
    for tag in "$EVALUATOR_TAG" "$AGENT_TAG" "$SMALL_EXTRACTOR"; do
        if ! ollama list | awk '{print $1}' | grep -qx "$tag"; then
            echo "The model $tag is not pulled." >&2
            exit 1
        fi
    done
    for arm in $ARMS; do
        out="./output_neg_${arm}"
        if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
            echo "$out already holds files; move it first." >&2
            exit 1
        fi
        ARM="$arm" EXPECTED_HINTS="$EXPECTED_HINTS" SETUP="$(setup_of "$arm")" "$PYTHON" - <<'EOF'
import os, sys
exec(os.environ["SETUP"])
import agents, config, pipeline, verifier
hints = pipeline._hint_fingerprint()["all"]
queries = pipeline._query_fingerprint()["all"]
prompt = agents.baseline_extractor_prompt()
print(f"{os.environ['ARM']}: mode {config.EXTRACTOR_MODE}, extractor {config.LLM_MODEL_NAME}, "
      f"evaluator {config.EVALUATOR_LLM_MODEL_NAME}, verifier "
      f"{verifier.verifier_settings().get('model')} {list(config.VERIFIER_INPUTS)}, "
      f"num_predict {config.LLM_NUM_PREDICT}, num_ctx {config.LLM_NUM_CTX}, chunks "
      f"{config.EHR_CHUNK_SIZE}/{config.EHR_CHUNK_OVERLAP}/{config.EHR_RETRIEVER_K}, hints {hints}, "
      f"queries {queries}, negation rule {config.EXTRACTOR_NEGATION_RULE}, rule in prompt "
      f"{agents.NEGATION_RULE in prompt}")
problems = []
if hints != os.environ["EXPECTED_HINTS"]:
    problems.append(f"hint fingerprint {hints}, expected {os.environ['EXPECTED_HINTS']}")
if (config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K) != (800, 150, 5):
    problems.append("chunks are not 800/150/5")
if config.LLM_TEMPERATURE != 0.0:
    problems.append(f"temperature {config.LLM_TEMPERATURE}")
if not config.VERIFIER_ENABLED or not config.BRIGHTON_CONTEXT_ENABLED:
    problems.append("verifier or guideline context off")
if agents.NEGATION_RULE not in prompt:
    problems.append("negation rule missing from the extractor prompt")
if problems:
    print("STOP: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
EOF
    done

    for arm in $ARMS; do
        out="./output_neg_${arm}"
        say "run $out"
        SETUP="$(setup_of "$arm")" "$PYTHON" - --output-dir "$out" <<'EOF' 2>&1 | tee "logs/$(basename "$out").log"
import os, sys
exec(os.environ["SETUP"])
sys.argv = ["run_synthetic_records.py"] + sys.argv[1:]
import run_synthetic_records
run_synthetic_records.main()
EOF
    done
fi

for arm in $ARMS; do
    out="./output_neg_${arm}"
    say "score $out"
    "$PYTHON" evaluate_predictions.py "$out"
    "$PYTHON" evaluate_verifier.py "$out"
    "$PYTHON" compare_runs.py ./output_a32_ref "$out"
done

say "done; reports in reports/, logs in logs/"
