#!/usr/bin/env bash
# The verifier on the DVT baselines: replay_verifier.py with qwen3.6:27b, the
# variants answer and reasoning, on the stored answers of
#
#   output_full_text              full_text, extractor llama3:8b-instruct
#   output_rag_nogate             rag, extractor llama3:8b-instruct
#   output_full_text_27b_nogate   full_text, extractor qwen3.6:27b
#   output_a32_raw                raw_record
#
# output_rag_nogate and output_full_text_27b_nogate are copies of output_rag and
# output_full_text_27b with X restored, on 3 and 2 records, to the answer Agent 2
# wrote before the keyword gate; _run_config.reconstructed says so. Each replay
# writes <run>_judge_qwen, then evaluate_verifier.py scores it. The replays go
# one after the other, never two at once.
#
# Optional, through environment variables:
#   RUNS="..."   the source directories, the four above by default
#   EVAL_ONLY=1  skip the replays and score the directories already on disk
#   PYTHON=...   the interpreter, python by default

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PYTHON="${PYTHON:-python}"
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

JUDGE="qwen3.6:27b"
RUNS="${RUNS:-output_full_text output_rag_nogate output_full_text_27b_nogate output_a32_raw}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo; echo "=== $(stamp) $*"; }

if [ "${EVAL_ONLY:-0}" != "1" ]; then
    say "preflight"
    echo "host $(hostname), $("$PYTHON" --version 2>&1), $(ollama --version 2>&1 | tail -1)"
    if ! ollama list | awk '{print $1}' | grep -qx "$JUDGE"; then
        echo "The model $JUDGE is not pulled." >&2
        exit 1
    fi
    for run in $RUNS; do
        if [ ! -d "$run" ]; then
            echo "$run does not exist." >&2
            exit 1
        fi
        out="${run}_judge_qwen"
        if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
            echo "$out already holds files; move it first." >&2
            exit 1
        fi
        echo "$run: $(ls "$run" | grep -c '_audit_log.json') audit logs"
    done

    for run in $RUNS; do
        say "replay $run"
        "$PYTHON" replay_verifier.py "./$run" --model "$JUDGE" \
            --inputs answer reasoning --output-dir "./${run}_judge_qwen" \
            2>&1 | tee "logs/${run}_judge_qwen.log"
    done
fi

for run in $RUNS; do
    say "score ${run}_judge_qwen"
    "$PYTHON" evaluate_verifier.py "./${run}_judge_qwen"
done

say "done; reports in reports/, logs in logs/"
