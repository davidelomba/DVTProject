#!/usr/bin/env bash
# Three myocarditis runs at temperature 0.5, every other setting fixed, then
# their scoring: myo/evaluate_myo.py and evaluate_verifier.py on each run,
# agreement_runs.py across the three.
#
# Before the first run it checks that Ollama answers, that the evaluator model
# is pulled, and that the hints, the queries and the temperature are the ones
# the runs are meant to carry. After each run it reads the audit logs back and
# prints what a run has to show: temperature, fingerprints, verifier results,
# failed sections. The runs go one after the other, never two at once.
#
# Optional, through environment variables:
#   T0=1         also a run at temperature 0, which separates the temperature
#                from the other changes of this arm
#   STRIPPED=1   also the DVT corpus stripped of hints, context, headings and
#                rules (myo/run_dvt_stripped.py), 46 errors to score the
#                verifier on
#   RUN_TAG=...  names the output directories myo/output_<RUN_TAG>_r1 to _r3
#                and myo/output_<RUN_TAG>_t0; t05 by default
#   EVAL_ONLY=1  skip the runs and score the directories already on disk
#   FORCE=1      run into an output directory that already holds files
#   PYTHON=...   the interpreter, python by default
#
# Usage, from the project root on mari:
#   mkdir -p logs && nohup bash run_myo_t05.sh > logs/run_myo_t05.log 2>&1 &
#   mkdir -p logs && T0=1 STRIPPED=1 nohup bash run_myo_t05.sh > logs/run_myo_t05.log 2>&1 &

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PYTHON="${PYTHON:-python}"
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

EXPECTED_HINTS="7d4be67fec0f"
EXPECTED_QUERIES="45a59c4e6d03"
GT="./myo/data/records"
RUN_TAG="${RUN_TAG:-t05}"
RUNS=(./myo/output_${RUN_TAG}_r1 ./myo/output_${RUN_TAG}_r2 ./myo/output_${RUN_TAG}_r3)
T0_DIR=./myo/output_${RUN_TAG}_t0
STRIPPED_DIR=./output_stripped_verifier

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo; echo "=== $(stamp) $*"; }

# ---------------------------------------------------------------- preflight

preflight() {
    say "preflight"
    echo "host $(hostname), $("$PYTHON" --version 2>&1), $(ollama --version 2>&1 | tail -1)"

    if ! curl -sf "${OLLAMA_HOST:-http://localhost:11434}/api/version" > /dev/null; then
        echo "Ollama does not answer at ${OLLAMA_HOST:-http://localhost:11434}." >&2
        exit 1
    fi

    local evaluator
    evaluator=$(grep -E '^EVALUATOR_LLM_MODEL_NAME' config.py | sed -E 's/.*"(.*)".*/\1/')
    if ! ollama list | awk '{print $1}' | grep -qx "$evaluator"; then
        echo "The evaluator model $evaluator is not pulled." >&2
        exit 1
    fi

    echo "config.py:"
    grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|LLM_NUM_PREDICT|LLM_NUM_CTX|EXTRACTOR_MODE|VERIFIER_ENABLED|VERIFIER_INPUTS|VERIFIER_TEMPERATURE|VERIFIER_NUM_CTX)' config.py

    # What a myocarditis run will send, read through the same code it runs
    EXPECTED_HINTS="$EXPECTED_HINTS" EXPECTED_QUERIES="$EXPECTED_QUERIES" "$PYTHON" - <<'EOF'
import os, sys
sys.path.insert(0, "myo")
import myo_schema  # noqa: F401
import agents, config, pipeline, run_myo
run_myo.apply_domain(guideline=True)
hints = pipeline._hint_fingerprint()["all"]
queries = pipeline._query_fingerprint()["all"]
print(f"sections {config.SECTION_ORDER}, temperature {config.LLM_TEMPERATURE}, "
      f"hints {hints}, queries {queries}, verifier {config.VERIFIER_ENABLED} "
      f"{list(config.VERIFIER_INPUTS)}, empty-answer sections "
      f"{sorted(agents.EMPTY_ANSWER_SECTIONS)}")
problems = []
if agents.EMPTY_ANSWER_SECTIONS:
    problems.append(f"empty-answer sections {sorted(agents.EMPTY_ANSWER_SECTIONS)}, expected none")
if hints != os.environ["EXPECTED_HINTS"]:
    problems.append(f"hint fingerprint {hints}, expected {os.environ['EXPECTED_HINTS']}")
if queries != os.environ["EXPECTED_QUERIES"]:
    problems.append(f"query fingerprint {queries}, expected {os.environ['EXPECTED_QUERIES']}")
if config.LLM_TEMPERATURE != 0.5:
    problems.append(f"temperature {config.LLM_TEMPERATURE}, expected 0.5")
if not config.VERIFIER_ENABLED:
    problems.append("VERIFIER_ENABLED is False")
if problems:
    print("STOP: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
EOF
}

# ------------------------------------------------------------- run helpers

check_empty() {
    local dir="$1"
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "$dir already holds files; move it or set FORCE=1." >&2
        exit 1
    fi
}

# Reads a run's audit logs back and prints what it has to show
check_run() {
    local dir="$1"
    "$PYTHON" - "$dir" <<'EOF'
import json, sys
from collections import Counter
from pathlib import Path

logs = sorted(Path(sys.argv[1]).glob("*_audit_log.json"))
if not logs:
    print(f"no audit log in {sys.argv[1]}")
    sys.exit(1)
temps, caps, hints, queries, empties, methods, errors, overflow, seconds = set(), set(), set(), set(), set(), Counter(), [], 0, []
for path in logs:
    audit = json.loads(path.read_text(encoding="utf-8"))
    rc = audit.get("_run_config", {})
    generation = rc.get("generation", {})
    temps.add(generation.get("temperature"))
    caps.add((generation.get("num_predict"), generation.get("num_ctx")))
    hints.add(rc.get("section_hints_fingerprint", {}).get("all"))
    queries.add(rc.get("section_queries_fingerprint", {}).get("all"))
    empties.add(tuple(rc.get("empty_answer_sections", ["not recorded"])))
    for key, entry in audit.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("error"):
            errors.append(f"{path.name[:12]} {key}")
        for outcome in (entry.get("verifier") or {}).values():
            methods[outcome.get("method")] += 1
            overflow += bool(outcome.get("overflow"))
            if outcome.get("seconds") is not None:
                seconds.append(outcome["seconds"])
print(f"{len(logs)} audit logs; temperature {temps}; (num_predict, num_ctx) {caps}; "
      f"hints {hints}; queries {queries}; empty-answer sections {empties}")
print(f"verifier methods {dict(methods)}; overflow {overflow}; "
      f"seconds per request median {sorted(seconds)[len(seconds) // 2] if seconds else 'n/a'}")
print(f"failed sections: {', '.join(errors) if errors else 'none'}")
EOF
}

run_myo() {
    local dir="$1"; shift
    check_empty "$dir"
    say "run $dir $*"
    "$PYTHON" myo/run_myo.py --guideline --output-dir "$dir" "$@" 2>&1 | tee "logs/$(basename "$dir").log"
    check_run "$dir"
}

score_myo() {
    local dir="$1"
    say "score $dir"
    "$PYTHON" myo/evaluate_myo.py "$dir"
    "$PYTHON" evaluate_verifier.py "$dir" --ground-truth "$GT"
}

# -------------------------------------------------------------------- main

if [ "${EVAL_ONLY:-0}" != "1" ]; then
    preflight
    for dir in "${RUNS[@]}"; do
        run_myo "$dir"
    done
    if [ "${T0:-0}" = "1" ]; then
        run_myo "$T0_DIR" --temperature 0
    fi
    if [ "${STRIPPED:-0}" = "1" ]; then
        check_empty "$STRIPPED_DIR"
        say "run $STRIPPED_DIR (DVT, stripped)"
        "$PYTHON" myo/run_dvt_stripped.py --output-dir "$STRIPPED_DIR" 2>&1 | tee "logs/$(basename "$STRIPPED_DIR").log"
        check_run "$STRIPPED_DIR"
    fi
fi

for dir in "${RUNS[@]}"; do
    score_myo "$dir"
done

say "agreement across ${RUNS[*]}"
"$PYTHON" agreement_runs.py "${RUNS[@]}" --ground-truth "$GT"

if [ -d "$T0_DIR" ]; then
    score_myo "$T0_DIR"
fi
if [ -d "$STRIPPED_DIR" ]; then
    say "score $STRIPPED_DIR"
    "$PYTHON" evaluate_predictions.py "$STRIPPED_DIR"
    "$PYTHON" evaluate_verifier.py "$STRIPPED_DIR"
fi

say "done; reports in reports/, run logs in logs/"
