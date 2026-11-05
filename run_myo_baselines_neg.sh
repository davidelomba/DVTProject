#!/usr/bin/env bash
# The DVT baselines with an extractor on the myocarditis cases, with
# config.EXTRACTOR_NEGATION_RULE on, three runs each at temperature 0.5, every
# other setting as in the v4 series (myo/output_t05_np2048_v4_r*). Four arms:
#
#   full_text_8b    full_text, extractor llama3:8b-instruct-q4_0
#   rag_8b          rag,       extractor llama3:8b-instruct-q4_0
#   full_text_27b   full_text, extractor Agent 1's tag (qwen3.6:27b)
#   rag_27b         rag,       extractor Agent 1's tag (qwen3.6:27b)
#
# config.py is not edited: each run sets EXTRACTOR_NEGATION_RULE in memory and
# then calls run_myo.main with --guideline, --mode and --extractor-model.
#
# Then their scoring: myo/evaluate_myo.py and evaluate_verifier.py on each run,
# agreement_runs.py across the three runs of each arm.
#
# Before the first run it checks that Ollama answers, that every model the arms
# need is pulled, that config.py carries the v4 token settings and that each arm
# runs the mode, the extractor, the hints, the queries and the temperature it is
# meant to, with agents.NEGATION_RULE in the extractor prompt. After each run it
# reads the audit logs back and prints the mode, the models, the fingerprints,
# the negation rule setting, the verifier results and the failed sections. The
# runs go one after the other, never two at once.
#
# Optional, through environment variables:
#   ARMS="..."   the arms to run, all four by default, e.g. ARMS="rag_27b"
#   EVAL_ONLY=1  skip the runs and score the directories already on disk
#   FORCE=1      run into an output directory that already holds files
#   PYTHON=...   the interpreter, python by default
#
# Output directories: myo/output_neg_<arm>_r1 to _r3.

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
EXPECTED_NUM_PREDICT=2048
EXPECTED_NUM_CTX=8192
GT="./myo/data/records"
SMALL_EXTRACTOR="llama3:8b-instruct-q4_0"
AGENT_TAG="$(grep -E '^AGENTIC_LLM_MODEL_NAME' config.py | sed -E 's/^[^"]*"([^"]*)".*/\1/')"
EVALUATOR_TAG="$(grep -E '^EVALUATOR_LLM_MODEL_NAME' config.py | sed -E 's/^[^"]*"([^"]*)".*/\1/')"
ARMS="${ARMS:-full_text_8b rag_8b full_text_27b rag_27b}"

# arm -> "mode extractor"
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

runs_of() { echo "./myo/output_neg_$1_r1 ./myo/output_neg_$1_r2 ./myo/output_neg_$1_r3"; }

# ---------------------------------------------------------------- preflight

preflight() {
    say "preflight"
    echo "host $(hostname), $("$PYTHON" --version 2>&1), $(ollama --version 2>&1 | tail -1)"

    if ! curl -sf "${OLLAMA_HOST:-http://localhost:11434}/api/version" > /dev/null; then
        echo "Ollama does not answer at ${OLLAMA_HOST:-http://localhost:11434}." >&2
        exit 1
    fi

    local tags=("$EVALUATOR_TAG" "$AGENT_TAG") arm spec
    for arm in $ARMS; do
        spec="$(arm_spec "$arm")"
        if [ "${spec#* }" != "-" ]; then tags+=("${spec#* }"); fi
    done
    local tag
    for tag in "${tags[@]}"; do
        if ! ollama list | awk '{print $1}' | grep -qx "$tag"; then
            echo "The model $tag is not pulled." >&2
            exit 1
        fi
    done

    echo "config.py:"
    grep -E '^(LLM_MODEL_NAME|EVALUATOR_LLM_MODEL_NAME|AGENTIC_LLM_MODEL_NAME|LLM_REASONING|LLM_NUM_PREDICT|LLM_NUM_CTX|EXTRACTOR_MODE|VERIFIER_ENABLED|VERIFIER_INPUTS|VERIFIER_TEMPERATURE|VERIFIER_NUM_CTX)' config.py

    # What each arm will send, read through the same code it runs
    local specs=""
    for arm in $ARMS; do
        specs+="$arm $(arm_spec "$arm");"
    done
    SPECS="$specs" EXPECTED_HINTS="$EXPECTED_HINTS" EXPECTED_QUERIES="$EXPECTED_QUERIES" \
    EXPECTED_NUM_PREDICT="$EXPECTED_NUM_PREDICT" EXPECTED_NUM_CTX="$EXPECTED_NUM_CTX" \
    "$PYTHON" - <<'EOF'
import os, sys
sys.path.insert(0, "myo")
import myo_schema  # noqa: F401
import agents, config, pipeline, run_myo, verifier
config.EXTRACTOR_NEGATION_RULE = True
problems = []
for spec in filter(None, os.environ["SPECS"].split(";")):
    arm, mode, extractor = spec.split()
    extractor = None if extractor == "-" else extractor
    run_myo.apply_domain(guideline=True, mode=mode, extractor_model=extractor)
    hints = pipeline._hint_fingerprint()["all"]
    queries = pipeline._query_fingerprint()["all"]
    used = None if mode == "raw_record" else config.LLM_MODEL_NAME
    print(f"{arm}: mode {config.EXTRACTOR_MODE}, extractor {used}, evaluator "
          f"{config.EVALUATOR_LLM_MODEL_NAME}, verifier "
          f"{verifier.verifier_settings().get('model')}, temperature {config.LLM_TEMPERATURE}, "
          f"hints {hints}, queries {queries}, verifier {config.VERIFIER_ENABLED} "
          f"{list(config.VERIFIER_INPUTS)}, context {config.BRIGHTON_CONTEXT_ENABLED}, "
          f"rules {config.CROSS_SECTION_RULES}, negation rule {config.EXTRACTOR_NEGATION_RULE}, "
          f"rule in prompt {agents.NEGATION_RULE in agents.baseline_extractor_prompt()}")
    if agents.NEGATION_RULE not in agents.baseline_extractor_prompt():
        problems.append(f"{arm}: negation rule missing from the extractor prompt")
    if config.EXTRACTOR_MODE != mode:
        problems.append(f"{arm}: EXTRACTOR_MODE {config.EXTRACTOR_MODE}")
    if extractor is not None and config.LLM_MODEL_NAME != extractor:
        problems.append(f"{arm}: extractor {config.LLM_MODEL_NAME}, expected {extractor}")
    if agents.EMPTY_ANSWER_SECTIONS:
        problems.append(f"{arm}: empty-answer sections {sorted(agents.EMPTY_ANSWER_SECTIONS)}")
    if hints != os.environ["EXPECTED_HINTS"]:
        problems.append(f"{arm}: hint fingerprint {hints}, expected {os.environ['EXPECTED_HINTS']}")
    if queries != os.environ["EXPECTED_QUERIES"]:
        problems.append(f"{arm}: query fingerprint {queries}, expected {os.environ['EXPECTED_QUERIES']}")
    if config.LLM_TEMPERATURE != 0.5:
        problems.append(f"{arm}: temperature {config.LLM_TEMPERATURE}, expected 0.5")
    if not config.VERIFIER_ENABLED:
        problems.append(f"{arm}: VERIFIER_ENABLED is False")
if config.LLM_NUM_PREDICT != int(os.environ["EXPECTED_NUM_PREDICT"]):
    problems.append(f"LLM_NUM_PREDICT {config.LLM_NUM_PREDICT}, expected {os.environ['EXPECTED_NUM_PREDICT']}")
if config.LLM_NUM_CTX != int(os.environ["EXPECTED_NUM_CTX"]):
    problems.append(f"LLM_NUM_CTX {config.LLM_NUM_CTX}, expected {os.environ['EXPECTED_NUM_CTX']}")
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
modes, temps, caps, hints, queries, models, neg = set(), set(), set(), set(), set(), set(), set()
methods, errors, overflow, seconds, empty_evidence = Counter(), [], 0, [], 0
for path in logs:
    audit = json.loads(path.read_text(encoding="utf-8"))
    rc = audit.get("_run_config", {})
    modes.add(rc.get("extractor_mode"))
    neg.add(rc.get("extractor_negation_rule"))
    models.add((rc.get("models", {}).get("extractor"), rc.get("models", {}).get("evaluator"),
                rc.get("verifier", {}).get("model")))
    generation = rc.get("generation", {})
    temps.add(generation.get("temperature"))
    caps.add((generation.get("num_predict"), generation.get("num_ctx")))
    hints.add(rc.get("section_hints_fingerprint", {}).get("all"))
    queries.add(rc.get("section_queries_fingerprint", {}).get("all"))
    for key, entry in audit.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("error"):
            errors.append(f"{path.name[:12]} {key}")
        if "NO RELEVANT EVIDENCE FOUND" in str(entry.get("evidence", "")):
            empty_evidence += 1
        for outcome in (entry.get("verifier") or {}).values():
            methods[outcome.get("method")] += 1
            overflow += bool(outcome.get("overflow"))
            if outcome.get("seconds") is not None:
                seconds.append(outcome["seconds"])
print(f"mode {modes}; (extractor, evaluator, verifier) {models}; negation rule {neg}")
print(f"{len(logs)} audit logs; temperature {temps}; (num_predict, num_ctx) {caps}; "
      f"hints {hints}; queries {queries}; empty evidence {empty_evidence}")
print(f"verifier methods {dict(methods)}; overflow {overflow}; "
      f"seconds per request median {sorted(seconds)[len(seconds) // 2] if seconds else 'n/a'}")
print(f"failed sections: {', '.join(errors) if errors else 'none'}")
EOF
}

run_arm() {
    local dir="$1" arm="$2" spec mode extractor
    spec="$(arm_spec "$arm")"
    mode="${spec%% *}"
    extractor="${spec#* }"
    local extra=()
    if [ "$extractor" != "-" ]; then extra=(--extractor-model "$extractor"); fi
    check_empty "$dir"
    say "run $dir, arm $arm (mode $mode, extractor $extractor)"
    "$PYTHON" - --guideline --mode "$mode" ${extra[@]+"${extra[@]}"} \
        --output-dir "$dir" <<'EOF' 2>&1 | tee "logs/$(basename "$dir").log"
import sys
sys.path.insert(0, "myo")
import myo_schema  # noqa: F401
import config
config.EXTRACTOR_NEGATION_RULE = True
sys.argv = ["myo/run_myo.py"] + sys.argv[1:]
import run_myo
run_myo.main()
EOF
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
    for arm in $ARMS; do
        for dir in $(runs_of "$arm"); do
            check_empty "$dir"
        done
    done
    for arm in $ARMS; do
        for dir in $(runs_of "$arm"); do
            run_arm "$dir" "$arm"
        done
    done
fi

for arm in $ARMS; do
    for dir in $(runs_of "$arm"); do
        score_myo "$dir"
    done
    say "agreement across the $arm runs"
    # shellcheck disable=SC2046
    "$PYTHON" agreement_runs.py $(runs_of "$arm") --ground-truth "$GT"
done

say "done; reports in reports/, run logs in logs/"
