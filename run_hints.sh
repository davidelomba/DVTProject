#!/bin/bash
# Three agentic_graph runs with the revised hints and Agent 3 on, one after
# the other, each in its own directory:
#   1. output_hints_v2               the reference configuration
#   2. output_hints_v2_descriptions  + SECTION_DESCRIPTIONS_ENABLED = True
#   3. output_hints_v2_small         + EHR chunks 200/40, retriever k 3
# config.py is copied before the first run and restored on exit, also when a
# run fails or the script is stopped.
#
# Launch from the project root:
#   systemd-run --user --unit=dvthints --working-directory=$HOME/DVTProject \
#     $HOME/DVTProject/run_hints.sh

cd "$HOME/DVTProject" || exit 1
PY="$HOME/DVTProject/.venv/bin/python"

cp config.py /tmp/config.py.bak
trap 'cp /tmp/config.py.bak config.py; echo "config.py restored"' EXIT

# Pre-flight: stop before any run if the working copy is not the one expected.
$PY - <<'PYCHECK' || exit 1
import sys
import config
import pipeline
checks = {
    "EVALUATOR_LLM_MODEL_NAME": (config.EVALUATOR_LLM_MODEL_NAME, "qwen3.6:27b"),
    "AGENTIC_LLM_MODEL_NAME": (config.AGENTIC_LLM_MODEL_NAME, "qwen3.6:27b"),
    "EXTRACTOR_MODE": (config.EXTRACTOR_MODE, "agentic_graph"),
    "SECTION_DESCRIPTIONS_ENABLED": (config.SECTION_DESCRIPTIONS_ENABLED, False),
    "EHR chunks": ((config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K), (800, 150, 5)),
    "CONFIDENCE_ENABLED": (config.CONFIDENCE_ENABLED, True),
    "hint fingerprint": (pipeline._hint_fingerprint()["all"], "0d4a00c11404"),
}
bad = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
print("pre-flight:", "OK" if not bad else "FAILED")
for line in bad:
    print("  " + line)
sys.exit(1 if bad else 0)
PYCHECK

echo "=== run 1: output_hints_v2 ==="
$PY run_synthetic_records.py --output-dir ./output_hints_v2

echo "=== run 2: output_hints_v2_descriptions ==="
sed -i 's/^SECTION_DESCRIPTIONS_ENABLED = False/SECTION_DESCRIPTIONS_ENABLED = True/' config.py
grep -n '^SECTION_DESCRIPTIONS_ENABLED' config.py
$PY run_synthetic_records.py --output-dir ./output_hints_v2_descriptions
cp /tmp/config.py.bak config.py

echo "=== run 3: output_hints_v2_small ==="
sed -i -e 's/^EHR_CHUNK_SIZE = 800/EHR_CHUNK_SIZE = 200/' \
       -e 's/^EHR_CHUNK_OVERLAP = 150/EHR_CHUNK_OVERLAP = 40/' \
       -e 's/^EHR_RETRIEVER_K = 5/EHR_RETRIEVER_K = 3/' config.py
grep -nE '^EHR_(CHUNK_SIZE|CHUNK_OVERLAP|RETRIEVER_K)' config.py
$PY run_synthetic_records.py --output-dir ./output_hints_v2_small
