#!/bin/bash
# Two runs with the revised hints and Agent 3 on, one after the other, each in
# its own directory:
#   1. output_hints_v2_raw         EXTRACTOR_MODE = "raw_record"
#   2. output_hints_v2_anchors_t3  guideline anchors chosen by principle
#                                  (Table 3, its rationale in 5.2.x and the
#                                  tables it cites; C and F not anchored)
# Run 2 sets the anchors in memory and leaves config.py as it is. Before it
# starts, a check builds every anchored section's Agent 2 prompt on the
# longest record and asks Ollama how many tokens it takes; run 2 is skipped
# when any prompt exceeds LLM_NUM_CTX minus LLM_NUM_PREDICT.
# config.py is copied before the first run and restored on exit, also when a
# run fails or the script is stopped.
#
# Launch from the project root:
#   systemd-run --user --unit=dvtnext --working-directory=$HOME/DVTProject \
#     /bin/bash $HOME/DVTProject/run_next.sh

cd "$HOME/DVTProject" || exit 1
PY="$HOME/DVTProject/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
rm -f __pycache__/config.*.pyc

cp config.py /tmp/config.py.bak
trap 'cp /tmp/config.py.bak config.py; echo "config.py restored"' EXIT

# The anchor map of run 2, shared by the token check and by the run itself.
cat > /tmp/anchors_t3.py <<'PYMAP'
import config

ANCHORS = {
    "A1":   ("Table 3", "5.2.2."),
    "A2":   ("Table 3",),
    "A3_1": ("Table 3", "Table 1"),
    "A3_2": ("Table 3", "Table 1"),
    "B1_1": ("Table 3", "5.2.1."),
    "B1_2": ("Table 3", "5.2.1."),
    "B2":   ("Table 3", "5.2.1."),
    "X":    ("Table 3", "5.2.7.", "Table 2"),
}


def apply():
    config.GUIDELINE_ANCHORS = ANCHORS
    config.GUIDELINE_ANCHORS_ENABLED = True
PYMAP

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
    "GUIDELINE_ANCHORS_ENABLED": (config.GUIDELINE_ANCHORS_ENABLED, False),
    "BRIGHTON_CONTEXT_ENABLED": (config.BRIGHTON_CONTEXT_ENABLED, True),
    "EHR chunks": ((config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K), (800, 150, 5)),
    "LLM_NUM_CTX": (config.LLM_NUM_CTX, 4096),
    "CONFIDENCE_ENABLED": (config.CONFIDENCE_ENABLED, True),
    "hint fingerprint": (pipeline._hint_fingerprint()["all"], "0d4a00c11404"),
}
bad = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
print("pre-flight:", "OK" if not bad else "FAILED")
for line in bad:
    print("  " + line)
sys.exit(1 if bad else 0)
PYCHECK

echo "=== run 1: output_hints_v2_raw ==="
sed -i 's/^EXTRACTOR_MODE = "agentic_graph"/EXTRACTOR_MODE = "raw_record"/' config.py
grep -n '^EXTRACTOR_MODE' config.py
$PY run_synthetic_records.py --output-dir ./output_hints_v2_raw
cp /tmp/config.py.bak config.py

echo "=== token check for run 2 ==="
$PY - <<'PYTOKENS' || { echo "run 2 skipped"; exit 1; }
# Agent 2's prompt for every anchored section on the longest record, with the
# evidence the reference run stored for it. Each request carries a distinct
# first line, so Ollama cannot reuse a cached prefix and prompt_eval_count
# counts the whole prompt; the check therefore overestimates by a few tokens.
import glob
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "/tmp")
import anchors_t3
import config
from agents import EVALUATOR_SYSTEM_PROMPT, _build_reasoning_prompt, _get_field_info
from confidence import OLLAMA_URL
from models import SECTION_MODELS
from rag_setup import load_brighton_pdf_text, resolve_guideline_anchors
from run_synthetic_records import BRIGHTON_PDF_PATH

anchors_t3.apply()
resolved = resolve_guideline_anchors(load_brighton_pdf_text(BRIGHTON_PDF_PATH))
missing = sorted(set(anchors_t3.ANCHORS) - set(resolved))
if missing:
    print("anchors that do not resolve, which would fall back to retrieval:", missing)
    sys.exit(1)

records = sorted(Path("data/synthetic_records").glob("*_v2.txt"), key=lambda p: p.stat().st_size)
longest = records[-1]
audits = sorted(glob.glob(f"output_hints_v2/{longest.stem}_*_audit_log.json"))
audit = json.load(open(audits[-1])) if audits else {}
limit = config.LLM_NUM_CTX - config.LLM_NUM_PREDICT
print(f"record {longest.stem}, {longest.stat().st_size} bytes; limit {limit} prompt tokens")

worst = 0
for n, section_key in enumerate(sorted(resolved)):
    evidence = audit.get(section_key, {}).get("evidence") or longest.read_text(encoding="utf-8")
    _, options, multi, _ = _get_field_info(SECTION_MODELS[section_key])
    prompt = _build_reasoning_prompt(
        evidence, resolved[section_key], options, multi,
        config.section_hint(section_key), "", context_is_anchored=True,
    )
    body = {
        "model": config.EVALUATOR_LLM_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": f"[token check {n}]\n" + EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"num_ctx": 8192, "num_predict": 1, "temperature": 0,
                    "num_gpu": config.LLM_NUM_GPU},
    }
    if config.LLM_REASONING is not None:
        body["think"] = config.LLM_REASONING
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=config.LLM_REQUEST_TIMEOUT) as response:
        tokens = json.loads(response.read())["prompt_eval_count"]
    worst = max(worst, tokens)
    print(f"  {section_key:5} {len(prompt):6} chars  {tokens:5} tokens  {'OK' if tokens <= limit else 'OVER'}")

print(f"largest prompt {worst} tokens against {limit}:", "OK" if worst <= limit else "TOO LONG")
sys.exit(0 if worst <= limit else 1)
PYTOKENS

echo "=== run 2: output_hints_v2_anchors_t3 ==="
$PY - <<'PYRUN'
import sys
sys.path.insert(0, "/tmp")
import anchors_t3
import config
anchors_t3.apply()
print("GUIDELINE_ANCHORS_ENABLED", config.GUIDELINE_ANCHORS_ENABLED, flush=True)
for key, labels in config.GUIDELINE_ANCHORS.items():
    print(f"  {key:5} {', '.join(labels)}", flush=True)
import run_synthetic_records
sys.argv = ["run_synthetic_records.py", "--output-dir", "./output_hints_v2_anchors_t3"]
run_synthetic_records.main()
PYRUN
