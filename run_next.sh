#!/bin/bash
# One run with the revised hints and Agent 3 on, in its own directory:
#   output_hints_v2_anchors_t3  guideline anchors chosen by principle
#                               (Table 3, its rationale in 5.2.x and the
#                               tables it cites; C and F not anchored)
# The anchors are set in memory and config.py is left as it is.
# Before the run, a check builds every anchored section's Agent 2 prompt on
# the longest record, asks Ollama how many tokens it takes and warns about
# any prompt above LLM_NUM_CTX minus LLM_NUM_PREDICT. After the run, a second
# check reads agent2_tokens from every audit log and lists the calls whose
# prompt and output together exceeded LLM_NUM_CTX.
# config.py is copied before the run and restored on exit, also when the run
# fails or the script is stopped.
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

# The anchor map, shared by the token check and by the run itself.
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

echo "=== token check before the run ==="
$PY - <<'PYTOKENS' || exit 1
# Agent 2's prompt for every anchored section on the longest record, with the
# evidence the reference run stored for it. Each request carries a distinct
# first line, so Ollama cannot reuse a cached prefix and prompt_eval_count
# counts the whole prompt; the check therefore overestimates by a few tokens.
# A prompt above the limit is reported and the run still starts; the check
# after the run says whether any call actually exceeded the window. The counts are saved to
# /tmp/token_check.json for that comparison.
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
counts = {}
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
    counts[section_key] = tokens
    print(f"  {section_key:5} {len(prompt):6} chars  {tokens:5} tokens  {'OK' if tokens <= limit else 'WARNING: over'}")

json.dump({"record": longest.stem, "tokens": counts}, open("/tmp/token_check.json", "w"))
print(f"largest prompt {worst} tokens against {limit}:",
      "OK" if worst <= limit else "WARNING, the run goes ahead and the check after it decides")
PYTOKENS

echo "=== run: output_hints_v2_anchors_t3 ==="
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

echo "=== token check after the run ==="
$PY - <<'PYAFTER'
# Every Agent 2 call of the run, from agent2_tokens in the audit logs. A call
# whose prompt and output together exceed LLM_NUM_CTX did not fit the window.
# The record checked before the run is compared with its own audit log: a
# prompt count lower there than before means Ollama reused a cached prefix and
# reports only the tokens it evaluated, so the sums below undercount.
import glob
import json
import os
import config

window = config.LLM_NUM_CTX
calls = over = 0
largest = (0, None)
for path in sorted(glob.glob("output_hints_v2_anchors_t3/*_audit_log.json")):
    audit = json.load(open(path))
    record = os.path.basename(path).split("_2026")[0]
    for section_key, entry in audit.items():
        if section_key.startswith("_"):
            continue
        for n, count in enumerate(entry.get("agent2_tokens") or [], 1):
            if count.get("prompt") is None or count.get("output") is None:
                print(f"  {record} {section_key} attempt {n}: counts missing")
                continue
            calls += 1
            total = count["prompt"] + count["output"]
            largest = max(largest, (total, f"{record} {section_key}"))
            if total > window:
                over += 1
                print(f"  OVER {record} {section_key} attempt {n}: "
                      f"{count['prompt']} + {count['output']} = {total} > {window}")

print(f"{calls} calls, largest {largest[0]} tokens ({largest[1]})")
print("no call exceeded LLM_NUM_CTX" if over == 0 else f"{over} calls exceeded LLM_NUM_CTX")

try:
    before = json.load(open("/tmp/token_check.json"))
except FileNotFoundError:
    before = None
if before:
    for path in glob.glob(f"output_hints_v2_anchors_t3/{before['record']}_*_audit_log.json"):
        audit = json.load(open(path))
        for section_key, tokens in before["tokens"].items():
            logged = (audit.get(section_key, {}).get("agent2_tokens") or [{}])[0].get("prompt")
            print(f"  {section_key:5} before the run {tokens:5}, in the audit log {logged}")
PYAFTER
