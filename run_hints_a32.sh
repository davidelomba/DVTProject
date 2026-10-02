#!/bin/bash
# Ten agentic_graph-based runs with the shortened A3_2 hint (hint fingerprint
# a78dc29e962f), Agent 3 on, one after the other, each in its own directory.
# They repeat the revised-hint rows of the thesis tables:
#
#   chunks 800/150/5
#    1. output_a32_ref               query set bfd9536a31fe (original queries)
#    2. output_a32_raw               bfd9536a31fe, EXTRACTOR_MODE raw_record
#    3. output_a32_descriptions      bfd9536a31fe, SECTION_DESCRIPTIONS_ENABLED
#    4. output_a32_queries_q         e0c31c90d6ce (all queries from the questionnaire)
#    5. output_a32_queries_mix       f7ed731f7e7d (query set in pipeline.py)
#    6. output_a32_e5large           f7ed731f7e7d, multilingual-e5-large
#   chunks 200/40/3
#    7. output_a32_small             bfd9536a31fe
#    8. output_a32_small_queries_q   e0c31c90d6ce
#    9. output_a32_small_queries_mix f7ed731f7e7d
#   10. output_a32_small_e5large     f7ed731f7e7d, multilingual-e5-large
#
# config.py and pipeline.py are copied before the first run, reset from the
# copies before each run, and restored on exit, also when a run fails or the
# script is stopped. Each run is preceded by a check of the configuration it
# is about to use and is skipped if the check fails. After each run, the calls
# that filled or overflowed LLM_NUM_CTX are listed from agent2_tokens.
#
# Launch from the project root:
#   systemd-run --user --unit=dvta32 --working-directory=$HOME/DVTProject \
#     /bin/bash $HOME/DVTProject/run_hints_a32.sh
# Follow it with:
#   journalctl --user -u dvta32 -f

cd "$HOME/DVTProject" || exit 1
PY="$HOME/DVTProject/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
rm -f __pycache__/config.*.pyc __pycache__/pipeline.*.pyc

cp config.py /tmp/config.py.a32bak
cp pipeline.py /tmp/pipeline.py.a32bak
trap 'cp /tmp/config.py.a32bak config.py; cp /tmp/pipeline.py.a32bak pipeline.py; echo "config.py and pipeline.py restored"' EXIT

# The three section query sets, as recorded in the audit logs of the runs that
# used them.
cat > /tmp/dvt_query_sets.json <<'JSON'
{
 "bfd9536a31fe": {
  "A1": "autopsy report, necropsy, post-mortem examination, autoptic findings",
  "A2": "thrombectomy, surgical procedure related to DVT",
  "A3_1": "ultrasound, CT, MRI, venography: imaging outcome for deep vein thrombosis",
  "A3_2": "type of imaging study performed: compression ultrasonography, doppler, venography",
  "B1_1": "reported symptoms or signs of deep vein thrombosis",
  "B1_2": "deep vein thrombosis lower extremity or upper extremity",
  "B2": "calf pain or tenderness, leg swelling or pitting oedema, redness, warmth or pain in any extremity, absent pulses in legs or arms",
  "C": "D-dimer value, test date, laboratory upper limit of normal",
  "F": "diagnosis of deep vein thrombosis reported by specialist",
  "X": "possible alternative etiologies; clinical syndromes to be differentiated from thrombosis; conditions explaining calf pain, redness, warmth or ankle edema other than DVT"
 },
 "e0c31c90d6ce": {
  "A1": "autopsy or post-mortem examination: pathologic or histopathologic findings of deep vein thrombosis",
  "A2": "thrombectomy or other surgical procedure that confirmed the presence of a deep vein thrombus",
  "A3_1": "imaging study findings consistent with deep vein thrombosis: whether an imaging study confirmed DVT or did not confirm it",
  "A3_2": "imaging study that confirmed deep vein thrombosis: compression ultrasonography, Doppler or duplex ultrasound, CT or MR venography, contrast venography, other imaging modality",
  "B1_1": "clinical presentation consistent with deep vein thrombosis: reported signs or symptoms, or a recognized or presumed DVT syndrome",
  "B1_2": "specific type of deep vein thrombosis: DVT of lower or upper limbs, lower extremity or upper extremity",
  "B2": "new onset non-specific clinical signs or symptoms suggesting DVT: calf pain or tenderness, leg swelling or pitting oedema, absent pulses in legs or arms, redness, warmth or pain in one or more extremities",
  "C": "D-dimer test: highest measured value within two weeks of the event, above or within the test lab's upper limit of normal",
  "F": "case of deep vein thrombosis reported by a specialist, with or without supporting clinical, imaging or laboratory details",
  "X": "alternative diagnosis or alternate etiology, other than deep vein thrombosis, that explained the acute clinical illness; disorders to be differentiated from thrombosis"
 },
 "f7ed731f7e7d": {
  "A1": "autopsy report, necropsy, post-mortem examination, autoptic findings",
  "A2": "thrombectomy or other surgical procedure that confirmed the presence of a deep vein thrombus",
  "A3_1": "ultrasound, CT, MRI, venography: imaging outcome for deep vein thrombosis",
  "A3_2": "imaging study that confirmed deep vein thrombosis: compression ultrasonography, Doppler or duplex ultrasound, CT or MR venography, contrast venography, other imaging modality",
  "B1_1": "reported symptoms or signs of deep vein thrombosis",
  "B1_2": "deep vein thrombosis lower extremity or upper extremity",
  "B2": "calf pain or tenderness, leg swelling or pitting oedema, redness, warmth or pain in any extremity, absent pulses in legs or arms",
  "C": "D-dimer test: highest measured value within two weeks of the event, above or within the test lab's upper limit of normal",
  "F": "case of deep vein thrombosis reported by a specialist, with or without supporting clinical, imaging or laboratory details",
  "X": "alternative diagnosis or alternate etiology, other than deep vein thrombosis, that explained the acute clinical illness; disorders to be differentiated from thrombosis"
 }
}
JSON

# Replaces the SECTION_QUERIES dict in pipeline.py with the set named by $1.
set_queries() {
$PY - "$1" <<'PYQUERIES'
import json
import sys
name = sys.argv[1]
queries = json.load(open("/tmp/dvt_query_sets.json"))[name]
src = open("pipeline.py", encoding="utf-8").read()
start = src.index("SECTION_QUERIES = {")
depth = 0
for end in range(start + len("SECTION_QUERIES = "), len(src)):
    if src[end] == "{":
        depth += 1
    elif src[end] == "}":
        depth -= 1
        if depth == 0:
            break
body = "SECTION_QUERIES = {\n" + "".join(
    f"    {key!r}: {value!r},\n" for key, value in queries.items()) + "}"
open("pipeline.py", "w", encoding="utf-8").write(src[:start] + body + src[end + 1:])
PYQUERIES
}

# Stops the run that follows if the working copy is not the one expected.
# Arguments: mode, descriptions (True/False), chunks (800 or 200), embedding
# model, query fingerprint.
check() {
$PY - "$@" <<'PYCHECK'
import sys
import config
import pipeline
mode, descriptions, chunks, embedding, queries = sys.argv[1:6]
expected_chunks = (800, 150, 5) if chunks == "800" else (200, 40, 3)
checks = {
    "EVALUATOR_LLM_MODEL_NAME": (config.EVALUATOR_LLM_MODEL_NAME, "qwen3.6:27b"),
    "AGENTIC_LLM_MODEL_NAME": (config.AGENTIC_LLM_MODEL_NAME, "qwen3.6:27b"),
    "EXTRACTOR_MODE": (config.EXTRACTOR_MODE, mode),
    "SECTION_DESCRIPTIONS_ENABLED": (config.SECTION_DESCRIPTIONS_ENABLED, descriptions == "True"),
    "GUIDELINE_ANCHORS_ENABLED": (config.GUIDELINE_ANCHORS_ENABLED, False),
    "BRIGHTON_CONTEXT_ENABLED": (config.BRIGHTON_CONTEXT_ENABLED, True),
    "EHR chunks": ((config.EHR_CHUNK_SIZE, config.EHR_CHUNK_OVERLAP, config.EHR_RETRIEVER_K), expected_chunks),
    "LLM_NUM_CTX": (config.LLM_NUM_CTX, 4096),
    "LLM_NUM_PREDICT": (config.LLM_NUM_PREDICT, 1024),
    "CONFIDENCE_ENABLED": (config.CONFIDENCE_ENABLED, True),
    "EMBEDDING_MODEL_NAME": (config.EMBEDDING_MODEL_NAME, embedding),
    "EMBEDDING_DEVICE": (config.EMBEDDING_DEVICE, "cpu"),
    "hint fingerprint": (pipeline._hint_fingerprint()["all"], "a78dc29e962f"),
    "query fingerprint": (pipeline._query_fingerprint()["all"], queries),
}
bad = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
print("pre-flight:", "OK" if not bad else "FAILED")
for line in bad:
    print("  " + line)
sys.exit(1 if bad else 0)
PYCHECK
}

token_check() {
$PY - "$1" <<'PYAFTER'
# Every Agent 2 call of the run, from agent2_tokens in the audit logs. A call
# whose prompt and output together reach LLM_NUM_CTX filled the window, and a
# retry logged with a shorter prompt than the attempt before it was truncated,
# since each retry only adds to the prompt.
import glob
import json
import os
import sys
import config

run_dir = sys.argv[1]
calls = over = retried = 0
for path in sorted(glob.glob(f"{run_dir}/*_audit_log.json")):
    audit = json.load(open(path))
    record = os.path.basename(path).split("_2026")[0]
    for section_key, entry in audit.items():
        if section_key.startswith("_"):
            continue
        counts = entry.get("agent2_tokens") or []
        if len(counts) > 1:
            retried += 1
            print(f"  retried: {record} {section_key}, {len(counts)} attempts")
        previous = None
        for n, count in enumerate(counts, 1):
            if count.get("prompt") is None or count.get("output") is None:
                continue
            calls += 1
            total = count["prompt"] + count["output"]
            if total >= config.LLM_NUM_CTX:
                over += 1
                print(f"  FULL {record} {section_key} attempt {n}: "
                      f"{count['prompt']} + {count['output']} = {total}")
            elif previous is not None and count["prompt"] < previous:
                over += 1
                print(f"  TRUNCATED {record} {section_key} attempt {n}: "
                      f"prompt {count['prompt']} after {previous}")
            previous = count["prompt"]
print(f"{run_dir}: {calls} calls, {retried} sections retried, "
      f"{over} calls that filled or overflowed LLM_NUM_CTX")
PYAFTER
}

SMALL='s/^EHR_CHUNK_SIZE = 800/EHR_CHUNK_SIZE = 200/;s/^EHR_CHUNK_OVERLAP = 150/EHR_CHUNK_OVERLAP = 40/;s/^EHR_RETRIEVER_K = 5/EHR_RETRIEVER_K = 3/'
RAW='s/^EXTRACTOR_MODE = "agentic_graph"/EXTRACTOR_MODE = "raw_record"/'
DESCRIPTIONS='s/^SECTION_DESCRIPTIONS_ENABLED = False/SECTION_DESCRIPTIONS_ENABLED = True/'
LARGE='s#^EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"#EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"#'
E5S="intfloat/multilingual-e5-small"
E5L="intfloat/multilingual-e5-large"

# Arguments: output directory, query fingerprint, sed edits for config.py
# (empty for none), then the arguments of check without the query fingerprint.
run() {
    local dir="$1" queries="$2" edits="$3"
    shift 3
    echo "=== run: $dir ==="
    cp /tmp/config.py.a32bak config.py
    cp /tmp/pipeline.py.a32bak pipeline.py
    [ -n "$edits" ] && sed -i "$edits" config.py
    set_queries "$queries"
    rm -f __pycache__/config.*.pyc __pycache__/pipeline.*.pyc
    if ! check "$@" "$queries"; then
        echo "=== skipped: $dir ==="
        return
    fi
    $PY run_synthetic_records.py --output-dir "./$dir"
    token_check "$dir"
}

run output_a32_ref               bfd9536a31fe ""                 agentic_graph False 800 "$E5S"
run output_a32_raw               bfd9536a31fe "$RAW"             raw_record    False 800 "$E5S"
run output_a32_descriptions      bfd9536a31fe "$DESCRIPTIONS"    agentic_graph True  800 "$E5S"
run output_a32_queries_q         e0c31c90d6ce ""                 agentic_graph False 800 "$E5S"
run output_a32_queries_mix       f7ed731f7e7d ""                 agentic_graph False 800 "$E5S"
run output_a32_e5large           f7ed731f7e7d "$LARGE"           agentic_graph False 800 "$E5L"
run output_a32_small             bfd9536a31fe "$SMALL"           agentic_graph False 200 "$E5S"
run output_a32_small_queries_q   e0c31c90d6ce "$SMALL"           agentic_graph False 200 "$E5S"
run output_a32_small_queries_mix f7ed731f7e7d "$SMALL"           agentic_graph False 200 "$E5S"
run output_a32_small_e5large     f7ed731f7e7d "$SMALL;$LARGE"    agentic_graph False 200 "$E5L"

echo "=== done ==="
