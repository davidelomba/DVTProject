"""
Replays Agent 2's call for stored sections through the Ollama HTTP API with
logprobs on. Checks the replay reproduces the stored response, then prints the
probability of each token on the FINAL_ANSWER line with its alternatives.

Temporary probe, not part of the pipeline. Run from the project root, with
config.py at the reference configuration:
    python probe_logprobs.py output_new_reference SYN_03:A2 SYN_10:B1_1 SYN_12:A2 SYN_01:A1 SYN_07:B2
"""
import glob
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.getcwd())
import config                              # noqa: E402
import agents                              # noqa: E402
from models import SECTION_MODELS          # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA


def newest_log(out_dir, record_id):
    files = sorted(glob.glob(os.path.join(out_dir, f"{record_id}_*_audit_log.json")))
    if not files:
        raise SystemExit(f"no audit log for {record_id} in {out_dir}")
    return files[-1]


def hint_for(section):
    if not config.SECTION_HINTS_ENABLED or section in config.SECTION_HINTS_DISABLED:
        return ""
    return config.SECTION_HINTS.get(section, "")


def chat(system, user):
    options = {"temperature": config.LLM_TEMPERATURE,
               "num_predict": config.LLM_NUM_PREDICT,
               "num_gpu": config.LLM_NUM_GPU,
               "num_ctx": config.LLM_NUM_CTX}
    body = {"model": config.EVALUATOR_LLM_MODEL_NAME, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {k: v for k, v in options.items() if v is not None},
            "logprobs": True, "top_logprobs": 5}
    if config.LLM_REASONING is not None:
        body["think"] = config.LLM_REASONING
    request = urllib.request.Request(f"{OLLAMA}/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=config.LLM_REQUEST_TIMEOUT or 900) as r:
        return json.loads(r.read())


def answer_line_tokens(logprobs):
    """The non-blank tokens after the last FINAL_ANSWER: up to the end of its line."""
    text, spans = "", []
    for token in logprobs:
        start = len(text)
        text += token["token"]
        spans.append((start, len(text), token))
    marker = text.rfind("FINAL_ANSWER:")
    if marker < 0:
        return []
    begin = marker + len("FINAL_ANSWER:")
    end = text.find("\n", begin)
    end = len(text) if end < 0 else end
    return [t for s, e, t in spans if e > begin and s < end and t["token"].strip()]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, targets = sys.argv[1], sys.argv[2:]
    for target in targets:
        record_id, section = target.split(":")
        path = newest_log(out_dir, record_id)
        log = json.load(open(path, encoding="utf-8"))
        run = log.get("_run_config", {})
        entry = log[section]

        print("=" * 78)
        print(f"{target}   {os.path.basename(path)}")
        logged = run.get("models", {}).get("evaluator")
        if logged != config.EVALUATOR_LLM_MODEL_NAME:
            print(f"!! log evaluator {logged} differs from config {config.EVALUATOR_LLM_MODEL_NAME}")
        if run.get("guideline_anchors_enabled"):
            print("!! log was produced with anchors on; the replay assumes off")
        host = run.get("environment", {}).get("hostname")
        print(f"log produced on {host}, replay against {OLLAMA}")

        _, options, multi, description = agents._get_field_info(SECTION_MODELS[section])
        prompt = agents._build_reasoning_prompt(
            entry["evidence"], entry.get("brighton_context") or "", options, multi,
            hint_for(section),
            description if config.SECTION_DESCRIPTIONS_ENABLED else "", False)

        reply = chat(agents.EVALUATOR_SYSTEM_PROMPT, prompt)
        content = reply["message"]["content"]
        stored = entry["reasoning"]
        same = content.strip() == stored.strip()
        print(f"replay identical to stored response: {same}")
        if not same:
            i = next((k for k, (a, b) in enumerate(zip(content, stored)) if a != b),
                     min(len(content), len(stored)))
            print(f"  first difference at char {i}")
            print(f"  replay : {content[max(0, i - 60):i + 60]!r}")
            print(f"  stored : {stored[max(0, i - 60):i + 60]!r}")

        if not reply.get("logprobs"):
            print("!! no logprobs in the reply: this Ollama does not return them")
            continue
        joint = 0.0
        for token in answer_line_tokens(reply["logprobs"]):
            joint += token["logprob"]
            alternatives = ", ".join(
                f"{a['token']!r} {math.exp(a['logprob']):.4f}"
                for a in token.get("top_logprobs", []) if a["token"] != token["token"])
            print(f"  {token['token']!r:8} p={math.exp(token['logprob']):.6f}   alt: {alternatives}")
        print(f"  joint p of the FINAL_ANSWER tokens: {math.exp(joint):.6f}")


if __name__ == "__main__":
    main()
