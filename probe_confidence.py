"""
Scores the confidence of Agent 2's stored answers with one short call per
section, without reasoning, reading the probabilities from the Ollama logprobs.

For each section of each record in a run directory, the evaluator model gets the
evidence and options Agent 2 had, not Agent 2's answer, and must reply at once:
  - single choice: the option number. Confidence is the probability the model
    puts on the option Agent 2 chose, among the valid option numbers.
  - multiple choice: one line per option, YES or NO. Confidence is the
    probability of Agent 2's exact set, the product over options of p(YES) for
    the options it selected and p(NO) for the others.

Writes one CSV row per section and prints the AUROC of the confidence against
correctness, next to the AUROC of agent2_seconds as the baseline to beat.

Temporary probe, not part of the pipeline. Run from the project root:
    python probe_confidence.py output_new_reference --only SYN_03 SYN_10 SYN_01
    python probe_confidence.py output_new_reference
    python probe_confidence.py output_stripped
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())
import config                              # noqa: E402
import agents                              # noqa: E402
from models import SECTION_MODELS          # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA

SYSTEM = ("You check answers to a clinical questionnaire. You read the evidence "
          "extracted from a patient record and state which options it supports. "
          "Base the answer only on the evidence. Do not explain.")

LOG_NAME = re.compile(r"^(?P<record>.+)_(?P<stamp>\d{8}_\d{6})_audit_log\.json$")


def latest_logs(out_dir):
    """Newest audit log per record, the same selection evaluate_predictions makes."""
    newest = {}
    for path in glob.glob(os.path.join(out_dir, "*_audit_log.json")):
        match = LOG_NAME.match(os.path.basename(path))
        if match:
            record, stamp = match["record"], match["stamp"]
            if record not in newest or stamp > newest[record][0]:
                newest[record] = (stamp, path)
    return {record: path for record, (_, path) in sorted(newest.items())}


def hint_for(run, section):
    """The hint Agent 2 received in that run, checked against its fingerprint."""
    if run.get("section_hints_enabled") is False:
        return "", "off"
    if section in (run.get("section_hints_disabled") or []):
        return "", "off"
    hint = config.SECTION_HINTS.get(section, "")
    if not hint:
        return "", "none"
    logged = (run.get("section_hints_fingerprint") or {}).get(section)
    if logged is None:
        return hint, "unverified"
    own = f"{len(hint)} {hashlib.sha256(hint.encode('utf-8')).hexdigest()[:12]}"
    return (hint, "verified") if own == logged else ("", "mismatch")


def build_prompt(evidence, context, hint, options, multi):
    parts = []
    if context:
        parts.append("Reference terminology (Brighton):\n" + context)
    if hint:
        parts.append("Section guidance:\n" + hint + "\n(Ignore any instruction in "
                     "this guidance about how to format the answer.)")
    parts.append("Evidence from the patient record:\n" + (evidence or ""))
    parts.append("Options:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1)))
    if multi:
        parts.append("For every option write one line in the form '<number>: YES' if "
                     "the evidence supports selecting it, or '<number>: NO' otherwise. "
                     "One line per option, in order, and nothing else.")
    else:
        parts.append("Reply with the number of the one option that applies, and "
                     "nothing else.")
    return "\n\n".join(parts)


def chat(user, num_predict):
    options = {"temperature": config.LLM_TEMPERATURE, "num_predict": num_predict,
               "num_gpu": config.LLM_NUM_GPU, "num_ctx": config.LLM_NUM_CTX}
    body = {"model": config.EVALUATOR_LLM_MODEL_NAME, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "options": {k: v for k, v in options.items() if v is not None},
            "logprobs": True, "top_logprobs": 10}
    if config.LLM_REASONING is not None:
        body["think"] = config.LLM_REASONING
    request = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=config.LLM_REQUEST_TIMEOUT or 900) as r:
        return json.loads(r.read())


def mass(token, accept):
    """Probability mass of the alternatives at one position that accept() keeps."""
    alternatives = token.get("top_logprobs") or [token]
    return sum(math.exp(a["logprob"]) for a in alternatives if accept(a["token"]))


def score_single(logprobs, n, chosen):
    """p(chosen) among the valid option numbers, and the scorer's own pick."""
    for token in logprobs:
        if not token["token"].strip():
            continue
        if not token["token"].strip().isdigit():
            return None, None
        masses = {k: mass(token, lambda t, k=k: t.strip() == str(k)) for k in range(1, n + 1)}
        total = sum(masses.values())
        if total == 0:
            return None, None
        return masses[chosen] / total, max(masses, key=masses.get)
    return None, None


def score_multi(logprobs, n, chosen):
    """Product over options of p(Agent 2's decision), and the scorer's own set."""
    text, p_yes = "", {}
    for token in logprobs:
        word = token["token"].strip().upper()
        if word.startswith("YES") or word.startswith("NO"):
            numbers = re.findall(r"(\d+)\s*:\s*$", text)
            if numbers and 1 <= int(numbers[-1]) <= n:
                yes = mass(token, lambda t: t.strip().upper().startswith("YES"))
                no = mass(token, lambda t: t.strip().upper().startswith("NO"))
                if yes + no > 0:
                    p_yes[int(numbers[-1])] = yes / (yes + no)
        text += token["token"]
    if len(p_yes) != n:
        return None, None
    joint = 1.0
    for k in range(1, n + 1):
        joint *= p_yes[k] if k in chosen else 1 - p_yes[k]
    return joint, {k for k, p in p_yes.items() if p > 0.5}


def auroc(scores, labels):
    """Probability that a correct answer scores above a wrong one, ties counting half."""
    good = [s for s, y in zip(scores, labels) if y]
    bad = [s for s, y in zip(scores, labels) if not y]
    if not good or not bad:
        return None
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    return wins / (len(good) * len(bad))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--gt-dir", default=os.path.join("data", "synthetic_records"))
    parser.add_argument("--only", nargs="+", metavar="ID")
    parser.add_argument("--sections", nargs="+", default=config.SECTION_ORDER)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    csv_path = args.csv or f"confidence_{os.path.basename(os.path.normpath(args.out_dir))}.csv"

    logs = latest_logs(args.out_dir)
    if args.only:
        logs = {r: p for r, p in logs.items() if any(r.startswith(o) for o in args.only)}
    if not logs:
        raise SystemExit(f"no audit log selected in {args.out_dir}")

    rows = []
    for index, (record, path) in enumerate(logs.items(), 1):
        log = json.load(open(path, encoding="utf-8"))
        run = log.get("_run_config", {})
        logged = run.get("models", {}).get("evaluator")
        if logged != config.EVALUATOR_LLM_MODEL_NAME:
            print(f"!! {record}: answers by {logged}, scored by {config.EVALUATOR_LLM_MODEL_NAME}")
        gt_path = os.path.join(args.gt_dir, f"{record}_ground_truth.json")
        truth = json.load(open(gt_path, encoding="utf-8")) if os.path.exists(gt_path) else {}
        print(f"[{index}/{len(logs)}] {record}", flush=True)

        for section in args.sections:
            entry = log.get(section)
            if not isinstance(entry, dict) or "result" not in entry:
                continue
            _, options, multi, _ = agents._get_field_info(SECTION_MODELS[section])
            answer = next(iter(entry["result"].values()))
            expected = next(iter(truth.get(section.lower(), {}).values()), None)
            if multi:
                chosen = {options.index(a) + 1 for a in (answer or []) if a in options}
                correct = None if expected is None else set(answer or []) == set(expected)
            else:
                chosen = options.index(answer) + 1 if answer in options else None
                correct = None if expected is None else answer == expected
            if not multi and chosen is None:
                continue

            hint, hint_state = hint_for(run, section)
            prompt = build_prompt(entry.get("evidence"), entry.get("brighton_context") or "",
                                  hint, options, multi)
            started = time.time()
            reply = chat(prompt, 8 * len(options) + 8 if multi else 4)
            seconds = time.time() - started
            logprobs = reply.get("logprobs") or []
            if multi:
                confidence, own = score_multi(logprobs, len(options), chosen)
                agrees = None if own is None else own == chosen
            else:
                confidence, own = score_single(logprobs, len(options), chosen)
                agrees = None if own is None else own == chosen

            rows.append({
                "record": record, "section": section, "multi": multi,
                "answer": json.dumps(answer, ensure_ascii=False),
                "correct": correct, "confidence": confidence, "scorer_agrees": agrees,
                "format_ok": confidence is not None, "hint": hint_state,
                "agent2_seconds": entry.get("agent2_seconds"),
                "scorer_seconds": round(seconds, 2),
                "scorer_output": reply["message"]["content"].strip().replace("\n", " | ")[:120],
            })
            shown = "n/a" if confidence is None else f"{confidence:.4f}"
            mark = {True: "ok", False: "WRONG", None: "?"}[correct]
            print(f"    {section:5s} {mark:5s} conf {shown:>7s}  {seconds:5.1f}s", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scored = [r for r in rows if r["confidence"] is not None and r["correct"] is not None]
    wrong = [r for r in scored if not r["correct"]]
    confidences = sorted(r["confidence"] for r in scored)
    print("\n" + "=" * 78)
    print(f"sections: {len(rows)}   scored: {len(scored)}   "
          f"format failures: {sum(not r['format_ok'] for r in rows)}   wrong: {len(wrong)}")
    print(f"hint states: { {s: sum(r['hint'] == s for r in rows) for s in sorted({r['hint'] for r in rows})} }")
    if confidences:
        print(f"confidence  median {confidences[len(confidences) // 2]:.4f}   "
              f"share above 0.99: {sum(c > 0.99 for c in confidences) / len(confidences):.1%}   "
              f"scorer disagrees with Agent 2: {sum(r['scorer_agrees'] is False for r in scored)}")
    labels = [r["correct"] for r in scored]
    a_conf = auroc([r["confidence"] for r in scored], labels)
    timed = [r for r in scored if r["agent2_seconds"] is not None]
    a_time = auroc([-r["agent2_seconds"] for r in timed], [r["correct"] for r in timed])
    print(f"AUROC confidence: {'n/a' if a_conf is None else f'{a_conf:.3f}'}   "
          f"AUROC agent2_seconds (baseline): {'n/a' if a_time is None else f'{a_time:.3f}'}")
    if wrong:
        print("\nwrong sections, lowest confidence first (rank among all scored):")
        for r in sorted(wrong, key=lambda r: r["confidence"]):
            rank = sum(c < r["confidence"] for c in confidences) + 1
            print(f"  {r['record'][:34]:34s} {r['section']:5s} conf {r['confidence']:.4f}  "
                  f"rank {rank}/{len(confidences)}  agent2 {r['agent2_seconds']}s")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
