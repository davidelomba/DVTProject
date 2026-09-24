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
    """p(chosen) among the valid option numbers, the scorer's own pick, and
    whether the number came after other text.

    Read at the first token that is a valid option number. When the model wrote
    something before it, as F does when its hint asks for a DETAILS_PRESENT
    line, the probability is conditioned on that text.
    """
    after_text = False
    for token in logprobs:
        word = token["token"].strip()
        if not word:
            continue
        if not (word.isdigit() and 1 <= int(word) <= n):
            after_text = True
            continue
        masses = {k: mass(token, lambda t, k=k: t.strip() == str(k)) for k in range(1, n + 1)}
        total = sum(masses.values())
        if total == 0:
            return None, None, after_text
        return masses[chosen] / total, max(masses, key=masses.get), after_text
    return None, None, after_text


def score_details(logprobs, options, chosen):
    """F's confidence from the DETAILS_PRESENT token and the number after it.

    F's hint makes present details imply "No", and absent details imply "Yes"
    only when a diagnosis was reported, "No" otherwise. The number the model
    writes after its DETAILS_PRESENT line carries that second condition, but
    only for the branch it wrote. So:
      - model wrote "no":  P(Yes) = p(no) q(Yes|no),
                           P(No)  = p(yes) + p(no) q(No|no)    exact under the hint
      - model wrote "yes": P(a) >= p(yes) q(a|yes)             lower bound, the
                           "no" branch not having been generated

    Returns (confidence of Agent 2's answer, scorer's own pick, method), or
    (None, None, None) when the DETAILS_PRESENT token or the number is missing.
    """
    if "Yes" not in options or "No" not in options:
        return None, None, None
    number = {"Yes": options.index("Yes") + 1, "No": options.index("No") + 1}
    text, p_yes = "", None
    for position, token in enumerate(logprobs):
        word = token["token"].strip().lower()
        if re.search(r"DETAILS_PRESENT\s*:\s*$", text) and (word.startswith("yes") or word.startswith("no")):
            yes = mass(token, lambda t: t.strip().lower().startswith("yes"))
            no = mass(token, lambda t: t.strip().lower().startswith("no"))
            if yes + no == 0:
                return None, None, None
            p_yes, branch = yes / (yes + no), ("yes" if word.startswith("yes") else "no")
            rest = logprobs[position + 1:]
            break
        text += token["token"]
    if p_yes is None:
        return None, None, None

    n = len(options)
    q = None
    for token in rest:
        word = token["token"].strip()
        if word.isdigit() and 1 <= int(word) <= n:
            masses = {k: mass(token, lambda t, k=k: t.strip() == str(k)) for k in range(1, n + 1)}
            total = sum(masses.values())
            if total:
                q = {label: masses[k] / total for label, k in number.items()}
            own = "Yes" if int(word) == number["Yes"] else "No"
            break
    if q is None:
        return None, None, None

    answer = "Yes" if chosen == number["Yes"] else "No"
    if branch == "no":
        p = {"Yes": (1 - p_yes) * q["Yes"], "No": p_yes + (1 - p_yes) * q["No"]}
        return p[answer], number[own], "details_exact"
    p_branch = {"Yes": p_yes * q["Yes"], "No": p_yes * q["No"]}
    return p_branch[answer], number[own], "details_lower_bound"


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
            # A single-choice answer is one token, but F's hint makes the model
            # write a DETAILS_PRESENT line first, which alone takes several.
            reply = chat(prompt, 8 * len(options) + 8 if multi else 32)
            seconds = time.time() - started
            logprobs = reply.get("logprobs") or []
            method = None
            if multi:
                confidence, own = score_multi(logprobs, len(options), chosen)
                after_text, method = False, "multi"
            else:
                confidence, own, method = (None, None, None)
                after_text = False
                if section == "F" and "DETAILS_PRESENT" in reply["message"]["content"]:
                    confidence, own, method = score_details(logprobs, options, chosen)
                if confidence is None:
                    confidence, own, after_text = score_single(logprobs, len(options), chosen)
                    method = "single"
            agrees = None if own is None else own == chosen

            rows.append({
                "record": record, "section": section, "multi": multi,
                "answer": json.dumps(answer, ensure_ascii=False),
                "correct": correct, "confidence": confidence, "scorer_agrees": agrees,
                "format_ok": confidence is not None, "method": method,
                "after_text": after_text,
                "hint": hint_state,
                "agent2_seconds": entry.get("agent2_seconds"),
                "scorer_seconds": round(seconds, 2),
                "done_reason": reply.get("done_reason"),
                "scorer_output": reply["message"]["content"].strip().replace("\n", " | ")[:120],
            })
            shown = "n/a" if confidence is None else f"{confidence:.4f}"
            mark = {True: "ok", False: "WRONG", None: "?"}[correct]
            note = "  (after text)" if after_text else ""
            if method and method.startswith("details"):
                note = f"  ({method})"
            if confidence is None:
                note += (f"  done_reason={reply.get('done_reason')} "
                         f"output={reply['message']['content'].strip()[:60]!r}")
            print(f"    {section:5s} {mark:5s} conf {shown:>7s}  {seconds:5.1f}s{note}", flush=True)

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
    print(f"answers read after other text: {sum(r['after_text'] for r in rows)}   "
          f"methods: { {m: sum(r['method'] == m for r in rows) for m in sorted({str(r['method']) for r in rows})} }")

    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    for kind, is_multi in (("single choice", False), ("multiple choice", True)):
        part = [r for r in scored if r["multi"] == is_multi]
        part_timed = [r for r in part if r["agent2_seconds"] is not None]
        values = sorted(r["confidence"] for r in part)
        median = f"{values[len(values) // 2]:.4f}" if values else "n/a"
        print(f"  {kind:16s} n={len(part):4d}  wrong={sum(not r['correct'] for r in part):3d}  "
              f"median conf {median}  "
              f"AUROC conf {fmt(auroc([r['confidence'] for r in part], [r['correct'] for r in part]))}  "
              f"AUROC time {fmt(auroc([-r['agent2_seconds'] for r in part_timed], [r['correct'] for r in part_timed]))}")
    if wrong:
        print("\nwrong sections, lowest confidence first (rank among all scored):")
        for r in sorted(wrong, key=lambda r: r["confidence"]):
            rank = sum(c < r["confidence"] for c in confidences) + 1
            print(f"  {r['record'][:34]:34s} {r['section']:5s} conf {r['confidence']:.4f}  "
                  f"rank {rank}/{len(confidences)}  agent2 {r['agent2_seconds']}s")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
