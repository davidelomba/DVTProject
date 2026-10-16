"""
Runs Agent 3's verifier again on a run already on disk, with a chosen model.

The verifier never changes an answer, and the audit log keeps everything it
reads: the evidence, the reasoning and, in the result file, the final answers.
So the verifier can be applied afresh to a stored run, and two judges compared
on exactly the same answers, without running Agents 1 and 2 again.

For every pair of result file and audit log in the run directory, the script
drops the verdicts already there, calls verifier.verify_record on the stored
answers, the same function the pipeline calls, with config.VERIFIER_MODEL_NAME
set to the chosen model and config.VERIFIER_INPUTS to the variants of
--inputs, and writes both files under the same names into the output
directory. The answers are copied unchanged. _run_config.verifier records the
model and the other verifier settings used (verifier.verifier_settings), plus
"replayed_from" (the source directory), "replayed_at", "hostname" and
"ollama_version" of the replay. evaluate_verifier.py and agreement_runs.py read
the output directory as they read a run.

The "_hint" variants send the hint the current code gives the section:
config.section_hint, with the myocarditis hints under --domain myo. Its digest
is recorded in _run_config.verifier.hints, and the script says so when it
differs from the hints the run itself was produced with.

Before the first record, one request checks that the model answers with token
probabilities; without them there is no P(YES) and the script stops.

Usage:
    python replay_verifier.py ./output_stripped --model medgemma:27b \
        --output-dir ./output_stripped_judge_medgemma
    python replay_verifier.py ./myo/output_t05_np2048_v2_r1 --domain myo \
        --model medgemma:27b --output-dir ./myo/output_t05_np2048_v2_r1_judge_medgemma
    python replay_verifier.py ./output_stripped --model qwen3.6:27b \
        --inputs answer reasoning answer_hint reasoning_hint \
        --output-dir ./output_stripped_judge_qwen_hint
"""

import argparse
import json
import socket
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("run_dir", type=Path, help="directory of the stored run")
    parser.add_argument("--model", required=True, help="Ollama tag of the judge")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory the replayed files are written to")
    parser.add_argument("--domain", choices=("dvt", "myo"), default="dvt",
                        help="questionnaire of the run (default dvt)")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="verifier variants (default config.VERIFIER_INPUTS): "
                             "answer, reasoning, answer_hint, reasoning_hint")
    parser.add_argument("--force", action="store_true",
                        help="write into an output directory that already holds files")
    return parser.parse_args()


def _install_domain(domain: str):
    """Installs the myocarditis schema and hints before verifier is imported."""

    if domain == "myo":
        sys.path.insert(0, str(PROJECT_ROOT / "myo"))
        import myo_schema  # noqa: F401  -- registers models_myo as `models`
        import config
        import models
        import prompts_myo
        config.SECTION_ORDER = list(models.SECTION_MODELS)
        config.SECTION_HINTS = prompts_myo.SECTION_HINTS
        config.SECTION_HINTS_DISABLED = set()


def _ollama_version(base_url: str):
    try:
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=10) as response:
            return json.loads(response.read()).get("version")
    except Exception:
        return None


def _check_model(verifier) -> None:
    """One request; stops the script when no verdict with probabilities comes back."""

    try:
        reply = verifier._chat("Is the following statement true: 2 + 2 = 4? Reply YES or NO.")
    except Exception as exc:
        sys.exit(f"The model {verifier.verifier_model()} did not answer: {exc}")
    agree, p_yes, method = verifier.read_verdict(reply.get("logprobs") or [])
    if method != "verdict":
        sys.exit(f"The model {verifier.verifier_model()} returned no YES/NO with "
                 f"token probabilities: {reply.get('message', {}).get('content')!r}")
    print(f"check: {verifier.verifier_model()} answered {'YES' if agree else 'NO'}, "
          f"P(YES) {p_yes:.4f}", flush=True)


def main():
    args = _parse_args()
    _install_domain(args.domain)

    import config
    import verifier
    from aggregation import VERIFIER_KEY, verifier_verdicts

    config.VERIFIER_MODEL_NAME = args.model
    if args.inputs:
        unknown = [v for v in args.inputs if v not in verifier.VARIANTS]
        if unknown:
            sys.exit(f"unknown variants {unknown}; valid: {list(verifier.VARIANTS)}")
        config.VERIFIER_INPUTS = tuple(args.inputs)
    config.VERIFIER_ENABLED = True

    if not args.run_dir.is_dir():
        sys.exit(f"{args.run_dir} is not a directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        sys.exit(f"{args.output_dir} already holds files; move it or pass --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for path in sorted(args.run_dir.glob("*.json")):
        if path.name.endswith("_audit_log.json"):
            continue
        audit_path = path.with_name(path.stem + "_audit_log.json")
        if audit_path.exists():
            pairs.append((path, audit_path))
    if not pairs:
        sys.exit(f"no result file with its audit log in {args.run_dir}")

    print(f"{len(pairs)} records from {args.run_dir}, judge {args.model}, "
          f"sections {config.SECTION_ORDER}, variants {list(config.VERIFIER_INPUTS)}",
          flush=True)
    settings = verifier.verifier_settings()
    if settings["hints"] is not None:
        first_audit = json.loads(pairs[0][1].read_text(encoding="utf-8"))
        run_hints = ((first_audit.get("_run_config") or {}).get("section_hints_fingerprint") or {}).get("all")
        same = "same as" if run_hints == settings["hints"] else "differs from"
        print(f"hints given to the verifier {settings['hints']}, {same} the run's {run_hints}",
              flush=True)
    _check_model(verifier)

    replay_info = {
        "replayed_from": str(args.run_dir),
        "replayed_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "ollama_version": _ollama_version(verifier.OLLAMA_URL),
    }

    for number, (path, audit_path) in enumerate(pairs, 1):
        prediction = json.loads(path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        print(f"\n[{number}/{len(pairs)}] {prediction.get('record_id', path.stem)}", flush=True)

        for key, entry in audit.items():
            if not key.startswith("_") and isinstance(entry, dict):
                entry.pop("verifier", None)

        form_data = {key.lower(): prediction.get(key.lower()) for key in config.SECTION_ORDER}
        verifier.verify_record(form_data, audit)

        run_config = audit.setdefault("_run_config", {})
        run_config["verifier"] = {**settings, **replay_info}

        prediction.pop(VERIFIER_KEY, None)
        verdicts = verifier_verdicts(audit)
        if verdicts:
            prediction[VERIFIER_KEY] = verdicts

        (args.output_dir / path.name).write_text(
            json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output_dir / audit_path.name).write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(pairs)} records written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
