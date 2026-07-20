"""
Step 0 evaluation script -- final architecture (reason freely, then extract
FINAL_ANSWER deterministically, with llama3:8b-instruct-q4_0).

Run this before launching pipeline.py on real data, to confirm the model
still behaves consistently (not just on a single lucky run).

Usage:
    python test_step0.py
"""

import time

from agents import build_llm, evaluate_section
from models import C_DDimer, B2_NewSymptoms

N_REPETITIONS = 10

TEST_CASES = [
    {
        "name": "C_DDimer (simple schema, single choice)",
        "model": C_DDimer,
        "evidence": (
            "The patient's D-dimer came back at 1200 ng/mL, "
            "the lab's upper limit of normal is 500 ng/mL."
        ),
        "expected_answer": "D-dimer exceeded test lab's upper limit of normal.",
    },
    {
        "name": "B2_NewSymptoms (schema with multi-select and validator)",
        "model": B2_NewSymptoms,
        "evidence": (
            "The patient has no edema. Reports pain in the left calf. "
            "No mention of redness or warmth."
        ),
        "expected_symptoms": ["Calf pain or tenderness"],
    },
]


def check_correctness(case, result) -> bool:
    if "expected_answer" in case:
        return result.answer == case["expected_answer"]
    if "expected_symptoms" in case:
        return set(result.symptoms) == set(case["expected_symptoms"])
    return True


def run_step0_evaluation():
    llm = build_llm()

    print(f"LLM under test: {llm.model}\n", flush=True)

    for case in TEST_CASES:
        print(f"--- {case['name']} ---", flush=True)

        schema_valid_count = 0
        correct_count = 0
        errors = []
        durations = []

        for i in range(N_REPETITIONS):
            print(f"  [attempt {i+1}/{N_REPETITIONS}] sending request to the model...", flush=True)
            start = time.time()
            try:
                result = evaluate_section(llm, case["model"], case["evidence"])
                elapsed = time.time() - start
                durations.append(elapsed)
                schema_valid_count += 1

                is_correct = check_correctness(case, result)
                if is_correct:
                    correct_count += 1
                    print(f"  [attempt {i+1}/{N_REPETITIONS}] OK (correct) in {elapsed:.1f}s -> {result}", flush=True)
                else:
                    print(f"  [attempt {i+1}/{N_REPETITIONS}] schema-valid but WRONG in {elapsed:.1f}s -> {result}", flush=True)
            except Exception as exc:
                elapsed = time.time() - start
                durations.append(elapsed)
                errors.append(f"attempt {i+1}: {type(exc).__name__} -- {exc}")
                print(f"  [attempt {i+1}/{N_REPETITIONS}] FAILED in {elapsed:.1f}s -> {type(exc).__name__}", flush=True)

        avg_time = sum(durations) / len(durations) if durations else 0
        print(
            f"\nSchema-valid: {schema_valid_count}/{N_REPETITIONS} "
            f"({schema_valid_count / N_REPETITIONS * 100:.0f}%) -- "
            f"Correct: {correct_count}/{N_REPETITIONS} "
            f"({correct_count / N_REPETITIONS * 100:.0f}%) -- "
            f"avg {avg_time:.1f}s/call"
        )
        if errors:
            print("Failure details:")
            for e in errors:
                print(f"  - {e}")
        print()

    print(
        "Note: 'Schema-valid' means a FINAL_ANSWER line was found and matched "
        "to a valid option. 'Correct' means the content was also factually "
        "right. Expect calls to take anywhere from ~10s to over 100s with "
        "this model on CPU -- see config.LLM_REQUEST_TIMEOUT if calls are "
        "being cut off."
    )


if __name__ == "__main__":
    run_step0_evaluation()
