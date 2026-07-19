"""
Step 0 evaluation script.

Run this BEFORE launching pipeline.py on real data. Requires Ollama to be
installed and running, and the model configured in config.py to already be
pulled (`ollama pull <model_name>`).

Usage:
    python test_step0.py

Interpreting the output:
- 100% success rate on both schemas, across repetitions -> you can trust
  .with_structured_output() and proceed with the pipeline as is.
- High but not 100% success rate (e.g. 8/10) -> the model mostly works but is
  not fully reliable: keep the retry mechanism already present in
  evaluate_section() (max_retries) active, and monitor failures in the logs
  during real use.
- Low success rate (below ~70%) or systematic exceptions -> do not trust
  native tool-calling: implement the JSON-mode fallback described in the
  evaluate_section() docstring in agents.py (explicit prompt with the schema
  + section_model.model_validate_json() with retry), instead of relying on
  .with_structured_output().
"""

from agents import build_llm
from models import C_DDimer, B2_NewSymptoms

N_REPETITIONS = 10

TEST_CASES = [
    {
        "name": "C_DDimer (simple schema, single choice)",
        "model": C_DDimer,
        "prompt": (
            "Evidence: the patient's D-dimer came back at 1200 ng/mL, "
            "the lab's upper limit of normal is 500 ng/mL. "
            "Fill in the schema."
        ),
    },
    {
        "name": "B2_NewSymptoms (schema with multi-select and validator)",
        "model": B2_NewSymptoms,
        "prompt": (
            "Evidence: the patient has no edema. Reports pain in the left "
            "calf. No mention of redness or warmth. "
            "Fill in the schema."
        ),
    },
]


def run_step0_evaluation():
    llm = build_llm()

    print(f"LLM under test: {llm.model}\n")

    for case in TEST_CASES:
        print(f"--- {case['name']} ---")
        structured_llm = llm.with_structured_output(case["model"])

        successes = 0
        errors = []

        for i in range(N_REPETITIONS):
            try:
                result = structured_llm.invoke(case["prompt"])
                if isinstance(result, case["model"]):
                    successes += 1
                else:
                    errors.append(f"attempt {i+1}: unexpected return type ({type(result)})")
            except Exception as exc:
                errors.append(f"attempt {i+1}: {type(exc).__name__} -- {exc}")

        rate = successes / N_REPETITIONS * 100
        print(f"Successes: {successes}/{N_REPETITIONS} ({rate:.0f}%)")
        if errors:
            print("Failure details:")
            for e in errors:
                print(f"  - {e}")
        print()

    print(
        "Read the results interpretation in this file's top docstring "
        "before deciding whether to proceed with .with_structured_output() "
        "or switch to the JSON-mode fallback."
    )


if __name__ == "__main__":
    run_step0_evaluation()
