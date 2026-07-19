"""
Script di valutazione dello Step 0.

Esegui questo PRIMA di lanciare pipeline.py su dati reali. Richiede che Ollama
sia installato e in esecuzione, e che il modello configurato in config.py sia
gia' stato scaricato (`ollama pull <nome_modello>`).

Uso:
    python test_step0.py

Interpretazione dell'output:
- Tasso di successo 100% su entrambi gli schemi, su piu' ripetizioni -> puoi
  fidarti di .with_structured_output() e procedere con la pipeline cosi' com'e'.
- Tasso di successo alto ma non 100% (es. 8/10) -> il modello regge ma non e'
  affidabile al 100%: tieni attivo il meccanismo di retry gia' presente in
  evaluate_section() (max_retries) e monitora i fallimenti nei log durante
  l'uso reale.
- Tasso di successo basso (sotto ~70%) o eccezioni sistematiche -> non fidarti
  del tool-calling nativo: implementa il fallback JSON-mode descritto nel
  docstring di evaluate_section() in agents.py (prompt esplicito con lo schema
  + section_model.model_validate_json() con retry), invece di affidarti a
  .with_structured_output().
"""

from agents import build_llm
from models import C_DDimer, B2_NewSymptoms

N_REPETITIONS = 10

TEST_CASES = [
    {
        "name": "C_DDimer (schema semplice, scelta singola)",
        "model": C_DDimer,
        "prompt": (
            "Evidenze: il D-dimero del paziente e' risultato 1200 ng/mL, "
            "il limite superiore di normalita' del laboratorio e' 500 ng/mL. "
            "Compila lo schema."
        ),
    },
    {
        "name": "B2_NewSymptoms (schema con multi-select e validator)",
        "model": B2_NewSymptoms,
        "prompt": (
            "Evidenze: il paziente non presenta edema. Riferisce dolore al "
            "polpaccio sinistro. Nessuna descrizione di arrossamento o calore. "
            "Compila lo schema."
        ),
    },
]


def run_step0_evaluation():
    llm = build_llm()

    print(f"Modello LLM in test: {llm.model}\n")

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
                    errors.append(f"tentativo {i+1}: tipo restituito inatteso ({type(result)})")
            except Exception as exc:
                errors.append(f"tentativo {i+1}: {type(exc).__name__} -- {exc}")

        rate = successes / N_REPETITIONS * 100
        print(f"Successi: {successes}/{N_REPETITIONS} ({rate:.0f}%)")
        if errors:
            print("Dettaglio fallimenti:")
            for e in errors:
                print(f"  - {e}")
        print()

    print(
        "Leggi l'interpretazione dei risultati nel docstring in cima a questo file "
        "prima di decidere se procedere con .with_structured_output() o passare "
        "al fallback JSON-mode."
    )


if __name__ == "__main__":
    run_step0_evaluation()
