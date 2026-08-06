"""
Data-augmentation script: generates synthetic Italian clinical records, paired
with matching ground-truth JSON (same shape as models.DVT_CriteriaForm), so the
pipeline's prompts/hints (config.SECTION_HINTS, EXTRACTOR_SYSTEM_PROMPT, the
deterministic gates in criteria_rules.py) can be validated on more than the
single real patient_001.txt they were originally tuned against.

TWO-PHASE DESIGN -- NOT a single LLM call:
  1. Every scenario's clinical facts AND its correct answer for all 10 Brighton
     sections are hardcoded below in plain Python, using models.py's exact
     Literal option strings. Ground truth is authored by construction, not
     guessed by an LLM -- otherwise it would not be trustworthy as a reference.
  2. A separate LLM call turns those facts into a natural, varied Italian
     clinical narrative. The prompt explicitly forbids inventing findings
     beyond the given facts, so the generated text can't silently contradict
     its own ground truth.

MODEL CHOICE: uses a model NOT used by ANY pipeline role (WRITER_MODEL_NAME
below, a plain literal -- deliberately not read from config.py), at
WRITER_TEMPERATURE=0.8 instead of the pipeline's deterministic 0.0. Reasons,
discussed with the user on 2026-08-06:
  - Reusing any pipeline model at temperature=0.0 would make every "style
    variant" of a scenario nearly identical (deterministic generation),
    defeating the point of testing phrasing robustness.
  - Writing test records with the SAME model that later extracts evidence
    from them risks a same-distribution bias -- the extractor may parse that
    model's own phrasing habits unrealistically well compared to genuine
    human-written records. This isn't limited to config.LLM_MODEL_NAME (the
    full_text/rag extractor): config.AGENTIC_LLM_MODEL_NAME (the
    agentic_graph search step) was considered and rejected for the same
    reason -- it would bias validation specifically for that mode. Using a
    model absent from every role in config.py keeps the synthetic set
    neutral regardless of which EXTRACTOR_MODE is being validated.
  - qwen2.5:7b-instruct was picked as a solid, officially-maintained Ollama
    tag (no risk of the template mismatch that broke cniongolo/biomistral)
    with strong multilingual/Italian support. Requires: ollama pull
    qwen2.5:7b-instruct.

NOTE ON LANGUAGE: scenario facts/descriptions and prompts are in ITALIAN
here, matching the language of the clinical records themselves (per the
user's request on 2026-08-06) -- keeps facts and generated output in the
same language, avoiding an extra translation step inside the writer prompt.
Everything else (docstrings, code, comments) stays in English.

KNOWN INTERPRETIVE ASSUMPTIONS -- worth checking against the Brighton paper
before treating this as ground truth for thesis results, not just pipeline
smoke-testing:
  - B1_1/B1_2 reflect what was REPORTED/suspected as a DVT syndrome, regardless
    of whether A3_1/X later confirm or rule it out (mirrors how the existing
    SECTION_HINTS are worded elsewhere in the project).
  - F's Yes/No follows criteria_rules.apply_details_gate's own convention:
    "No" covers both "reported WITH clinical details" and "not reported at
    all"; "Yes" is reserved for "reported WITHOUT details".
  - B2 has no generic "swelling" option for arms (only "Leg swelling or
    pitting oedema", worded lower-extremity-specific) -- the upper-extremity
    scenario below (03) deliberately does NOT count arm swelling toward any
    B2 option, rather than silently stretching the schema's wording.

COVERAGE: every Literal option in every one of the 10 sections is hit by at
least one scenario below (validated programmatically -- see conversation on
2026-08-06). Scenarios 11-18 were added specifically to close gaps found in
the initial 10-scenario set (A1's negative-but-done branch, A2's non-
thrombectomy branch, A3_2's Contrast venography/Other, C's within-normal-range
branch, B1_1's clearly-negative branch, F's Yes branch, B2's Absent-pulses
true positive).

Usage: python generate_synthetic_records.py
Output: data/synthetic_records/<scenario_id>_<style_id>.txt (the record) and
        data/synthetic_records/<scenario_id>_<style_id>_ground_truth.json (answers).
"""

import json
from pathlib import Path

from agents import build_llm

# Deliberately a plain literal, NOT read from config.py -- see the module
# docstring's MODEL CHOICE section: this must stay independent of whichever
# model any pipeline role (extractor, agentic search, evaluator) happens to
# use, in either mode, now or after future config changes.
WRITER_MODEL_NAME = "qwen2.5:7b-instruct"
WRITER_TEMPERATURE = 0.8

OUTPUT_DIR = Path(__file__).parent / "data" / "synthetic_records"

WRITER_SYSTEM_PROMPT = """Sei un medico di pronto soccorso che scrive cartelle
cliniche in ITALIANO per un paziente con sospetta trombosi venosa profonda (TVP).

REGOLE FERREE:
1. Usa SOLO ED ESCLUSIVAMENTE i fatti clinici elencati dall'utente. Non
   aggiungere reperti, esami, sintomi, valori di laboratorio o diagnosi che
   non siano stati esplicitamente forniti.
2. Non omettere nessuno dei fatti forniti: devono comparire tutti nel testo.
3. Non aggiungere una frase di sintesi diagnostica esplicita (es. "Si conferma
   diagnosi di TVP") a meno che non sia uno dei fatti forniti.
4. Segui la direttiva di stile indicata, ma il CONTENUTO CLINICO deve restare
   identico indipendentemente dallo stile.
5. Output SOLO il testo della cartella clinica, nessun commento, nessuna nota,
   nessuna intestazione tipo "Ecco la cartella clinica:".
"""

STYLE_VARIANTS = [
    {
        "id": "v1",
        "directive": (
            "Stile pronto soccorso, telegrafico e sintetico. Struttura in sezioni "
            "esplicite nell'ordine: Anamnesi patologica remota, Anamnesi Prossima, "
            "Esame obiettivo, Esami di laboratorio/strumentali. Usa abbreviazioni "
            "cliniche italiane comuni (PA, FC, FR, SpO2, TC) per i parametri vitali."
        ),
    },
    {
        "id": "v2",
        "directive": (
            "Stile discorsivo e narrativo, come una lettera di dimissione, SENZA "
            "intestazioni di sezione rigide. Ordina il racconto partendo "
            "dall'esame obiettivo e dagli esami strumentali, poi l'anamnesi. Scrivi "
            "i termini clinici per esteso, evita abbreviazioni."
        ),
    },
]

# ---------------------------------------------------------------------------
# Scenarios: (facts fed to the writer LLM) + (ground truth, authored by hand)
# ---------------------------------------------------------------------------
# Ground truth dict keys/values must match models.py's Literal strings EXACTLY
# -- copy-paste from models.py, do not retype from memory.

SCENARIOS = [
    {
        "id": "SYN_01_lower_dvt_doppler",
        "description": "TVP arto inferiore confermata a ecocolordoppler, quadro classico",
        "facts": [
            "Paziente donna, 58 anni",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Volo intercontinentale di 9 ore effettuato 5 giorni prima dell'esordio (fattore di rischio, non menzionare altre diagnosi)",
            "Da 3 giorni dolore al polpaccio destro ed edema progressivo dell'arto",
            "Aumento della temperatura cutanea locale al polpaccio destro",
            "D-dimero: 2.100 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra, flusso assente nel segmento trombizzato",
            "Nessuna menzione di diagnosi alternativa",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema", "Redness, warmth, or pain in one or more extremities"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_02_dvt_ruled_out_cellulitis",
        "description": "TVP esclusa, diagnosi alternativa di cellulite",
        "facts": [
            "Paziente uomo, 45 anni",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Da 2 giorni arrossamento, calore e dolore alla gamba sinistra, con febbre (TC 38.3°C)",
            "Compressione ecografica (ecografia compressiva) venosa arto inferiore sinistro: vene comprimibili, NESSUNA evidenza di trombosi",
            "D-dimero non eseguito",
            "Diagnosi dello specialista: cellulite dell'arto inferiore sinistro, confermata da leucocitosi e PCR elevata, nessun coinvolgimento venoso profondo",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done but didn't confirm DVT"},
            "a3_2": {"studies": ["Compression ultrasonography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Redness, warmth, or pain in one or more extremities"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "An alternative diagnosis was found that explained the acute illness"},
        },
    },
    {
        "id": "SYN_03_upper_extremity_catheter",
        "description": "TVP arto superiore da catetere venoso centrale, confermata a CT/RM venografia",
        "facts": [
            "Paziente donna, 62 anni, portatrice di catetere venoso centrale per chemioterapia",
            "Nessuna storia di autopsia o intervento chirurgico recente diverso dal posizionamento del catetere",
            "Da 4 giorni dolore e arrossamento al braccio destro (sede del catetere), con sensazione di calore locale",
            "Nessun edema riportato",
            "D-dimero: 1.800 ng/mL, superiore al limite di laboratorio",
            "CT venografia arto superiore destro: trombosi della vena succlavia destra associata al catetere",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["CT or MR venography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Upper extremity DVT"]},
            "b2": {"symptoms": ["Redness, warmth, or pain in one or more extremities"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_04_multi_imaging_confirmed",
        "description": "TVP confermata da due studi di imaging diversi (compressione + doppler), nessun arrossamento/calore",
        "facts": [
            "Paziente uomo, 70 anni, recente sostituzione protesica del ginocchio (intervento NON di trombectomia, solo fattore di rischio)",
            "Nessuna autopsia",
            "Da 5 giorni dolore al polpaccio sinistro ed edema dell'arto",
            "Nessun arrossamento o aumento di temperatura locale riportato",
            "D-dimero: 3.050 ng/mL, superiore al limite di laboratorio",
            "Ecografia compressiva venosa arto inferiore sinistro: vena poplitea non comprimibile",
            "Ecocolordoppler dello stesso arto, eseguito lo stesso giorno: conferma trombosi della vena poplitea sinistra",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Compression ultrasonography", "Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_05_absent_flow_not_pulses_trap",
        "description": "Flusso assente SOLO su imaging, nessun esame dei polsi periferici menzionato -- verifica che B2 non selezioni 'Absent pulses'",
        "facts": [
            "Paziente donna, 51 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 2 giorni dolore gravativo al polpaccio destro con edema progressivo",
            "Esame obiettivo NON menziona alcun esame dei polsi periferici (non riportare polsi presenti né assenti)",
            "D-dimero: 1.650 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra con flusso assente nel segmento trombizzato",
            "Vene femorali comuni e superficiali pervie",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_06_no_imaging_clinical_only",
        "description": "Nessun imaging eseguito, diagnosi clinica dello specialista con dettagli",
        "facts": [
            "Paziente uomo, 39 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 24 ore dolore al polpaccio sinistro ed edema lieve",
            "Nessun esame di imaging venoso eseguito (paziente dimesso prima di poterlo eseguire)",
            "D-dimero: 980 ng/mL, superiore al limite di laboratorio",
            "Nota dello specialista: sospetta trombosi venosa profonda in base al quadro clinico e al D-dimero elevato, in attesa di ecocolordoppler ambulatoriale",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_07_thrombectomy_positive",
        "description": "TVP confermata, trattata con trombectomia chirurgica d'urgenza (ramo positivo di A2, raro)",
        "facts": [
            "Paziente uomo, 66 anni",
            "Nessuna autopsia",
            "Da 6 ore dolore acuto e edema improvviso dell'arto inferiore sinistro, con cute fredda e pallida",
            "Ecocolordoppler venoso urgente: trombosi ileo-femorale sinistra estesa",
            "Sottoposto a trombectomia chirurgica d'urgenza per trombosi venosa ileo-femorale, con conferma intraoperatoria del trombo",
            "D-dimero: 4.400 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "Thrombectomy related to DVT performed"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_08_autopsy_positive",
        "description": "Paziente deceduto, TVP/embolia polmonare confermata solo all'autopsia (ramo positivo di A1, raro)",
        "facts": [
            "Paziente uomo, 74 anni, deceduto in Pronto Soccorso poco dopo l'arrivo",
            "Anamnesi familiare riferisce dolore ed edema al polpaccio destro comparsi 3 giorni prima del decesso",
            "Nessun esame di imaging o D-dimero eseguito prima del decesso",
            "Nessun intervento chirurgico eseguito",
            "Riscontro autoptico: trombosi venosa profonda della vena poplitea destra con embolia polmonare massiva come causa del decesso",
            "Nessuna diagnosi alternativa identificata all'autopsia",
        ],
        "ground_truth": {
            "a1": {"answer": "Autopsy showed presence of DVT"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_09_risk_factor_not_alt_diagnosis",
        "description": "Fattori di rischio (volo, terapia ormonale) esplicitamente presenti ma NON trattati come diagnosi alternativa -- regressione mirata sul fix di X",
        "facts": [
            "Paziente donna, 34 anni, in terapia ormonale contraccettiva da 2 anni",
            "Volo intercontinentale di 12 ore effettuato 6 giorni prima dell'esordio dei sintomi",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 3 giorni dolore al polpaccio sinistro ed edema progressivo",
            "D-dimero: 1.500 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto inferiore sinistro: trombosi venosa poplitea sinistra",
            "Nessuna diagnosi alternativa identificata; il volo e la terapia ormonale sono annotati solo come fattori di rischio anamnestici, non come diagnosi",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_10_unclear_sparse_record",
        "description": "Cartella clinica molto scarna e vaga -- verifica le opzioni 'unknown'/di default su piu' sezioni contemporaneamente",
        "facts": [
            "Paziente uomo, 55 anni",
            "Riferisce generico 'fastidio alla gamba' da qualche giorno, senza ulteriori dettagli su sede, intensita' o caratteristiche",
            "Nessun esame obiettivo dettagliato riportato in cartella",
            "Nessun esame di imaging eseguito",
            "Nessun D-dimero eseguito",
            "Nessuna nota di diagnosi da parte di uno specialista presente in cartella",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "It is unknown if there was a report of a DVT syndrome"},
            "b1_2": {"types": []},
            "b2": {"symptoms": ["None of the above were present or it is unknown if any of 1-4 were present"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_11_autopsy_negative",
        "description": "Decesso per causa non correlata, autopsia eseguita ma NEGATIVA per TVP (ramo A1 mancante)",
        "facts": [
            "Paziente donna, 80 anni, deceduta in ospedale per arresto cardiaco improvviso",
            "Nessun intervento chirurgico recente",
            "Nessun sintomo di gonfiore, dolore, arrossamento o calore agli arti riportato prima del decesso",
            "Nessun esame di imaging venoso o D-dimero eseguito in vita",
            "Riscontro autoptico: esame del sistema venoso profondo degli arti inferiori negativo, nessuna evidenza di trombosi venosa profonda",
            "Causa del decesso: infarto miocardico acuto, non correlato a fenomeni tromboembolici venosi",
        ],
        "ground_truth": {
            "a1": {"answer": "Autopsy done but showed no evidence of DVT"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "There was no report of a recognized DVT syndrome"},
            "b1_2": {"types": []},
            "b2": {"symptoms": ["None of the above were present or it is unknown if any of 1-4 were present"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_12_other_procedure_positive",
        "description": "TVP confermata da una procedura diversa dalla trombectomia (trombolisi con venografia intraprocedurale) -- ramo A2 mancante",
        "facts": [
            "Paziente uomo, 52 anni",
            "Nessuna autopsia",
            "Da 2 giorni dolore al polpaccio ed edema severo dell'arto inferiore sinistro",
            "Ecocolordoppler venoso: trombosi venosa ileo-femorale sinistra estesa",
            "Sottoposto a trombolisi farmaco-meccanica per via percutanea con catetere; la venografia intraprocedurale ha confermato la presenza del trombo nella vena iliaca sinistra",
            "D-dimero: 3.700 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "Other procedure done that confirmed presence of DVT"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_13_contrast_venography_only",
        "description": "Ecografia non dirimente, TVP confermata solo con flebografia a contrasto -- ramo A3_2 mancante",
        "facts": [
            "Paziente donna, 47 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 4 giorni dolore al polpaccio ed edema dell'arto inferiore destro",
            "Ecografia compressiva risultata non dirimente per artefatti tecnici (NON ha confermato nulla)",
            "Flebografia con mezzo di contrasto arto inferiore destro: difetto di riempimento centrale a carico della vena poplitea destra, compatibile con trombosi venosa profonda",
            "D-dimero: 2.400 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Contrast venography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_14_other_imaging_incidental",
        "description": "TVP scoperta incidentalmente su TC total body eseguita per altro motivo -- ramo A3_2 'Other' mancante",
        "facts": [
            "Paziente uomo, 68 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "TC total body eseguita per stadiazione oncologica (motivo NON correlato a sospetta TVP) ha incidentalmente mostrato trombosi della vena femorale sinistra",
            "Successivamente riferito dolore lieve al polpaccio sinistro, presente da alcuni giorni",
            "D-dimero: 1.900 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Other"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_15_ddimer_negative_low_probability",
        "description": "Bassa probabilita' clinica, D-dimero negativo, TVP esclusa senza necessita' di imaging -- ramo C mancante",
        "facts": [
            "Paziente donna, 29 anni, bassa probabilita' clinica pre-test",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 1 giorno lieve fastidio al polpaccio sinistro, senza edema ne' arrossamento",
            "D-dimero: 320 ng/mL, entro il range di normalita' del laboratorio",
            "Data la bassa probabilita' clinica e il D-dimero negativo, nessun esame di imaging venoso eseguito; TVP esclusa clinicamente",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness"]},
            "c": {"answer": "D-dimer tested and was within test lab's range of normal"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_16_no_dvt_syndrome_reported",
        "description": "Accesso per motivo non correlato, nega esplicitamente sintomi di TVP -- ramo B1_1 negativo mancante",
        "facts": [
            "Paziente uomo, 41 anni, giunto in Pronto Soccorso per dolore toracico atipico",
            "Nessuna autopsia o intervento chirurgico recente",
            "Nega esplicitamente dolore, gonfiore, arrossamento o calore agli arti inferiori o superiori",
            "Nessun esame di imaging venoso eseguito",
            "D-dimero non eseguito",
            "Dolore toracico attribuito a causa muscoloscheletrica, nessun sospetto di TVP sollevato durante la valutazione",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "There was no report of a recognized DVT syndrome"},
            "b1_2": {"types": []},
            "b2": {"symptoms": ["None of the above were present or it is unknown if any of 1-4 were present"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_17_diagnosis_no_details",
        "description": "Diagnosi di TVP riferita dal medico curante SENZA alcun dettaglio clinico di supporto -- ramo F='Yes' mancante",
        "facts": [
            "Paziente donna, 60 anni",
            "Riferita dal medico di base con diagnosi di trombosi venosa profonda dell'arto inferiore sinistro",
            "Nessun dettaglio clinico, esame obiettivo, di laboratorio o di imaging disponibile in questa sede oltre alla diagnosi riferita",
            "Nessuna autopsia o intervento chirurgico recente",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "It is unknown if there was a report of a DVT syndrome"},
            "b1_2": {"types": []},
            "b2": {"symptoms": ["None of the above were present or it is unknown if any of 1-4 were present"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "Yes"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_18_absent_pulses_true_positive",
        "description": "Polsi periferici realmente assenti ALL'ESAME OBIETTIVO (flegmasia cerulea dolens) -- controparte positiva della trappola dello scenario 05",
        "facts": [
            "Paziente uomo, 63 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 12 ore edema massivo e dolore severo al polpaccio e all'arto inferiore sinistro, cute cianotica",
            "Esame obiettivo: polsi periferici (pedidio e tibiale posteriore) NON palpabili all'arto inferiore sinistro",
            "Ecocolordoppler venoso: trombosi venosa ileo-femorale sinistra estesa, quadro compatibile con flegmasia cerulea dolens",
            "D-dimero: 5.200 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema", "Absent pulses in legs or arms"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
]


def facts_to_prompt(scenario: dict, style: dict) -> str:
    facts_block = "\n".join(f"- {fact}" for fact in scenario["facts"])
    return (
        f"Direttiva di stile: {style['directive']}\n\n"
        f"Fatti clinici da includere (tutti, nessuno escluso, nessuno aggiunto):\n{facts_block}"
    )


def build_ground_truth(record_id: str, scenario: dict) -> dict:
    gt = {"record_id": record_id}
    gt.update(scenario["ground_truth"])
    return gt


def generate_record(llm, scenario: dict, style: dict) -> str:
    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", facts_to_prompt(scenario, style)),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    llm = build_llm(WRITER_MODEL_NAME, temperature=WRITER_TEMPERATURE)

    for scenario in SCENARIOS:
        for style in STYLE_VARIANTS:
            record_id = f"{scenario['id']}_{style['id']}"
            print(f"[{record_id}] generating ({scenario['description']})...", flush=True)

            record_text = generate_record(llm, scenario, style)

            txt_path = OUTPUT_DIR / f"{record_id}.txt"
            json_path = OUTPUT_DIR / f"{record_id}_ground_truth.json"

            txt_path.write_text(record_text, encoding="utf-8")
            json_path.write_text(
                json.dumps(build_ground_truth(record_id, scenario), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[{record_id}] saved -> {txt_path.name}, {json_path.name}", flush=True)

    print(f"\nDone: {len(SCENARIOS) * len(STYLE_VARIANTS)} records in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
