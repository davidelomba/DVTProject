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
2026-08-06). Scenarios 11-18 were added to close gaps found in the initial
10-scenario set (A1's negative-but-done branch, A2's non-thrombectomy branch,
A3_2's Contrast venography/Other, C's within-normal-range branch, B1_1's
clearly-negative branch, F's Yes branch, B2's Absent-pulses true positive).
Scenarios 19-24 were added on 2026-08-07 to thicken branches that were still
down to a single example each -- notably autopsy (A1's two positive branches
only had one scenario apiece) and upper-extremity DVT (B1_2's "Upper
extremity DVT" only had SYN_03, always paired with CT venography) -- and to
decouple two branches that had been confounded in a single scenario each
(A3_1's "didn't confirm" and X's "alternative diagnosis found" both used to
live only in SYN_02, together). Scenarios 25-29 (same day) covered the
remaining single-example branches: A2's "Thrombectomy", A3_2's "CT or MR
venography"/"Contrast venography"/"Other", B2's "Absent pulses in legs or
arms", and F's "Yes" -- every Literal option across all 10 sections now has
at least 2 scenarios (validated programmatically). Scenario 30 was added on
2026-08-16 for a different reason: not to cover a missing option, but to break
a spurious CORRELATION between two sections that per-option coverage checks
cannot detect (elevated D-dimer always co-occurring with confirmed DVT) -- see
that scenario's own comment, and note the same audit could usefully be applied
to other section pairs.

STYLE: only one narrative style (STYLE_VARIANTS' "v2") is generated per
scenario -- the earlier telegraphic/abbreviation-heavy "v1" style was dropped
per the user's request on 2026-08-07 (kept producing records terse enough to
lose clinical nuance the ground truth depended on).

STYLE FIDELITY: STYLE_VARIANTS' directive is modelled on the single real
record supplied by clinicians (data/patient_001.txt) -- inline section labels,
abbreviated vitals, home medications, pertinent negatives, one continuous
block, ~1450 characters. See that constant's comment for the measured
divergences this is meant to close. Scenario dicts may also carry a
"writer_notes" list: constraints ABOUT the writing (never content), kept out
of "facts" because the writer used to copy such parentheticals verbatim into
the record.

RECORD LENGTH -- KNOWN LIMITATION, deliberately accepted for now: records are
short (~1450 characters even after the style change above), i.e. roughly ONE
config.EHR_CHUNK_SIZE chunk, while
config.EHR_RETRIEVER_K asks for 5. Retrieval therefore returns the whole
record every time -- measured across a full 29-record run, the evidence Agent
1 passed to Agent 2 had a median length of 1.00x the source text (min 1.00,
max 1.16). That means "rag" and "agentic_graph" currently produce the same
evaluator input as "full_text", so the three config.EXTRACTOR_MODEs cannot be
compared meaningfully on this dataset. A padding mechanism (routine
DVT-irrelevant background, sized to span several chunks) was implemented and
then reverted on 2026-08-16: real clinical records are expected from
collaborators, and Davide prefers to run the mode comparison on those rather
than on synthetically-lengthened text. Revisit when those arrive.

FIDELITY CHECK: every generated record is verified before being saved (see
check_record and generate_checked_record) and re-rolled if it fails, because
the writer has repeatedly been caught dropping or altering supplied facts.
The same checks can be run over an already-generated corpus, without calling
any LLM, via:  python generate_synthetic_records.py --check

Usage: python generate_synthetic_records.py
Output: data/synthetic_records/<scenario_id>_<style_id>.txt (the record) and
        data/synthetic_records/<scenario_id>_<style_id>_ground_truth.json (answers).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from agents import build_llm

# Deliberately a plain literal, NOT read from config.py -- see the module
# docstring's MODEL CHOICE section: this must stay independent of whichever
# model any pipeline role (extractor, agentic search, evaluator) happens to
# use, in either mode, now or after future config changes.
WRITER_MODEL_NAME = "qwen2.5:7b-instruct"
WRITER_TEMPERATURE = 0.8
# Overrides config.LLM_NUM_PREDICT (512 tokens, sized for the pipeline's short
# structured answers) for the writer only -- see agents.build_llm's num_predict
# parameter for the incident this prevents. 512 tokens is about 1750 Italian
# characters, i.e. just under the length this generator asks for, so the cap
# was cutting records off mid-sentence instead of visibly failing.
WRITER_NUM_PREDICT = 3072
# Regeneration attempts per record when check_record() rejects the output.
# The writer runs at temperature 0.8, so a retry genuinely resamples rather
# than reproducing the same defective text.
WRITER_MAX_ATTEMPTS = 3

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
6. Scrivi come un medico che documenta un paziente, MAI come qualcuno che
   compila o commenta un questionario: non scrivere frasi del tipo "non sono
   state menzionate diagnosi alternative", "non e' stato riportato X" o "questo
   e' solo un fattore di rischio". Se un elemento non fa parte dei fatti,
   semplicemente non compare nel referto: non dichiararne l'assenza.
7. La direttiva di stile puo' chiederti di aggiungere elementi di contorno
   realistici (parametri vitali, terapia domiciliare, negazioni pertinenti):
   questi sono l'UNICA eccezione consentita alla regola 1, e non devono mai
   contraddire i fatti clinici forniti ne' riguardare gli stessi sintomi,
   esami o procedure di cui i fatti parlano.
"""

# Informed by -- but deliberately NOT a copy of -- the one real record
# available (data/patient_001.txt, supplied by clinicians). A 2026-08-16
# comparison found the generated records diverged from it on almost every
# stylistic axis: ~900 characters against the real record's ~1450, no section
# labels (6/29), no vital signs (4/29), no pertinent negatives (6/29), no home
# medication list (0/29), virtually no incidental quantitative detail (1/29).
# Closing that gap matters for external validity: measuring the pipeline only
# on clean, signal-dense text overstates how it would do on real records.
#
# WHY THIS IS PHRASED GENERICALLY: the directive asks for the STRUCTURAL
# features that Italian hospital records share as a genre (labelled sections,
# abbreviated vitals, home medications, pertinent negatives, continuous prose),
# and deliberately avoids dictating the exact section labels, drug names or
# negation wording. An earlier draft quoted patient_001 almost verbatim -- but
# a writer LLM copies the examples it is given, so prescribing them would have
# produced 27 near-identical variants of a single real document: homogeneous
# rather than realistic, and overfitted to the idiosyncrasies of one clinician
# at one department (n=1). Leaving the surface wording free lets the writer's
# temperature 0.8 supply the variation that the removed second style variant
# used to provide, without doubling the number of records to generate and run.
STYLE_VARIANTS = [
    {
        "id": "v2",
        "directive": (
            "Scrivi il documento come un referto ospedaliero italiano reale, in "
            "prosa continua (nessun elenco puntato). Organizza il contenuto nelle "
            "sezioni tipiche di un referto italiano -- anamnesi remota, anamnesi "
            "prossima, esame obiettivo, esami di laboratorio e strumentali -- "
            "introducendole con una breve etichetta in linea nel testo; scegli tu "
            "la formulazione esatta delle etichette e l'ordine piu' naturale. "
            "Riporta i parametri vitali con le abbreviazioni cliniche italiane "
            "d'uso comune e valori plausibili nella norma, salvo diversa "
            "indicazione nei fatti. "
            "Includi elementi anamnestici di contorno realistici e non correlati "
            "al quesito diagnostico: una breve terapia domiciliare, lo stato "
            "allergologico, e alcune negazioni pertinenti su sintomi NON gia' "
            "citati nei fatti. Varia il lessico e non riutilizzare formule fisse. "
            "Non negare MAI qualcosa che i fatti riportano come presente. "
            "Lunghezza complessiva indicativa: 1300-1600 caratteri."
        ),
        # Scenarios whose ground truth is F="Yes" ("diagnosis reported WITHOUT
        # details") cannot use the directive above: a single vital sign, lab
        # value or examination finding would make the record detailed and flip
        # F's correct answer to "No". They get this instead -- which is also
        # what such a document looks like in reality, since a bare referral or
        # inter-hospital transfer note genuinely is short and finding-free.
        "directive_no_details": (
            "Scrivi il documento come una breve nota di segnalazione o di "
            "trasferimento ospedaliera italiana reale, in prosa continua. "
            "Riporta i dati anagrafici, la provenienza della segnalazione e la "
            "diagnosi riferita, e indica che non e' disponibile altra "
            "documentazione clinica. "
            "NON inventare e NON riportare alcun parametro vitale, valore di "
            "laboratorio, reperto di esame obiettivo o risultato strumentale: la "
            "loro assenza e' il contenuto stesso del documento. "
            "Puoi includere solo elementi anagrafici o amministrativi. "
            "Lunghezza complessiva indicativa: 500-700 caratteri."
        ),
    },
]


def directive_for(scenario: dict, style: dict) -> str:
    """Picks the style directive appropriate to this scenario.

    See STYLE_VARIANTS' "directive_no_details" comment: F="Yes" scenarios are
    defined by the ABSENCE of clinical detail, so they cannot be written in the
    detail-rich house style without invalidating their own ground truth.
    """
    if scenario["ground_truth"]["f"]["answer"] == "Yes":
        return style["directive_no_details"]
    return style["directive"]


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
            "Volo intercontinentale di 9 ore effettuato 5 giorni prima dell'esordio",
            "Da 3 giorni dolore al polpaccio destro ed edema progressivo dell'arto",
            "Aumento della temperatura cutanea locale al polpaccio destro",
            "D-dimero: 2.100 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra, flusso assente nel segmento trombizzato",
        ],
        "writer_notes": [
            "Il volo va riportato come semplice dato anamnestico, senza commentarne il ruolo",
            "Non nominare alcuna diagnosi alternativa e non dichiarare che non ce ne sono",
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
            "Paziente uomo, 70 anni, recente sostituzione protesica del ginocchio",
            "Nessuna autopsia",
            "Da 5 giorni dolore al polpaccio sinistro ed edema dell'arto",
            "Nessun arrossamento o aumento di temperatura locale riportato",
            "D-dimero: 3.050 ng/mL, superiore al limite di laboratorio",
            "Ecografia compressiva venosa arto inferiore sinistro: vena poplitea non comprimibile",
            "Ecocolordoppler dello stesso arto, eseguito lo stesso giorno: conferma trombosi della vena poplitea sinistra",
        ],
        "writer_notes": [
            "La protesi di ginocchio va riportata solo come dato anamnestico remoto; non descriverla come intervento eseguito per la trombosi ne' collegarla alla diagnosi attuale",
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
            "D-dimero: 1.650 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra con flusso assente nel segmento trombizzato",
            "Vene femorali comuni e superficiali pervie",
        ],
        "writer_notes": [
            "Non riportare NULLA sui polsi periferici, ne' presenti ne' assenti ne' non valutabili: l'esame dei polsi non deve comparire in alcuna forma",
            "Nelle negazioni pertinenti non includere nulla che riguardi i polsi o la perfusione arteriosa",
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
            "Ecografia compressiva risultata non dirimente per artefatti tecnici",
            "Flebografia con mezzo di contrasto arto inferiore destro: difetto di riempimento centrale a carico della vena poplitea destra, compatibile con trombosi venosa profonda",
            "D-dimero: 2.400 ng/mL, superiore al limite di laboratorio",
        ],
        "writer_notes": [
            "L'ecografia compressiva non deve risultare in alcun modo confermativa: descrivila come non diagnostica per limiti tecnici, senza attribuirle reperti",
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
            "TC total body eseguita per stadiazione oncologica, che ha incidentalmente mostrato trombosi della vena femorale sinistra",
            "Successivamente riferito dolore lieve al polpaccio sinistro, presente da alcuni giorni",
            "D-dimero: 1.900 ng/mL, superiore al limite di laboratorio",
        ],
        "writer_notes": [
            "La TC total body va presentata come esame oncologico di stadiazione con reperto incidentale, non come indagine richiesta per sospetta trombosi",
            "Non nominare ecografia, ecocolordoppler o flebografia: l'unica indagine per immagini eseguita e' la TC total body",
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
    {
        "id": "SYN_19_autopsy_positive_upper_extremity",
        "description": "Secondo caso di autopsia positiva, contesto diverso da SYN_08 e con TVP dell'arto SUPERIORE -- rinforza sia A1 positivo che B1_2 arto superiore",
        "facts": [
            "Paziente donna, 68 anni, portatrice di catetere venoso centrale per chemioterapia sul braccio sinistro, deceduta improvvisamente in reparto oncologico",
            "Nei giorni precedenti riferito dolore e arrossamento al braccio sinistro, sede del catetere",
            "Nessun esame di imaging venoso ne' D-dimero eseguiti prima del decesso",
            "Nessun intervento chirurgico eseguito",
            "Riscontro autoptico: trombosi venosa profonda della vena succlavia sinistra con embolia polmonare massiva come causa del decesso",
            "Nessuna diagnosi alternativa identificata all'autopsia",
        ],
        "ground_truth": {
            "a1": {"answer": "Autopsy showed presence of DVT"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "No imaging studies done, unknown if done, or done but results unknown"},
            "a3_2": {"studies": []},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Upper extremity DVT"]},
            "b2": {"symptoms": ["Redness, warmth, or pain in one or more extremities"]},
            "c": {"answer": "D-dimer not tested, or tested but results unknown or not available"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_20_autopsy_negative_trauma",
        "description": "Secondo caso di autopsia negativa per TVP, contesto traumatico diverso da SYN_11 (arresto cardiaco in anziana)",
        "facts": [
            "Paziente uomo, 45 anni, deceduto in seguito a politrauma da incidente stradale",
            "Nessun sintomo di gonfiore, dolore, arrossamento o calore agli arti riportato prima del decesso",
            "Nessun esame di imaging venoso o D-dimero eseguito in vita",
            "Nessun intervento chirurgico correlato a TVP eseguito",
            "Riscontro autoptico: esame del sistema venoso profondo degli arti negativo, nessuna evidenza di trombosi venosa profonda",
            "Causa del decesso: emorragia interna massiva da trauma, non correlata a fenomeni tromboembolici venosi",
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
        "id": "SYN_21_alt_diagnosis_bakers_cyst",
        "description": "Diagnosi alternativa di cisti di Baker rotta (invece di cellulite come in SYN_02) -- rinforza X positivo e C nella norma",
        "facts": [
            "Paziente donna, 52 anni",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Da 3 giorni dolore al polpaccio e gonfiore dell'arto inferiore destro, comparsi dopo un'attivita' fisica intensa",
            "D-dimero: 410 ng/mL, entro il range di normalita' del laboratorio",
            "Ecografia compressiva venosa arto inferiore destro: vene comprimibili, nessuna evidenza di trombosi; visualizzata raccolta liquida compatibile con cisti di Baker rotta nella regione poplitea",
            "Diagnosi dello specialista: rottura di cisti di Baker, nessun coinvolgimento del sistema venoso profondo",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done but didn't confirm DVT"},
            "a3_2": {"studies": ["Compression ultrasonography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer tested and was within test lab's range of normal"},
            "f": {"answer": "No"},
            "x": {"answer": "An alternative diagnosis was found that explained the acute illness"},
        },
    },
    {
        "id": "SYN_22_upper_extremity_effort_thrombosis",
        "description": "TVP primaria dell'arto superiore da sforzo (sindrome di Paget-Schroetter), confermata a ecocolordoppler (non CT come in SYN_03) -- diversifica sia B1_2 arto superiore che la modalita' di imaging associata",
        "facts": [
            "Paziente uomo, 24 anni, atleta, nessun catetere venoso ne' storia oncologica",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Da 2 giorni gonfiore e dolore improvvisi al braccio destro dominante, comparsi dopo intensa attivita' di sollevamento pesi",
            "Cute del braccio destro arrossata e calda al tatto",
            "D-dimero: 2.200 ng/mL, superiore al limite di laboratorio",
            "Ecocolordoppler venoso arto superiore destro: trombosi della vena succlavia destra da sforzo, quadro compatibile con sindrome di Paget-Schroetter",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Upper extremity DVT"]},
            "b2": {"symptoms": ["Redness, warmth, or pain in one or more extremities"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_23_other_procedure_ivc_filter",
        "description": "Secondo caso A2 'Other procedure', diverso dalla trombolisi di SYN_12 -- posizionamento di filtro cavale con venografia intraprocedurale che conferma il trombo",
        "facts": [
            "Paziente donna, 58 anni, controindicazione alla terapia anticoagulante",
            "Nessuna autopsia",
            "Da 3 giorni dolore al polpaccio ed edema severo dell'arto inferiore sinistro",
            "Ecocolordoppler venoso: trombosi venosa ileo-femorale sinistra estesa",
            "Sottoposta a posizionamento di filtro cavale per via percutanea; la venografia intraprocedurale ha confermato la presenza del trombo nella vena iliaca sinistra",
            "D-dimero: 3.100 ng/mL, superiore al limite di laboratorio",
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
        "id": "SYN_24_imaging_inconclusive_no_alt_diagnosis",
        "description": "Ecografia negativa per TVP ma NESSUNA diagnosi alternativa identificata -- decoupla A3_1='didn't confirm' da X='alternative found', confondi solo insieme in SYN_02",
        "facts": [
            "Paziente donna, 37 anni",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Da 2 giorni dolore al polpaccio sinistro, senza edema ne' arrossamento evidenti",
            "D-dimero: 890 ng/mL, superiore al limite di laboratorio",
            "Ecografia compressiva venosa arto inferiore sinistro: vene comprimibili, nessuna evidenza di trombosi",
            "Paziente dimessa con diagnosi incerta; nessuna diagnosi alternativa identificata, consigliato controllo clinico a distanza di pochi giorni",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done but didn't confirm DVT"},
            "a3_2": {"studies": ["Compression ultrasonography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_25_thrombectomy_postpartum",
        "description": "Secondo caso A2 'Thrombectomy', contesto diverso da SYN_07 (puerperio invece di presentazione acuta generica)",
        "facts": [
            "Paziente donna, 31 anni, in puerperio (3 settimane dopo parto cesareo)",
            "Nessuna storia di autopsia",
            "Da 1 giorno dolore severo al polpaccio ed edema massivo dell'intero arto inferiore sinistro, cute tesa e dolente",
            "Ecocolordoppler venoso urgente: trombosi ileo-femorale sinistra estesa con interessamento della vena cava inferiore",
            "Sottoposta a trombectomia chirurgica d'urgenza, con conferma intraoperatoria del trombo",
            "D-dimero: 5.800 ng/mL, superiore al limite di laboratorio",
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
        "id": "SYN_26_ct_venography_pregnancy_pelvic",
        "description": "Secondo caso A3_2 'CT or MR venography', diverso da SYN_03 (TVP pelvica in gravidanza, ecografia non dirimente per limiti anatomici)",
        "facts": [
            "Paziente donna, 29 anni, in gravidanza (28 settimane)",
            "Nessuna storia di autopsia o intervento chirurgico recente",
            "Da 4 giorni dolore al polpaccio e gonfiore dell'intero arto inferiore sinistro, fino alla regione inguinale",
            "Ecografia compressiva venosa arto inferiore sinistro: risultata non dirimente per la sede pelvica del sospetto trombo, limitata dall'utero gravido",
            "RM venografia pelvica: trombosi della vena iliaca comune sinistra",
            "D-dimero: 2.600 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["CT or MR venography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_27_contrast_venography_absent_pulses",
        "description": "Secondo caso A3_2 'Contrast venography' E secondo caso B2 'Absent pulses' vero positivo, diversi da SYN_13/SYN_18 -- edema massivo con compressione arteriosa",
        "facts": [
            "Paziente uomo, 57 anni, arteriopatia periferica nota",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 18 ore dolore severo al polpaccio ed edema massivo dell'arto inferiore destro, cute marezzata",
            "Esame obiettivo: polsi periferici (pedidio e tibiale posteriore) NON palpabili all'arto inferiore destro",
            "Ecografia compressiva risultata tecnicamente limitata per l'esteso edema",
            "Flebografia con mezzo di contrasto arto inferiore destro: difetto di riempimento esteso a carico della vena iliaca e femorale destra, compatibile con trombosi venosa profonda",
            "D-dimero: 4.900 ng/mL, superiore al limite di laboratorio",
        ],
        "writer_notes": [
            "L'ecografia compressiva non deve risultare in alcun modo confermativa: descrivila come non diagnostica per limiti tecnici, senza attribuirle reperti",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Contrast venography"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema", "Absent pulses in legs or arms"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_28_other_modality_plethysmography",
        "description": "Secondo caso A3_2 'Other', diverso da SYN_14 -- pletismografia ad impedenza invece di TC incidentale",
        "facts": [
            "Paziente donna, 66 anni, portatrice di pacemaker (controindicazione relativa alla risonanza magnetica)",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 3 giorni dolore al polpaccio e gonfiore dell'arto inferiore sinistro",
            "Pletismografia ad impedenza dell'arto inferiore sinistro: alterazione del reflusso venoso compatibile con trombosi venosa profonda prossimale",
            "D-dimero: 2.050 ng/mL, superiore al limite di laboratorio",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Other"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer exceeded test lab's upper limit of normal."},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
    {
        "id": "SYN_29_diagnosis_no_details_second",
        "description": "Secondo caso F='Yes', diverso da SYN_17 -- nota di trasferimento inter-ospedaliero invece di segnalazione del medico di base",
        "facts": [
            "Paziente uomo, 49 anni",
            "Nota di trasferimento inter-ospedaliero: 'paziente con diagnosi di trombosi venosa profonda dell'arto inferiore destro'",
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
        "id": "SYN_30_confirmed_dvt_normal_ddimer",
        "description": (
            "TVP confermata a ecocolordoppler CON D-dimero nella norma -- rompe la "
            "correlazione spuria D-dimero elevato <-> TVP confermata presente in tutti "
            "gli altri 29 scenari (vedi il commento sotto)"
        ),
        # Added 2026-08-16 on the basis of the Brighton paper itself
        # (data/reference/1-s2.0-S0264410X22010854-main.pdf), which states
        # explicitly that laboratory results "can be normal in the presence of
        # thrombotic or thromboembolic events or abnormal in the absence of
        # thrombosis or thromboembolism". A coverage audit of the other 29
        # scenarios found that a normal D-dimer ONLY ever appeared in records
        # where DVT was ruled out (SYN_15, SYN_21), and that no scenario at all
        # paired a confirmed DVT with a normal D-dimer. The corpus therefore
        # taught a correlation the source guideline calls false, and a model
        # exploiting that shortcut would have scored better than it deserved.
        # This scenario breaks it: imaging is unambiguously positive while C is
        # unambiguously normal, so section C cannot be inferred from the DVT
        # outcome (or vice versa).
        #
        # NOTE ON HOW BRIGHTON IS USED HERE: as a clinical authority for
        # SCENARIO DESIGN only. The paper is deliberately NOT used to word the
        # generated records -- pipeline.py already feeds Agent 2 retrieved
        # context from this same PDF as reference terminology, so phrasing the
        # test records with it would make the evaluator match text against its
        # own glossary (circular validation) and overstate accuracy relative to
        # real clinician-written records.
        "facts": [
            "Paziente uomo, 61 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 5 giorni dolore al polpaccio destro con edema dell'arto inferiore omolaterale",
            "D-dimero: 380 ng/mL, entro il range di normalita' del laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra con flusso assente nel segmento trombizzato",
        ],
        "writer_notes": [
            "Riporta il valore del D-dimero come dato di laboratorio fra gli altri, senza commentarne la discordanza con il reperto ecografico e senza spiegazioni fisiopatologiche",
            "Il referto ecografico deve restare inequivocabilmente positivo per trombosi",
        ],
        "ground_truth": {
            "a1": {"answer": "No autopsy done, unknown if done, or done but results unavailable"},
            "a2": {"answer": "No surgical procedure done; or, done but either did not confirm presence of DVT or findings unknown; or unknown if done"},
            "a3_1": {"answer": "≥1 imaging study was done and confirmed DVT"},
            "a3_2": {"studies": ["Doppler/Duplex Ultrasound"]},
            "b1_1": {"answer": "≥1 symptom or sign of DVT was reported"},
            "b1_2": {"types": ["Lower extremity DVT"]},
            "b2": {"symptoms": ["Calf pain or tenderness", "Leg swelling or pitting oedema"]},
            "c": {"answer": "D-dimer tested and was within test lab's range of normal"},
            "f": {"answer": "No"},
            "x": {"answer": "No alternative diagnosis was found to explain the acute illness"},
        },
    },
]


def facts_to_prompt(scenario: dict, style: dict) -> str:
    facts_block = "\n".join(f"- {fact}" for fact in scenario["facts"])
    prompt = (
        f"Direttiva di stile: {directive_for(scenario, style)}\n\n"
        f"Fatti clinici da includere (tutti, nessuno escluso, nessuno aggiunto):\n{facts_block}"
    )
    # Constraints ABOUT the writing, kept strictly apart from the clinical
    # facts and marked as not-to-be-written. They used to live inside the
    # facts as parentheticals ("(fattore di rischio, non menzionare altre
    # diagnosi)"), and the writer copied them verbatim into the record: three
    # generated records ended up containing sentences like "Non sono state
    # menzionate diagnosi alternative" or "non vengono considerati diagnosi a
    # se' stanti". A real record never comments on the questionnaire like
    # that, and worse, such a sentence hands the evaluator section X's answer
    # instead of requiring it to be inferred from the clinical picture.
    notes = scenario.get("writer_notes")
    if notes:
        notes_block = "\n".join(f"- {n}" for n in notes)
        prompt += (
            f"\n\nVincoli di scrittura (istruzioni per te, NON scriverle nel "
            f"referto e non parafrasarle):\n{notes_block}"
        )
    return prompt


def build_ground_truth(record_id: str, scenario: dict) -> dict:
    gt = {"record_id": record_id}
    gt.update(scenario["ground_truth"])
    return gt


# ---------------------------------------------------------------------------
# Fidelity check
# ---------------------------------------------------------------------------
# The writer does not reliably obey the "include every fact, invent nothing"
# rules: individual records have been caught dropping a diagnosis name, flipping
# an affected limb, genericising a named imaging modality, and (2026-08-16, on
# a corpus-wide scale) losing whole trailing sections to a token cap. Every one
# of those defects reached the pipeline unnoticed and was later mistaken for a
# pipeline error, because nothing between the writer and the evaluation looked
# at the generated text at all. These checks close that gap.
#
# SCOPE -- what this can and cannot catch. It verifies that decisive facts are
# textually PRESENT and that the record is structurally complete. It cannot
# judge whether they are used correctly: SYN_30's generated record contained
# "380 ng/mL" but described it as "risultato positivo" while also editorialising
# about the discordance with the ultrasound, which is a semantic violation this
# check passes. Catching that class would need a second LLM as a judge --
# deliberately not done here, to keep the check deterministic and fast.

# Distinctive, hard-to-paraphrase markers extracted from a scenario's facts.
# Names of specific procedures/tests are used rather than whole sentences: the
# writer legitimately rewords sentences, but it cannot rename "flebografia" and
# still be reporting the same study. Each entry maps a term as it appears in
# the facts to the alternatives that count as the same thing in the record.
_FACT_MARKERS = {
    "d-dimero": ["d-dimero", "d dimero", "ddimero"],
    "ecocolordoppler": ["ecocolordoppler", "eco-color-doppler", "ecocolor doppler", "ecodoppler"],
    "ecografia compressiva": ["ecografia compressiva", "compressiva", "compressione ecografica"],
    "flebografia": ["flebografia", "flebografico"],
    "venografia": ["venografia", "venografico"],
    "tc total body": ["tc total body", "tc totale", "total body"],
    "pletismografia": ["pletismografia", "pletismografico"],
    "trombectomia": ["trombectomia", "trombectomico"],
    "filtro cavale": ["filtro cavale", "filtro in cava"],
    "autoptico": ["autopsia", "autoptic", "riscontro autoptico"],
    "riscontro autoptico": ["autopsia", "autoptic", "riscontro autoptico"],
}


def _expected_markers(scenario: dict) -> list[tuple[str, list[str]]]:
    """(label, accepted_variants) for every decisive marker this scenario's
    facts contain. Numeric values (lab results) are added verbatim: a dropped
    or altered measurement is exactly the defect that made section C wrong on
    three records, and a number is the one token the writer cannot paraphrase.

    A scenario can override the auto-derived list with an explicit
    "must_contain" key when the heuristic is not the right one for it.
    """
    if "must_contain" in scenario:
        return [(term, [term]) for term in scenario["must_contain"]]

    markers = []
    for term, variants in _FACT_MARKERS.items():
        # Only require a marker the facts actually ASSERT. A NEGATIVE fact
        # ("Nessun esame di imaging venoso ne' D-dimero eseguiti") is about
        # something that did NOT happen, and a real record often just stays
        # silent about it rather than spelling out its absence -- so demanding
        # the word would reject records that are perfectly faithful to the
        # scenario. The sections concerned are covered anyway: an absent test
        # leads the evaluator to the "not done / unknown" option, which is
        # what the ground truth says in these cases, and A1/A2/X additionally
        # have keyword gates in criteria_rules.py for the hallucinated-positive
        # direction.
        #
        # Each fact is tested on its own rather than on the concatenation of
        # all of them: facts carry no trailing punctuation, so on a joined
        # string the negation pattern below reached across a fact boundary,
        # and a "Nessun intervento chirurgico eseguito" in one fact suppressed
        # the requirement for "Riscontro autoptico" asserted in the next --
        # which is how SYN_19's missing autopsy slipped past this check when
        # it was first written.
        asserted = any(
            term in fact.lower()
            and not re.search(rf"(nessun[ao]?|non\s+\w+\s+)[^,;]*{re.escape(term)}", fact.lower())
            for fact in scenario["facts"]
        )
        # Deduplicated on the accepted-variants set: several keys deliberately
        # map to the same synonyms ("autoptico" / "riscontro autoptico"), and
        # without this a single missing finding would be reported twice.
        if asserted and not any(set(v) == set(variants) for _, v in markers):
            markers.append((term, variants))

    for number in re.findall(r"\b\d{1,3}(?:[.,]\d{3})*\s*ng/mL", " ".join(scenario["facts"])):
        digits = re.sub(r"\s*ng/mL", "", number)
        # Accept either separator: the writer freely switches "2.100"/"2,100".
        markers.append((number, [digits, digits.replace(".", ","), digits.replace(",", ".")]))

    return markers


def check_record(text: str, scenario: dict) -> list[str]:
    """Returns a list of problems found in a generated record; empty means OK."""
    problems = []
    stripped = text.strip()

    if not stripped:
        return ["empty record"]

    # Truncation: a record cut off by the token cap ends mid-sentence. Checked
    # first because it is the defect that silently invalidated a whole corpus,
    # and because it usually causes the missing-marker failures below too.
    if not stripped.endswith((".", "!", "?")):
        problems.append(f"truncated mid-sentence (ends with {stripped[-40:]!r})")

    for label, variants in _expected_markers(scenario):
        if not any(v.lower() in stripped.lower() for v in variants):
            problems.append(f"decisive fact missing from the record: {label!r}")

    return problems


def generate_record(llm, scenario: dict, style: dict) -> str:
    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", facts_to_prompt(scenario, style)),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def generate_checked_record(llm, scenario: dict, style: dict, record_id: str) -> tuple[str, list[str]]:
    """Generates a record, re-rolling while check_record() rejects it.

    Returns (best_text, remaining_problems). Retrying is worthwhile because the
    writer runs at temperature 0.8, so each attempt is a genuine resample
    rather than a repeat. After WRITER_MAX_ATTEMPTS the least-bad attempt is
    returned WITH its problems, rather than raising: one stubborn scenario
    should not abort a 30-record generation run, but it must not be saved
    silently either -- main() reports it and the caller decides.
    """
    best_text, best_problems = None, None

    for attempt in range(1, WRITER_MAX_ATTEMPTS + 1):
        text = generate_record(llm, scenario, style)
        problems = check_record(text, scenario)

        if not problems:
            return text, []

        if best_problems is None or len(problems) < len(best_problems):
            best_text, best_problems = text, problems

        print(
            f"[{record_id}] attempt {attempt}/{WRITER_MAX_ATTEMPTS} rejected: "
            f"{'; '.join(problems)}",
            flush=True,
        )

    return best_text, best_problems


def check_existing_records() -> int:
    """Runs check_record() over the records already on disk, without
    regenerating anything. Returns the number of defective records.

    Same checks as the inline ones, exposed separately so an existing corpus
    can be audited (or re-audited after a manual fix) without spending an
    Ollama run -- and so a corpus generated before this check existed can be
    triaged. Usage: python generate_synthetic_records.py --check
    """
    defective = 0
    for scenario in SCENARIOS:
        for style in STYLE_VARIANTS:
            record_id = f"{scenario['id']}_{style['id']}"
            path = OUTPUT_DIR / f"{record_id}.txt"
            if not path.exists():
                print(f"[{record_id}] MISSING: {path.name} does not exist", flush=True)
                defective += 1
                continue
            problems = check_record(path.read_text(encoding="utf-8"), scenario)
            if problems:
                defective += 1
                print(f"[{record_id}] DEFECTIVE:", flush=True)
                for p in problems:
                    print(f"    - {p}", flush=True)

    total = len(SCENARIOS) * len(STYLE_VARIANTS)
    print(f"\n{total - defective}/{total} records pass the fidelity check.", flush=True)
    return defective


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only re-check the records already in data/synthetic_records/ "
             "against check_record(); generate nothing and call no LLM.",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(1 if check_existing_records() else 0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    llm = build_llm(
        WRITER_MODEL_NAME,
        temperature=WRITER_TEMPERATURE,
        num_predict=WRITER_NUM_PREDICT,
    )

    still_defective = []
    for scenario in SCENARIOS:
        for style in STYLE_VARIANTS:
            record_id = f"{scenario['id']}_{style['id']}"
            print(f"[{record_id}] generating ({scenario['description']})...", flush=True)

            record_text, problems = generate_checked_record(llm, scenario, style, record_id)

            txt_path = OUTPUT_DIR / f"{record_id}.txt"
            json_path = OUTPUT_DIR / f"{record_id}_ground_truth.json"

            txt_path.write_text(record_text, encoding="utf-8")
            json_path.write_text(
                json.dumps(build_ground_truth(record_id, scenario), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if problems:
                # Saved anyway (a partial record is still inspectable), but
                # recorded so the run ends with an explicit list instead of
                # letting a defective record slip into the evaluation set.
                still_defective.append((record_id, problems))
                print(f"[{record_id}] SAVED WITH PROBLEMS -> {txt_path.name}", flush=True)
            else:
                print(f"[{record_id}] saved -> {txt_path.name}, {json_path.name}", flush=True)

    if still_defective:
        print(
            f"\n!! {len(still_defective)} record(s) still failed the fidelity check "
            f"after {WRITER_MAX_ATTEMPTS} attempts:",
            flush=True,
        )
        for record_id, problems in still_defective:
            print(f"  {record_id}: {'; '.join(problems)}", flush=True)
        print(
            "Review them (and fix them by hand if needed) BEFORE running the "
            "pipeline on this set.",
            flush=True,
        )

    print(f"\nDone: {len(SCENARIOS) * len(STYLE_VARIANTS)} records in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
