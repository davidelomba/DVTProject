"""
Data-augmentation script: generates synthetic Italian clinical records paired
with matching ground-truth JSON (same shape as models.DVT_CriteriaForm), so the
pipeline can be validated on more than the single real record it was originally
tuned against.

GROUND TRUTH BY CONSTRUCTION: every scenario below carries both its clinical
facts and its correct answer for all 10 sections, written by hand with
models.py's exact Literal strings. No model ever guesses the reference, which
is what makes it usable as one. The ground-truth JSONs are always rewritten,
since producing them involves no LLM.

RECORDS: the 30 records in data/synthetic_records/ were not produced by the
writer below. They were drafted with Claude (a general-purpose model, outside
every pipeline role) from each scenario's facts and reviewed against them,
after generated ones were repeatedly found to contradict their own ground
truth. The writer model (WRITER_MODEL_NAME, a literal rather than a config
import so it stays outside every pipeline role too) remains available for new
scenarios, at a non-zero temperature for lexical variation. Records are only
written when missing, unless --force, so a plain run cannot overwrite them.

FIDELITY CHECK: a record is verified against its scenario's facts before being
saved and regenerated if it fails; see check_record. The same checks run over
an existing corpus without calling any LLM:
    python generate_synthetic_records.py --check

INTERPRETIVE ASSUMPTIONS, worth re-checking against the Brighton paper:
  - B1_1/B1_2 record what was REPORTED as a DVT syndrome, whether or not
    A3_1/X later confirm or rule it out.
  - F follows criteria_rules.apply_details_gate: "No" covers both "reported
    WITH details" and "not reported at all"; "Yes" only "reported WITHOUT
    details".
  - B2 offers no generic swelling option for arms, so upper-extremity
    scenarios count arm swelling toward no option rather than stretching
    "Leg swelling or pitting oedema".

COVERAGE: every option of every section appears in at least two scenarios,
verified programmatically. SYN_30 is there for a different reason: it breaks a
correlation rather than covering an option. Before it, an elevated D-dimer
co-occurred with a confirmed DVT in every record, which per-option coverage
cannot detect and which lets a model answer section C without reading the
value.

KNOWN LIMITATION: a record is around 1000 characters, about one
config.EHR_CHUNK_SIZE chunk, while config.EHR_RETRIEVER_K asks for 5 --
retrieval returns the whole record every time (measured median evidence length:
1.00x the source). The three EXTRACTOR_MODEs therefore give Agent 2 the same
input and cannot be compared on this dataset. Padding was tried and reverted:
the comparison belongs on the real records expected from collaborators.

Usage: python generate_synthetic_records.py [--check] [--force] [--only ID...]
Output: data/synthetic_records/<scenario_id>_<style_id>.txt and the matching
        _ground_truth.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from agents import build_llm

# A literal, not a config import: the writer must stay independent of whichever
# model the pipeline roles use, now or after a config change.
WRITER_MODEL_NAME = "qwen2.5:7b-instruct"
WRITER_TEMPERATURE = 0.8

# config.LLM_NUM_PREDICT (512) is about 1750 Italian characters, just under the
# length asked for here: it cuts records off mid-sentence without any error.
WRITER_NUM_PREDICT = 3072

# Regeneration attempts when check_record() rejects the output. Each is a real
# resample thanks to the non-zero temperature.
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

# Modelled on the one real record available (data/patient_001.txt), not copied
# from it. Left to itself a writer produces something far thinner: no section
# labels, no vitals, no home medications, no pertinent negatives. Measuring the
# pipeline on clean, signal-dense text would overstate how it does on real ones.
#
# Generic on purpose: it asks for the structural features Italian hospital
# records share, never for the exact labels, drug names or negation wording. A
# writer copies whatever examples it is given, so prescribing those would
# produce near-identical variants of a single document, overfitted to one
# clinician. Left free, the variation comes from the temperature instead.
STYLE_VARIANTS = [
    {
        "id": "v2",
        "directive": (
            "Scrivi il documento come un referto ospedaliero italiano reale, in "
            "prosa continua (nessun elenco puntato). Organizza il contenuto nelle "
            "sezioni tipiche di un referto italiano (anamnesi remota, anamnesi "
            "prossima, esame obiettivo, esami di laboratorio e strumentali) "
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
    """Picks the style directive appropriate to a scenario.

    Args:
        scenario: an entry of SCENARIOS.
        style: an entry of STYLE_VARIANTS.

    Returns:
        The detail-rich house directive, or the detail-free one for scenarios
        whose ground truth is F="Yes". Those are defined by the ABSENCE of
        clinical detail, so a single vital sign or lab value written into them
        would flip F's correct answer.
    """

    if scenario["ground_truth"]["f"]["answer"] == "Yes":
        return style["directive_no_details"]
    return style["directive"]


# Scenarios: each case's clinical facts plus its correct answers. The facts
# feed the writer and are what check_record verifies the record against; the
# ground truth is written out here, never inferred from the text. Its values
# must match models.py's Literal strings EXACTLY (copy them, don't retype).

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
            "b1_2": {"types": []},
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
            "b1_2": {"types": []},
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
            "b1_2": {"types": ["Lower extremity DVT"]},
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
        "writer_notes": [
            "Il paziente e' deceduto e non e' mai stato in grado di riferire sintomi: non attribuirgli dichiarazioni, negazioni o anamnesi raccolte direttamente da lui",
            "L'assenza di sintomi agli arti va resa come constatazione clinica del personale, non come qualcosa che il paziente ha negato",
            "Non scrivere frasi che descrivono la cartella stessa (del tipo 'non sono state riportate/menzionate...'): riporta solo fatti clinici",
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
            "b1_2": {"types": []},
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
            "b1_2": {"types": []},
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
            "b1_2": {"types": ["Lower extremity DVT"]},
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
            "correlazione spuria D-dimero elevato <-> TVP confermata (vedi il commento)"
        ),

        # Exists to break a correlation, not to cover an option: in every other
        # scenario a normal D-dimer appears only where DVT was ruled out, so
        # section C can be answered from the outcome alone without reading the
        # value. Here imaging is positive and C normal, so neither section can
        # be inferred from the other. The Brighton paper backs the combination:
        # labs "can be normal in the presence of thrombotic events".
        #
        # It is used for scenario DESIGN only, never to word the records:
        # Agent 2 already receives retrieved context from that same PDF, so
        # borrowing its vocabulary would have the evaluator match text against
        # its own glossary.
        "facts": [
            "Paziente uomo, 61 anni",
            "Nessuna autopsia o intervento chirurgico recente",
            "Da 5 giorni dolore al polpaccio destro con edema dell'arto inferiore omolaterale",
            "D-dimero: 380 ng/mL, entro il range di normalita' del laboratorio",
            "Ecocolordoppler venoso arto inferiore destro: trombosi venosa poplitea destra con flusso assente nel segmento trombizzato",
        ],
        "writer_notes": [
            "Riporta il valore del D-dimero come un dato di laboratorio qualsiasi, insieme agli altri, senza aggiungere alcun commento sul suo significato",
            "NON scrivere da nessuna parte che il D-dimero e' discorde, inatteso, sorprendente o in contrasto con l'ecografia, e non aggiungere spiegazioni fisiopatologiche: la discordanza non deve essere segnalata al lettore in alcun modo",
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
    """Assembles the human message sent to the writer for one record.

    Args:
        scenario: an entry of SCENARIOS.
        style: an entry of STYLE_VARIANTS.

    Returns:
        The prompt: style directive, clinical facts, and the scenario's
        writer_notes if it has any.
    """

    facts_block = "\n".join(f"- {fact}" for fact in scenario["facts"])
    prompt = (
        f"Direttiva di stile: {directive_for(scenario, style)}\n\n"
        f"Fatti clinici da includere (tutti, nessuno escluso, nessuno aggiunto):\n{facts_block}"
    )

    # writer_notes are constraints ABOUT the writing, kept in their own block
    # and marked as not-to-be-written. Inside "facts" they get copied verbatim
    # into the record, which both breaks the illusion of a clinical document
    # and can hand the evaluator an answer it should have had to infer.
    notes = scenario.get("writer_notes")
    if notes:
        notes_block = "\n".join(f"- {n}" for n in notes)
        prompt += (
            f"\n\nVincoli di scrittura (istruzioni per te, NON scriverle nel "
            f"referto e non parafrasarle):\n{notes_block}"
        )
    return prompt


def build_ground_truth(record_id: str, scenario: dict) -> dict:
    """Builds the reference answers saved alongside a generated record.

    Args:
        record_id: "<scenario id>_<style id>", the key the evaluation uses to
            pair a prediction with its reference.
        scenario: an entry of SCENARIOS.

    Returns:
        The scenario's hand-authored answers, with record_id added. No model is
        involved at any point, which is what makes this usable as a reference.
    """
    gt = {"record_id": record_id}
    gt.update(scenario["ground_truth"])
    return gt



# Fidelity check

# A record can silently contradict its own scenario: facts get dropped, a limb
# flipped, a named imaging modality genericised. Nothing between writing and
# evaluation inspects the text, so such defects reach the pipeline looking like
# pipeline errors.
#
# SCOPE: verifies that decisive facts are textually PRESENT and the record is
# structurally complete. It cannot judge whether a fact is used correctly (a
# record stating a normal lab value then calling it positive passes). That would
# need an LLM as judge, left out to keep the check deterministic and fast.

# Distinctive markers, keyed by the term as it appears in a scenario's facts,
# mapped to the spellings that count as the same thing. Names of tests rather
# than whole sentences: wording is free, but "flebografia" cannot be renamed
# and still report the same study.
_FACT_MARKERS = {
    "d-dimero": ["d-dimero", "d dimero", "ddimero"],

    # "doppler" on its own is accepted: the writer misspells the compound
    # ("ecocoloredoppler") often enough that requiring a full spelling rejects
    # records where the study is plainly reported.
    "ecocolordoppler": ["doppler", "ecodoppler"],

    # "comprimibil" covers the clinically equivalent way of reporting the same
    # study (describing the veins as compressible rather than naming the
    # technique) which is how a real record often puts it.
    "ecografia compressiva": [
        "ecografia compressiva", "compressiva", "compressione ecografica", "comprimibil",
    ],
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
    """Works out which facts a generated record must mention.

    Args:
        scenario: an entry of SCENARIOS. A "must_contain" key overrides the
            heuristic below with an explicit list of required terms.

    Returns:
        (label, accepted_variants) pairs, including any lab value found in the
        facts: a number is the one token the writer cannot paraphrase.
    """

    if "must_contain" in scenario:
        return [(term, [term]) for term in scenario["must_contain"]]

    markers = []
    for term, variants in _FACT_MARKERS.items():

        # Required only where the fact ASSERTS something: on a negative fact
        # ("Nessun D-dimero eseguito") a faithful record may stay silent and
        # the missing test leads to the "not done" option anyway.
        # Tested fact by fact, not on them joined: facts carry no trailing
        # punctuation, so on a joined string one fact's "Nessun..." would
        # reach across the boundary and suppress the next fact's requirement.
        asserted = any(
            term in fact.lower()
            and not re.search(rf"(nessun[ao]?|non\s+\w+\s+)[^,;]*{re.escape(term)}", fact.lower())
            for fact in scenario["facts"]
        )
        # Deduplicated on the variants, since several keys share synonyms.
        if asserted and not any(set(v) == set(variants) for _, v in markers):
            markers.append((term, variants))

    for number in re.findall(r"\b\d{1,3}(?:[.,]\d{3})*\s*ng/mL", " ".join(scenario["facts"])):
        digits = re.sub(r"\s*ng/mL", "", number)
        # Accept either separator: the writer freely switches "2.100"/"2,100".
        markers.append((number, [digits, digits.replace(".", ","), digits.replace(",", ".")]))

    return markers


def check_record(text: str, scenario: dict) -> list[str]:
    """Checks a generated record against the scenario it was written from.

    Args:
        text: the generated record.
        scenario: an entry of SCENARIOS.

    Returns:
        A list of human-readable problems; empty means the record passed.
    """

    problems = []
    stripped = text.strip()

    if not stripped:
        return ["empty record"]

    # A record cut off by the token cap ends mid-sentence. Checked first
    # because truncation is usually what causes the missing markers below.
    if not stripped.endswith((".", "!", "?")):
        problems.append(f"truncated mid-sentence (ends with {stripped[-40:]!r})")

    for label, variants in _expected_markers(scenario):
        if not any(v.lower() in stripped.lower() for v in variants):
            problems.append(f"decisive fact missing from the record: {label!r}")

    return problems


def generate_record(llm, scenario: dict, style: dict) -> str:
    """Asks the writer for one record, without checking the result.

    Args:
        llm: the writer model.
        scenario: an entry of SCENARIOS.
        style: an entry of STYLE_VARIANTS.

    Returns:
        The generated record text.
    """

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", facts_to_prompt(scenario, style)),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def generate_checked_record(llm, scenario: dict, style: dict, record_id: str) -> tuple[str, list[str]]:
    """Generates a record, regenerating while check_record rejects it.

    Args:
        llm: the writer model.
        scenario: an entry of SCENARIOS.
        style: an entry of STYLE_VARIANTS.
        record_id: used for progress output only.

    Returns:
        (text, problems). Problems is empty when an attempt passed; otherwise
        the least-bad attempt is returned along with what is still wrong with
        it. Returning rather than raising keeps one stubborn scenario from
        aborting a whole generation run, while main() makes sure the failure
        is reported instead of saved silently.
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
    """Audits the records already on disk, regenerating nothing.

    Same checks as the inline ones, exposed separately so a corpus can be
    inspected (or re-inspected after a manual fix) without spending an
    Ollama run.

    Returns:
        The number of records that are missing or defective.
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
    """Generates every scenario's record and ground truth or audits an
    existing corpus when called with --check.

    Records that still fail the fidelity check after WRITER_MAX_ATTEMPTS are
    saved anyway (a partial record is still worth inspecting) but are
    listed at the end so none of them reaches the evaluation unnoticed.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only re-check the records already in data/synthetic_records/ "
             "against check_record(); generate nothing and call no LLM.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="ID",
        help="Restrict the run to the scenarios whose id contains one of these "
             "strings (e.g. --only SYN_20 SYN_30). Every other record is left "
             "untouched, so a single defective one can be re-rolled without "
             "spending a full generation run or disturbing the rest of the set.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite records that already exist. Without it only missing "
             "ones are written, so an accidental run cannot destroy records "
             "that were edited or authored by hand.",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(1 if check_existing_records() else 0)

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if any(frag in s["id"] for frag in args.only)]
        if not scenarios:
            sys.exit(f"No scenario id matches {args.only}.")
        print(f"Restricting the run to {len(scenarios)} of {len(SCENARIOS)} scenarios: "
              f"{', '.join(s['id'] for s in scenarios)}\n", flush=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Ground truth is derived from SCENARIOS with no model involved, so it is
    # always rewritten: that keeps the reference answers in step with the
    # scenarios even when no record is regenerated.
    for scenario in scenarios:
        for style in STYLE_VARIANTS:
            record_id = f"{scenario['id']}_{style['id']}"
            (OUTPUT_DIR / f"{record_id}_ground_truth.json").write_text(
                json.dumps(build_ground_truth(record_id, scenario), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Records are only written when missing, unless --force. Records edited or
    # written by hand are indistinguishable from generated ones on disk and
    # regenerating one silently replaces work that took real effort to get
    # right.
    pending = [
        (scenario, style)
        for scenario in scenarios
        for style in STYLE_VARIANTS
        if args.force or not (OUTPUT_DIR / f"{scenario['id']}_{style['id']}.txt").exists()
    ]
    skipped = len(scenarios) * len(STYLE_VARIANTS) - len(pending)
    if skipped:
        print(f"{skipped} record(s) already exist and are left untouched "
              f"(use --force to regenerate them).", flush=True)
    if not pending:
        print("Nothing to generate. Ground truth files refreshed.", flush=True)
        return

    llm = build_llm(
        WRITER_MODEL_NAME,
        temperature=WRITER_TEMPERATURE,
        num_predict=WRITER_NUM_PREDICT,
    )

    still_defective = []
    for scenario, style in pending:
        record_id = f"{scenario['id']}_{style['id']}"
        print(f"[{record_id}] generating ({scenario['description']})...", flush=True)

        record_text, problems = generate_checked_record(llm, scenario, style, record_id)

        txt_path = OUTPUT_DIR / f"{record_id}.txt"
        txt_path.write_text(record_text, encoding="utf-8")

        if problems:
            still_defective.append((record_id, problems))
            print(f"[{record_id}] SAVED WITH PROBLEMS -> {txt_path.name}", flush=True)
        else:
            print(f"[{record_id}] saved -> {txt_path.name}", flush=True)

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

    written = len(pending) - len(still_defective)
    print(
        f"\nDone: {written} record(s) written, {len(still_defective)} saved with "
        f"problems, {skipped} left untouched -- in {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
