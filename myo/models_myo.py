"""
Schema of the myocarditis sections the pipeline fills: A1, B, C, E and F.

The option strings are copied from CriteriaForm_SEVALIDMYOCARDITIS, including
its commas inside terms (`ST, segment`, `T, wave`, `r, wave`, `PR, depression`)
and its asterisks: the form is the reference the answers are scored against, so
a tidied string would no longer be the option the questionnaire offers. Two
things the printed form carries next to an option are not part of it: the
footnote numbers that point to the normal ranges and the `Highest measured
value {field}` slots of C, which are separate REDCap fields.

Every section is a checkbox on the form and is therefore List[Literal[...]],
which is the same shape as A3_2, B1_2 and B2 in models.py.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class A1_CardiacSymptoms(BaseModel):
    """A1. Cardiac symptoms."""

    symptoms: List[Literal[
        "acute chest pain or pressure",
        "palpitation",
        "dyspnea at rest, with exercise or lying down",
        "diaphoresis",
        "sudden death",
        "none of the above",
        "unknown if any of 1-5 present or absent",
    ]] = Field(default_factory=list, description="A1: Cardiac Symptom — check all that apply")


class B_PhysicalExam(BaseModel):
    """B. Physical exam features."""

    features: List[Literal[
        "pericardial friction rub",
        "pulsus paradoxus",
        "distant heart sounds (infant and children)",
        "none of the above",
        "unknown if any of 1-3 present or absent",
        "*Confirmed physical exam conforming Pericarditis, without details",
    ]] = Field(default_factory=list, description="B: Physical Exam Features — check all that apply")


class C_InflammatoryMarkers(BaseModel):
    """C. Inflammatory markers.

    The form's footnotes give the normal ranges: ESR below 15 mm/hr for men
    under 50, 20 for men over 50 and women under 50, 30 for women over 50,
    0-2 for newborns; CRP above 10 mg/L; hsCRP above 3 mg/L. D-dimer has none.
    """

    markers: List[Literal[
        "None tested, or tested but no results or unknown if tested",
        "ESR elevated",
        "CRP elevated",
        "High sensitivity (hs) CRP elevated",
        "D-Dimer elevated",
        "None of the above (choose this option if whichever ones were tested, "
        "were within normal range)",
        "*confirmed positive inflammatory marker conforming myocarditis without details.",
    ]] = Field(default_factory=list, description="C: Inflammatory Markers — check all that apply")


class E_Electrocardiogram(BaseModel):
    """E. Electrocardiogram findings."""

    findings: List[Literal[
        "ECG not done, or done but no results available or unknown if done",
        "Paroxysmal or sustained atrial or ventricular arrhythmias (premature "
        "atrial or ventricular beats, and/or supraventricular or ventricular "
        "tachycardia, interventricular conduction delay, abnormal Q waves, low "
        "voltages): :e.g (Sinus tachycardia, SVT, atrial fibrillation, PVCs, VT, VF)",
        "AV nodal conduction delays or intraventricular conduction defects (AV "
        "block (grade I, III), new bundle branch block)",
        "Continuous ambulatory ECG monitoring that detects frequent atrial or "
        "ventricular ectopy",
        "ST, segment or T, wave abnormalities (elevation or inversion)",
        "Newly reduced r, wave height, low voltage, or abnormal q waves",
        "Premature atrial and premature ventricular contractions (PACs, PVCs)",
        "*confirmed ECG abnormality conforming myocarditis without details",
        "Diffuse concave, upward ST, segment elevation",
        "ST, segment depression in aVR (ECG electrode lead placed on R arm)",
        "*PR, depression throughout the leads without reciprocal ST, segment changes",
        "*Nonspecific abnormalities other than those listed above.",
        "No abnormalities seen on ECG",
        "*confirmed ECG abnormality conforming Pericarditis without details",
    ]] = Field(default_factory=list, description="E: Electrocardiogram — check all that apply")


class F_Echocardiogram(BaseModel):
    """F. Echocardiogram findings."""

    findings: List[Literal[
        "ECHO not done, or done but no results or unknown if done",
        "New focal or diffuse left or right ventricular function abnormalities "
        "(e.g. decreased ejection fraction)",
        "Segmental wall motion abnormalities",
        "Global systolic or diastolic function depression/abnormality - "
        "(*Decreased longitudinal and circumferential strain and strain rates on "
        "tissue Doppler)",
        "Ventricular dilation",
        "Wall thickness change",
        "confirmed Echocardiogram conforming myocarditis without details",
        "Evidence of abnormal fluid collection or pericardial inflammation",
        "No abnormalities seen on ECHO",
        "*confirmed Echocardiogram conforming pericarditis without details",
    ]] = Field(default_factory=list, description="F: Echocardiogram — check all that apply")


class MYO_CriteriaForm(BaseModel):
    """The filled-in sections for one clinical record."""

    record_id: str

    a1: A1_CardiacSymptoms | None = None
    b: B_PhysicalExam | None = None
    c: C_InflammatoryMarkers | None = None
    e: E_Electrocardiogram | None = None
    f: F_Echocardiogram | None = None


# pipeline.py imports this name from the module installed as `models`.
DVT_CriteriaForm = MYO_CriteriaForm

SECTION_MODELS = {
    "A1": A1_CardiacSymptoms,
    "B": B_PhysicalExam,
    "C": C_InflammatoryMarkers,
    "E": E_Electrocardiogram,
    "F": F_Echocardiogram,
}
