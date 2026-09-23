"""
Schema of the two myocarditis sections the transfer experiment fills, E and F.

The option strings are copied from CriteriaForm_SEVALIDMYOCARDITIS, including
its commas inside terms (`ST, segment`, `T, wave`, `r, wave`, `PR, depression`)
and its asterisks: the form is the reference the answers are scored against, so
a tidied string would no longer be the option the questionnaire offers.

Both sections are checkbox on the form and are therefore List[Literal[...]],
which is the same shape as A3_2, B1_2 and B2 in models.py.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


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
    """The two filled-in sections for one clinical record."""

    record_id: str

    e: E_Electrocardiogram | None = None
    f: F_Echocardiogram | None = None


# pipeline.py imports this name from the module installed as `models`.
DVT_CriteriaForm = MYO_CriteriaForm

SECTION_MODELS = {
    "E": E_Electrocardiogram,
    "F": F_Echocardiogram,
}
