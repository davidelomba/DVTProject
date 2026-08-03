"""
Pydantic models for the questionnaire (Criteria Form).

Full coverage of the "checkbox" questions only (single choice / multi-select).
Deliberately excluded: dates, free-text fields, descriptions,
"specify other" (e.g. a3_2_other, x_description).

Each class corresponds to ONE question in the form, so Agent 2 can be invoked
section by section (see pipeline.py) instead of having to fill in the whole
form in one go.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Criterion A: Confirmation of DVT
# ---------------------------------------------------------------------------

class A1_Autopsy(BaseModel):
    answer: Literal[
        "Autopsy showed presence of DVT",
        "Autopsy done but showed no evidence of DVT",
        "No autopsy done, unknown if done, or done but results unavailable",
    ] = Field(description="A1. Pathologic findings from autopsy — check the one best answer")


class A2_SurgicalProcedure(BaseModel):
    answer: Literal[
        "Thrombectomy related to DVT performed",
        "Other procedure done that confirmed presence of DVT",
        "No surgical procedure done; or, done but either did not confirm presence of "
        "DVT or findings unknown; or unknown if done",
    ] = Field(description="A2. Surgical procedure — check the one best answer")


class A3_1_ImagingOutcome(BaseModel):
    answer: Literal[
        "\u22651 imaging study was done and confirmed DVT",
        "\u22651 imaging study was done but didn't confirm DVT",
        "No imaging studies done, unknown if done, or done but results unknown",
    ] = Field(description="A3.1. Imaging studies confirmed the presence of DVT — one best answer")


class A3_2_ImagingStudies(BaseModel):
    studies: List[Literal[
        "Compression ultrasonography",
        "CT or MR venography",
        "Contrast venography",
        "Doppler/Duplex Ultrasound",
        "Other",
    ]] = Field(default_factory=list, description="A3.2. Which studies confirmed DVT — check all that apply")


# ---------------------------------------------------------------------------
# Criterion B: Clinical evidence for presence of DVT
# ---------------------------------------------------------------------------

class B1_1_SymptomsReported(BaseModel):
    answer: Literal[
        "\u22651 symptom or sign of DVT was reported",
        "There was no report of a recognized DVT syndrome",
        "It is unknown if there was a report of a DVT syndrome",
    ] = Field(description="B1.1. Signs and symptoms of DVT reported — one best answer")


class B1_2_DVTType(BaseModel):
    types: List[Literal[
        "Lower extremity DVT",
        "Upper extremity DVT",
    ]] = Field(default_factory=list, description="B1.2. Specific type(s) of DVT — check all that apply")


class B2_NewSymptoms(BaseModel):
    symptoms: List[Literal[
        "Calf pain or tenderness",
        "Leg swelling or pitting oedema",
        "Absent pulses in legs or arms",
        "Redness, warmth, or pain in one or more extremities",
        "None of the above were present or it is unknown if any of 1-4 were present",
    ] ] = Field(default_factory=list, description="B2. New onset clinical symptoms or signs — check all that apply")

    @model_validator(mode="after")
    def none_is_exclusive(self):
        none_label = "None of the above were present or it is unknown if any of 1-4 were present"
        if none_label in self.symptoms and len(self.symptoms) > 1:
            raise ValueError(
                f"'{none_label}' cannot coexist with other selections in B2 "
                f"(found: {self.symptoms})"
            )
        return self


# ---------------------------------------------------------------------------
# Criterion C: D-Dimer
# ---------------------------------------------------------------------------

class C_DDimer(BaseModel):
    answer: Literal[
        "D-dimer exceeded test lab's upper limit of normal.",
        "D-dimer tested and was within test lab's range of normal",
        "D-dimer not tested, or tested but results unknown or not available",
    ] = Field(description="C. D-Dimer — one best answer")


# ---------------------------------------------------------------------------
# Criterion F
# ---------------------------------------------------------------------------

class F_ReportedBySpecialist(BaseModel):
    answer: Literal["Yes", "No"] = Field(
        description="F. Reported as a case of DVT by specialist, without details"
    )


# ---------------------------------------------------------------------------
# Criterion X: Alternative diagnosis
# ---------------------------------------------------------------------------

class X_AlternativeDiagnosis(BaseModel):
    answer: Literal[
        "An alternative diagnosis was found that explained the acute illness",
        "No alternative diagnosis was found to explain the acute illness",
    ] = Field(description="X. Alternative diagnosis — one best answer")


# ---------------------------------------------------------------------------
# Container for the full form (union of all sections)
# ---------------------------------------------------------------------------

class DVT_CriteriaForm(BaseModel):
    record_id: str

    a1: A1_Autopsy | None = None
    a2: A2_SurgicalProcedure | None = None
    a3_1: A3_1_ImagingOutcome | None = None
    a3_2: A3_2_ImagingStudies | None = None

    b1_1: B1_1_SymptomsReported | None = None
    b1_2: B1_2_DVTType | None = None
    b2: B2_NewSymptoms | None = None

    c: C_DDimer | None = None
    f: F_ReportedBySpecialist | None = None
    x: X_AlternativeDiagnosis | None = None


# Section-name -> Pydantic-class map, used by the loop in pipeline.py
SECTION_MODELS = {
    "A1": A1_Autopsy,
    "A2": A2_SurgicalProcedure,
    "A3_1": A3_1_ImagingOutcome,
    "A3_2": A3_2_ImagingStudies,
    "B1_1": B1_1_SymptomsReported,
    "B1_2": B1_2_DVTType,
    "B2": B2_NewSymptoms,
    "C": C_DDimer,
    "F": F_ReportedBySpecialist,
    "X": X_AlternativeDiagnosis,
}
