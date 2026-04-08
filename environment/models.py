"""
MedTriageEnv — Typed Pydantic Models (OpenEnv Spec Compliant)

All Observation, Action, and Reward types are defined here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────── Enums ───────────────────────────────────────────

class ESILevel(int, Enum):
    """Emergency Severity Index (1 = most critical, 5 = least urgent)."""
    IMMEDIATE    = 1   # Life-/limb-threatening; needs immediate intervention
    EMERGENT     = 2   # High risk; should not wait
    URGENT       = 3   # Stable but requires 2+ resources
    LESS_URGENT  = 4   # Stable; needs 1 resource
    NON_URGENT   = 5   # Stable; no resources needed


class BedType(str, Enum):
    TRAUMA_BAY    = "trauma_bay"
    CCU           = "ccu"
    ACUTE         = "acute"
    GENERAL       = "general"
    WAITING       = "waiting"


class DiagnosticTest(str, Enum):
    ECG           = "ecg"
    CHEST_XRAY    = "chest_xray"
    CBC           = "cbc"
    BMP           = "bmp"
    TROPONIN      = "troponin"
    D_DIMER       = "d_dimer"
    CT_HEAD       = "ct_head"
    CT_CHEST      = "ct_chest"
    CT_ABDOMEN    = "ct_abdomen"
    URINALYSIS    = "urinalysis"
    LACTATE       = "lactate"
    BNP           = "bnp"
    LIPASE        = "lipase"
    LFT           = "lft"
    COAGS         = "coags"
    BLOOD_CULTURE = "blood_culture"
    URINE_CULTURE = "urine_culture"


class ActionType(str, Enum):
    TRIAGE              = "triage"
    ORDER_DIAGNOSTICS   = "order_diagnostics"
    ALLOCATE_RESOURCES  = "allocate_resources"


# ─────────────────────────── Sub-models ──────────────────────────────────────

class VitalSigns(BaseModel):
    """Objective patient measurements."""
    bp_systolic:       int   = Field(..., ge=0,   le=250, description="Systolic BP (mmHg); 0=cardiac arrest")
    bp_diastolic:      int   = Field(..., ge=0,   le=150, description="Diastolic BP (mmHg); 0=cardiac arrest")
    heart_rate:        int   = Field(..., ge=0,   le=250, description="Heart rate (bpm); 0=cardiac arrest")
    respiratory_rate:  int   = Field(..., ge=0,   le=60,  description="Respiratory rate (/min); 0=apnoeic")
    spo2:              float = Field(..., ge=0.0, le=100, description="O₂ saturation (%)")
    temperature:       float = Field(..., ge=32.0,le=43.0,description="Temperature (°C)")
    gcs:               int   = Field(..., ge=3,   le=15,  description="Glasgow Coma Scale")
    pain_score:        int   = Field(..., ge=0,   le=10,  description="Pain (0–10 NRS)")


class Patient(BaseModel):
    """Full patient presentation as seen by triage staff."""
    patient_id:             str
    age:                    int
    sex:                    str                      # "M" | "F"
    chief_complaint:        str
    vitals:                 VitalSigns
    symptom_duration_hours: float
    history:                str
    medications:            List[str]
    allergies:              List[str]
    arrival_mode:           str                      # walk-in | ambulance | helicopter
    additional_info:        Optional[Dict[str, Any]] = None


class DepartmentStatus(BaseModel):
    """Real-time ED capacity snapshot (Task 3 only)."""
    trauma_bays_total:   int
    trauma_bays_used:    int
    ccu_beds_total:      int
    ccu_beds_used:       int
    acute_beds_total:    int
    acute_beds_used:     int
    general_beds_total:  int
    general_beds_used:   int
    patients_waiting:    int
    elapsed_minutes:     float


# ──────────────────────────── Action ─────────────────────────────────────────

class Action(BaseModel):
    """
    Unified action model for all three tasks.

    Task 1  → action_type = "triage"             (esi_level required)
    Task 2  → action_type = "order_diagnostics"  (diagnostics list required)
    Task 3  → action_type = "allocate_resources" (esi_level + bed_type required)
    """
    action_type:  str            = Field(..., description=ActionType.__doc__)
    patient_id:   str            = Field(..., description="ID of patient this action targets")
    esi_level:    Optional[int]  = Field(None, ge=1, le=5, description="ESI level 1–5")
    diagnostics:  Optional[List[str]] = Field(None, description="List of DiagnosticTest values")
    bed_type:     Optional[str]  = Field(None, description="BedType for resource allocation")
    interventions:Optional[List[str]] = Field(None, description="Immediate interventions ordered")
    rationale:    Optional[str]  = Field(None, description="Clinical reasoning (optional but scored)")


class StepRequest(BaseModel):
    """Wrapper for POST /step request body."""
    action: Action


# ──────────────────────────── Observation ────────────────────────────────────

class Observation(BaseModel):
    """Returned by reset() and step()."""
    task_id:           str
    episode_id:        str
    step:              int
    max_steps:         int
    current_patient:   Optional[Patient]          = None
    patient_queue:     Optional[List[Patient]]    = None   # Task 3 only
    department_status: Optional[DepartmentStatus] = None   # Task 3 only
    instructions:      str
    context:           Optional[str]              = None
    valid_action_types:List[str]                  = Field(default_factory=list)
    action_schema:     Dict[str, Any]             = Field(default_factory=dict)


# ──────────────────────────── Reward ─────────────────────────────────────────

class PartialScores(BaseModel):
    accuracy:        Optional[float] = Field(None, ge=0.0, le=1.0)
    appropriateness: Optional[float] = Field(None, ge=0.0, le=1.0)
    efficiency:      Optional[float] = Field(None, ge=0.0, le=1.0)
    safety:          Optional[float] = Field(None, ge=0.0, le=1.0)


class Reward(BaseModel):
    """Returned at every step."""
    score:          float         = Field(..., ge=0.0, le=1.0, description="Step reward")
    partial_scores: PartialScores = Field(default_factory=PartialScores)
    feedback:       str
    done:           bool
    episode_score:  Optional[float] = Field(None, ge=0.0, le=1.0,
                                            description="Set only when done=True")


# ──────────────────────────── State ──────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Returned by state() — full serialisable snapshot."""
    task_id:               str
    episode_id:            str
    step:                  int
    max_steps:             int
    step_scores:           List[float]
    current_episode_score: float
    status:                str                     # "running" | "done"
    patients_processed:    int
    patients_remaining:    int
    department_status:     Optional[DepartmentStatus] = None


# ──────────────────────────── Task metadata ──────────────────────────────────

class TaskInfo(BaseModel):
    task_id:                str
    name:                   str
    description:            str
    difficulty:             str        # "easy" | "medium" | "hard"
    max_steps:              int
    valid_action_types:     List[str]
    action_schema:          Dict[str, Any]
    observation_description:str
    scoring_criteria:       str
    baseline_score:         Optional[float] = None


# ──────────────────────────── Baseline result ────────────────────────────────

class BaselineResult(BaseModel):
    task_id:        str
    score:          float
    step_scores:    List[float]
    model:          str
    total_steps:    int
