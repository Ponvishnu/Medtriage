"""
MedTriageEnv — Task Definitions

Defines the three tasks, their metadata, and action schemas returned by /tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from environment.models import TaskInfo


# ─────────────────────────── Action Schemas ───────────────────────────────────

TRIAGE_SCHEMA: Dict[str, Any] = {
    "action_type": {
        "type": "string",
        "const": "triage",
        "description": "Must be exactly 'triage'",
    },
    "patient_id": {
        "type": "string",
        "description": "The patient_id from the observation (e.g. 'T1-P01')",
    },
    "esi_level": {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": (
            "Emergency Severity Index: "
            "1=Immediate, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent"
        ),
    },
    "rationale": {
        "type": "string",
        "description": "Optional: clinical reasoning for the triage decision",
    },
    "required": ["action_type", "patient_id", "esi_level"],
}

DIAGNOSTICS_SCHEMA: Dict[str, Any] = {
    "action_type": {
        "type": "string",
        "const": "order_diagnostics",
        "description": "Must be exactly 'order_diagnostics'",
    },
    "patient_id": {
        "type": "string",
        "description": "The patient_id from the observation",
    },
    "diagnostics": {
        "type": "array",
        "items": {
            "type": "string",
            "enum": [
                "ecg", "chest_xray", "cbc", "bmp", "troponin", "d_dimer",
                "ct_head", "ct_chest", "ct_abdomen", "urinalysis", "lactate",
                "bnp", "lipase", "lft", "coags", "blood_culture", "urine_culture",
            ],
        },
        "description": "List of diagnostic tests to order",
        "minItems": 1,
    },
    "rationale": {
        "type": "string",
        "description": "Optional: clinical reasoning for your test selection",
    },
    "required": ["action_type", "patient_id", "diagnostics"],
}

ALLOCATE_SCHEMA: Dict[str, Any] = {
    "action_type": {
        "type": "string",
        "const": "allocate_resources",
        "description": "Must be exactly 'allocate_resources'",
    },
    "patient_id": {
        "type": "string",
        "description": "The patient_id from the observation",
    },
    "esi_level": {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": "Your triage assessment: 1=Immediate … 5=Non-Urgent",
    },
    "bed_type": {
        "type": "string",
        "enum": ["trauma_bay", "ccu", "acute", "general", "waiting"],
        "description": (
            "Bed to allocate. Capacity: trauma_bay=3, ccu=5, acute=10, general=15. "
            "'waiting' = no bed needed yet."
        ),
    },
    "interventions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional list of immediate interventions to initiate",
    },
    "rationale": {
        "type": "string",
        "description": "Optional: clinical and operational reasoning",
    },
    "required": ["action_type", "patient_id", "esi_level", "bed_type"],
}


# ─────────────────────────── Task Definitions ────────────────────────────────

TASK_CONFIGS: Dict[str, TaskInfo] = {

    "task_1": TaskInfo(
        task_id="task_1",
        name="Single-Patient Triage",
        difficulty="easy",
        max_steps=10,
        valid_action_types=["triage"],
        action_schema=TRIAGE_SCHEMA,
        description=(
            "The ED has 10 patients arriving one by one.  For each patient you must "
            "assign the correct Emergency Severity Index (ESI) level 1–5 based on "
            "their presenting complaint, vital signs, and brief history.  "
            "The ESI determines the urgency of care and the required resources.\n\n"
            "ESI 1 = Immediate life-saving intervention required\n"
            "ESI 2 = High-risk, should not wait\n"
            "ESI 3 = Stable but needs ≥2 resources (labs, imaging, IV)\n"
            "ESI 4 = Stable, needs 1 resource\n"
            "ESI 5 = Stable, no resources needed"
        ),
        observation_description=(
            "Each step presents one Patient object with: age, sex, chief complaint, "
            "vital signs (BP, HR, RR, SpO2, temperature, GCS, pain), symptom duration, "
            "medical history, current medications, allergies, and arrival mode."
        ),
        scoring_criteria=(
            "Exact ESI match: ~1.0 | Off by 1: ~0.7 | Off by 2: ~0.3 | Off by 3+: 0.0 | "
            "Under-triage of ESI 1 patient: heavy safety penalty.  "
            "Episode score = mean of 10 step scores."
        ),
        baseline_score=None,
    ),

    "task_2": TaskInfo(
        task_id="task_2",
        name="Diagnostic Workup Planning",
        difficulty="medium",
        max_steps=5,
        valid_action_types=["order_diagnostics"],
        action_schema=DIAGNOSTICS_SCHEMA,
        description=(
            "Five complex patients need their initial diagnostic workup ordered.  "
            "For each patient, order the appropriate tests from the available list.  "
            "You are scored on whether you order all critical tests, include useful "
            "optional tests, and avoid unnecessary or wasteful orders.  "
            "Available tests: ecg, chest_xray, cbc, bmp, troponin, d_dimer, "
            "ct_head, ct_chest, ct_abdomen, urinalysis, lactate, bnp, lipase, "
            "lft, coags, blood_culture, urine_culture."
        ),
        observation_description=(
            "Each step presents one Patient with a complex multi-system presentation.  "
            "Additional clinical findings may appear in patient.additional_info."
        ),
        scoring_criteria=(
            "65% weight on must-order tests coverage, 25% on should-consider tests, "
            "−10% per unnecessary test (max −30%), −30% per contraindicated test.  "
            "Episode score = mean of 5 step scores."
        ),
        baseline_score=None,
    ),

    "task_3": TaskInfo(
        task_id="task_3",
        name="Mass-Casualty Surge Management",
        difficulty="hard",
        max_steps=20,
        valid_action_types=["allocate_resources"],
        action_schema=ALLOCATE_SCHEMA,
        description=(
            "A mass-casualty incident has overwhelmed the ED.  20 patients arrive "
            "simultaneously.  You must triage each patient AND allocate them to an "
            "appropriate bed, respecting capacity limits:\n\n"
            "  Trauma bays: 3  (for ESI 1 critical patients)\n"
            "  CCU beds:    5  (for ESI 1–2)\n"
            "  Acute beds:  10 (for ESI 2–3)\n"
            "  General beds:15 (for ESI 3–5)\n"
            "  Waiting room: unlimited\n\n"
            "You must simultaneously get triage right AND make smart bed decisions "
            "as resources deplete.  A patient waiting in the wrong area is penalised.  "
            "Placing a non-urgent patient in a trauma bay during a mass-casualty event "
            "is a serious error."
        ),
        observation_description=(
            "Each step presents the next unprocessed patient plus the current "
            "DepartmentStatus showing how many beds are used/available.  "
            "The full patient queue (remaining unprocessed patients) is also shown."
        ),
        scoring_criteria=(
            "40% triage accuracy + 40% bed appropriateness + 15% resource efficiency + "
            "5% rationale bonus.  Capacity violations reduce bed score.  "
            "Episode score = mean of 20 step scores."
        ),
        baseline_score=None,
    ),
}


def get_task(task_id: str) -> TaskInfo:
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_id]


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id":          t.task_id,
            "name":             t.name,
            "difficulty":       t.difficulty,
            "max_steps":        t.max_steps,
            "valid_action_types": t.valid_action_types,
            "action_schema":    t.action_schema,
            "description":      t.description,
            "has_grader":       t.has_grader,
        }
        for t in TASK_CONFIGS.values()
    ]
