"""
MedTriageEnv — Patient Scenario Bank

All scenarios are fixed (seed-based determinism) so baseline scores reproduce
exactly across runs.  Medical accuracy is grounded in ESI v4 guidelines and
validated against published case studies.

Each scenario includes the ground-truth ESI level and diagnostic protocol key
used by the graders — these fields are NEVER exposed in the Observation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from environment.models import Patient, VitalSigns


def _patient(pid: str, age: int, sex: str, cc: str,
             sbp: int, dbp: int, hr: int, rr: int, spo2: float,
             temp: float, gcs: int, pain: int,
             duration_h: float, history: str,
             meds: List[str], allergies: List[str],
             arrival: str, true_esi: int,
             protocol_key: str = "",
             extra: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "patient": Patient(
            patient_id=pid,
            age=age,
            sex=sex,
            chief_complaint=cc,
            vitals=VitalSigns(
                bp_systolic=sbp, bp_diastolic=dbp,
                heart_rate=hr, respiratory_rate=rr,
                spo2=spo2, temperature=temp,
                gcs=gcs, pain_score=pain,
            ),
            symptom_duration_hours=duration_h,
            history=history,
            medications=meds,
            allergies=allergies,
            arrival_mode=arrival,
            additional_info=extra or {},
        ),
        "true_esi":     true_esi,
        "protocol_key": protocol_key,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Single-patient triage  (10 patients, ESI 1–5 coverage)
# ══════════════════════════════════════════════════════════════════════════════

TASK1_PATIENTS: List[Dict] = [

    # ── ESI 1 ────────────────────────────────────────────────────────────────
    _patient(
        "T1-P01", 65, "M",
        "Crushing central chest pain, heavy sweating, feeling faint",
        88, 52, 122, 24, 91.0, 36.8, 15, 9,
        0.5, "Known CAD, HTN, T2DM.  Previous CABG 2018.",
        ["metoprolol", "aspirin", "metformin", "lisinopril"],
        ["penicillin"],
        "ambulance",
        true_esi=1,
        protocol_key="stemi",
        extra={"ecg_shown": "ST elevation V1–V4", "rhythm": "sinus tachycardia"},
    ),

    _patient(
        "T1-P02", 78, "F",
        "Acute confusion, not responding normally, brought by family",
        74, 40, 138, 34, 86.0, 39.9, 9, 0,
        6.0, "T2DM, CHF, CKD stage 3.  Lives alone.",
        ["furosemide", "atorvastatin", "insulin glargine"],
        ["sulfa drugs"],
        "ambulance",
        true_esi=1,
        protocol_key="septic_shock",
        extra={"family_note": "Found on floor, urine soaked, confused since morning"},
    ),

    _patient(
        "T1-P03", 7, "M",
        "Seizure at home, now drowsy",
        108, 68, 152, 30, 94.0, 40.3, 10, 0,
        0.3, "No prior seizures.  No neurological history.  Up to date vaccinations.",
        [],
        ["amoxicillin"],
        "ambulance",
        true_esi=1,
        extra={"seizure_duration_min": 4, "post_ictal": True},
    ),

    # ── ESI 2 ────────────────────────────────────────────────────────────────
    _patient(
        "T1-P04", 72, "M",
        "Sudden right facial droop, left arm weakness, can't speak clearly",
        192, 106, 96, 18, 96.0, 37.0, 13, 2,
        0.75, "HTN, atrial fibrillation (on warfarin), hyperlipidaemia.",
        ["warfarin", "atorvastatin", "amlodipine"],
        [],
        "ambulance",
        true_esi=2,
        protocol_key="acute_stroke",
        extra={"last_known_well": "45 min ago", "nihss_estimated": 12},
    ),

    _patient(
        "T1-P05", 45, "M",
        "Worst headache of my life, sudden onset, neck stiffness",
        148, 88, 94, 20, 98.0, 37.5, 14, 8,
        1.5, "No significant past medical history.",
        [],
        [],
        "walk-in",
        true_esi=2,
        protocol_key="subarachnoid_hemorrhage",
        extra={"headache_onset": "thunderclap", "photophobia": True, "neck_stiffness": True},
    ),

    _patient(
        "T1-P06", 52, "M",
        "Severe epigastric pain radiating to back, repeated vomiting, yellow eyes",
        138, 82, 126, 26, 93.0, 37.3, 15, 10,
        8.0, "Gallstones diagnosed 1 year ago, heavy alcohol use.",
        ["omeprazole"],
        [],
        "ambulance",
        true_esi=2,
        protocol_key="pancreatitis",
        extra={"jaundice": True, "murphy_sign": "positive"},
    ),

    # ── ESI 3 ────────────────────────────────────────────────────────────────
    _patient(
        "T1-P07", 58, "M",
        "Chest tightness, mild shortness of breath on exertion, started 2 hours ago",
        142, 90, 96, 20, 96.0, 36.9, 15, 5,
        2.0, "Hypertension, family history of MI.  Smoker 20 pack-years.",
        ["amlodipine", "atorvastatin"],
        ["nsaids"],
        "walk-in",
        true_esi=3,
        protocol_key="stemi",
        extra={"ecg_shown": "Non-specific ST changes", "diaphoresis": False},
    ),

    _patient(
        "T1-P08", 34, "F",
        "Right lower quadrant abdominal pain with fever",
        124, 80, 102, 22, 98.0, 38.7, 15, 7,
        18.0, "Appendectomy 2018.  IUD in situ.",
        ["oral contraceptive"],
        [],
        "walk-in",
        true_esi=3,
        protocol_key="appendicitis",
        extra={"rebound_tenderness": True, "guarding": "mild"},
    ),

    # ── ESI 4 ────────────────────────────────────────────────────────────────
    _patient(
        "T1-P09", 19, "M",
        "Laceration right forearm from kitchen knife, bleeding controlled",
        120, 78, 78, 16, 99.0, 36.6, 15, 3,
        0.5, "No significant history.",
        [],
        [],
        "walk-in",
        true_esi=4,
        extra={"wound_length_cm": 3, "bleeding_controlled": True, "neurovascular_intact": True},
    ),

    # ── ESI 5 ────────────────────────────────────────────────────────────────
    _patient(
        "T1-P10", 28, "F",
        "Urinary burning and frequency for 2 days",
        116, 74, 74, 16, 99.0, 36.6, 15, 2,
        48.0, "Recurrent UTIs, otherwise healthy.",
        ["oral contraceptive"],
        ["sulfa drugs"],
        "walk-in",
        true_esi=5,
        extra={"no_fever": True, "vaginal_discharge": False, "back_pain": False},
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Diagnostic workup planning  (5 complex presentations)
# ══════════════════════════════════════════════════════════════════════════════

TASK2_PATIENTS: List[Dict] = [

    _patient(
        "T2-P01", 61, "M",
        "10/10 crushing central chest pain radiating to left jaw, diaphoresis",
        96, 62, 114, 22, 93.0, 36.7, 15, 10,
        1.0, "Known ischaemic heart disease, previous NSTEMI 2020, T2DM, HTN.",
        ["clopidogrel", "atorvastatin", "ramipril", "metformin"],
        ["penicillin"],
        "ambulance",
        true_esi=1,
        protocol_key="stemi",
        extra={"ecg_shown": "ST elevation II, III, aVF", "rhythm": "sinus tachycardia"},
    ),

    _patient(
        "T2-P02", 47, "F",
        "Sudden onset worst headache of life, neck stiffness, vomited twice",
        162, 94, 88, 18, 98.0, 37.1, 14, 9,
        2.0, "Migraines (but this is 'completely different'), on OCP.",
        ["sumatriptan", "oral contraceptive"],
        [],
        "ambulance",
        true_esi=2,
        protocol_key="subarachnoid_hemorrhage",
        extra={"thunderclap": True, "photophobia": True},
    ),

    _patient(
        "T2-P03", 38, "M",
        "Right lower quadrant pain, fever, nausea — unable to eat since yesterday",
        128, 82, 104, 22, 97.0, 38.9, 15, 7,
        20.0, "No prior surgeries.  Healthy.",
        [],
        ["codeine"],
        "walk-in",
        true_esi=3,
        protocol_key="appendicitis",
        extra={"rovsings_sign": True, "psoas_sign": True},
    ),

    _patient(
        "T2-P04", 68, "F",
        "Acute breathlessness, right leg swelling since returning from a long flight",
        136, 88, 112, 28, 88.0, 36.8, 15, 6,
        12.0, "Breast cancer (on chemotherapy), recent 14-hour flight.",
        ["tamoxifen", "ondansetron"],
        [],
        "ambulance",
        true_esi=2,
        protocol_key="pulmonary_embolism",
        extra={"wells_score": 7.5, "right_leg_diameter_increase_cm": 3},
    ),

    _patient(
        "T2-P05", 24, "M",
        "High fever, confusion, photophobia, non-blanching skin spots",
        84, 48, 136, 32, 88.0, 40.2, 10, 0,
        6.0, "University student, returned from freshers' week last week.",
        [],
        [],
        "ambulance",
        true_esi=1,
        protocol_key="meningitis",
        extra={"petechiae": True, "kernigs_sign": True, "brudzinskis_sign": True},
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Mass-casualty surge management  (20 patients, mixed acuity)
# ══════════════════════════════════════════════════════════════════════════════

TASK3_PATIENTS: List[Dict] = [

    # ── Wave 1: Major road traffic collision + bystanders ────────────────────
    _patient(
        "T3-P01", 35, "M", "Major blunt chest trauma, reduced breath sounds right",
        78, 42, 132, 32, 84.0, 36.5, 12, 8,
        0.2, "Unrestrained driver, RTC at 80 km/h.",
        [], [], "helicopter", true_esi=1,
        extra={"mechanism": "RTC", "suspected": "tension pneumothorax"}
    ),
    _patient(
        "T3-P02", 28, "F", "Pelvic fracture, uncontrolled haemorrhage",
        66, 34, 148, 36, 82.0, 36.2, 11, 9,
        0.2, "Front-seat passenger same RTC.",
        [], [], "helicopter", true_esi=1,
        extra={"mechanism": "RTC", "pelvis_xray": "open book fracture"}
    ),
    _patient(
        "T3-P03", 52, "M", "Head injury, brief LOC, now GCS 14, vomited once",
        146, 90, 92, 20, 97.0, 36.8, 14, 5,
        0.3, "Driver of other vehicle, moderate speed.",
        ["aspirin"], [], "ambulance", true_esi=2,
        extra={"mechanism": "RTC", "anticoagulated": True}
    ),
    _patient(
        "T3-P04", 19, "F", "Ankle fracture, severe pain, neurovascular intact",
        122, 76, 98, 18, 99.0, 36.7, 15, 8,
        0.3, "Pedestrian struck by vehicle.",
        [], [], "ambulance", true_esi=3,
        extra={"xray_finding": "distal fibula fracture"}
    ),
    _patient(
        "T3-P05", 66, "M", "Witnessed cardiac arrest, bystander CPR in progress",
        0, 0, 0, 0, 0.0, 36.0, 3, 0,
        0.1, "Bystander at RTC, collapsed while watching.",
        ["bisoprolol", "ramipril"], [], "ambulance", true_esi=1,
        extra={"rhythm": "VF", "cpr_ongoing": True}
    ),

    # ── Wave 2: Medical presentations ────────────────────────────────────────
    _patient(
        "T3-P06", 70, "F", "Acute hemiplegia, facial droop, slurred speech onset 1 hr ago",
        182, 100, 88, 18, 96.0, 36.9, 13, 2,
        1.0, "HTN, AF (on apixaban), hyperlipidaemia.",
        ["apixaban", "atorvastatin", "lisinopril"], [], "ambulance", true_esi=2,
        protocol_key="acute_stroke",
    ),
    _patient(
        "T3-P07", 55, "M", "Fever, rigors, hypotension, catheterised last week",
        80, 44, 130, 30, 90.0, 39.8, 13, 0,
        12.0, "Prostate cancer, recent urological procedure.",
        ["tamsulosin", "finasteride"], ["penicillin"], "ambulance", true_esi=2,
        protocol_key="septic_shock",
    ),
    _patient(
        "T3-P08", 42, "F", "Anaphylaxis to bee sting — throat tightening, widespread urticaria",
        88, 54, 128, 28, 91.0, 37.1, 14, 6,
        0.25, "Bee allergy, forgot EpiPen.",
        [], ["bee stings"], "ambulance", true_esi=2,
        extra={"stridor": True, "urticaria": True}
    ),
    _patient(
        "T3-P09", 48, "M", "Severe central chest pain, ECG done in ambulance",
        104, 68, 98, 22, 95.0, 36.8, 15, 8,
        1.5, "Hypertension, smoker, family history MI.",
        ["amlodipine"], ["aspirin"], "ambulance", true_esi=2,
        protocol_key="stemi",
        extra={"ecg_shown": "STEMI anterior"}
    ),
    _patient(
        "T3-P10", 33, "F", "Productive cough, fever 38.5°C, pleuritic chest pain",
        124, 78, 100, 22, 95.0, 38.5, 15, 5,
        72.0, "Asthma on salbutamol.",
        ["salbutamol inhaler"], [], "walk-in", true_esi=3,
    ),

    # ── Wave 3: Mixed ─────────────────────────────────────────────────────────
    _patient(
        "T3-P11", 82, "M", "Fall from standing, hip pain, unable to weight-bear",
        134, 82, 88, 18, 97.0, 36.7, 14, 7,
        1.0, "Osteoporosis, warfarin for AF.",
        ["warfarin", "calcium carbonate"], [], "ambulance", true_esi=3,
        extra={"suspected": "NOF fracture", "anticoagulated": True}
    ),
    _patient(
        "T3-P12", 29, "M", "First episode psychosis, agitated, threatening staff",
        144, 88, 108, 20, 98.0, 37.0, 15, 0,
        0.5, "No psychiatric history.",
        [], [], "walk-in", true_esi=2,
        extra={"behaviour": "aggressive", "risk_to_self": True}
    ),
    _patient(
        "T3-P13", 61, "F", "Breathlessness, bilateral leg swelling, orthopnoea",
        160, 96, 108, 26, 91.0, 36.8, 15, 5,
        48.0, "Known heart failure, HTN, non-compliant with diuretics.",
        ["furosemide", "carvedilol", "spironolactone"], ["sulfa drugs"], "walk-in", true_esi=2,
        protocol_key="heart_failure",
    ),
    _patient(
        "T3-P14", 16, "M", "Asthma attack, unable to complete sentences, not responding to nebuliser",
        128, 80, 130, 36, 88.0, 36.9, 14, 7,
        2.0, "Severe asthma, 3 previous ICU admissions.",
        ["salbutamol", "budesonide", "montelukast"], [], "ambulance", true_esi=1,
        extra={"peak_flow_predicted": 25, "silent_chest": False}
    ),
    _patient(
        "T3-P15", 44, "F", "Moderate headache, 3/10, tension type, no red flags",
        118, 76, 78, 16, 99.0, 36.5, 15, 3,
        6.0, "Frequent tension headaches.",
        ["paracetamol (prn)"], [], "walk-in", true_esi=4,
    ),

    # ── Wave 4: Lower acuity ──────────────────────────────────────────────────
    _patient(
        "T3-P16", 23, "M", "Painful eye — something went in while cutting wood",
        118, 74, 76, 16, 99.0, 36.6, 15, 4,
        2.0, "Healthy.",
        [], [], "walk-in", true_esi=3,
        extra={"suspected": "corneal foreign body"}
    ),
    _patient(
        "T3-P17", 50, "M", "Moderate back pain after lifting, no neurological symptoms",
        122, 80, 80, 16, 99.0, 36.7, 15, 5,
        24.0, "Chronic back pain.",
        ["ibuprofen"], [], "walk-in", true_esi=4,
    ),
    _patient(
        "T3-P18", 38, "F", "Sprained wrist, mild swelling, distal pulses intact",
        120, 78, 82, 16, 99.0, 36.6, 15, 4,
        1.0, "Healthy.",
        [], [], "walk-in", true_esi=4,
    ),
    _patient(
        "T3-P19", 31, "M", "Minor cut on finger — needs a plaster and tetanus check",
        118, 76, 74, 16, 99.0, 36.5, 15, 1,
        0.5, "No significant history.",
        [], [], "walk-in", true_esi=5,
    ),
    _patient(
        "T3-P20", 26, "F", "Prescription refill for oral contraceptive",
        116, 72, 72, 14, 99.0, 36.5, 15, 0,
        0.0, "Healthy.",
        ["oral contraceptive"], [], "walk-in", true_esi=5,
    ),
]


def get_task1_patients() -> List[Dict]:
    return list(TASK1_PATIENTS)


def get_task2_patients() -> List[Dict]:
    return list(TASK2_PATIENTS)


def get_task3_patients() -> List[Dict]:
    return list(TASK3_PATIENTS)
