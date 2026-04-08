"""
MedTriageEnv — Clinical Knowledge Base

Encodes evidence-based guidelines for:
  • Emergency Severity Index (ESI) v4 classification criteria
  • Diagnostic appropriateness for common presentations
  • Resource requirements per acuity level
  • Bed allocation guidelines

Sources:
  • Gilboy N et al. Emergency Severity Index (ESI): A Triage Tool for Emergency
    Department Care, Version 4. AHRQ Publication, 2012.
  • Choosing Wisely / ACR Appropriateness Criteria
  • ACEP Clinical Policies
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List


# ─────────────────────────── ESI Criteria ────────────────────────────────────

ESI_CRITERIA: Dict[int, Dict] = {
    1: {
        "label": "Immediate",
        "description": "Requires immediate life-saving intervention",
        "examples": [
            "Cardiac/respiratory arrest",
            "Unresponsive / GCS ≤ 8 with airway compromise",
            "Severe respiratory distress (accessory muscle use, SpO2 < 90%)",
            "Systolic BP < 80 with signs of shock",
            "Active major haemorrhage",
            "Paediatric seizure with altered consciousness",
        ],
        "vital_flags": {
            "gcs_max": 8,
            "spo2_max": 89,
            "sbp_max": 80,
        },
    },
    2: {
        "label": "Emergent",
        "description": "High-risk situation; should not wait in queue",
        "examples": [
            "Suspected STEMI / ACS with haemodynamic changes",
            "Acute stroke within treatment window",
            "Anaphylaxis",
            "Altered mental status with haemodynamic instability",
            "Sepsis (SIRS criteria + suspected source)",
            "Threatened limb / ischaemia",
            "Worst-headache-of-life (subarachnoid haemorrhage until proven otherwise)",
            "New-onset severe hypertensive emergency (SBP > 180 + end-organ signs)",
        ],
        "vital_flags": {
            "hr_max": 130,
            "hr_min": 40,
            "rr_max": 28,
            "spo2_max": 93,
        },
    },
    3: {
        "label": "Urgent",
        "description": "Stable but requires 2+ diagnostic resources",
        "examples": [
            "Moderate chest pain (non-STEMI, needs ECG + troponin + CXR)",
            "Moderate abdominal pain (needs labs + imaging)",
            "Head injury (alert and oriented, needs CT + neuro obs)",
            "Fever with unknown source (needs labs + cultures)",
            "Back pain with neurological symptoms",
            "Moderate allergic reaction",
        ],
        "vital_flags": {},
    },
    4: {
        "label": "Less Urgent",
        "description": "Stable; needs only 1 diagnostic resource",
        "examples": [
            "Simple laceration requiring sutures",
            "Ankle/wrist sprain needing X-ray",
            "Urinary tract infection symptoms",
            "Mild allergic reaction (local hives only)",
            "Minor head injury (no LOC, no amnesia, GCS 15)",
            "Earache / ear infection",
        ],
        "vital_flags": {},
    },
    5: {
        "label": "Non-Urgent",
        "description": "Stable; no diagnostic resources needed",
        "examples": [
            "Cold / minor upper respiratory infection",
            "Prescription refill request",
            "Chronic back pain (no change from baseline)",
            "Suture / staple removal",
            "Minor skin rash (stable, no airway involvement)",
        ],
        "vital_flags": {},
    },
}


# ─────────────────────────── Diagnostic Protocols ────────────────────────────

# Each protocol encodes which tests are required, recommended, unnecessary, or
# contraindicated for a given clinical presentation.

DIAGNOSTIC_PROTOCOLS: Dict[str, Dict[str, List[str]]] = {

    # ── Cardiac ──────────────────────────────────────────────────────────────
    "stemi": {
        "must_order":      ["ecg", "troponin", "chest_xray"],
        "should_consider": ["cbc", "bmp", "coags"],
        "unnecessary":     ["lipase", "urinalysis", "ct_abdomen", "lft"],
        "contraindicated": [],
        "presentation":    "ST-elevation myocardial infarction / acute coronary syndrome",
    },
    "heart_failure": {
        "must_order":      ["ecg", "bnp", "chest_xray"],
        "should_consider": ["cbc", "bmp", "troponin"],
        "unnecessary":     ["lipase", "ct_abdomen", "blood_culture"],
        "contraindicated": [],
        "presentation":    "Acute decompensated heart failure",
    },

    # ── Neurological ─────────────────────────────────────────────────────────
    "subarachnoid_hemorrhage": {
        "must_order":      ["ct_head"],
        "should_consider": ["cbc", "bmp", "coags"],
        "unnecessary":     ["ecg", "lipase", "bnp", "troponin"],
        "contraindicated": [],
        "presentation":    "Thunderclap headache / suspected subarachnoid haemorrhage",
    },
    "acute_stroke": {
        "must_order":      ["ct_head", "cbc", "bmp", "coags"],
        "should_consider": ["ecg", "troponin"],
        "unnecessary":     ["lipase", "d_dimer", "bnp"],
        "contraindicated": [],
        "presentation":    "Acute ischaemic/haemorrhagic stroke",
    },

    # ── Abdominal ─────────────────────────────────────────────────────────────
    "appendicitis": {
        "must_order":      ["cbc", "bmp"],
        "should_consider": ["ct_abdomen", "urinalysis", "lipase"],
        "unnecessary":     ["troponin", "bnp", "d_dimer", "ecg"],
        "contraindicated": [],
        "presentation":    "Suspected appendicitis / RLQ pain with fever",
    },
    "pancreatitis": {
        "must_order":      ["lipase", "cbc", "bmp"],
        "should_consider": ["lft", "ct_abdomen"],
        "unnecessary":     ["troponin", "bnp", "d_dimer", "blood_culture"],
        "contraindicated": [],
        "presentation":    "Acute pancreatitis",
    },

    # ── Pulmonary ─────────────────────────────────────────────────────────────
    "pulmonary_embolism": {
        "must_order":      ["ecg", "chest_xray", "d_dimer"],
        "should_consider": ["troponin", "bnp", "cbc", "bmp"],
        "unnecessary":     ["lipase", "ct_abdomen", "urinalysis", "lft"],
        "contraindicated": [],
        "presentation":    "Suspected pulmonary embolism / acute dyspnoea with hypoxia",
    },

    # ── Infectious ────────────────────────────────────────────────────────────
    "septic_shock": {
        "must_order":      ["blood_culture", "lactate", "cbc", "bmp", "coags"],
        "should_consider": ["urinalysis", "urine_culture", "chest_xray"],
        "unnecessary":     ["lipase", "bnp"],
        "contraindicated": [],
        "presentation":    "Septic shock / severe sepsis",
    },
    "meningitis": {
        "must_order":      ["ct_head", "blood_culture", "cbc", "bmp"],
        "should_consider": ["coags", "lactate"],
        "unnecessary":     ["lipase", "ecg", "bnp", "troponin"],
        "contraindicated": [],
        "presentation":    "Suspected bacterial meningitis",
    },
}


# ─────────────────────────── Resource Guidelines ─────────────────────────────

BED_APPROPRIATENESS: Dict[int, Dict[str, float]] = {
    # ESI level → {bed_type: score}
    1: {
        "trauma_bay": 1.0,
        "ccu":        0.9,
        "acute":      0.3,   # Dangerous delay
        "general":    0.0,
        "waiting":    0.0,
    },
    2: {
        "trauma_bay": 0.8,   # Overkill but acceptable in surge
        "ccu":        1.0,
        "acute":      0.9,
        "general":    0.5,
        "waiting":    0.0,
    },
    3: {
        "trauma_bay": 0.5,   # Wasteful but not harmful
        "ccu":        0.6,
        "acute":      1.0,
        "general":    0.9,
        "waiting":    0.4,
    },
    4: {
        "trauma_bay": 0.2,
        "ccu":        0.3,
        "acute":      0.8,
        "general":    1.0,
        "waiting":    0.9,
    },
    5: {
        "trauma_bay": 0.1,
        "ccu":        0.2,
        "acute":      0.5,
        "general":    0.8,
        "waiting":    1.0,
    },
}

RESOURCE_CAPACITY = {
    "trauma_bay": 3,
    "ccu":        5,
    "acute":      10,
    "general":    15,
}


# ─────────────────────────── Scoring Helpers ─────────────────────────────────

ESI_SCORE_MATRIX: Dict[int, Dict[int, float]] = {
    # true_esi → predicted_esi → score
    1: {1: 1.0, 2: 0.4, 3: 0.0, 4: 0.0, 5: 0.0},  # Under-triage of critical is dangerous
    2: {1: 0.7, 2: 1.0, 3: 0.5, 4: 0.0, 5: 0.0},
    3: {1: 0.5, 2: 0.8, 3: 1.0, 4: 0.7, 5: 0.0},
    4: {1: 0.3, 2: 0.5, 3: 0.8, 4: 1.0, 5: 0.8},
    5: {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.9, 5: 1.0},
}
# Note: Over-triaging (assigning ESI 1 to ESI 5) wastes scarce resources,
# hence symmetrical penalties are appropriate but slightly less severe than
# under-triaging (assigning ESI 5 to ESI 1) which is clinically dangerous.


def score_esi(predicted: int, true: int) -> float:
    """Return triage accuracy score using the ESI scoring matrix."""
    return ESI_SCORE_MATRIX.get(true, {}).get(predicted, 0.0)


def score_diagnostics(ordered: list[str], protocol_key: str) -> dict:
    """
    Score a diagnostic order set against the protocol for a given presentation.

    Returns a dict with:
      - score (float 0–1)
      - must_hit (int)
      - should_hit (int)
      - unnecessary_ordered (int)
      - contraindicated_ordered (int)
      - feedback (str)
    """
    if protocol_key not in DIAGNOSTIC_PROTOCOLS:
        return {"score": 0.5, "feedback": "Unknown protocol — unable to grade"}

    proto   = DIAGNOSTIC_PROTOCOLS[protocol_key]
    ordered_set: FrozenSet[str] = frozenset(t.lower() for t in ordered)

    must_set    = frozenset(proto["must_order"])
    should_set  = frozenset(proto["should_consider"])
    unnec_set   = frozenset(proto["unnecessary"])
    contra_set  = frozenset(proto["contraindicated"])

    must_hit        = len(ordered_set & must_set)
    should_hit      = len(ordered_set & should_set)
    unnec_ordered   = len(ordered_set & unnec_set)
    contra_ordered  = len(ordered_set & contra_set)

    must_total   = max(len(must_set),   1)
    should_total = max(len(should_set), 1)

    must_score   = must_hit   / must_total
    should_score = should_hit / should_total

    penalty = min(0.3 * unnec_ordered, 0.3) + min(0.5 * contra_ordered, 0.5)

    raw_score = 0.65 * must_score + 0.25 * should_score - penalty
    score     = max(0.0, min(1.0, raw_score))

    parts = []
    if must_hit < len(must_set):
        missing = sorted(must_set - ordered_set)
        parts.append(f"Missing critical tests: {', '.join(missing)}")
    if contra_ordered:
        bad = sorted(contra_set & ordered_set)
        parts.append(f"CONTRAINDICATED ordered: {', '.join(bad)}")
    if unnec_ordered:
        extra = sorted(unnec_set & ordered_set)
        parts.append(f"Unnecessary tests increase cost/time: {', '.join(extra)}")
    if not parts:
        parts.append("Excellent diagnostic workup")

    return {
        "score":                  round(score, 4),
        "must_hit":               must_hit,
        "should_hit":             should_hit,
        "unnecessary_ordered":    unnec_ordered,
        "contraindicated_ordered":contra_ordered,
        "feedback":               "; ".join(parts),
    }
