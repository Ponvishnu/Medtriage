"""
MedTriageEnv — Grading Engine

Three deterministic graders, one per task.  All return float in [0.0, 1.0].

Grader contracts:
  grade_task1_step(action, patient_record)           → (score, partial, feedback)
  grade_task2_step(action, patient_record)           → (score, partial, feedback)
  grade_task3_step(action, patient_record, dept)     → (score, partial, feedback)
  finalize_episode(step_scores)                      → episode_score
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from environment.clinical_knowledge import (
    BED_APPROPRIATENESS,
    RESOURCE_CAPACITY,
    score_diagnostics,
    score_esi,
)
from environment.models import Action, DepartmentStatus, PartialScores


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _rationale_bonus(rationale: str | None) -> float:
    """Small bonus (up to 0.05) for providing non-trivial clinical reasoning."""
    if not rationale:
        return 0.0
    words = rationale.strip().split()
    if len(words) >= 6:
        return 0.05
    if len(words) >= 3:
        return 0.02
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Single-patient triage
# ══════════════════════════════════════════════════════════════════════════════

def grade_task1_step(
    action: Action,
    patient_record: Dict[str, Any],
) -> Tuple[float, PartialScores, str]:
    """
    Grade a single triage action.

    Scoring:
      • 65%  Accuracy: ESI score from clinical matrix
      • 30%  Safety:   Heavy penalty if critical patient (ESI 1) is under-triaged
      • 5%   Rationale bonus
    """
    true_esi = patient_record["true_esi"]
    pred_esi = action.esi_level

    if pred_esi is None:
        return 0.0, PartialScores(accuracy=0.0, safety=0.0), \
               "No ESI level provided — action must include esi_level for task_1"

    accuracy_score = score_esi(pred_esi, true_esi)

    # Safety dimension: under-triaging a critical patient is clinically dangerous
    safety_score = 1.0
    if true_esi == 1 and pred_esi >= 3:
        safety_score = 0.0   # Missed a life-threatening emergency
    elif true_esi == 1 and pred_esi == 2:
        safety_score = 0.5   # Better but still a miss
    elif true_esi == 2 and pred_esi >= 4:
        safety_score = 0.3

    bonus = _rationale_bonus(action.rationale)
    raw_score = 0.65 * accuracy_score + 0.30 * safety_score + bonus
    score = _clamp(raw_score)

    # Human-readable feedback
    if pred_esi == true_esi:
        fb = f"✓ Correct triage — ESI {true_esi} ({_esi_label(true_esi)})"
    elif pred_esi < true_esi:
        fb = (f"Over-triaged: assigned ESI {pred_esi} but correct is ESI {true_esi}. "
              f"Wastes scarce resuscitation resources.")
    else:
        fb = (f"Under-triaged: assigned ESI {pred_esi} but correct is ESI {true_esi}. "
              f"{'⚠ SAFETY RISK — patient needed immediate care.' if true_esi <= 2 else ''}")

    partial = PartialScores(accuracy=round(accuracy_score, 4),
                            safety=round(safety_score, 4))
    return round(score, 4), partial, fb


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Diagnostic workup planning
# ══════════════════════════════════════════════════════════════════════════════

def grade_task2_step(
    action: Action,
    patient_record: Dict[str, Any],
) -> Tuple[float, PartialScores, str]:
    """
    Grade a diagnostic order set.

    Scoring:
      • 65%  Must-order coverage
      • 25%  Should-consider coverage
      • −penalty  Unnecessary or contraindicated tests
      • 5%   Rationale bonus
    """
    protocol_key = patient_record.get("protocol_key", "")
    ordered: List[str] = action.diagnostics or []

    if not ordered:
        return 0.0, PartialScores(appropriateness=0.0), \
               "No diagnostics ordered — action must include diagnostics list for task_2"

    result = score_diagnostics(ordered, protocol_key)
    base_score = result["score"]
    bonus = _rationale_bonus(action.rationale)
    score = _clamp(base_score + bonus)

    partial = PartialScores(appropriateness=round(base_score, 4))
    return round(score, 4), partial, result["feedback"]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Mass-casualty surge management
# ══════════════════════════════════════════════════════════════════════════════

def grade_task3_step(
    action: Action,
    patient_record: Dict[str, Any],
    dept: DepartmentStatus,
    resource_usage: Dict[str, int],
) -> Tuple[float, PartialScores, str, Dict[str, int]]:
    """
    Grade a single resource-allocation action during surge.

    Scoring per patient:
      • 40%  Triage accuracy (ESI matrix)
      • 40%  Bed appropriateness
      • 15%  Resource efficiency (avoid over-allocating scarce beds to low-acuity)
      • 5%   Rationale bonus

    Returns (score, partial_scores, feedback, updated_resource_usage).
    """
    true_esi = patient_record["true_esi"]
    pred_esi = action.esi_level
    bed_type  = action.bed_type or "waiting"

    if pred_esi is None:
        return 0.0, PartialScores(), "esi_level required for task_3 actions", resource_usage

    # ── ESI accuracy ──────────────────────────────────────────────────────────
    esi_acc = score_esi(pred_esi, true_esi)

    # ── Bed appropriateness ───────────────────────────────────────────────────
    bed_score_map = BED_APPROPRIATENESS.get(true_esi, {})
    bed_score = bed_score_map.get(bed_type, 0.5)

    # Capacity check: if bed type is full, penalise — can't physically assign it
    capacity = RESOURCE_CAPACITY.get(bed_type, 999)
    used_so_far = resource_usage.get(bed_type, 0)
    capacity_violation = used_so_far >= capacity

    if capacity_violation and bed_type != "waiting":
        bed_score = max(0.0, bed_score - 0.4)
        cap_note = f" ⚠ {bed_type} is FULL ({used_so_far}/{capacity})"
    else:
        cap_note = ""

    # ── Resource efficiency ───────────────────────────────────────────────────
    # Penalise placing low-acuity patients in high-value beds
    efficiency_score = 1.0
    if true_esi >= 4 and bed_type in ("trauma_bay", "ccu"):
        efficiency_score = 0.2   # Seriously wasteful during surge
    elif true_esi == 3 and bed_type == "trauma_bay":
        efficiency_score = 0.5

    # Update resource tracking
    updated_usage = dict(resource_usage)
    if bed_type != "waiting":
        updated_usage[bed_type] = used_so_far + 1

    bonus = _rationale_bonus(action.rationale)
    raw = 0.40 * esi_acc + 0.40 * bed_score + 0.15 * efficiency_score + bonus
    score = _clamp(raw)

    # Compose feedback
    parts = []
    if pred_esi != true_esi:
        parts.append(f"Triage: assigned ESI {pred_esi}, correct ESI {true_esi}")
    else:
        parts.append(f"Triage ✓ ESI {true_esi}")
    if bed_score < 0.7:
        parts.append(f"Bed assignment '{bed_type}' suboptimal for ESI {true_esi}")
    if capacity_violation:
        parts.append(cap_note)
    if efficiency_score < 0.5:
        parts.append("Placing low-acuity patient in critical bed wastes surge capacity")
    if not parts or all("✓" in p for p in parts):
        parts = [f"Good allocation — ESI {true_esi} → {bed_type}"]

    partial = PartialScores(
        accuracy=round(esi_acc, 4),
        appropriateness=round(bed_score, 4),
        efficiency=round(efficiency_score, 4),
    )
    return round(score, 4), partial, "; ".join(parts), updated_usage


# ══════════════════════════════════════════════════════════════════════════════
# Episode finalization
# ══════════════════════════════════════════════════════════════════════════════

def finalize_episode(step_scores: List[float]) -> float:
    """Aggregate step scores into a final episode score (mean)."""
    if not step_scores:
        return 0.0
    return round(sum(step_scores) / len(step_scores), 4)


def grader_summary(task_id: str, step_scores: List[float]) -> Dict[str, Any]:
    """Return a full grader report for a completed episode."""
    ep_score = finalize_episode(step_scores)
    return {
        "task_id":        task_id,
        "episode_score":  ep_score,
        "step_scores":    step_scores,
        "num_steps":      len(step_scores),
        "min_step":       min(step_scores) if step_scores else 0.0,
        "max_step":       max(step_scores) if step_scores else 0.0,
        "grade":          _letter_grade(ep_score),
    }


def _letter_grade(score: float) -> str:
    if score >= 0.90: return "A"
    if score >= 0.80: return "B"
    if score >= 0.70: return "C"
    if score >= 0.60: return "D"
    return "F"


def _esi_label(esi: int) -> str:
    labels = {1: "Immediate", 2: "Emergent", 3: "Urgent",
              4: "Less Urgent", 5: "Non-Urgent"}
    return labels.get(esi, "Unknown")
