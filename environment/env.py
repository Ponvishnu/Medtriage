"""
MedTriageEnv — Core Environment  (OpenEnv Spec Compliant)

Implements:
  reset(task_id)           → Observation
  step(action)             → (Observation, Reward, bool, dict)
  state()                  → EnvironmentState
  get_tasks()              → list of task metadata dicts
  get_grader_score()       → grader summary for current/last episode
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from environment.graders import (
    finalize_episode,
    grade_task1_step,
    grade_task2_step,
    grade_task3_step,
    grader_summary,
)
from environment.models import (
    Action,
    DepartmentStatus,
    EnvironmentState,
    Observation,
    PartialScores,
    Patient,
    Reward,
)
from environment.patient_generator import (
    get_task1_patients,
    get_task2_patients,
    get_task3_patients,
)
from environment.tasks import TASK_CONFIGS, list_tasks


# ══════════════════════════════════════════════════════════════════════════════
class MedTriageEnvironment:
    """
    Emergency Department Triage Environment.

    State is fully in-memory.  Call reset() to begin a new episode.
    One episode = one complete run through a task's patient list.
    """

    def __init__(self) -> None:
        self._task_id:        str                = "task_1"
        self._episode_id:     str                = ""
        self._step:           int                = 0
        self._max_steps:      int                = 10
        self._patients:       List[Dict]         = []
        self._step_scores:    List[float]        = []
        self._done:           bool               = True
        self._resource_usage: Dict[str, int]     = {}
        self._last_grader:    Optional[Dict]     = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_dept_status(self) -> DepartmentStatus:
        used = self._resource_usage
        return DepartmentStatus(
            trauma_bays_total=3,   trauma_bays_used=used.get("trauma_bay", 0),
            ccu_beds_total=5,      ccu_beds_used=used.get("ccu", 0),
            acute_beds_total=10,   acute_beds_used=used.get("acute", 0),
            general_beds_total=15, general_beds_used=used.get("general", 0),
            patients_waiting=used.get("waiting", 0),
            elapsed_minutes=float(self._step * 3),   # ~3 min per decision
        )

    def _current_patient(self) -> Optional[Patient]:
        if self._step < len(self._patients):
            return self._patients[self._step]["patient"]
        return None

    def _remaining_queue(self) -> List[Patient]:
        """Return patients not yet processed (for Task 3 visibility)."""
        return [p["patient"] for p in self._patients[self._step + 1:]]

    def _task_config(self):
        return TASK_CONFIGS[self._task_id]

    def _build_observation(self, step_override: Optional[int] = None) -> Observation:
        step = step_override if step_override is not None else self._step
        cfg  = self._task_config()

        dept_status = self._make_dept_status() if self._task_id == "task_3" else None
        queue       = self._remaining_queue()   if self._task_id == "task_3" else None
        patient     = self._current_patient()

        return Observation(
            task_id=self._task_id,
            episode_id=self._episode_id,
            step=step,
            max_steps=self._max_steps,
            current_patient=patient,
            patient_queue=queue if queue else None,
            department_status=dept_status,
            instructions=cfg.description,
            context=cfg.observation_description,
            valid_action_types=cfg.valid_action_types,
            action_schema=cfg.action_schema,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_1") -> Observation:
        """Start a new episode for the given task.  Returns initial observation."""
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid tasks: {list(TASK_CONFIGS.keys())}"
            )

        self._task_id     = task_id
        self._episode_id  = str(uuid.uuid4())
        self._step        = 0
        self._step_scores = []
        self._done        = False
        self._last_grader = None

        if task_id == "task_1":
            self._patients = get_task1_patients()
        elif task_id == "task_2":
            self._patients = get_task2_patients()
        else:
            self._patients = get_task3_patients()
            self._resource_usage = {
                "trauma_bay": 0, "ccu": 0, "acute": 0, "general": 0, "waiting": 0
            }

        self._max_steps = len(self._patients)
        return self._build_observation(step_override=0)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action.

        Returns:
          observation  — next state (or terminal state if done)
          reward       — Reward pydantic model
          done         — True if episode has ended
          info         — auxiliary metadata dict
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() to start a new one.")

        if self._step >= len(self._patients):
            raise RuntimeError(f"Step {self._step} exceeds patient count {len(self._patients)}")

        patient_record = self._patients[self._step]
        task_id = self._task_id

        # ── Grade this step ───────────────────────────────────────────────────
        if task_id == "task_1":
            score, partial, feedback = grade_task1_step(action, patient_record)

        elif task_id == "task_2":
            score, partial, feedback = grade_task2_step(action, patient_record)

        else:  # task_3
            dept = self._make_dept_status()
            score, partial, feedback, self._resource_usage = grade_task3_step(
                action, patient_record, dept, self._resource_usage
            )

        self._step_scores.append(score)
        self._step += 1

        # Check terminal condition
        done = (self._step >= len(self._patients))
        self._done = done

        episode_score: Optional[float] = None
        if done:
            episode_score = finalize_episode(self._step_scores)
            self._last_grader = grader_summary(task_id, self._step_scores)

        # ── Build next observation ────────────────────────────────────────────
        next_obs = self._build_observation()

        reward = Reward(
            score=score,
            partial_scores=partial,
            feedback=feedback,
            done=done,
            episode_score=episode_score,
        )

        info: Dict[str, Any] = {
            "episode_id":           self._episode_id,
            "step":                 self._step,
            "step_scores_so_far":   list(self._step_scores),
            "running_mean_score":   round(sum(self._step_scores) / len(self._step_scores), 4),
        }
        if done and episode_score is not None:
            info["episode_score"] = episode_score
            info["grade"]         = self._last_grader["grade"]

        return next_obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return the full serialisable environment state."""
        remaining = max(0, len(self._patients) - self._step)
        dept = self._make_dept_status() if self._task_id == "task_3" else None
        return EnvironmentState(
            task_id=self._task_id,
            episode_id=self._episode_id,
            step=self._step,
            max_steps=self._max_steps,
            step_scores=list(self._step_scores),
            current_episode_score=(
                round(sum(self._step_scores) / len(self._step_scores), 4)
                if self._step_scores else 0.0
            ),
            status="done" if self._done else "running",
            patients_processed=self._step,
            patients_remaining=remaining,
            department_status=dept,
        )

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Return all task metadata (for GET /tasks)."""
        return list_tasks()

    def get_grader_score(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Return grader summary for the last completed episode."""
        if self._last_grader is None:
            if not self._done:
                return {
                    "status":  "in_progress",
                    "message": "Episode still running — grader score available after completion",
                    "step":    self._step,
                    "running_score": (
                        round(sum(self._step_scores) / len(self._step_scores), 4)
                        if self._step_scores else 0.0
                    ),
                }
            return {"status": "no_episode", "message": "No episode completed yet — call reset() first"}
        return {"status": "complete", **self._last_grader}
