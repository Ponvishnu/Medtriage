"""
MedTriageEnv — FastAPI Server

Exposes the full OpenEnv interface:
  POST /reset          → initial Observation
  POST /step           → Observation + Reward + done + info
  GET  /state          → EnvironmentState

Plus required extra endpoints:
  GET  /tasks          → list of tasks and action schemas
  POST /baseline       → run baseline inference, return scores
  POST /grader         → grader score for completed episode

And utility endpoints:
  GET  /               → health/info
  GET  /health         → liveness probe

Run locally:
  uvicorn app:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment.env import MedTriageEnvironment
from environment.models import Action, StepRequest

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedTriageEnv",
    description=(
        "Emergency Department Clinical Triage and Decision Support — "
        "OpenEnv compatible real-world environment for training and evaluating AI agents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env = MedTriageEnvironment()


# ──────────────────────────────────────────────────────────────────────────────
# Health / Info
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """Root endpoint — returns environment metadata."""
    return {
        "name":        "MedTriageEnv",
        "version":     "1.0.0",
        "description": (
            "Emergency Department Clinical Triage and Decision Support Environment. "
            "An AI agent acts as a triage clinician making real-time decisions about "
            "patient acuity, diagnostic workup, and resource allocation."
        ),
        "tasks": ["task_1 (easy)", "task_2 (medium)", "task_3 (hard)"],
        "openenv_spec": True,
        "endpoints": {
            "reset":    "POST /reset?task_id=task_1",
            "step":     "POST /step",
            "state":    "GET  /state",
            "tasks":    "GET  /tasks",
            "baseline": "POST /baseline",
            "grader":   "POST /grader",
        },
    }


@app.get("/health", tags=["Info"])
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Core OpenEnv Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", tags=["OpenEnv"])
def reset(task_id: str = Query(default="task_1",
                               description="One of: task_1, task_2, task_3")) -> Dict[str, Any]:
    """
    **Reset** — begin a new episode for the specified task.

    Returns the initial Observation (first patient + instructions).
    Must be called before the first step().
    """
    try:
        obs = _env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step", tags=["OpenEnv"])
def step(request: StepRequest) -> Dict[str, Any]:
    """
    **Step** — execute one action and advance the environment.

    Request body must include an `action` object.  The required fields depend
    on the current task (see GET /tasks for the full schema).

    Returns:
    - `observation`  — next patient (or terminal state)
    - `reward`       — score, partial scores, feedback, done flag
    - `done`         — True if the episode has ended
    - `info`         — auxiliary metadata (running score, episode_id, …)
    """
    try:
        obs, reward, done, info = _env.step(request.action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Step failed: {e}\n{traceback.format_exc()}")


@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    """
    **State** — return the full serialisable environment state.

    Useful for debugging or checkpointing an agent's progress mid-episode.
    """
    return _env.state().model_dump()


# ──────────────────────────────────────────────────────────────────────────────
# Required Extra Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/tasks", tags=["OpenEnv"])
def tasks() -> Dict[str, Any]:
    """
    **Tasks** — return all task descriptions and their action schemas.

    Use this to understand what fields your action must include for each task.
    """
    return {"tasks": _env.get_tasks()}


@app.post("/baseline", tags=["OpenEnv"])
def baseline() -> Dict[str, Any]:
    """
    **Baseline** — run the built-in deterministic rule-based baseline agent
    against all three tasks and return reproducible scores.

    This does NOT require an OpenAI API key.  The rule-based baseline uses
    structured clinical heuristics to achieve consistent reference scores.

    Use `baseline/run_baseline.py` (with OPENAI_API_KEY) for LLM-powered scores.
    """
    try:
        from baseline.baseline_agent import RuleBasedAgent
        agent  = RuleBasedAgent()
        result = agent.run_all_tasks()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline run failed: {e}\n{traceback.format_exc()}"
        )


@app.post("/grader", tags=["OpenEnv"])
def grader(episode_id: Optional[str] = None) -> Dict[str, Any]:
    """
    **Grader** — return the grader score for the most recently completed episode.

    If an episode is still in progress, returns the running partial score.
    If no episode has been started, returns an informative message.
    """
    return _env.get_grader_score(episode_id=episode_id)


# ──────────────────────────────────────────────────────────────────────────────
# Example action endpoint (developer convenience)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/example_action", tags=["Info"])
def example_action(task_id: str = Query(default="task_1")) -> Dict[str, Any]:
    """Return an example valid action for the specified task."""
    examples = {
        "task_1": {
            "action": {
                "action_type": "triage",
                "patient_id":  "T1-P01",
                "esi_level":   2,
                "rationale":   "Haemodynamically unstable with chest pain — high suspicion STEMI",
            }
        },
        "task_2": {
            "action": {
                "action_type": "order_diagnostics",
                "patient_id":  "T2-P01",
                "diagnostics": ["ecg", "troponin", "chest_xray", "cbc", "bmp"],
                "rationale":   "Suspected STEMI — need cardiac workup urgently",
            }
        },
        "task_3": {
            "action": {
                "action_type":  "allocate_resources",
                "patient_id":   "T3-P01",
                "esi_level":    1,
                "bed_type":     "trauma_bay",
                "interventions": ["oxygen", "large_bore_iv", "trauma_alert"],
                "rationale":    "Major trauma with haemodynamic instability — trauma bay + immediate team",
            }
        },
    }
    if task_id not in examples:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    return examples[task_id]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
