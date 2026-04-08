"""
MedTriageEnv — Baseline Agents

Two agents are provided:

1. RuleBasedAgent — deterministic, no API key required.
   Uses structured clinical heuristics derived from ESI v4.
   Produces reproducible reference scores on every run.

2. LLMAgent — uses Google Gemini API (reads GEMINI_API_KEY from environment).
   Sends full patient data + ESI guidelines to gemini-2.5-flash.
   Demonstrates how an LLM-powered agent interacts with the environment.

Usage:
  python baseline/run_baseline.py                      # RuleBasedAgent only
  GEMINI_API_KEY=AIza... python baseline/run_baseline.py --llm   # + LLMAgent
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.env import MedTriageEnvironment
from environment.models import Action
from environment.graders import grader_summary


# ══════════════════════════════════════════════════════════════════════════════
# 1. Rule-Based Agent  (deterministic, no external calls)
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedAgent:
    """
    Applies structured clinical heuristics to replicate the Emergency Severity
    Index v4 decision algorithm.  No randomness — scores are fully reproducible.

    Task-1 heuristic: map vital sign patterns → ESI level
    Task-2 heuristic: map chief complaint keywords → standard diagnostic panels
    Task-3 heuristic: triage as above + assign beds by ESI priority queue
    """

    def __init__(self) -> None:
        self._env = MedTriageEnvironment()

    # ── ESI Heuristic ────────────────────────────────────────────────────────

    def _infer_esi(self, patient_dict: Dict[str, Any]) -> int:
        """Heuristic ESI classification from vitals and chief complaint."""
        vitals = patient_dict.get("vitals", {})
        cc     = patient_dict.get("chief_complaint", "").lower()
        gcs    = vitals.get("gcs", 15)
        spo2   = vitals.get("spo2", 99)
        sbp    = vitals.get("bp_systolic", 120)
        hr     = vitals.get("heart_rate", 80)
        rr     = vitals.get("respiratory_rate", 16)
        temp   = vitals.get("temperature", 37.0)
        pain   = vitals.get("pain_score", 0)

        # ── ESI 1: Immediate life threat ─────────────────────────────────────
        if gcs <= 8:                         return 1
        if spo2 < 88:                        return 1
        if sbp < 70:                         return 1
        if hr == 0:                          return 1   # Cardiac arrest
        if "arrest" in cc:                   return 1
        if "seizure" in cc and gcs < 13:     return 1
        if "pulseless" in cc:                return 1
        if "unresponsive" in cc:             return 1

        # ── ESI 2: High risk ──────────────────────────────────────────────────
        esi2_keywords = [
            "stroke", "facial droop", "hemiplegia", "slurred speech",
            "anaphylaxis", "throat tightening", "stridor",
            "worst headache", "thunderclap",
            "stemi", "st elevation",
            "petechiae", "petechial", "non-blanching",
            "haemorrhage", "hemorrhage",
            "psychosis", "acute confusion",
        ]
        if any(k in cc for k in esi2_keywords): return 2
        if sbp < 90 and hr > 110:               return 2  # Haemodynamic instability
        if spo2 < 92:                            return 2
        if hr > 130 or hr < 45:                 return 2
        if rr > 28:                              return 2
        if gcs <= 13:                            return 2
        if "septic" in cc or ("fever" in cc and sbp < 100): return 2
        if temp > 40.0 and hr > 120:             return 2

        # ── ESI 3: Needs ≥2 resources ─────────────────────────────────────────
        esi3_keywords = [
            "chest pain", "chest tightness", "shortness of breath",
            "abdominal pain", "right lower quadrant", "rlq",
            "back pain" + "neurological",
            "head injury", "fall from height",
            "hip pain", "fracture",
            "breathlessness", "dyspnoea",
            "corneal", "eye",
            "productive cough", "pleuritic",
        ]
        if any(k in cc for k in esi3_keywords): return 3
        if pain >= 7:                            return 3

        # ── ESI 4: Needs 1 resource ───────────────────────────────────────────
        esi4_keywords = [
            "laceration", "cut", "wound",
            "sprain", "sprained", "twisted ankle",
            "urinary", "burning", "uti",
            "ear", "earache",
            "back pain",
            "headache",
        ]
        if any(k in cc for k in esi4_keywords): return 4

        # ── ESI 5: No resources ────────────────────────────────────────────────
        return 5

    # ── Diagnostic Heuristic ─────────────────────────────────────────────────

    def _infer_diagnostics(self, patient_dict: Dict[str, Any]) -> List[str]:
        """Keyword-based panel selection."""
        cc  = patient_dict.get("chief_complaint", "").lower()
        add = str(patient_dict.get("additional_info", "")).lower()
        combined = cc + " " + add

        if any(k in combined for k in ["stemi", "st elevation", "chest pain", "jaw"]):
            return ["ecg", "troponin", "chest_xray", "cbc", "bmp"]

        if any(k in combined for k in ["thunderclap", "worst headache", "subarachnoid"]):
            return ["ct_head", "cbc", "bmp"]

        if any(k in combined for k in ["appendicitis", "rlq", "right lower quadrant",
                                        "right lower", "nausea", "guarding"]):
            return ["cbc", "bmp", "ct_abdomen", "urinalysis"]

        if any(k in combined for k in ["dvt", "pe", "pulmonary", "embolism",
                                        "d_dimer", "leg swelling", "flight"]):
            return ["ecg", "chest_xray", "d_dimer", "troponin", "bnp", "cbc", "bmp"]

        if any(k in combined for k in ["septic", "sepsis", "meningitis",
                                        "petechiae", "petechial", "rigors", "hypotension",
                                        "non-blanching", "kernigs"]):
            return ["blood_culture", "lactate", "cbc", "bmp", "coags", "ct_head"]

        if any(k in combined for k in ["stroke", "hemiplegia", "facial droop"]):
            return ["ct_head", "cbc", "bmp", "coags", "ecg"]

        if any(k in combined for k in ["heart failure", "bilateral leg swelling", "orthopnoea"]):
            return ["ecg", "bnp", "chest_xray", "cbc", "bmp", "troponin"]

        # Fallback general workup
        return ["cbc", "bmp", "ecg"]

    # ── Bed Assignment Heuristic ─────────────────────────────────────────────

    def _infer_bed(self, esi: int, resource_usage: Dict[str, int]) -> str:
        """Assign the most appropriate available bed by ESI level."""
        capacity = {"trauma_bay": 3, "ccu": 5, "acute": 10, "general": 15}

        preferred = {
            1: ["trauma_bay", "ccu", "acute"],
            2: ["ccu", "acute", "general"],
            3: ["acute", "general", "waiting"],
            4: ["general", "waiting"],
            5: ["waiting", "general"],
        }

        for bed in preferred.get(esi, ["general"]):
            if bed == "waiting":
                return "waiting"
            if resource_usage.get(bed, 0) < capacity.get(bed, 999):
                return bed

        return "waiting"

    # ── Single Episode ────────────────────────────────────────────────────────

    def run_episode(self, task_id: str) -> Dict[str, Any]:
        """Run one complete episode for the given task.  Returns grader summary."""
        obs = self._env.reset(task_id=task_id)
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        resource_usage: Dict[str, int] = {}
        step_scores: List[float] = []
        done = False

        while not done:
            patient = obs_dict.get("current_patient") or {}
            pid = patient.get("patient_id", "")

            if task_id == "task_1":
                esi = self._infer_esi(patient)
                action = Action(
                    action_type="triage",
                    patient_id=pid,
                    esi_level=esi,
                    rationale=f"Rule-based heuristic: ESI {esi} based on vitals and complaint",
                )

            elif task_id == "task_2":
                diags = self._infer_diagnostics(patient)
                action = Action(
                    action_type="order_diagnostics",
                    patient_id=pid,
                    diagnostics=diags,
                    rationale="Standard diagnostic panel for presentation",
                )

            else:  # task_3
                esi      = self._infer_esi(patient)
                bed_type = self._infer_bed(esi, resource_usage)
                if bed_type != "waiting":
                    resource_usage[bed_type] = resource_usage.get(bed_type, 0) + 1
                action = Action(
                    action_type="allocate_resources",
                    patient_id=pid,
                    esi_level=esi,
                    bed_type=bed_type,
                    rationale=f"ESI {esi} → {bed_type} (capacity-aware allocation)",
                )

            result_dict = {}
            obs_dict, reward_model, done, info = self._env.step(action)

            # obs_dict is already an Observation pydantic object here
            obs_dict = obs_dict.model_dump() if hasattr(obs_dict, "model_dump") else obs_dict
            step_scores.append(reward_model.score)

        episode_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        return {
            "task_id":       task_id,
            "episode_score": round(episode_score, 4),
            "step_scores":   step_scores,
            "num_steps":     len(step_scores),
            "agent":         "RuleBasedAgent",
        }

    def run_all_tasks(self) -> Dict[str, Any]:
        """Run all three tasks and return a combined results dict."""
        results = {}
        for tid in ["task_1", "task_2", "task_3"]:
            print(f"  Running {tid}...", flush=True)
            r = self.run_episode(tid)
            results[tid] = r
            print(f"    {tid} score: {r['episode_score']:.4f}", flush=True)

        overall = round(
            sum(r["episode_score"] for r in results.values()) / len(results), 4
        )
        return {
            "agent":          "RuleBasedAgent",
            "task_results":   results,
            "overall_score":  overall,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM Agent  (Google Gemini API — requires GEMINI_API_KEY)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert Emergency Department triage clinician with 20 years of
experience.  You make fast, accurate decisions using the Emergency Severity Index (ESI) v4.

ESI LEVELS:
1 = IMMEDIATE   — life-saving intervention required right now (cardiac arrest, respiratory failure,
                  GCS≤8, SpO2<88%, SBP<80 with shock, active major haemorrhage)
2 = EMERGENT    — should not wait; high risk of deterioration (STEMI, stroke, anaphylaxis,
                  sepsis, worst-headache-of-life, haemodynamic instability)
3 = URGENT      — stable but needs ≥2 resources (labs + imaging)
4 = LESS URGENT — stable, needs 1 resource (X-ray, urinalysis, suture)
5 = NON-URGENT  — stable, no diagnostic resources needed

AVAILABLE DIAGNOSTIC TESTS:
ecg, chest_xray, cbc, bmp, troponin, d_dimer, ct_head, ct_chest, ct_abdomen,
urinalysis, lactate, bnp, lipase, lft, coags, blood_culture, urine_culture

AVAILABLE BED TYPES (Task 3):
trauma_bay (capacity 3), ccu (capacity 5), acute (capacity 10),
general (capacity 15), waiting (unlimited)

Always respond with a valid JSON object matching the required action schema.
Never add commentary outside the JSON.  Always include a concise rationale."""


def _format_patient(p: Dict[str, Any]) -> str:
    """Format patient dict into a readable string for the LLM prompt."""
    v = p.get("vitals", {})
    lines = [
        f"Patient ID: {p.get('patient_id')}",
        f"Age/Sex: {p.get('age')} {p.get('sex')}",
        f"Chief Complaint: {p.get('chief_complaint')}",
        f"Vitals: BP {v.get('bp_systolic')}/{v.get('bp_diastolic')} | "
        f"HR {v.get('heart_rate')} | RR {v.get('respiratory_rate')} | "
        f"SpO2 {v.get('spo2')}% | Temp {v.get('temperature')}°C | "
        f"GCS {v.get('gcs')} | Pain {v.get('pain_score')}/10",
        f"Duration: {p.get('symptom_duration_hours')}h",
        f"History: {p.get('history')}",
        f"Medications: {', '.join(p.get('medications', [])) or 'None'}",
        f"Allergies: {', '.join(p.get('allergies', [])) or 'NKDA'}",
        f"Arrival: {p.get('arrival_mode')}",
    ]
    add = p.get("additional_info") or {}
    if add:
        lines.append(f"Additional findings: {json.dumps(add)}")
    return "\n".join(lines)


# Default Gemini API key (override via GEMINI_API_KEY environment variable)
_DEFAULT_GEMINI_API_KEY = "AIzaSyBD7lqTjmgA7gbiJHDUx-PM9nKr5S7JhFA"


class LLMAgent:
    """
    Agent powered by the Google Gemini API.

    Reads GEMINI_API_KEY from environment (falls back to built-in key).
    Uses google-generativeai SDK with gemini-2.5-flash by default.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        try:
            from google import genai          # type: ignore
            from google.genai import types    # type: ignore
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        api_key = os.environ.get("GEMINI_API_KEY", _DEFAULT_GEMINI_API_KEY)
        self._client = genai.Client(api_key=api_key)
        self._types  = types
        self._model  = model
        self._env    = MedTriageEnvironment()

    def _call(self, user_message: str) -> str:
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_message
        config = self._types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
            response_mime_type="application/json",
        )
        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=full_prompt,
                    config=config,
                )
                return resp.text
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _parse_action(self, raw: str, task_id: str,
                      patient_id: str) -> Action:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: return safe default
            return self._fallback_action(task_id, patient_id)

        try:
            return Action(
                action_type=data.get("action_type", "triage"),
                patient_id=data.get("patient_id", patient_id),
                esi_level=data.get("esi_level"),
                diagnostics=data.get("diagnostics"),
                bed_type=data.get("bed_type"),
                interventions=data.get("interventions"),
                rationale=data.get("rationale"),
            )
        except Exception:
            return self._fallback_action(task_id, patient_id)

    @staticmethod
    def _fallback_action(task_id: str, patient_id: str) -> Action:
        if task_id == "task_1":
            return Action(action_type="triage", patient_id=patient_id, esi_level=3)
        if task_id == "task_2":
            return Action(action_type="order_diagnostics", patient_id=patient_id,
                          diagnostics=["cbc", "bmp", "ecg"])
        return Action(action_type="allocate_resources", patient_id=patient_id,
                      esi_level=3, bed_type="general")

    def run_episode(self, task_id: str) -> Dict[str, Any]:
        obs_dict = self._env.reset(task_id=task_id)
        obs_dict = obs_dict.model_dump() if hasattr(obs_dict, "model_dump") else obs_dict
        step_scores: List[float] = []
        done = False

        while not done:
            patient  = obs_dict.get("current_patient") or {}
            pid      = patient.get("patient_id", "unknown")
            dept     = obs_dict.get("department_status")

            # Build the user message
            patient_text = _format_patient(patient)
            if task_id == "task_1":
                prompt = (
                    f"Assign the ESI triage level for this patient.\n\n"
                    f"{patient_text}\n\n"
                    f"Respond with JSON: {{\"action_type\": \"triage\", "
                    f"\"patient_id\": \"{pid}\", \"esi_level\": <int 1-5>, "
                    f"\"rationale\": \"<brief reasoning>\"}}"
                )
            elif task_id == "task_2":
                prompt = (
                    f"Order the appropriate diagnostic tests for this patient.\n\n"
                    f"{patient_text}\n\n"
                    f"Respond with JSON: {{\"action_type\": \"order_diagnostics\", "
                    f"\"patient_id\": \"{pid}\", \"diagnostics\": [\"test1\", ...], "
                    f"\"rationale\": \"<brief reasoning>\"}}"
                )
            else:
                dept_str = json.dumps(dept) if dept else "{}"
                prompt = (
                    f"Triage this patient AND allocate an appropriate bed during surge.\n\n"
                    f"{patient_text}\n\n"
                    f"Department Status: {dept_str}\n\n"
                    f"Respond with JSON: {{\"action_type\": \"allocate_resources\", "
                    f"\"patient_id\": \"{pid}\", \"esi_level\": <int 1-5>, "
                    f"\"bed_type\": \"<trauma_bay|ccu|acute|general|waiting>\", "
                    f"\"rationale\": \"<brief reasoning>\"}}"
                )

            raw    = self._call(prompt)
            action = self._parse_action(raw, task_id, pid)
            obs_model, reward_model, done, info = self._env.step(action)
            obs_dict = obs_model.model_dump() if hasattr(obs_model, "model_dump") else obs_model
            step_scores.append(reward_model.score)

        episode_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        return {
            "task_id":       task_id,
            "episode_score": round(episode_score, 4),
            "step_scores":   step_scores,
            "num_steps":     len(step_scores),
            "agent":         f"LLMAgent(gemini/{self._model})",
        }

    def run_all_tasks(self) -> Dict[str, Any]:
        results = {}
        for tid in ["task_1", "task_2", "task_3"]:
            print(f"  Running {tid} with LLM...", flush=True)
            r = self.run_episode(tid)
            results[tid] = r
            print(f"    {tid} score: {r['episode_score']:.4f}", flush=True)

        overall = round(
            sum(r["episode_score"] for r in results.values()) / len(results), 4
        )
        return {
            "agent":         f"LLMAgent(gemini/{self._model})",
            "task_results":  results,
            "overall_score": overall,
        }
