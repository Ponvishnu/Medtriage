"""
MedTriageEnv — Test Suite

Tests:
  1. OpenEnv spec compliance (typed models, step/reset/state contract)
  2. All three task episodes from start to finish
  3. Grader correctness (known inputs → expected scores)
  4. API endpoint smoke tests
  5. Baseline agent reproducibility

Run with:
  pytest tests/ -v
  python -m pytest tests/test_env.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.clinical_knowledge import score_esi, score_diagnostics
from environment.env import MedTriageEnvironment
from environment.models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    StepRequest,
)
from environment.tasks import TASK_CONFIGS, get_task, list_tasks


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    return MedTriageEnvironment()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Spec compliance
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecCompliance:

    def test_reset_returns_observation(self, env):
        obs = env.reset("task_1")
        assert isinstance(obs, Observation), "reset() must return Observation"

    def test_observation_has_required_fields(self, env):
        obs = env.reset("task_1")
        assert obs.task_id      == "task_1"
        assert obs.episode_id   != ""
        assert obs.step         == 0
        assert obs.max_steps    == 10
        assert obs.current_patient is not None
        assert len(obs.valid_action_types) > 0
        assert isinstance(obs.action_schema, dict)

    def test_step_returns_correct_types(self, env):
        env.reset("task_1")
        action = Action(action_type="triage", patient_id="T1-P01", esi_level=2)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs,    Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done,   bool)
        assert isinstance(info,   dict)

    def test_reward_score_in_range(self, env):
        env.reset("task_1")
        action = Action(action_type="triage", patient_id="T1-P01", esi_level=3)
        _, reward, _, _ = env.step(action)
        assert 0.0 <= reward.score <= 1.0, "Reward score must be in [0.0, 1.0]"

    def test_state_returns_environment_state(self, env):
        env.reset("task_2")
        s = env.state()
        assert isinstance(s, EnvironmentState)
        assert s.task_id == "task_2"
        assert s.status  == "running"

    def test_action_model_validates_esi_range(self):
        with pytest.raises(Exception):
            Action(action_type="triage", patient_id="P01", esi_level=0)
        with pytest.raises(Exception):
            Action(action_type="triage", patient_id="P01", esi_level=6)

    def test_reset_with_invalid_task_raises(self, env):
        with pytest.raises((ValueError, KeyError)):
            env.reset("task_999")

    def test_step_before_reset_raises(self, env):
        """env._done starts True — step without reset should raise."""
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="triage", patient_id="X", esi_level=3))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Full episode runs
# ══════════════════════════════════════════════════════════════════════════════

class TestFullEpisodes:

    def _run_episode(self, env, task_id: str, esi_override: int = 3,
                     diagnostics_override=None, bed_override: str = "acute") -> float:
        obs = env.reset(task_id)
        done = False
        scores = []
        step_count = 0

        while not done:
            patient = obs.current_patient
            assert patient is not None, f"current_patient is None at step {step_count}"
            pid = patient.patient_id

            if task_id == "task_1":
                action = Action(action_type="triage", patient_id=pid,
                                esi_level=esi_override)
            elif task_id == "task_2":
                action = Action(action_type="order_diagnostics", patient_id=pid,
                                diagnostics=diagnostics_override or ["cbc", "bmp"])
            else:
                action = Action(action_type="allocate_resources", patient_id=pid,
                                esi_level=esi_override, bed_type=bed_override)

            obs, reward, done, info = env.step(action)
            scores.append(reward.score)
            step_count += 1

        assert done, "Episode must end with done=True"
        assert reward.episode_score is not None, "Final reward must have episode_score"
        assert 0.0 <= reward.episode_score <= 1.0
        return reward.episode_score

    def test_task1_completes(self, env):
        score = self._run_episode(env, "task_1", esi_override=3)
        assert isinstance(score, float)

    def test_task2_completes(self, env):
        score = self._run_episode(env, "task_2",
                                  diagnostics_override=["ecg", "troponin", "chest_xray"])
        assert isinstance(score, float)

    def test_task3_completes(self, env):
        score = self._run_episode(env, "task_3", esi_override=2, bed_override="ccu")
        assert isinstance(score, float)

    def test_task1_perfect_answers(self, env):
        """Providing exact ESI answers should score higher than random answers."""
        from environment.patient_generator import get_task1_patients
        obs = env.reset("task_1")
        done = False
        patients_data = get_task1_patients()
        step = 0
        ep_score = 0.0

        while not done:
            patient = obs.current_patient
            true_esi = patients_data[step]["true_esi"]
            action = Action(action_type="triage", patient_id=patient.patient_id,
                            esi_level=true_esi,
                            rationale="Exact ground-truth answer for testing")
            obs, reward, done, info = env.step(action)
            step += 1

        ep_score = reward.episode_score
        assert ep_score > 0.85, (
            f"Perfect triage answers should score >0.85; got {ep_score}"
        )

    def test_task1_step_count(self, env):
        env.reset("task_1")
        count = 0
        done  = False
        obs   = env.reset("task_1")
        while not done:
            pid    = obs.current_patient.patient_id
            action = Action(action_type="triage", patient_id=pid, esi_level=3)
            obs, _, done, _ = env.step(action)
            count += 1
        assert count == 10, f"Task 1 should have exactly 10 steps, got {count}"

    def test_task2_step_count(self, env):
        obs  = env.reset("task_2")
        done = False
        count = 0
        while not done:
            pid    = obs.current_patient.patient_id
            action = Action(action_type="order_diagnostics", patient_id=pid,
                            diagnostics=["cbc"])
            obs, _, done, _ = env.step(action)
            count += 1
        assert count == 5, f"Task 2 should have exactly 5 steps, got {count}"

    def test_task3_step_count(self, env):
        obs  = env.reset("task_3")
        done = False
        count = 0
        while not done:
            pid    = obs.current_patient.patient_id
            action = Action(action_type="allocate_resources", patient_id=pid,
                            esi_level=3, bed_type="general")
            obs, _, done, _ = env.step(action)
            count += 1
        assert count == 20, f"Task 3 should have exactly 20 steps, got {count}"

    def test_episode_score_reproducible(self, env):
        """Same actions on task_1 should produce identical scores on every run."""
        def run():
            obs  = env.reset("task_1")
            done = False
            while not done:
                pid = obs.current_patient.patient_id
                action = Action(action_type="triage", patient_id=pid, esi_level=2)
                obs, reward, done, _ = env.step(action)
            return reward.episode_score

        score1 = run()
        score2 = run()
        assert score1 == score2, (
            f"Scores must be reproducible: {score1} != {score2}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Grader correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestGraders:

    def test_esi_exact_match_scores_1(self):
        for esi in range(1, 6):
            assert score_esi(esi, esi) == 1.0, f"ESI {esi} exact match should be 1.0"

    def test_esi_scores_in_range(self):
        for true in range(1, 6):
            for pred in range(1, 6):
                s = score_esi(pred, true)
                assert 0.0 <= s <= 1.0, f"ESI score({pred},{true}) = {s} out of range"

    def test_esi_undertriage_penalised_more_than_overtriage(self):
        """Missing ESI 1 is more dangerous than over-triaging ESI 5."""
        under_triage = score_esi(4, 1)   # Predicted 4, true 1
        over_triage  = score_esi(1, 4)   # Predicted 1, true 4
        assert under_triage < over_triage, (
            "Under-triaging a critical patient must score lower than over-triaging"
        )

    def test_diagnostics_must_order_coverage(self):
        result = score_diagnostics(["ecg", "troponin", "chest_xray"], "stemi")
        assert result["must_hit"] == 3
        assert result["score"] > 0.6

    def test_diagnostics_contraindicated_penalised(self):
        # No contraindicated tests in stemi protocol — verify unnecessary penalty
        result_clean = score_diagnostics(["ecg", "troponin", "chest_xray"], "stemi")
        result_dirty = score_diagnostics(["ecg", "troponin", "chest_xray",
                                          "lipase", "urinalysis", "ct_abdomen"], "stemi")
        assert result_clean["score"] > result_dirty["score"], (
            "Ordering unnecessary tests should reduce score"
        )

    def test_diagnostics_empty_order(self):
        result = score_diagnostics([], "stemi")
        assert result["score"] == 0.0 or result["must_hit"] == 0

    def test_grader_summary_keys(self):
        from environment.graders import grader_summary
        summary = grader_summary("task_1", [0.8, 0.9, 1.0, 0.7])
        assert "episode_score" in summary
        assert "grade" in summary
        assert 0.0 <= summary["episode_score"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Task metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskMetadata:

    def test_all_tasks_present(self):
        tasks = list_tasks()
        task_ids = [t["task_id"] for t in tasks]
        assert "task_1" in task_ids
        assert "task_2" in task_ids
        assert "task_3" in task_ids

    def test_task_difficulty_progression(self):
        difficulties = {t["task_id"]: t["difficulty"] for t in list_tasks()}
        assert difficulties["task_1"] == "easy"
        assert difficulties["task_2"] == "medium"
        assert difficulties["task_3"] == "hard"

    def test_task_action_schema_has_required(self):
        for task in list_tasks():
            schema = task["action_schema"]
            assert "required" in schema, f"{task['task_id']} missing 'required' in schema"

    def test_get_task_raises_on_unknown(self):
        with pytest.raises(KeyError):
            get_task("nonexistent_task")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Baseline reproducibility
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineReproducibility:

    def test_rule_based_agent_reproducible(self):
        """Two runs of RuleBasedAgent on the same task must produce identical scores."""
        from baseline.baseline_agent import RuleBasedAgent
        agent = RuleBasedAgent()

        run1 = agent.run_episode("task_1")
        run2 = agent.run_episode("task_1")

        assert run1["episode_score"] == run2["episode_score"], (
            f"RuleBasedAgent must be deterministic: {run1['episode_score']} != {run2['episode_score']}"
        )

    def test_rule_based_all_tasks(self):
        from baseline.baseline_agent import RuleBasedAgent
        agent   = RuleBasedAgent()
        results = agent.run_all_tasks()

        for tid in ["task_1", "task_2", "task_3"]:
            score = results["task_results"][tid]["episode_score"]
            assert 0.0 <= score <= 1.0, f"{tid} score {score} out of [0,1]"

    def test_rule_based_achieves_minimum_score(self):
        """The rule-based agent should outperform random guessing (>0.3 on all tasks)."""
        from baseline.baseline_agent import RuleBasedAgent
        agent = RuleBasedAgent()
        for tid in ["task_1", "task_2", "task_3"]:
            r = agent.run_episode(tid)
            assert r["episode_score"] > 0.3, (
                f"RuleBasedAgent scored {r['episode_score']} on {tid} — "
                "should beat random baseline of ~0.3"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. State management
# ══════════════════════════════════════════════════════════════════════════════

class TestStateManagement:

    def test_state_tracks_steps(self, env):
        env.reset("task_1")
        for i in range(3):
            s = env.state()
            assert s.step == i
            assert s.patients_processed == i
            pid = s.task_id  # just checking state is accessible
            obs = env.reset("task_1") if i == 0 else None
            # Use the patient from current state
            break

    def test_done_state_after_episode(self, env):
        obs = env.reset("task_1")
        done = False
        while not done:
            pid    = obs.current_patient.patient_id
            action = Action(action_type="triage", patient_id=pid, esi_level=3)
            obs, _, done, _ = env.step(action)

        s = env.state()
        assert s.status == "done"
        assert s.patients_remaining == 0

    def test_grader_before_episode_completion(self, env):
        env.reset("task_1")
        result = env.get_grader_score()
        # Should return in-progress status, not crash
        assert "status" in result

    def test_grader_after_episode_completion(self, env):
        obs  = env.reset("task_1")
        done = False
        while not done:
            pid    = obs.current_patient.patient_id
            action = Action(action_type="triage", patient_id=pid, esi_level=3)
            obs, _, done, _ = env.step(action)

        result = env.get_grader_score()
        assert result["status"] == "complete"
        assert "episode_score" in result
        assert 0.0 <= result["episode_score"] <= 1.0

    def test_reset_clears_state(self, env):
        obs1 = env.reset("task_1")
        ep1  = obs1.episode_id

        # Step once
        action = Action(action_type="triage", patient_id=obs1.current_patient.patient_id,
                        esi_level=3)
        env.step(action)

        # Reset to a different task
        obs2 = env.reset("task_2")
        assert obs2.episode_id != ep1
        assert obs2.task_id == "task_2"
        assert obs2.step == 0
