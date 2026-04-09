import os
import json
import time

from openai import OpenAI

from environment.env import MedTriageEnvironment
from environment.models import Action
from baseline.baseline_agent import SYSTEM_PROMPT, _format_patient

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("MEDTRIAGE_TASK", "task_1")
BENCHMARK = os.getenv("MEDTRIAGE_BENCHMARK", "medtriage-env")
TEMPERATURE = 0.0

client = OpenAI(
    api_key=API_KEY if API_KEY else "dummy-key",
    base_url=API_BASE_URL
)

def fallback_action(task_id: str, patient_id: str) -> Action:
    if task_id == "task_1":
        return Action(action_type="triage", patient_id=patient_id, esi_level=3)
    if task_id == "task_2":
        return Action(action_type="order_diagnostics", patient_id=patient_id, diagnostics=["cbc", "bmp", "ecg"])
    return Action(action_type="allocate_resources", patient_id=patient_id, esi_level=3, bed_type="general")

def main():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    env = MedTriageEnvironment()
    obs = env.reset(task_id=TASK_NAME)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
    
    done = False
    step_count = 0
    rewards = []
    
    while not done:
        step_count += 1
        patient = obs_dict.get("current_patient") or {}
        pid = patient.get("patient_id", "unknown")
        dept = obs_dict.get("department_status")
        
        patient_text = _format_patient(patient)
        
        if TASK_NAME == "task_1":
            prompt = (
                f"Assign the ESI triage level for this patient.\n\n"
                f"{patient_text}\n\n"
                f"Respond with JSON: {{\"action_type\": \"triage\", "
                f"\"patient_id\": \"{pid}\", \"esi_level\": <int 1-5>, "
                f"\"rationale\": \"<brief reasoning>\"}}"
            )
        elif TASK_NAME == "task_2":
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

        raw_action = "{}"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=512
            )
            raw_action = response.choices[0].message.content
            # Clean possible markdown block
            raw_action = raw_action.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            pass
            
        action_data = {}
        try:
            action_data = json.loads(raw_action)
        except json.JSONDecodeError:
            pass
            
        action_data["action_type"] = action_data.get("action_type") or ("triage" if TASK_NAME == "task_1" else "order_diagnostics" if TASK_NAME == "task_2" else "allocate_resources")
        action_data["patient_id"] = pid
        
        action = None
        error_msg = "null"
        try:
            action = Action(**action_data)
        except Exception as e:
            action = fallback_action(TASK_NAME, pid)
            error_msg = f"\"{str(e)}\"".replace("\n", " ").replace("\"", "'")

        action_str = json.dumps(action.model_dump()).replace(" ", "").replace("\"", "'")
            
        obs_model, reward_model, done, info = env.step(action)
        obs_dict = obs_model.model_dump() if hasattr(obs_model, "model_dump") else obs_model
        
        reward_val = getattr(reward_model, "score", 0.0)
        rewards.append(reward_val)
        
        done_str = "true" if done else "false"
        
        print(f"[STEP] step={step_count} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}", flush=True)

    avg_score = sum(rewards) / len(rewards) if rewards else 0.0
    success_str = "true" if done else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    print(f"[END] success={success_str} steps={step_count} score={avg_score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()
