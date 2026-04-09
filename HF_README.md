---
title: MedTriageEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - healthcare
  - triage
  - reinforcement-learning
  - clinical-decision-support
  - emergency-medicine
  - real-world
pinned: false
license: mit
short_description: "Emergency Department triage & resource allocation — OpenEnv real-world environment"
---

# MedTriageEnv on Hugging Face Spaces

This repository is compliant with the **OpenEnv Hackathon Pre-Submission Checklist**.

## Evaluation Script

The primary evaluation entry point is `inference.py` in the root directory. 
It conforms to the structured JSON stdout rules (`[START]`, `[STEP]`, `[END]`) and utilizes the `openai` Python SDK.

### Mandatory Environment Variables

To run the LLM inference correctly, you **must** supply these environment variables:

- `API_BASE_URL`: The API endpoint for the LLM (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME`: The model identifier to use (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` (or `API_KEY`): Your Hugging Face / API key

## Local & Space API Endpoints

**Quick test:**
```text
POST /reset?task_id=task_1
POST /step
GET  /state
GET  /tasks
POST /baseline
POST /grader
```

API docs: `https://<space-url>/docs`
See the full `README.md` for complete documentation on clinical schemas and local usage.
