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

See the full README in the repository for complete documentation.

**Quick test:**
```
POST /reset?task_id=task_1
POST /step
GET  /state
GET  /tasks
POST /baseline
POST /grader
```

API docs: `https://<space-url>/docs`
