---
title: Enterprise AI Meeting Assistant Backend
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Enterprise AI Meeting Assistant — FastAPI Backend

FastAPI backend API powering neural audio representation, Whisper ASR, and FLAN-T5 LLM summarization.

## Endpoints

- `GET  /api/health` — System health check & CUDA status
- `POST /api/meetings/upload` — Upload meeting audio for async processing
- `GET  /api/jobs/{id}` — Poll pipeline job status
- `POST /api/meetings/{id}/access` — Authenticate for meeting access
- `GET  /api/meetings/{id}` — Fetch full meeting analysis
- `GET  /api/models/evaluation` — Return neural model metrics & evaluation
