# =========================================================================
# Hugging Face Spaces entry point (ZeroGPU)
#
# IMPORTANT: `import spaces` must be the very first import in this file,
# before sys/os/uvicorn/gradio and especially before anything that could
# pull in torch. ZeroGPU patches CUDA interception at import time — if
# torch gets touched first (even indirectly), GPU allocation and/or the
# startup detection of @spaces.GPU functions can silently break.
# =========================================================================
import spaces

import sys
import os
import uvicorn
import gradio as gr

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app as fastapi_app


# -------------------------------------------------------------------------
# Single shared GPU entry point.
#
# Every code path that needs the GPU — the demo button below AND your real
# FastAPI /api/meetings/upload route — must call through a function
# decorated with @spaces.GPU. ZeroGPU only allocates a GPU time-slice to
# calls that go through a decorated function; a plain FastAPI handler that
# calls run_pipeline() directly, without this wrapper, will never get GPU
# access, even once the Space itself starts successfully.
#
# duration=120 gives headroom for AE/VAE/GAN/Diffusion + Whisper + FLAN-T5
# to run in one call — the default is 60s, which a full pipeline pass can
# exceed. Raise this further if you still see timeouts.
# -------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_pipeline_gpu(audio_path: str):
    """The one GPU-decorated call every caller (Gradio UI and FastAPI) goes through."""
    from app.pipeline.runner import run_pipeline
    return run_pipeline(audio_path)


def process_audio_zero_gpu(audio_file):
    """Gradio-facing wrapper — this is what makes ZeroGPU's startup scanner
    detect the Space, since it's wired to a real event listener (btn.click)."""
    if not audio_file:
        return "No audio file provided."
    return str(run_pipeline_gpu(audio_file))


# -------------------------------------------------------------------------
# NOTE: update your FastAPI route (in app/main.py or wherever
# /api/meetings/upload lives) to call `run_pipeline_gpu(audio_path)` from
# this module instead of importing `run_pipeline` directly, e.g.:
#
#     from app import run_pipeline_gpu
#     result = run_pipeline_gpu(saved_audio_path)
#
# Otherwise the HTTP endpoint your frontend actually calls will never
# receive a GPU allocation, even after this startup error is fixed.
# -------------------------------------------------------------------------


# Create Gradio UI
with gr.Blocks(title="Enterprise AI Meeting Assistant API") as demo:
    gr.Markdown(
        """
        # 🎙️ Enterprise AI Meeting Assistant API

        The FastAPI backend server is **Live and Running** on Hugging Face Spaces (ZeroGPU).

        ### Available API Endpoints (mounted at `/api`):
        - `GET  /api/health` — System health check & CUDA status
        - `POST /api/meetings/upload` — Upload meeting audio for async processing
        - `GET  /api/jobs/{id}` — Poll pipeline job status
        - `POST /api/meetings/{id}/access` — Authenticate for meeting access
        - `GET  /api/meetings/{id}` — Fetch full meeting analysis
        - `GET  /api/models/evaluation` — Return neural model metrics & evaluation
        """
    )

    with gr.Accordion("Test GPU Pipeline", open=False):
        audio_in = gr.Audio(type="filepath", label="Upload Meeting Audio")
        btn = gr.Button("Process Audio on ZeroGPU", variant="primary")
        out_text = gr.Textbox(label="Result", lines=5)

        btn.click(fn=process_audio_zero_gpu, inputs=[audio_in], outputs=[out_text])

# Mount FastAPI app onto Gradio. `app` is the primary ASGI app.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)