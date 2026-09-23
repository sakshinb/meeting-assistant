import sys
import os
import gradio as gr
import spaces

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app as fastapi_app

# ZeroGPU requires exact @spaces.GPU syntax bound to a Gradio event listener
@spaces.GPU
def process_audio_zero_gpu(audio_file):
    """ZeroGPU registered GPU inference function."""
    if not audio_file:
        return "No audio file provided."
    from app.pipeline.runner import run_pipeline
    return str(run_pipeline(audio_file))


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

# Mount FastAPI routes onto Gradio's underlying app (Gradio stays the primary/root app,
# which is required for the ZeroGPU startup scanner to detect @spaces.GPU functions)
demo.app.mount("/api", fastapi_app)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)