import sys
import os
import gradio as gr

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app as fastapi_app

# Create Gradio UI
with gr.Blocks(title="Enterprise AI Meeting Assistant API") as demo:
    gr.Markdown(
        """
        # 🎙️ Enterprise AI Meeting Assistant API
        
        The FastAPI backend server is **Live and Running** on Hugging Face Spaces.
        
        ### Available API Endpoints:
        - `GET  /api/health` — System health check & CUDA status
        - `POST /api/meetings/upload` — Upload meeting audio for async processing
        - `GET  /api/jobs/{id}` — Poll pipeline job status
        - `POST /api/meetings/{id}/access` — Authenticate for meeting access
        - `GET  /api/meetings/{id}` — Fetch full meeting analysis
        - `GET  /api/models/evaluation` — Return neural model metrics & evaluation
        """
    )

# Mount FastAPI app onto Gradio. `app` is the primary ASGI app.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
