FROM python:3.10-slim

# Install system dependencies (ffmpeg, libsndfile1 for librosa/torchaudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete repository content
COPY --chown=user . $HOME/app

# Hugging Face Spaces default port
EXPOSE 7860

# Launch FastAPI app with Uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
