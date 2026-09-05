# ==============================================================================
# ADAPTIVE-DEFENCE ANC: Master Edge & Evaluation Container
# DRDO Smart India Hackathon 2026 — Problem Statement ID: 26052
# ==============================================================================
FROM python:3.10-slim-bullseye

LABEL maintainer="DRDO SIH 2026 Adaptive ANC Team <khanjunaidtmsl2024-bot>"
LABEL description="Authoritative Hub for Hybrid AI-DSP Adaptive Noise Cancellation"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system audio dependencies (ALSA, libsndfile, portaudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libportaudio2 \
    libasound2 \
    libasound2-plugins \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Expose Web Audio Demo port
EXPOSE 8000

# Default entrypoint runs the 20-experiment validation suite
CMD ["python", "main.py", "experiments"]
