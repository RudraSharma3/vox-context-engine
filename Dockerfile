# Use a slim Python base image for fast builds and low footprint
FROM python:3.11-slim

# Force stdin/stdout streams to be unbuffered for instant Docker log flushing
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/.hf_cache

WORKDIR /app

# Install system dependencies needed for compiling certain packages if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the application structures into the workspace
COPY src/ ./src
COPY data/ ./data

EXPOSE 8000

# Start uvicorn server targeting application lifecycle lifespans
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]