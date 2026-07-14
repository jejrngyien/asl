FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the model definition and the API service
COPY src/ src/
COPY api/ api/

# The C3D weights are downloaded from the GitHub Release on first startup
# (override with ASL_MODEL_URL / ASL_MODEL_PATH).
EXPOSE 8000
CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
