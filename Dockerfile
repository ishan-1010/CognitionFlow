# CognitionFlow API (Python 3.12 for onnxruntime compatibility)
FROM python:3.12-slim

WORKDIR /app

# Install torch CPU-only first (much smaller, no CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

ENV PYTHONPATH=/app/src
ENV COGNITIONFLOW_WORKSPACE=/app/project_workspace

EXPOSE 8000
ENV PORT=8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
