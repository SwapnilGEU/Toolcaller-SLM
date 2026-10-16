# serverv2 gateway: guardrails + LangGraph routing + Langfuse tracing.
# The model itself is served separately (vLLM / Ollama); point VLLM_URL / OLLAMA_URL at it.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY serverv2/ serverv2/

RUN useradd --create-home app
USER app

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')"

# e.g. docker run -p 8100:8100 -e VLLM_URL=http://host.docker.internal:8000/v1 --env-file .env <image>
CMD ["uvicorn", "serverv2.gateway:app", "--host", "0.0.0.0", "--port", "8100"]
