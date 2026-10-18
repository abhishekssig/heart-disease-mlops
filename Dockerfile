FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt && \
    useradd --create-home --uid 1001 appuser

COPY heartml/ heartml/
COPY models/model.joblib models/metadata.json models/
COPY sample_input.json .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "heartml.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
