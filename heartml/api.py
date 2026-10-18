import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from heartml import __version__, config
from heartml.predict import HeartModel


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


def setup_logging():
    logger = logging.getLogger("heart_api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(JsonFormatter())
                logger.addHandler(fh)
            except OSError as exc:
                logger.warning("cannot open LOG_FILE %s: %s", log_file, exc)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


log = setup_logging()

REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram("heart_api_prediction_confidence", "Prediction confidence",
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
DISEASE_PROB = Histogram("heart_api_disease_probability", "Predicted probability of disease",
                         buckets=tuple(i / 10 for i in range(1, 11)))
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded")
ERRORS = Counter("heart_api_errors_total", "Unhandled errors", ["endpoint"])

KNOWN_PATHS = {"/", "/health", "/model-info", "/predict", "/predict/batch", "/metrics", "/docs", "/openapi.json"}


class Patient(BaseModel):
    age: float = Field(..., ge=18, le=100, examples=[57])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[4])
    trestbps: float = Field(..., ge=70, le=220, examples=[140])
    chol: float = Field(..., ge=100, le=600, examples=[241])
    fbs: int = Field(..., ge=0, le=1, examples=[0])
    restecg: int = Field(..., ge=0, le=2, examples=[0])
    thalach: float = Field(..., ge=60, le=220, examples=[123])
    exang: int = Field(..., ge=0, le=1, examples=[1])
    oldpeak: float = Field(..., ge=0, le=7, examples=[0.2])
    slope: int = Field(..., ge=1, le=3, examples=[2])
    ca: Optional[int] = Field(None, ge=0, le=3, examples=[0])
    thal: Optional[int] = Field(None, ge=3, le=7, examples=[7])


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float


class BatchRequest(BaseModel):
    patients: List[Patient] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    model: str
    results: List[Prediction]


state = {}


@asynccontextmanager
async def lifespan(app):
    path = os.getenv("MODEL_PATH", str(config.MODEL_FILE))
    try:
        state["model"] = HeartModel(path)
        MODEL_LOADED.set(1)
        log.info("model loaded", extra={"extra_fields": {"path": path, "model": state["model"].name}})
    except FileNotFoundError:
        MODEL_LOADED.set(0)
        log.error("model file not found", extra={"extra_fields": {"path": path}})
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    endpoint = request.url.path if request.url.path in KNOWN_PATHS else "other"
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ERRORS.labels(endpoint).inc()
        REQUESTS.labels(request.method, endpoint, "500").inc()
        log.exception("unhandled error", extra={"extra_fields": {"request_id": request_id, "path": endpoint}})
        raise
    elapsed = time.perf_counter() - start
    if endpoint != "/metrics":
        REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        log.info("request", extra={"extra_fields": {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
            "client": request.client.host if request.client else None,
        }})
    response.headers["x-request-id"] = request_id
    return response


def get_model():
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def record(results):
    for r in results:
        PREDICTIONS.labels(r["label"]).inc()
        CONFIDENCE.observe(r["confidence"])
        DISEASE_PROB.observe(r["probability_disease"])


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": get_model().name}


@app.get("/model-info")
def model_info():
    return get_model().metadata


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    result = get_model().predict([patient.model_dump()])[0]
    record([result])
    log.info("prediction", extra={"extra_fields": {"input": patient.model_dump(), **result}})
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    model = get_model()
    results = model.predict([p.model_dump() for p in batch.patients])
    record(results)
    return {"model": model.name, "results": results}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
