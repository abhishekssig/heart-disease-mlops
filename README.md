# Heart Disease Risk - MLOps Pipeline

AIMLCZG523 MLOps Assignment 1. Binary classifier on the UCI Heart Disease (Cleveland) dataset, tracked with MLflow, tested and trained in GitHub Actions, served by FastAPI in Docker, deployed to Kubernetes (Minikube) and monitored with Prometheus + Grafana.

## Layout

```
heartml/            package: config, data, eda, features, train, predict, api
tests/              pytest unit tests (data, features, model, api)
notebooks/          EDA + training notebook (uses the package)
data/processed/     cleaned dataset (raw is downloaded by heartml.data)
models/             trained pipeline (model.joblib) + metadata.json
reports/            EDA figures, eda_summary.json, model_comparison.csv
k8s/                namespace, deployment, service (LoadBalancer), ingress, hpa, monitoring/
monitoring/         prometheus.yml, alerts.yml, grafana provisioning + dashboard
scripts/            setup_vm.sh, run_pipeline.sh, deploy_minikube.sh, smoke_test.sh, load_test.py
.github/workflows/  ci.yml (lint -> test -> train -> docker build + run)
report/             written report
screenshots/        screenshots used in the report
```

## Quick start (Linux / Prayogshala VM)

```bash
cd ~/workspace
git clone https://github.com/<your-user>/heart-disease-mlops.git
cd heart-disease-mlops
bash scripts/setup_vm.sh
source .venv/bin/activate
bash scripts/run_pipeline.sh
```

`run_pipeline.sh` runs lint, tests, download + cleaning, EDA, training (3 models, grid search, 5-fold CV) and a batch prediction.

Prayogshala notes: only `~/workspace` survives a VM stop, sessions idle out after 15 minutes, and a stopped VM is deleted after 60 minutes. Push to GitHub often.

## Step by step

| Step | Command |
|------|---------|
| Download + clean | `python -m heartml.data` |
| EDA figures | `python -m heartml.eda` |
| Train + track | `python -m heartml.train` (`--quick` for a fast smoke run) |
| MLflow UI | `mlflow ui --backend-store-uri ./mlruns --port 5000` |
| Tests | `pytest --cov=heartml` |
| Lint | `flake8 heartml tests` |
| CLI inference | `python -m heartml.predict sample_input.json` |
| Local API | `uvicorn heartml.api:app --port 8000` then open `/docs` |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | liveness/readiness, 503 if no model |
| GET | `/model-info` | model name, params, metrics |
| POST | `/predict` | one patient -> prediction, label, probability, confidence |
| POST | `/predict/batch` | `{"patients": [...]}` up to 500 |
| GET | `/metrics` | Prometheus metrics |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
{"prediction":1,"label":"Heart disease","probability_disease":0.8128,"confidence":0.8128}
```

Every request is logged as one JSON line (request id, path, status, latency); set `LOG_FILE` to also write to a file.

## Docker

```bash
docker build -t heart-api:latest .
docker run --rm -p 8000:8000 heart-api:latest
bash scripts/smoke_test.sh http://localhost:8000
```

The image only contains the serving dependencies (`requirements-serve.txt`), runs as a non-root user and has a HEALTHCHECK.

## Monitoring with Docker Compose

```bash
docker compose up -d --build
python scripts/load_test.py -n 300
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (dashboard "Heart Disease API", admin/admin)

## Kubernetes (Minikube)

```bash
bash scripts/install_k8s_tools.sh
bash scripts/deploy_minikube.sh
minikube service heart-api -n heart --url
bash scripts/smoke_test.sh <url-from-above>
```

LoadBalancer: run `minikube tunnel` in a second terminal, then `kubectl -n heart get svc heart-api` shows an external IP.
Ingress: add `$(minikube ip) heart-api.local` to `/etc/hosts` and call `http://heart-api.local/health`.
Prometheus and Grafana are exposed on NodePorts 30090 and 30300.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. **lint** - flake8, report uploaded
2. **test** - pytest with JUnit + coverage reports uploaded
3. **train** - download, EDA, full training, quality gate (test ROC-AUC >= 0.80), inference smoke test; uploads model, mlruns, figures, logs
4. **docker** - builds the image with the trained model, runs it, hits `/health`, `/predict`, `/metrics`; uploads container logs and the image

Any failing step fails the workflow.

## Results (seed 42, 80/20 stratified split)

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Test recall |
|-------|-----------:|-------------:|--------------:|------------:|
| Logistic regression (selected) | 0.906 | 0.968 | 0.885 | 0.929 |
| Random forest | 0.892 | 0.960 | 0.902 | 0.929 |
| Gradient boosting | 0.883 | 0.959 | 0.902 | 0.929 |

Selection is by mean 5-fold CV ROC-AUC on the training split, not the test set.
