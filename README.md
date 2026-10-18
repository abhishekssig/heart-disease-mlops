# Heart Disease Risk - MLOps

AIMLCZG523 Assignment 1 - UCI Heart Disease (Cleveland)

## Setup

```bash
cd ~/workspace
git clone https://github.com/abhishekssig/heart-disease-mlops.git
cd heart-disease-mlops
bash scripts/setup_vm.sh
source .venv/bin/activate
```

## Pipeline

```bash
bash scripts/run_pipeline.sh
```

| Step | Command |
|------|---------|
| Data | `python -m heartml.data` |
| EDA | `python -m heartml.eda` |
| Train | `python -m heartml.train` |
| MLflow | `mlflow ui --backend-store-uri ./mlruns --port 5000` |
| Tests | `pytest --cov=heartml` |
| Lint | `flake8 heartml tests` |
| Predict | `python -m heartml.predict sample_input.json` |
| API | `uvicorn heartml.api:app --port 8000` |

## API

| Method | Path |
|--------|------|
| GET | `/health` |
| GET | `/model-info` |
| POST | `/predict` |
| POST | `/predict/batch` |
| GET | `/metrics` |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

## Docker

```bash
docker build -t heart-api:latest .
docker run --rm -p 8000:8000 heart-api:latest
bash scripts/smoke_test.sh http://localhost:8000
```

## Monitoring

```bash
docker compose up -d --build
python scripts/load_test.py -n 300
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Kubernetes

```bash
bash scripts/install_k8s_tools.sh
bash scripts/deploy_minikube.sh
minikube service heart-api -n heart --url
```

## CI/CD

`.github/workflows/ci.yml`: lint -> test -> train -> docker

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy |
|-------|-----------:|-------------:|--------------:|
| Logistic regression | 0.906 | 0.968 | 0.885 |
| Random forest | 0.892 | 0.960 | 0.902 |
| Gradient boosting | 0.883 | 0.959 | 0.902 |
