# API Access Instructions

Abhishek Pagariya | BITS ID 2025ae05133 | AIMLCZG523 Assignment 1

The API is deployed on a local Kubernetes cluster (Minikube) on the BITS Prayogshala virtual lab. It is not exposed on the public internet, so it is accessed with the steps below.

## Option 1: Kubernetes (Minikube)

```bash
git clone https://github.com/abhishekssig/heart-disease-mlops.git
cd heart-disease-mlops
bash scripts/setup_vm.sh
bash scripts/run_pipeline.sh
bash scripts/deploy_minikube.sh
```

| Access path | How |
|-------------|-----|
| NodePort | `minikube service heart-api -n heart --url` |
| LoadBalancer | run `minikube tunnel` in a second terminal, then `kubectl -n heart get svc heart-api` and use the EXTERNAL-IP |
| Ingress | `echo "$(minikube ip) heart-api.local" \| sudo tee -a /etc/hosts`, then `http://heart-api.local` |
| Prometheus | `http://$(minikube ip):30090` |
| Grafana | `http://$(minikube ip):30300` (dashboard "Heart Disease API") |

## Option 2: Docker only

```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```

API at `http://localhost:8000`, docs at `http://localhost:8000/docs`.

## Test requests

```bash
curl http://<HOST>/health
curl -X POST http://<HOST>/predict -H "Content-Type: application/json" -d @sample_input.json
curl -X POST http://<HOST>/predict/batch -H "Content-Type: application/json" -d @sample_batch.json
bash scripts/smoke_test.sh http://<HOST>
python scripts/load_test.py --url http://<HOST> -n 300
```

Expected `/predict` response for `sample_input.json`:

```json
{"prediction":1,"label":"Heart disease","probability_disease":0.8128,"confidence":0.8128}
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | readiness and liveness |
| GET | /model-info | model name, parameters, metrics |
| POST | /predict | one patient, returns prediction and confidence |
| POST | /predict/batch | up to 500 patients |
| GET | /metrics | Prometheus metrics |
| GET | /docs | interactive API documentation |

## Verified deployment (Prayogshala, 18 Oct 2026)

| Item | Value |
|------|-------|
| Cluster | Minikube 1.35, Kubernetes 1.32, Docker driver |
| LoadBalancer EXTERNAL-IP | 10.111.39.183 (via minikube tunnel) |
| Ingress host | heart-api.local -> 192.168.49.2 |
| Replicas | 2 baseline, HPA scaled to 4 under load |

Screenshots: `screenshots/12_k8s_endpoints_smoke_test.jpg`, `screenshots/13_k8s_resources_hpa_scaled_logs.jpg`.
