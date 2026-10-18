#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

[ -f models/model.joblib ] || { echo "models/model.joblib missing - run scripts/run_pipeline.sh first"; exit 1; }

minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=4096
minikube addons enable ingress
minikube addons enable metrics-server

eval "$(minikube docker-env)"
docker build -t heart-api:latest .
eval "$(minikube docker-env -u)"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml || echo "ingress controller not ready yet - re-run: kubectl apply -f k8s/ingress.yaml"

kubectl -n heart create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/heart-api.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/

kubectl -n heart rollout status deployment/heart-api --timeout=180s
kubectl -n heart get pods,svc,ingress,hpa -o wide

echo
echo "API via NodePort tunnel:  minikube service heart-api -n heart --url"
echo "LoadBalancer IP:          run 'minikube tunnel' in another terminal, then: kubectl -n heart get svc heart-api"
echo "Ingress:                  echo \"\$(minikube ip) heart-api.local\" | sudo tee -a /etc/hosts; curl http://heart-api.local/health"
echo "Prometheus:               $(minikube ip):30090"
echo "Grafana:                  $(minikube ip):30300"
