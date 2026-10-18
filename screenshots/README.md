# Screenshots

| File | What to capture |
|------|-----------------|
| pipeline_run.png | `bash scripts/run_pipeline.sh` output on the Prayogshala VM |
| eda_*.png | figures from `reports/figures/` (already generated) |
| mlflow_runs.png | MLflow UI experiment list showing the three model runs + children |
| mlflow_run_detail.png | one run: params, metrics, plots |
| ci_success.png | GitHub Actions run with all four jobs green |
| ci_artifacts.png | artefacts section of that run |
| ci_failure.png | a run failing on a broken test |
| docker_run.png | `docker build`, `docker run`, curl `/predict` |
| k8s_resources.png | `kubectl get pods,svc,ingress,hpa -n heart` |
| k8s_predict.png | `/predict` through the LoadBalancer or Ingress |
| prometheus_targets.png | Prometheus Status -> Targets |
| grafana.png | Grafana dashboard after `scripts/load_test.py` |
| api_docs.png | FastAPI `/docs` page |
