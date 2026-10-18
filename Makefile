.PHONY: setup lint test data eda train pipeline serve docker-build docker-run compose-up compose-down deploy smoke load mlflow

setup:
	bash scripts/setup_vm.sh

lint:
	flake8 heartml tests

test:
	pytest --cov=heartml --cov-report=term-missing

data:
	python -m heartml.data

eda:
	python -m heartml.eda

train:
	python -m heartml.train

pipeline:
	bash scripts/run_pipeline.sh

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

serve:
	uvicorn heartml.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-api:latest .

docker-run:
	docker run --rm -d --name heart-api -p 8000:8000 heart-api:latest

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

deploy:
	bash scripts/deploy_minikube.sh

smoke:
	bash scripts/smoke_test.sh $(URL)

load:
	python scripts/load_test.py --url $(or $(URL),http://localhost:8000)
