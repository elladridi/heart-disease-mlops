prepare:
	python main.py --step prepare

train:
	python main.py --step train

evaluate:
	python main.py --step evaluate

all:
	python main.py --step all

test:
	pytest -v

track:
	python main.py --step all --track --run-name v2-impute

track-v1:
	python main.py --step all --track --keep-zeros --run-name v1-raw-zeros

ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t heart-disease-mlops .

docker-run:
	mkdir -p models
	docker run --rm -v $(CURDIR)/models:/app/models heart-disease-mlops python main.py --step all --model-path models/model.joblib

api:
	uvicorn api:app --host 127.0.0.1 --port 8000

cv:
	python main.py --step cv

track-cv:
	python main.py --step cv --track --run-name cv-5fold

lint:
	flake8 .
	black --check .

format:
	black .

security:
	bandit -q -r model_pipeline.py main.py api.py check_model.py

gate:
	python check_model.py
