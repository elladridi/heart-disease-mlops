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
