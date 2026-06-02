.PHONY: install load train anomaly cluster api mlflow-ui docker-build docker-run

install:
	pip install -e ".[dev]"

load:
	python data/load_data.py

train:
	python src/train.py

anomaly:
	python src/anomaly.py

cluster:
	python src/cluster.py

all-models: load train anomaly cluster

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui

docker-build:
	docker build -t facape-despesas .

docker-run:
	docker run -p 8000:8000 facape-despesas
