.PHONY: install load fetch train anomaly cluster api mlflow-ui docker-build docker-run up down status logs

install:
	pip install -e ".[dev]"
	pip install mcp-brasil

fetch:
	python data/ingest_mcp.py 2021 2022 2023 2024 2025

load:
	python data/load_data.py

train:
	python src/train.py

anomaly:
	python src/anomaly.py

cluster:
	python src/cluster.py

all-models: load train anomaly cluster

# Inicia toda a stack (API + Prometheus + Grafana + DB)
up: docker-build
	docker-compose up -d
	@echo "-------------------------------------------------------"
	@echo "Stack iniciada!"
	@echo "API: http://localhost:8000/docs"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "-------------------------------------------------------"

# Para todos os serviços
down:
	docker-compose down

# Verifica o status dos containers
status:
	docker-compose ps

# Acompanha os logs
logs:
	docker-compose logs -f

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	docker run -p 5000:5000 facape-despesas mlflow ui --host 0.0.0.0

docker-build:
	docker build -t facape-despesas .

docker-run:
	docker run -p 8000:8000 facape-despesas
