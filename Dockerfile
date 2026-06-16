FROM python:3.11-slim

WORKDIR /app

# Instala ferramentas de build necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de configuração
COPY pyproject.toml Makefile ./

# atualiza o pip e instala as ferramentas de build
RUN pip install --upgrade pip
RUN pip install setuptools wheel

# Instala dependências
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir psycopg2-binary 

# Copia o código fonte e bases de dados para reprodutibilidade
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY facape_mlflow.db ./facape_mlflow.db

# Configura o ambiente para o MLflow dentro do container
ENV MLFLOW_TRACKING_URI=sqlite:///app/facape_mlflow.db

# Cria o diretório de dados se não existir
RUN mkdir -p data models

EXPOSE 8000

# O comando de inicialização (roda API e Exporter em paralelo)
CMD python src/business_exporter.py & uvicorn src.api:app --host 0.0.0.0 --port 8000
