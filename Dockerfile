FROM python:3.11-slim

WORKDIR /app

# Instala dependências
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copia código e dados
COPY data/ ./data/
COPY src/ ./src/
COPY despesas*.JSON ./

# Treina o modelo durante o build
RUN python src/train.py

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
