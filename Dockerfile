FROM python:3.11-slim

WORKDIR /app

# 1. Instala dependências (Boa prática mantida!)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# 2. Copia apenas o código da API e o modelo já treinado
COPY src/api.py ./src/
# O modelo treinado (.pkl, .joblib, etc) deve ser gerado antes e copiado, 
# ou idealmente montado via volume.
COPY models/model.pkl ./models/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]